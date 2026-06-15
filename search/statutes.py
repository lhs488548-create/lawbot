"""Unified statutes + precedent search (lawbot.org ``POST /v1/statutes/search``).

This module implements the **core, generation-free data-infra surface** of the
contract (``_BUILD_CONTRACT.md`` §(d) → *Statutes search*, aligned to
``분석/09_청킹_임베딩_헤더_설계.md`` §A). It sits *on top of* the dense retriever
(:mod:`search.retriever`) and turns raw vector hits into clean, auditable result
rows that any LLM (or front-end) can consume directly:

* **No generation cost.** Pure retrieval — embed the query once, run a Qdrant
  similarity search, format the rows. The only paid call is the single query
  embedding (a few tokens); there is no GPT call here.
* **Unified law + precedent** in one ranked list, with article/section
  precision (``article_no`` carries "제4조" for statutes or "판결요지" for
  precedents).
* **Common response meta on every row** (09 §A):
  ``{trust_grade, source_url, license, as_of_date, effective_from}`` so callers
  can audit provenance and licensing without a second request.
* **Point-in-time (`as_of_date`) filtering** (09 §A/E): only rows whose
  ``effective_from <= as_of_date`` are returned, modelling "what was the current
  law on this date". The date cut is delegated to :func:`search.retriever.search`
  (the single source of truth for point-in-time semantics); this module only
  applies the same rule itself as a fallback for an older retriever signature.
  Per the retriever's safe-default policy, a row whose ``effective_from`` is
  missing/blank or later than ``as_of_date`` is **excluded** (we never present a
  row as the law "as of" a date without proving it was in force then).
* **Pagination** (``offset`` / ``k``) for large result sets, with a stable
  ordering (descending score, ``doc_id`` as a deterministic tie-breaker).

Design contract::

    def statutes_search(query, k=config.DEFAULT_TOP_K, filter=None,
                        as_of_date=None) -> list[dict]

Each returned ``dict`` is::

    {
      "doc_id":        "LAW:014565:법률",          # owning document id
      "doc_type":      "law",                       # law|ordinance|admrule|precedent
      "title":         "민법",                      # 법령명 / 사건명
      "article_no":    "제4조",                     # 조/항/호 · 판례 섹션
      "score":         0.83,                        # cosine similarity
      "text":          "[민법 제4조 …] …",          # the embedded chunk text
      # --- common meta (09 §A) ---
      "trust_grade":   "A",                         # A=원문있음, B=메타만
      "source_url":    "https://law.go.kr/...",
      "license":       "<config.DEFAULT_LICENSE>",
      "as_of_date":    "2026-06-15" | None,         # echoes the request as_of
      "effective_from":"2024-07-03" | None,
    }

This module owns only result shaping; it delegates retrieval to
:func:`search.retriever.search` (the single, shared retriever) and never calls a
generation model.

Run an offline self-check (in-memory Qdrant, no OpenAI/cloud)::

    cd /home/user1/lawbot && .venv/bin/python -m search.statutes --selftest
"""

from __future__ import annotations

import argparse
import inspect
import sys
from typing import Any, Final, Mapping

import config
from search import retriever

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
# Hard upper bound on a single page so a caller cannot ask for, say, 10_000 rows
# and force a huge Qdrant scan + payload transfer. Pagination beyond this is via
# repeated ``offset`` requests.
MAX_PAGE_SIZE: Final[int] = 100

# When ``as_of_date`` is given we must over-fetch from Qdrant, because some of
# the top hits may be filtered out for being not-yet-effective. We grow the
# fetch window by this multiplicative factor (and the additive floor below) so
# that, in the common case, a single retrieval still fills the requested page.
_ASOF_OVERFETCH_FACTOR: Final[int] = 4
_ASOF_OVERFETCH_FLOOR: Final[int] = 20

def _retriever_supports_as_of() -> bool:
    """Whether the shared retriever's ``search`` already accepts ``as_of_date``.

    The contract allows :func:`search.retriever.search` to gain an ``as_of_date``
    keyword additively. When it has one we forward the date straight through;
    otherwise we apply the point-in-time filter here. This is resolved **per
    call** (not frozen at import) so the behaviour tracks the *current* retriever
    — important when the function is monkeypatched in tests or swapped at runtime.

    Returns:
        ``True`` if ``retriever.search`` exposes an ``as_of_date`` parameter.
    """
    try:
        return "as_of_date" in inspect.signature(retriever.search).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False


# --------------------------------------------------------------------------- #
# Helpers (pure — unit-tested without any network)                            #
# --------------------------------------------------------------------------- #
def _payload_of(hit: Any) -> dict[str, Any]:
    """Return a hit's payload as a plain dict (never ``None``)."""
    payload = getattr(hit, "payload", None)
    return dict(payload) if payload else {}


def _doc_id_of(hit: Any, payload: Mapping[str, Any]) -> str:
    """Best-effort owning-document id for a retrieved child chunk.

    The retriever's point id is a UUID5 (good for citation ``source_id`` but not
    human-readable), so we prefer the stable ``doc_id``/``parent_id`` carried in
    the payload, then fall back to deriving it from ``chunk_id`` (everything
    before the first ``'#'``), and finally to the raw point id.

    Args:
        hit: A retrieved hit exposing ``.id`` and ``.payload``.
        payload: The hit's payload dict.

    Returns:
        The owning ``doc_id`` string (e.g. ``"LAW:014565:법률"``), or the point
        id as a last resort.
    """
    for key in ("doc_id", "parent_id"):
        value = payload.get(key)
        if value:
            return str(value)
    chunk_id = payload.get("chunk_id")
    if chunk_id:
        # chunk_id == "{doc_id}#{article_no}#{part_idx}" -> take the doc_id part.
        return str(chunk_id).split("#", 1)[0]
    return str(getattr(hit, "id", "") or "")


def _as_str_or_none(value: Any) -> str | None:
    """Coerce a payload value to a non-empty trimmed string, else ``None``."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _is_effective_on(effective_from: str | None, as_of_date: str) -> bool:
    """Point-in-time predicate: was a row effective on/before ``as_of_date``?

    Dates are compared as ISO-ish strings (``YYYY-MM-DD``), which sort
    lexicographically in the same order as chronologically.

    This is only the **fallback** used when the shared retriever does not itself
    apply ``as_of_date`` (older signature / a monkeypatched stub). It mirrors the
    retriever's authoritative policy exactly: a row whose ``effective_from`` is
    missing/blank or later than ``as_of_date`` is **excluded** — the safe default
    for a current-law query (we never present a row as the law "as of" a date
    when we cannot prove it was in force then).

    Args:
        effective_from: The row's enforcement/decision date, or ``None``.
        as_of_date: The requested point-in-time date (``YYYY-MM-DD``).

    Returns:
        ``True`` only if ``effective_from`` is known and ``<= as_of_date``.
    """
    if not effective_from:
        return False
    # Compare on the leading 10 chars so a stray time component never trips us.
    return effective_from[:10] <= as_of_date[:10]


def _format_row(hit: Any, as_of_date: str | None) -> dict[str, Any]:
    """Shape a single retriever :class:`Hit` into a public result row.

    The row carries the search essentials plus the **common meta** mandated by
    09 §A. Missing optional fields are filled with sensible, honest defaults
    (``license`` from :data:`config.DEFAULT_LICENSE`; ``trust_grade`` "A").

    Args:
        hit: A retrieved hit (``.id``, ``.score``, ``.payload``).
        as_of_date: The requested as-of date, echoed back on every row, or
            ``None``.

    Returns:
        The public result dict described in the module docstring.
    """
    payload = _payload_of(hit)
    license_value = _as_str_or_none(payload.get("license")) or config.DEFAULT_LICENSE
    score = getattr(hit, "score", None)
    return {
        "doc_id": _doc_id_of(hit, payload),
        "doc_type": _as_str_or_none(payload.get("doc_type")),
        "title": _as_str_or_none(payload.get("title")),
        "article_no": _as_str_or_none(payload.get("article_no")),
        "score": float(score) if score is not None else 0.0,
        "text": str(payload.get("text", "")),
        # --- common response meta (09 §A) ----------------------------------- #
        "trust_grade": str(payload.get("trust_grade") or "A"),
        "source_url": _as_str_or_none(payload.get("source_url")),
        "license": license_value,
        "as_of_date": as_of_date,
        "effective_from": _as_str_or_none(payload.get("effective_from")),
    }


def _validate_as_of_date(as_of_date: str | None) -> str | None:
    """Validate the optional ``as_of_date`` is an ISO ``YYYY-MM-DD`` string.

    Args:
        as_of_date: The requested date, or ``None``.

    Returns:
        The normalized date string, or ``None`` if not supplied.

    Raises:
        ValueError: If a non-empty value is not a valid ``YYYY-MM-DD`` date.
    """
    if as_of_date is None:
        return None
    s = str(as_of_date).strip()
    if not s:
        return None
    # date.fromisoformat accepts exactly YYYY-MM-DD (plus full datetimes); we
    # want a calendar date, so validate strictly on the 10-char prefix.
    from datetime import date

    try:
        date.fromisoformat(s[:10])
    except ValueError as exc:
        raise ValueError(
            f"as_of_date must be an ISO date 'YYYY-MM-DD'; got {as_of_date!r}."
        ) from exc
    return s[:10]


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def statutes_search(
    query: str,
    k: int = config.DEFAULT_TOP_K,
    filter: dict[str, Any] | None = None,  # noqa: A002 - matches public contract
    as_of_date: str | None = None,
    *,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Unified law + precedent search with article/section precision.

    This is the implementation behind ``POST /v1/statutes/search``. It performs
    a single dense retrieval (no generation), applies the optional metadata
    pre-filter and point-in-time ``as_of_date`` constraint, paginates the
    result, and returns rows enriched with the common provenance meta.

    Args:
        query: The natural-language search query. Must be non-empty.
        k: Page size — the maximum number of rows to return. Coerced into
            ``[1, MAX_PAGE_SIZE]``.
        filter: Optional flat metadata pre-filter AND-ed into the search, e.g.
            ``{"doc_type": "law"}`` or ``{"jurisdiction": "전라남도"}`` or
            ``{"law_kind": ["법률", "시행령"]}``. Keys must be among the
            retriever's indexed filter keys (``doc_type``, ``jurisdiction``,
            ``law_kind``, ``effective_from``); unknown keys raise ``ValueError``.
        as_of_date: Optional ISO ``YYYY-MM-DD`` point-in-time date. Only rows
            whose ``effective_from <= as_of_date`` are returned; rows whose
            ``effective_from`` is missing or later are excluded (the retriever's
            safe current-law policy). Echoed back on every row.
        offset: Zero-based pagination offset (number of leading rows to skip).
            Coerced to ``>= 0``.

    Returns:
        A list of result dicts (see the module docstring), ordered by descending
        similarity score with ``doc_id`` as a deterministic tie-breaker. Empty
        if nothing matches.

    Raises:
        ValueError: For a blank ``query``, an invalid ``filter`` (propagated from
            the retriever), a malformed ``as_of_date``, or a negative ``offset``.
    """
    if not (query or "").strip():
        raise ValueError("query must be a non-empty string")

    page_size = max(1, min(int(k), MAX_PAGE_SIZE))
    offset = int(offset)
    if offset < 0:
        raise ValueError("offset must be >= 0")

    as_of = _validate_as_of_date(as_of_date)

    # How many raw hits we need from the retriever to satisfy this page.
    #   * We always need to fetch past the requested offset.
    #   * When as_of filtering happens *here* (retriever lacks the kwarg), some
    #     fetched rows will be dropped, so we over-fetch to still fill the page.
    retriever_as_of = _retriever_supports_as_of()
    filter_here = as_of is not None and not retriever_as_of

    needed = offset + page_size
    if filter_here:
        fetch_k = max(needed * _ASOF_OVERFETCH_FACTOR, needed + _ASOF_OVERFETCH_FLOOR)
    else:
        fetch_k = needed

    # Delegate retrieval to the single shared retriever. Pass ``as_of_date``
    # through only if that retriever already understands it (forward-compatible
    # with a future revision); otherwise filter here.
    if retriever_as_of:
        hits = retriever.search(query, k=fetch_k, flt=filter, as_of_date=as_of)
    else:
        hits = retriever.search(query, k=fetch_k, flt=filter)

    rows = [_format_row(hit, as_of) for hit in hits]

    # Point-in-time filter (only if the retriever did not already do it).
    if filter_here:
        rows = [r for r in rows if _is_effective_on(r["effective_from"], as_of)]

    # Stable ordering: retriever already returns descending score, but make the
    # tie-break deterministic so pagination is consistent across requests.
    rows.sort(key=lambda r: (-r["score"], r["doc_id"]))

    return rows[offset : offset + page_size]


# --------------------------------------------------------------------------- #
# Offline self-test (in-memory Qdrant, faked embeddings — no OpenAI/cloud)    #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Exercise shaping, as_of filtering and pagination end to end.

    Builds an in-memory Qdrant collection, stubs the query embedding (so no
    OpenAI call is made), and asserts the public behaviours of
    :func:`statutes_search`. Returns a process exit code (0 == pass).
    """
    import uuid

    from qdrant_client import QdrantClient, models

    collection = config.COLLECTION
    dim = 4
    qc = QdrantClient(location=":memory:")
    qc.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    # Four rows: a law (effective 2013), a precedent (2020), an ordinance
    # effective in the *future* (2030 — excluded by an as_of in 2025), and a law
    # with NO effective_from (also excluded under as_of: the retriever's safe
    # current-law policy drops rows it cannot prove were in force).
    fixtures = [
        {
            "vec": [1.0, 0.0, 0.0, 0.0],
            "payload": {
                "chunk_id": "LAW:000001:법률#제4조#0",
                "doc_id": "LAW:000001:법률",
                "parent_id": "LAW:000001:법률",
                "text": "[민법 제4조 성년] 사람은 19세로 성년에 이르게 된다.",
                "doc_type": "law",
                "title": "민법",
                "jurisdiction": "국가",
                "law_kind": "법률",
                "article_no": "제4조",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "trust_grade": "A",
            },
        },
        {
            "vec": [0.92, 0.39, 0.0, 0.0],
            "payload": {
                "chunk_id": "PREC:424370#판결요지#0",
                "doc_id": "PREC:424370",
                "text": "[대법원 2020다12345 판결요지] 손해배상 책임 인정.",
                "doc_type": "precedent",
                "title": "손해배상청구",
                "jurisdiction": "대법원",
                "law_kind": "민사",
                "article_no": "판결요지",
                "effective_from": "2020-05-14",
                "source_url": "https://law.go.kr/prec",
                "trust_grade": "A",
            },
        },
        {
            "vec": [0.85, 0.0, 0.53, 0.0],
            "payload": {
                "chunk_id": "ORD:전라남도:2200001#제2조#0",
                "doc_id": "ORD:전라남도:2200001",
                "text": "[전라남도 미래 조례 제2조] 2030년 시행 정의 규정.",
                "doc_type": "ordinance",
                "title": "전라남도 미래 조례",
                "jurisdiction": "전라남도",
                "law_kind": "조례",
                "article_no": "제2조",
                "effective_from": "2030-01-01",
                "source_url": "https://law.go.kr/ord",
                "trust_grade": "A",
                # No "license" key on purpose -> must default to DEFAULT_LICENSE.
            },
        },
        {
            "vec": [0.80, 0.0, 0.0, 0.60],
            "payload": {
                "chunk_id": "LAW:000002:법률#제1조#0",
                "doc_id": "LAW:000002:법률",
                "text": "[연혁미상법 제1조] 시행일 메타 결측 조문.",
                "doc_type": "law",
                "title": "연혁미상법",
                "jurisdiction": "국가",
                "law_kind": "법률",
                "article_no": "제1조",
                # effective_from intentionally absent.
                "source_url": "https://law.go.kr/none",
                "trust_grade": "B",
            },
        },
    ]
    qc.upsert(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, fx["payload"]["chunk_id"])),
                vector=fx["vec"],
                payload=fx["payload"],
            )
            for fx in fixtures
        ],
    )

    # Wire the retriever to this in-memory client and stub its query embedding so
    # no OpenAI call happens. The stub returns the "law" axis, so the 민법 row
    # ranks first.
    retriever.set_clients(qdrant_client=qc)
    real_embed = retriever.embed_query

    def _fake_embed(_query: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    retriever.embed_query = _fake_embed  # type: ignore[assignment]
    try:
        failures: list[str] = []

        # 1) Basic search: rows carry the full common-meta surface.
        rows = statutes_search("성년 나이", k=4)
        if not rows:
            failures.append("basic search returned no rows")
        else:
            top = rows[0]
            required = {
                "doc_id",
                "doc_type",
                "title",
                "article_no",
                "score",
                "text",
                "trust_grade",
                "source_url",
                "license",
                "as_of_date",
                "effective_from",
            }
            missing = required - set(top)
            if missing:
                failures.append(f"row missing keys: {sorted(missing)}")
            if top["doc_id"] != "LAW:000001:법률":
                failures.append(f"top doc_id {top['doc_id']!r} != 'LAW:000001:법률'")
            if top["doc_type"] != "law":
                failures.append(f"top doc_type {top['doc_type']!r} != 'law'")
            if top["as_of_date"] is not None:
                failures.append("as_of_date should be None when not requested")

        # 2) license defaults to config.DEFAULT_LICENSE when payload lacks it.
        ord_row = next((r for r in rows if r["doc_type"] == "ordinance"), None)
        if ord_row is None:
            failures.append("ordinance row missing from unfiltered search")
        elif ord_row["license"] != config.DEFAULT_LICENSE:
            failures.append("missing license did not default to DEFAULT_LICENSE")

        # 3) doc_type filter narrows to laws only.
        law_rows = statutes_search("아무 질의", k=4, filter={"doc_type": "law"})
        if {r["doc_type"] for r in law_rows} != {"law"}:
            failures.append(
                f"doc_type=law filter returned {[r['doc_type'] for r in law_rows]}"
            )

        # 4) as_of_date excludes the future (2030) ordinance AND the date-less
        #    law (safe current-law policy), keeps the in-force 2013 law.
        asof_rows = statutes_search("아무 질의", k=4, as_of_date="2025-12-31")
        ids = {r["doc_id"] for r in asof_rows}
        if "ORD:전라남도:2200001" in ids:
            failures.append("as_of=2025 wrongly included a 2030-effective ordinance")
        if "LAW:000002:법률" in ids:
            failures.append("as_of=2025 wrongly included a row with no effective_from")
        if "LAW:000001:법률" not in ids:
            failures.append("as_of=2025 dropped the in-force 2013 law")
        if any(r["as_of_date"] != "2025-12-31" for r in asof_rows):
            failures.append("as_of_date not echoed onto every row")

        # 5) pagination: page 1 then page 2 are disjoint and complete.
        page1 = statutes_search("아무 질의", k=2, offset=0)
        page2 = statutes_search("아무 질의", k=2, offset=2)
        ids1 = [r["doc_id"] for r in page1]
        ids2 = [r["doc_id"] for r in page2]
        if len(ids1) != 2:
            failures.append(f"page1 size {len(ids1)} != 2")
        if set(ids1) & set(ids2):
            failures.append("pagination pages overlap")
        if len(set(ids1) | set(ids2)) != 4:
            failures.append("pagination did not cover all 4 rows across 2 pages")

        # 6) deterministic ordering: identical request yields identical order.
        again = statutes_search("성년 나이", k=4)
        if [r["doc_id"] for r in again] != [r["doc_id"] for r in rows]:
            failures.append("ordering is not deterministic across identical calls")

        # 7) input validation.
        for bad_call, label in (
            (lambda: statutes_search("   "), "blank query"),
            (lambda: statutes_search("q", as_of_date="2026/01/01"), "bad as_of_date"),
            (lambda: statutes_search("q", offset=-1), "negative offset"),
            (
                lambda: statutes_search("q", filter={"unknown_key": "x"}),
                "invalid filter key",
            ),
        ):
            try:
                bad_call()
            except ValueError:
                pass
            else:
                failures.append(f"{label} was not rejected")

        if failures:
            print("SELFTEST FAILED:")
            for f in failures:
                print("  -", f)
            return 1
        print(
            "SELFTEST PASSED: 7 checks "
            "(shaping, license default, filter, as_of, pagination, ordering, validation)."
        )
        return 0
    finally:
        retriever.embed_query = real_embed  # type: ignore[assignment]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search.statutes",
        description="Unified law+precedent search (/v1/statutes/search).",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run an offline self-check against an in-memory Qdrant collection.",
    )
    parser.add_argument("query", nargs="?", help="Query to run against the live collection.")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K, help="Page size.")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset.")
    parser.add_argument("--doc-type", help="Optional doc_type filter.")
    parser.add_argument("--as-of-date", help="Optional ISO YYYY-MM-DD point-in-time date.")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.query:
        print("Provide a QUERY or use --selftest.", file=sys.stderr)
        return 2
    flt = {"doc_type": args.doc_type} if args.doc_type else None
    rows = statutes_search(
        args.query, k=args.k, filter=flt, as_of_date=args.as_of_date, offset=args.offset
    )
    for i, r in enumerate(rows, start=1):
        print(
            f"[{i}] score={r['score']:.4f} {r['doc_type']} {r['title']} "
            f"{r['article_no']} id={r['doc_id']} 시행={r['effective_from']}"
        )
        print(f"    {r['text'][:160]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
