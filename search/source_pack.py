"""Source Pack builder (09 §E-3, lawbot.org ``/v1/source-pack``).

Turn a query / fact pattern into an **LLM-citable markdown bundle** of the
relevant primary sources (statute articles, ordinances, admin rules,
precedent sections), assembled **deterministically with no generation cost**.

Pipeline (09 §E-1/E-3, BUILD CONTRACT (d) → Source Pack):

    retrieve child hits (dense, optional metadata pre-filter + as_of_date)
      -> promote each child hit to its parent (the whole law / precedent)
      -> keep <= config.SOURCE_PACK_MAX_PARENTS distinct parents, best-score first
      -> resolve each parent's full original text
      -> emit a markdown bundle (법령 > 조문 원문 + 시행일 + source_url
         + trust_grade + license) ready to paste into an LLM prompt as cited
         context, plus a structured ``sources[]`` carrying the common response
         meta {trust_grade, source_url, license, as_of_date, effective_from}.

Why parents, not raw child chunks: child chunks are small and precise (good for
*finding* the right law), but an LLM citing the law needs the *whole article /
precedent* in front of it. So child hits are the retrieval handle and parents
are the payload (09 §B-1 parent/child, "child 정밀 + parent 맥락").

Design notes / robustness:

* This module owns **only** source-pack assembly. It depends on the
  Contracts-owned ``config`` and ``ingest.schema`` and on the retriever's public
  ``search`` interface. It never modifies shared files and never calls a
  generation model (cost rule — assembly is pure string work, $0).
* **Parent resolution is defensive** so the source pack works whether or not the
  embed/retriever builders have materialized parents yet:
    1. ``search.retriever.get_parent`` if that function exists (preferred);
    2. else ``config.PARENTS_JSONL`` read directly (the parents sidecar);
    3. else fall back to assembling the parent from the retrieved child hits'
       own text (degraded but always-available — flagged in the source meta).
* ``as_of_date`` (ISO ``YYYY-MM-DD``) restricts results to sources whose
  ``effective_from <= as_of_date`` (point-in-time current-law lookup, 09 §A/E).
  It is applied here as a post-filter so the pack is correct even if the
  retriever has not yet pushed the date predicate into Qdrant.

Run a quick offline self-check (fakes the retriever — no OpenAI/Qdrant needed)::

    cd /home/user1/lawbot && .venv/bin/python -m search.source_pack --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import lru_cache
from typing import Any, Callable, Final, Optional

import config
from ingest.schema import parent_id_of

logger = logging.getLogger(__name__)

# Hard cap on how much parent text we paste into one bundle entry. Source packs
# are meant to be dropped into an LLM context window, so an individual parent
# (e.g. an entire long statute) is truncated to keep the bundle bounded. This is
# a *display* limit only; the full text always remains at ``source_url``.
_MAX_PARENT_CHARS: Final[int] = 12000

# Per-doc_type human label for the markdown bundle headings.
_DOCTYPE_LABEL: Final[dict[str, str]] = {
    "law": "법령",
    "ordinance": "자치법규",
    "admrule": "행정규칙",
    "precedent": "판례",
}


# --------------------------------------------------------------------------- #
# Parent resolution (3-tier, defensive)                                       #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _retriever_get_parent() -> Optional[Callable[[str], Optional[dict[str, Any]]]]:
    """Return ``search.retriever.get_parent`` if the retriever exposes it.

    The retriever builder may add ``get_parent`` (per the contract) after this
    module is written; we look it up lazily and cache the result so the source
    pack transparently uses it when available and falls back otherwise. The
    import is wrapped so a partially-built retriever never breaks this module.

    Returns:
        The ``get_parent`` callable, or ``None`` when it is not (yet) provided.
    """
    try:
        from search import retriever  # noqa: PLC0415 - lazy to avoid import cost
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("retriever import unavailable for get_parent: %s", exc)
        return None
    fn = getattr(retriever, "get_parent", None)
    return fn if callable(fn) else None


@lru_cache(maxsize=1)
def _parents_index() -> dict[str, dict[str, Any]]:
    """Load and cache the parents sidecar (``config.PARENTS_JSONL``).

    The parents JSONL (one ``{parent_id, ..., full_text}`` record per line) is
    produced by the chunking stage. It may be absent until that stage has run;
    in that case an empty index is returned and the caller falls back to
    assembling parents from child-hit text.

    Returns:
        A mapping ``parent_id -> parent record``. Empty when the file is missing
        or unreadable.
    """
    path = config.PARENTS_JSONL
    index: dict[str, dict[str, Any]] = {}
    if not path.exists():
        logger.debug("PARENTS_JSONL not found at %s; using child-text fallback.", path)
        return index
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    logger.warning("Skip malformed parents.jsonl:%d: %s", line_no, exc)
                    continue
                pid = rec.get("parent_id") or rec.get("doc_id")
                if pid:
                    index[str(pid)] = rec
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Could not read PARENTS_JSONL (%s): %s", path, exc)
    return index


def get_parent(parent_id: str) -> Optional[dict[str, Any]]:
    """Resolve a parent's full record by id, trying each source in turn.

    Resolution order (first hit wins): the retriever's own ``get_parent`` (if
    present), then the parents sidecar JSONL, else ``None`` (the caller then
    reconstructs from child hits). Kept here so callers have a single,
    fallback-aware entry point.

    Args:
        parent_id: The parent (document) id, e.g. ``"LAW:014565:법률"``.

    Returns:
        A parent record dict, or ``None`` if no materialized parent is found.
    """
    if not parent_id:
        return None
    fn = _retriever_get_parent()
    if fn is not None:
        try:
            rec = fn(parent_id)
            if rec:
                return dict(rec)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("retriever.get_parent(%r) failed: %s", parent_id, exc)
    return _parents_index().get(parent_id)


# --------------------------------------------------------------------------- #
# Hit / payload helpers                                                        #
# --------------------------------------------------------------------------- #
def _payload_of(hit: Any) -> dict[str, Any]:
    """Return a hit's payload as a plain dict (never ``None``)."""
    payload = getattr(hit, "payload", None)
    return dict(payload) if payload else {}


def _parent_id_of_hit(hit: Any, payload: dict[str, Any]) -> str:
    """Best-effort parent id for a retrieved child hit.

    Prefers an explicit ``parent_id`` in the payload (09 §D-2), then ``doc_id``,
    then derives it from a ``chunk_id`` via :func:`ingest.schema.parent_id_of`.

    Args:
        hit: A retrieved hit.
        payload: Its payload (already extracted).

    Returns:
        The parent/document id, or ``""`` when nothing identifies it.
    """
    pid = payload.get("parent_id") or payload.get("doc_id")
    if pid:
        return str(pid)
    chunk_id = payload.get("chunk_id") or getattr(hit, "chunk_id", None)
    if chunk_id:
        return parent_id_of(str(chunk_id))
    return ""


def _passes_as_of(effective_from: Any, as_of_date: Optional[str]) -> bool:
    """Whether a source effective on ``effective_from`` is current at ``as_of_date``.

    ISO ``YYYY-MM-DD`` strings compare correctly lexicographically. Sources with
    no effective date are kept (we cannot prove they are *not* in force, and
    dropping them would silently hide otherwise-relevant law). Args:

    Args:
        effective_from: The source's enforcement/decision date (ISO-ish string)
            or ``None``.
        as_of_date: The point-in-time cutoff, or ``None`` to disable the filter.

    Returns:
        ``True`` if the source should be included.
    """
    if not as_of_date:
        return True
    if not effective_from:
        return True
    return str(effective_from)[:10] <= as_of_date[:10]


# --------------------------------------------------------------------------- #
# Source assembly                                                              #
# --------------------------------------------------------------------------- #
def _source_meta(
    parent_id: str,
    parent: Optional[dict[str, Any]],
    rep_payload: dict[str, Any],
    full_text: str,
    text_origin: str,
) -> dict[str, Any]:
    """Build one ``sources[]`` entry with the common response meta.

    Fields prefer the materialized parent record; otherwise they come from the
    representative (best-scoring) child hit's payload. Always carries the common
    meta ``{trust_grade, source_url, license, as_of_date, effective_from}`` plus
    enough to render and re-fetch the source.

    Args:
        parent_id: The parent/document id.
        parent: The resolved parent record, or ``None`` (degraded fallback).
        rep_payload: Payload of the best child hit for this parent.
        full_text: The assembled full text used in the bundle.
        text_origin: ``"parent"`` when full text came from a materialized parent,
            ``"child"`` when reconstructed from child hits (flagged for honesty).

    Returns:
        A JSON-serializable source-meta dict.
    """
    src = parent or {}

    def pick(key: str, *fallbacks: str) -> Any:
        for k in (key, *fallbacks):
            if src.get(k) not in (None, ""):
                return src.get(k)
            if rep_payload.get(k) not in (None, ""):
                return rep_payload.get(k)
        return None

    return {
        "doc_id": parent_id,
        "doc_type": pick("doc_type"),
        "title": pick("title"),
        "law_kind": pick("law_kind"),
        "jurisdiction": pick("jurisdiction"),
        "article_no": rep_payload.get("article_no"),
        "trust_grade": pick("trust_grade") or "A",
        "source_url": pick("source_url"),
        "license": pick("license") or config.DEFAULT_LICENSE,
        "effective_from": pick("effective_from"),
        "as_of_date": rep_payload.get("as_of"),
        "text_origin": text_origin,
        "chars": len(full_text),
    }


def _resolve_full_text(
    parent: Optional[dict[str, Any]],
    child_payloads: list[dict[str, Any]],
) -> tuple[str, str]:
    """Return the parent's full text and where it came from.

    Prefers the materialized parent's ``full_text``. When no parent record is
    available, reconstructs a best-effort body by concatenating the retrieved
    child chunks' text in article order (de-duplicated). This guarantees a
    non-empty, citable body even before the parents sidecar exists.

    Args:
        parent: The resolved parent record, or ``None``.
        child_payloads: Payloads of the child hits that promoted to this parent
            (best-score order).

    Returns:
        ``(full_text, origin)`` with ``origin`` in ``{"parent", "child"}``.
    """
    if parent and str(parent.get("full_text") or "").strip():
        return str(parent["full_text"]).strip(), "parent"

    # Degraded fallback: stitch the child chunk texts. De-dup while preserving
    # order; child ``text`` already carries the two-layer header + body.
    seen: set[str] = set()
    pieces: list[str] = []
    for p in child_payloads:
        body = str(p.get("text") or "").strip()
        if body and body not in seen:
            seen.add(body)
            pieces.append(body)
    return "\n\n".join(pieces).strip(), "child"


def _render_markdown(sources: list[dict[str, Any]], bodies: list[str]) -> str:
    """Render the ordered sources + bodies into one citable markdown bundle.

    The bundle leads with a short provenance preamble, then one section per
    source with a heading (종류·제목·조문/섹션·시행일), an attribution line
    (출처·등급·라이선스), and the (truncated) original text in a fenced quote so an
    LLM can quote it verbatim with an accurate citation.

    Args:
        sources: The ordered source-meta dicts.
        bodies: The full-text bodies aligned 1:1 with ``sources``.

    Returns:
        A markdown string. Returns a clear "no sources" notice when empty.
    """
    if not sources:
        return (
            "# 소스 팩\n\n"
            "_관련 원문을 찾지 못했습니다. 질의를 더 구체화하거나 필터/시점을 "
            "조정해 주십시오._\n"
        )

    out: list[str] = [
        "# 소스 팩 (인용 가능 원문 번들)",
        "",
        "> 아래 각 출처의 원문만 근거로 인용하십시오. 본문에 없는 사실은 추가하지 "
        "마십시오. 각 출처의 시행일·등급·라이선스를 함께 명시하십시오.",
        "",
    ]
    for i, (meta, body) in enumerate(zip(sources, bodies), start=1):
        label = _DOCTYPE_LABEL.get(str(meta.get("doc_type")), "출처")
        title = meta.get("title") or "(제목 없음)"
        article = meta.get("article_no") or ""
        eff = meta.get("effective_from") or "미상"
        heading = f"## [{i}] {label} · {title}"
        if article:
            heading += f" · {article}"
        out.append(heading)

        attribution = (
            f"- 시행/선고일: {eff} · 신뢰등급: {meta.get('trust_grade', 'A')} "
            f"· 라이선스: {meta.get('license')}"
        )
        out.append(attribution)
        if meta.get("source_url"):
            out.append(f"- 출처: {meta['source_url']}")
        if meta.get("doc_id"):
            out.append(f"- 식별자: `{meta['doc_id']}`")
        if meta.get("text_origin") == "child":
            out.append(
                "- ⚠️ 주의: 전체 원문 미확보 — 검색된 조문/섹션 일부만 포함됩니다."
            )
        if str(meta.get("trust_grade")) == "B":
            out.append(
                "- ⚠️ 주의: 본문 없음(B등급) — 메타데이터에 한정된 출처입니다."
            )
        out.append("")

        text = body.strip() or "_(원문 본문 없음)_"
        if len(text) > _MAX_PARENT_CHARS:
            text = text[:_MAX_PARENT_CHARS].rstrip() + "\n…(이하 생략, 전문은 출처 참조)"
        # Blockquote the original so it is visually separated and easy to cite.
        out.append("\n".join(f"> {ln}" if ln else ">" for ln in text.splitlines()))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def build(
    query: str,
    k: int = config.DEFAULT_TOP_K,
    filter: dict[str, Any] | None = None,  # noqa: A002 - matches contract name
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Build an LLM-citable source-pack bundle for ``query``.

    Retrieves the top child chunks, promotes them to at most
    ``config.SOURCE_PACK_MAX_PARENTS`` distinct parents (ranked by their best
    child score), resolves each parent's full original text, and returns a
    deterministic markdown bundle plus a structured ``sources[]`` carrying the
    common response meta. **No generation model is used** (assembly is pure
    string work, $0).

    Args:
        query: The legal question or fact pattern. Must be non-empty.
        k: Number of child chunks to retrieve before parent promotion. Defaults
            to ``config.DEFAULT_TOP_K``. A larger ``k`` widens parent coverage.
        filter: Optional flat ``{payload_key: value}`` pre-filter AND-ed into the
            retrieval (e.g. ``{"doc_type": "law"}``,
            ``{"jurisdiction": "전라남도"}``). Keys must be retriever-allowed.
        as_of_date: Optional ISO ``YYYY-MM-DD`` cutoff; sources whose
            ``effective_from`` is after this date are excluded (point-in-time
            current-law view).

    Returns:
        ``{"markdown": str, "sources": [ {..common meta..} ], "as_of_date": str|None}``
        per ``_BUILD_CONTRACT.md`` (d)/(e). ``sources`` is ordered best-first and
        capped at ``config.SOURCE_PACK_MAX_PARENTS``.

    Raises:
        ValueError: If ``query`` is empty/whitespace or ``k`` < 1.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError("k must be >= 1")

    # Lazy import keeps this module importable without a live vector store (so
    # the markdown/assembly logic stays unit-testable in isolation).
    from search.retriever import search  # noqa: PLC0415

    hits = list(search(query, k=k, flt=filter))

    # Group child hits by parent, preserving the best (first-seen, since hits are
    # score-ordered) score and collecting each parent's child payloads in rank
    # order. Apply the as_of_date post-filter per child.
    order: list[str] = []
    by_parent: dict[str, dict[str, Any]] = {}
    for hit in hits:
        payload = _payload_of(hit)
        if not _passes_as_of(payload.get("effective_from"), as_of_date):
            continue
        pid = _parent_id_of_hit(hit, payload)
        if not pid:
            continue
        bucket = by_parent.get(pid)
        if bucket is None:
            order.append(pid)
            by_parent[pid] = {
                "best_score": float(getattr(hit, "score", 0.0) or 0.0),
                "rep_payload": payload,  # best (first) child hit's payload
                "child_payloads": [payload],
            }
        else:
            bucket["child_payloads"].append(payload)

    # Cap to the configured maximum number of parents (already best-first).
    selected = order[: config.SOURCE_PACK_MAX_PARENTS]

    sources: list[dict[str, Any]] = []
    bodies: list[str] = []
    for pid in selected:
        bucket = by_parent[pid]
        parent = get_parent(pid)
        full_text, origin = _resolve_full_text(parent, bucket["child_payloads"])
        meta = _source_meta(pid, parent, bucket["rep_payload"], full_text, origin)
        meta["score"] = round(bucket["best_score"], 6)
        if as_of_date:
            meta["as_of_date"] = as_of_date
        sources.append(meta)
        bodies.append(full_text)

    markdown = _render_markdown(sources, bodies)
    return {"markdown": markdown, "sources": sources, "as_of_date": as_of_date}


__all__ = ["build", "get_parent"]


# --------------------------------------------------------------------------- #
# Offline self-test (no OpenAI / Qdrant): fake the retriever + parents         #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Exercise grouping, parent promotion, as_of_date and markdown rendering.

    The retriever's ``search`` is monkeypatched to return canned hits, and the
    parents index is stubbed, so the test is free, offline and deterministic.
    Returns a process exit code (0 = pass).
    """
    from dataclasses import dataclass, field

    @dataclass
    class _FakeHit:
        id: str
        score: float
        payload: dict[str, Any] = field(default_factory=dict)

    # Two parents: 민법 (two child articles) and a precedent (one section).
    fake_hits = [
        _FakeHit(
            id="h1",
            score=0.82,
            payload={
                "chunk_id": "LAW:000001:법률#제4조#0",
                "parent_id": "LAW:000001:법률",
                "doc_type": "law",
                "title": "민법",
                "law_kind": "법률",
                "jurisdiction": "국가",
                "article_no": "제4조",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "trust_grade": "A",
                "text": "[민법 제4조 성년] 사람은 19세로 성년에 이르게 된다.",
            },
        ),
        _FakeHit(
            id="h2",
            score=0.61,
            payload={
                "chunk_id": "LAW:000001:법률#제5조#0",
                "parent_id": "LAW:000001:법률",
                "doc_type": "law",
                "title": "민법",
                "law_kind": "법률",
                "jurisdiction": "국가",
                "article_no": "제5조",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "trust_grade": "A",
                "text": "[민법 제5조] 미성년자의 법률행위.",
            },
        ),
        _FakeHit(
            id="h3",
            score=0.55,
            payload={
                "chunk_id": "PREC:424370#판결요지#0",
                "parent_id": "PREC:424370",
                "doc_type": "precedent",
                "title": "손해배상청구",
                "law_kind": "민사",
                "jurisdiction": "대법원",
                "article_no": "판결요지",
                "effective_from": "2025-05-14",  # future-dated for as_of test
                "source_url": "https://law.go.kr/prec",
                "trust_grade": "A",
                "text": "[대법원 2020다12345 판결요지] 손해배상 책임 인정.",
            },
        ),
    ]

    # Operate on *this* module object. When run as ``python -m
    # search.source_pack`` the module is imported twice (once as ``__main__``,
    # once as ``search.source_pack``); ``build``/``get_parent`` close over THIS
    # module's globals, so the stubs must be installed here, not on a re-import.
    mod = sys.modules[__name__]
    from search import retriever as _retriever_mod

    real_search = getattr(_retriever_mod, "search")
    real_get_parent = getattr(_retriever_mod, "get_parent", None)

    def _fake_search(query: str, k: int = 8, flt: Any = None) -> list[_FakeHit]:
        out = fake_hits
        if flt and "doc_type" in flt:
            out = [h for h in out if h.payload.get("doc_type") == flt["doc_type"]]
        return out[:k]

    _retriever_mod.search = _fake_search  # type: ignore[assignment]
    if real_get_parent is not None:
        _retriever_mod.get_parent = None  # type: ignore[assignment]
    mod._retriever_get_parent.cache_clear()

    # Stub the parents index so 민법 resolves via a materialized full_text
    # ("parent" origin) while the precedent has no parent record and must fall
    # back to child-hit text ("child" origin). We swap the module attribute
    # itself (not the lru_cache internals) to keep this deterministic.
    _orig_parents_index = mod._parents_index

    def _stub_parents_index() -> dict[str, Any]:
        return {
            "LAW:000001:법률": {
                "parent_id": "LAW:000001:법률",
                "doc_type": "law",
                "title": "민법",
                "law_kind": "법률",
                "jurisdiction": "국가",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "license": config.DEFAULT_LICENSE,
                "trust_grade": "A",
                "full_text": "제4조(성년) 사람은 19세로 성년에 이른다.\n\n제5조 …",
            }
        }

    mod._parents_index = _stub_parents_index  # type: ignore[assignment]

    failures: list[str] = []
    try:
        # 1) Basic build: two parents (민법, 판례), 민법 first (best score).
        pack = build("성년 나이와 손해배상", k=8)
        if not isinstance(pack, dict) or set(pack) != {"markdown", "sources", "as_of_date"}:
            failures.append(f"unexpected top-level keys: {sorted(pack)}")
        srcs = pack.get("sources", [])
        if len(srcs) != 2:
            failures.append(f"expected 2 parents, got {len(srcs)}")
        if srcs and srcs[0]["doc_id"] != "LAW:000001:법률":
            failures.append(f"first parent should be 민법, got {srcs[0]['doc_id']}")
        if srcs and srcs[0].get("text_origin") != "parent":
            failures.append("민법 should resolve via parent full_text")
        if len(srcs) > 1 and srcs[1].get("text_origin") != "child":
            failures.append("precedent should fall back to child text")
        # Common meta present on every source.
        for s in srcs:
            for key in ("trust_grade", "source_url", "license", "effective_from"):
                if key not in s:
                    failures.append(f"source missing common-meta key {key!r}")
        # Markdown carries the law text and an attribution line.
        md = pack["markdown"]
        if "민법" not in md or "라이선스" not in md or "성년" not in md:
            failures.append("markdown missing expected content")

        # 2) as_of_date excludes the future-dated precedent (2025-05-14).
        pack2 = build("성년 나이와 손해배상", k=8, as_of_date="2020-01-01")
        ids2 = [s["doc_id"] for s in pack2["sources"]]
        if "PREC:424370" in ids2:
            failures.append("as_of_date should exclude future-dated precedent")
        if "LAW:000001:법률" not in ids2:
            failures.append("as_of_date wrongly excluded in-force statute")
        if pack2["as_of_date"] != "2020-01-01":
            failures.append("as_of_date not echoed back")

        # 3) doc_type filter forwarded to the retriever.
        pack3 = build("성년", k=8, filter={"doc_type": "precedent"})
        ids3 = [s["doc_id"] for s in pack3["sources"]]
        if ids3 != ["PREC:424370"]:
            failures.append(f"doc_type filter not honored: {ids3}")

        # 4) parent cap respected.
        capped = build("x", k=8)
        if len(capped["sources"]) > config.SOURCE_PACK_MAX_PARENTS:
            failures.append("SOURCE_PACK_MAX_PARENTS not enforced")

        # 5) blank query rejected.
        try:
            build("   ")
        except ValueError:
            pass
        else:
            failures.append("blank query not rejected")
    finally:
        _retriever_mod.search = real_search  # type: ignore[assignment]
        if real_get_parent is not None:
            _retriever_mod.get_parent = real_get_parent  # type: ignore[assignment]
        mod._parents_index = _orig_parents_index
        mod._retriever_get_parent.cache_clear()
        mod._parents_index.cache_clear()

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SELFTEST PASSED: 5 checks (promotion, fallback, as_of_date, filter, cap).")
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="search.source_pack",
        description="Build an LLM-citable source-pack bundle (09 §E-3).",
    )
    parser.add_argument("--selftest", action="store_true", help="Run offline self-check.")
    parser.add_argument("query", nargs="?", help="Query to build a source pack for.")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--doc-type", help="Optional doc_type filter.")
    parser.add_argument("--as-of-date", help="Optional ISO YYYY-MM-DD cutoff.")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.query:
        print("Provide a QUERY or use --selftest.", file=sys.stderr)
        return 2
    flt = {"doc_type": args.doc_type} if args.doc_type else None
    pack = build(args.query, k=args.k, filter=flt, as_of_date=args.as_of_date)
    print(pack["markdown"])
    print(f"\n--- {len(pack['sources'])} sources ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
