"""Golden-set evaluation harness for lawbot (Playbook 08 Task 5.1).

Quantifies retrieval and answer quality so we can make data-driven model
decisions (``text-embedding-3-small`` ↔ ``-large``, ``gpt-4o-mini`` ↔ ``gpt-4o``,
dense ↔ hybrid). It reads ``eval/golden_set.jsonl`` (20 lawyer-style questions
with expected target ``doc_type`` / title / article / keywords) and reports a
baseline scorecard.

Two execution modes
-------------------
* ``--mode http`` (default): drive the **running service** over HTTP — the most
  faithful "real-service" measurement. Requires the API up (see ``DEPLOY.md``)
  and an API key (``--api-key`` or ``$LAWBOT_API_KEY``). This exercises auth,
  filters, the retriever, and (optionally) the RAG citation firewall exactly as
  a tenant would.
* ``--mode direct``: import ``search.statutes`` / ``search.rag`` in-process. No
  HTTP server needed; still needs Qdrant + an embedded collection. Handy in CI
  smoke runs and local debugging.

Metrics
-------
* **Hit@K**     — does any of the top-K results match the expected target?
  A row "matches" when its ``doc_type`` equals the expected one *and* (when
  given) its ``title`` contains an expected substring *and/or* its text/title
  contains the expected keywords. ``article`` match is reported separately.
* **MRR@K**     — mean reciprocal rank of the first matching row.
* **Article-hit** — fraction of (article-bearing) questions whose expected
  ``article_no`` appears in the top-K.
* **Citation accuracy** (``--ask`` only) — fraction of answer citations whose
  ``source_id`` is present in the question's retrieved context (post-verified by
  ``search.rag.ask`` already; we re-check as an independent audit).
* **Grounding** (``--ask`` only) — fraction of answers that either carry ≥1
  verified citation or explicitly say "근거 불충분"/"확인 필요" (no silent
  fabrication). This is the anti-hallucination signal; Stanford legal-RAG
  baselines hallucinate 17–33%, so we want grounding ≫ that.

Cost discipline (hard rule)
---------------------------
* Retrieval-only scoring uses ``/v1/statutes/search`` (or ``statutes_search``):
  **no generation cost**. This is the default.
* The LLM answer pass (``--ask``) is **opt-in** and capped by ``--ask-limit``
  (default 3) so a full run costs only a few cents. Never embeds the corpus.

Usage
-----
::

    # Retrieval-only baseline against a running API (cheap, recommended):
    cd /home/user1/lawbot && .venv/bin/python -m eval.run_eval \\
        --mode http --base-url http://localhost:8000 --api-key "$LAWBOT_API_KEY"

    # Add a small LLM answer/citation pass (≤3 questions, a few cents):
    ... --ask --ask-limit 3

    # In-process (no HTTP server), retrieval only:
    cd /home/user1/lawbot && .venv/bin/python -m eval.run_eval --mode direct

Exit code is ``0`` on a successful run (regardless of score) and non-zero only
on a harness/configuration error, so CI can gate on "the harness ran" while
humans read the scorecard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# Project root on sys.path so ``import config`` / ``search.*`` resolve when run
# as ``python -m eval.run_eval`` from /home/user1/lawbot.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

GOLDEN_PATH: Path = Path(__file__).resolve().parent / "golden_set.jsonl"


# --------------------------------------------------------------------------- #
# Golden-set loading                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GoldenItem:
    """One evaluation question with its expected retrieval target."""

    id: str
    query: str
    doc_type: Optional[str]
    expect_title_contains: list[str]
    expect_article: Optional[str]
    expect_keywords: list[str]
    as_of_date: Optional[str]
    notes: str = ""


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    """Parse the JSONL golden set into typed items.

    Args:
        path: Path to ``golden_set.jsonl``.

    Returns:
        The list of golden items, in file order.

    Raises:
        FileNotFoundError: If the golden set is missing.
        ValueError: If a line is malformed JSON or lacks an ``id``/``query``.
    """
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")
    items: list[GoldenItem] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not obj.get("id") or not obj.get("query"):
                raise ValueError(f"{path}:{lineno}: each item needs 'id' and 'query'")
            items.append(
                GoldenItem(
                    id=str(obj["id"]),
                    query=str(obj["query"]),
                    doc_type=obj.get("doc_type"),
                    expect_title_contains=list(obj.get("expect_title_contains") or []),
                    expect_article=obj.get("expect_article"),
                    expect_keywords=list(obj.get("expect_keywords") or []),
                    as_of_date=obj.get("as_of_date"),
                    notes=str(obj.get("notes") or ""),
                )
            )
    return items


# --------------------------------------------------------------------------- #
# Matching logic (shared by both modes)                                        #
# --------------------------------------------------------------------------- #
def _row_matches(item: GoldenItem, row: dict[str, Any]) -> bool:
    """Return True if a single result row satisfies the golden expectation.

    A row matches when its ``doc_type`` equals the expected one (if specified)
    and the expected title substrings / keywords are found in the row's title or
    text. Keyword matching is lenient (any keyword counts) to tolerate phrasing.
    """
    if item.doc_type and str(row.get("doc_type") or "") != item.doc_type:
        return False

    title = str(row.get("title") or "")
    text = str(row.get("text") or "")
    haystack = f"{title}\n{text}"

    if item.expect_title_contains:
        if not any(sub in title for sub in item.expect_title_contains):
            return False

    if item.expect_keywords:
        if not any(kw in haystack for kw in item.expect_keywords):
            return False

    return True


def _article_in_rows(item: GoldenItem, rows: list[dict[str, Any]]) -> Optional[bool]:
    """Whether the expected article appears in any row (None if no expectation)."""
    if not item.expect_article:
        return None
    return any(str(r.get("article_no") or "") == item.expect_article for r in rows)


def _first_match_rank(item: GoldenItem, rows: list[dict[str, Any]]) -> Optional[int]:
    """1-based rank of the first matching row, or None if none match."""
    for idx, row in enumerate(rows, 1):
        if _row_matches(item, row):
            return idx
    return None


# --------------------------------------------------------------------------- #
# Backends — HTTP and direct                                                   #
# --------------------------------------------------------------------------- #
class Backend:
    """Abstract scoring backend."""

    def search(self, item: GoldenItem, k: int) -> list[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def ask(self, item: GoldenItem, k: int) -> Optional[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError


class HttpBackend(Backend):
    """Drive the running service over HTTP (real-service measurement)."""

    def __init__(self, base_url: str, api_key: Optional[str], timeout: float = 60.0) -> None:
        import httpx  # local import: only needed in http mode

        self._base = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(f"{self._base}{path}", json=body)
        resp.raise_for_status()
        return resp.json()

    def search(self, item: GoldenItem, k: int) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"query": item.query, "k": k}
        if item.as_of_date:
            body["as_of_date"] = item.as_of_date
        data = self._post("/v1/statutes/search", body)
        return list(data.get("results") or [])

    def ask(self, item: GoldenItem, k: int) -> Optional[dict[str, Any]]:
        body: dict[str, Any] = {"query": item.query, "k": k}
        if item.as_of_date:
            body["as_of_date"] = item.as_of_date
        return self._post("/v1/ask", body)


class DirectBackend(Backend):
    """Import the search modules in-process (no HTTP server needed)."""

    def __init__(self) -> None:
        from search import statutes  # noqa: PLC0415 - lazy

        self._statutes = statutes
        self._rag = None  # imported on demand to avoid OpenAI client init cost

    def search(self, item: GoldenItem, k: int) -> list[dict[str, Any]]:
        return self._statutes.statutes_search(
            item.query, k=k, as_of_date=item.as_of_date
        )

    def ask(self, item: GoldenItem, k: int) -> Optional[dict[str, Any]]:
        if self._rag is None:
            from search import rag  # noqa: PLC0415 - lazy, only when --ask

            self._rag = rag
        return dict(self._rag.ask(item.query, k=k, as_of_date=item.as_of_date))


# --------------------------------------------------------------------------- #
# Scoring                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ItemResult:
    """Per-question scoring outcome."""

    id: str
    n_rows: int
    hit: bool
    rank: Optional[int]
    article_hit: Optional[bool]
    error: Optional[str] = None
    # answer-pass fields
    asked: bool = False
    n_citations: int = 0
    citation_ok: Optional[bool] = None
    grounded: Optional[bool] = None


@dataclass
class Report:
    """Aggregate scorecard."""

    k: int
    items: list[ItemResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len([i for i in self.items if i.error is None])

    @property
    def hit_at_k(self) -> float:
        scored = [i for i in self.items if i.error is None]
        return _frac(sum(1 for i in scored if i.hit), len(scored))

    @property
    def mrr(self) -> float:
        scored = [i for i in self.items if i.error is None]
        if not scored:
            return 0.0
        return sum((1.0 / i.rank) if i.rank else 0.0 for i in scored) / len(scored)

    @property
    def article_hit(self) -> float:
        scored = [i for i in self.items if i.error is None and i.article_hit is not None]
        return _frac(sum(1 for i in scored if i.article_hit), len(scored))

    @property
    def citation_accuracy(self) -> Optional[float]:
        asked = [i for i in self.items if i.asked and i.citation_ok is not None]
        if not asked:
            return None
        return _frac(sum(1 for i in asked if i.citation_ok), len(asked))

    @property
    def grounding(self) -> Optional[float]:
        asked = [i for i in self.items if i.asked and i.grounded is not None]
        if not asked:
            return None
        return _frac(sum(1 for i in asked if i.grounded), len(asked))


def _frac(num: int, den: int) -> float:
    return (num / den) if den else 0.0


_NO_BASIS_MARKERS = ("근거 불충분", "확인 필요", "확인 불가", "근거가 불충분")


def _score_answer(item: GoldenItem, answer: dict[str, Any], context_rows: list[dict[str, Any]]) -> tuple[int, bool, bool]:
    """Audit an /ask answer for citation accuracy and grounding.

    Returns:
        (n_citations, citation_ok, grounded).
        * citation_ok: every citation's source_id is present in the retrieved
          context ids (independent re-check of the firewall).
        * grounded: the answer has ≥1 citation OR explicitly says 근거 불충분.
    """
    citations = list(answer.get("citations") or [])
    # Build the set of valid context ids the model was shown. Prefer the answer's
    # own used_context (authoritative), fall back to the search rows.
    used = answer.get("used_context") or context_rows
    valid_ids: set[str] = set()
    for blk in used:
        for key in ("source_id", "id", "chunk_id", "doc_id"):
            val = blk.get(key) if isinstance(blk, dict) else None
            if val:
                valid_ids.add(str(val))

    if citations:
        citation_ok = all(str(c.get("source_id") or "") in valid_ids for c in citations)
    else:
        citation_ok = True  # zero citations cannot be wrong; grounding covers it

    text = str(answer.get("answer") or "")
    grounded = bool(citations) or any(m in text for m in _NO_BASIS_MARKERS)
    return len(citations), citation_ok, grounded


def evaluate(
    backend: Backend,
    items: Iterable[GoldenItem],
    *,
    k: int,
    ask: bool,
    ask_limit: int,
) -> Report:
    """Run the golden set through a backend and aggregate a Report."""
    report = Report(k=k)
    asked = 0
    for item in items:
        try:
            rows = backend.search(item, k)
        except Exception as exc:  # one bad question must not abort the run
            report.items.append(ItemResult(id=item.id, n_rows=0, hit=False, rank=None, article_hit=None, error=str(exc)))
            continue

        rank = _first_match_rank(item, rows)
        res = ItemResult(
            id=item.id,
            n_rows=len(rows),
            hit=rank is not None,
            rank=rank,
            article_hit=_article_in_rows(item, rows),
        )

        if ask and asked < ask_limit:
            try:
                answer = backend.ask(item, k)
                if answer is not None:
                    res.asked = True
                    asked += 1
                    n_c, cit_ok, grounded = _score_answer(item, answer, rows)
                    res.n_citations = n_c
                    res.citation_ok = cit_ok
                    res.grounded = grounded
            except Exception as exc:
                res.error = f"ask: {exc}"

        report.items.append(res)
    return report


# --------------------------------------------------------------------------- #
# Output                                                                        #
# --------------------------------------------------------------------------- #
def print_report(report: Report, *, json_out: bool) -> None:
    """Print the scorecard as a table (default) or machine-readable JSON."""
    if json_out:
        payload = {
            "k": report.k,
            "n": report.n,
            "hit_at_k": round(report.hit_at_k, 4),
            "mrr": round(report.mrr, 4),
            "article_hit": round(report.article_hit, 4),
            "citation_accuracy": (
                round(report.citation_accuracy, 4)
                if report.citation_accuracy is not None
                else None
            ),
            "grounding": (
                round(report.grounding, 4) if report.grounding is not None else None
            ),
            "items": [
                {
                    "id": i.id,
                    "hit": i.hit,
                    "rank": i.rank,
                    "n_rows": i.n_rows,
                    "article_hit": i.article_hit,
                    "asked": i.asked,
                    "n_citations": i.n_citations,
                    "citation_ok": i.citation_ok,
                    "grounded": i.grounded,
                    "error": i.error,
                }
                for i in report.items
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("\n=== lawbot golden-set evaluation ===")
    print(f"questions: {len(report.items)}  scored: {report.n}  top-K: {report.k}\n")
    print(f"{'id':<14} {'hit':<4} {'rank':<5} {'rows':<5} {'art':<5} {'cite':<5} {'gnd':<4} note")
    print("-" * 64)
    for i in report.items:
        if i.error:
            print(f"{i.id:<14} ERR  {i.error[:40]}")
            continue
        art = "-" if i.article_hit is None else ("Y" if i.article_hit else "n")
        cite = "-" if i.citation_ok is None else ("Y" if i.citation_ok else "n")
        gnd = "-" if i.grounded is None else ("Y" if i.grounded else "n")
        print(
            f"{i.id:<14} {'Y' if i.hit else 'n':<4} "
            f"{str(i.rank or '-'):<5} {i.n_rows:<5} {art:<5} {cite:<5} {gnd:<4}"
        )
    print("-" * 64)
    print(f"Hit@{report.k}        : {report.hit_at_k:.1%}")
    print(f"MRR@{report.k}        : {report.mrr:.3f}")
    print(f"Article-hit@{report.k}: {report.article_hit:.1%}")
    if report.citation_accuracy is not None:
        print(f"Citation accuracy : {report.citation_accuracy:.1%}  (answer pass)")
    if report.grounding is not None:
        print(f"Grounding rate    : {report.grounding:.1%}  (>= cited or '근거 불충분')")
    print()


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval.run_eval",
        description="lawbot golden-set evaluation (retrieval + optional cited answers).",
    )
    p.add_argument("--mode", choices=["http", "direct"], default="http",
                   help="http: drive the running API (default); direct: import search modules.")
    p.add_argument("--base-url", default=os.getenv("LAWBOT_BASE_URL", "http://localhost:8000"),
                   help="API base URL for --mode http.")
    p.add_argument("--api-key", default=os.getenv("LAWBOT_API_KEY"),
                   help="Bearer API key for --mode http (or $LAWBOT_API_KEY).")
    p.add_argument("-k", "--top-k", type=int, default=8, help="top-K results to score (default 8).")
    p.add_argument("--ask", action="store_true",
                   help="Also run a small LLM answer/citation pass (opt-in, costs a few cents).")
    p.add_argument("--ask-limit", type=int, default=3,
                   help="Max questions sent to /ask when --ask is set (cost cap, default 3).")
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH, help="Path to golden_set.jsonl.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--out", type=Path, default=None, help="Optional path to also write the JSON report.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    try:
        items = load_golden(args.golden)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.mode == "http":
        if not args.api_key:
            print(
                "error: --mode http needs an API key (--api-key or $LAWBOT_API_KEY).\n"
                "       Bootstrap one with:  .venv/bin/python -m api.auth   (prints an admin key once)\n"
                "       then issue a tenant key via POST /v1/keys, or use the admin key directly.",
                file=sys.stderr,
            )
            return 2
        try:
            backend: Backend = HttpBackend(args.base_url, args.api_key)
        except ImportError:
            print("error: httpx is required for --mode http (pip install httpx).", file=sys.stderr)
            return 2
    else:
        try:
            backend = DirectBackend()
        except Exception as exc:  # noqa: BLE001 - surface config/import errors clearly
            print(f"error: could not initialize direct backend: {exc}", file=sys.stderr)
            return 2

    report = evaluate(
        backend,
        items,
        k=args.top_k,
        ask=args.ask,
        ask_limit=max(0, args.ask_limit),
    )
    print_report(report, json_out=args.json)

    if args.out is not None:
        # Always write JSON to --out regardless of console format.
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(report, json_out=True)
        args.out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"[wrote {args.out}]", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
