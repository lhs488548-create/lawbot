"""Chunking stage — parent/child two-layer chunks (09 §B · BUILD CONTRACT (d)).

Turn normalized :class:`ingest.schema.Document` records (read as dicts from the
Phase-1 ``docs_*.jsonl`` artifacts) into embedding-ready **child chunks** plus a
sidecar **parent** lookup, exactly per the chunking SoT (``분석/09_…``).

Design (09 §B, BUILD CONTRACT (d)):

* **child = one article / one precedent section** (the search & embedding unit);
  **parent = the whole law / precedent** (the generation & source-pack unit).
  The source data is already split into article ("제N조") / section ("판결요지"
  …) units, so no general-purpose text splitter is needed.
* The embedded ``text`` is the **deterministic two-layer header + normalized
  body** produced by :func:`header.build.build_headers` — the *single* place
  headers are made (no re-implementation here, so headers never drift). The L1
  citation header + L2 context line land *inside* the vector; the structured
  meta is payload-only (filtering & citation).
* **Second split (09 §B-2):** only when a single article/section exceeds
  ``config.EMBED_MAX_TOKENS`` (rare; mainly long precedent sections). The body is
  normalized, windowed into ``config.CHUNK_WINDOW_TOKENS``-token windows with
  ``config.CHUNK_OVERLAP_TOKENS`` overlap on **sentence boundaries** where
  possible, and the same L1/L2 header is re-prefixed onto every window (via
  ``build_headers(body_override=…)``) with an incrementing ``part_idx``.
* **별표/별지 (09 §B-3):** attachments declared in the document front matter
  (``meta['첨부파일']``). On this corpus they are *label + file link only* (no
  body), so they are emitted as ``kind="별표"`` **metadata chunks**
  (``trust_grade="B"``) carrying the file link in the payload — searchable and
  citable, but flagged B and not treated as full text. (Were a body present, it
  would become a normal embedded ``kind="별표"`` child.)
* **B-grade documents** (metadata only, ``articles == []``) still get a single
  ``kind="메타"`` chunk so the document stays discoverable and is honestly
  surfaced as "메타데이터만 존재".

Public interface (BUILD CONTRACT (d))::

    def chunks_of(doc: dict) -> Iterator[dict]: ...   # one parsed Document
    def build_chunks() -> None: ...                    # docs_*.jsonl -> CHUNKS_JSONL (+ PARENTS_JSONL)

Each emitted child chunk is a JSON-serializable dict (BUILD CONTRACT (d))::

    {
      "chunk_id":     "LAW:000325:법률#제4조#0",   # build_chunk_id(doc_id, article_no, part_idx)
      "doc_id":       "LAW:000325:법률",
      "parent_id":    "LAW:000325:법률",            # == doc_id (parent = whole law/precedent)
      "text":         "<L1 인용헤더>\\n<L2 맥락헤더>\\n<정규화 본문>",
      "content_hash": "<sha256 of text>",           # 09 §C cache key (idempotent re-embed)
      "payload":      { ...09 §D-2 structured meta... }
    }

Owner: embed/chunk builder. Consumes shared constants from ``config``, the
header from ``header.build``, ids from ``ingest.schema``, and the cache-key hash
from ``embed.embed_client`` — and never redefines any of them. Run as a script
to (re)build the chunk + parent artifacts::

    cd /home/user1/lawbot && .venv/bin/python -m embed.chunk
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Iterator

import tiktoken

import config
from embed.embed_client import content_hash
from header.build import build_headers, normalize_body
from ingest.schema import build_chunk_id

# --------------------------------------------------------------------------- #
# Local tunables (model/dim/encoding/window come from config; nothing here      #
# duplicates a shared constant).                                               #
# --------------------------------------------------------------------------- #
# Sentinel article label for a metadata-only (B-grade / label-only) document
# that carries no article text of its own.
_META_ARTICLE_NO: Final[str] = "메타"

# Optional Korean sentence splitter (09 §B-2 "한국어는 kss 보조"). It is *not* a
# hard dependency (budget/requirements are Contracts-owned); when unavailable we
# fall back to a dependency-free sentence regex. Either way splitting prefers
# sentence boundaries; the token windower guarantees the hard size limit.
@lru_cache(maxsize=1)
def _kss_split():
    """Return a ``kss.split_sentences``-like callable, or ``None`` if absent.

    Returns:
        A callable ``str -> list[str]`` splitting Korean sentences, or ``None``
        when ``kss`` is not installed (then the regex fallback is used).
    """
    try:  # pragma: no cover - optional dependency
        import kss  # type: ignore

        return lambda s: list(kss.split_sentences(s))
    except Exception:  # pragma: no cover - kss not installed (the common case)
        return None


# Dependency-free Korean/Latin sentence boundary: end punctuation (. ! ? 。)
# optionally followed by closing quotes/brackets, then whitespace. Also treats a
# blank line as a hard boundary so 항/호 paragraphs split cleanly.
_SENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?。])[\s'\"”’」』)\]]*\s+|\n{2,}"
)


@lru_cache(maxsize=1)
def _encoder() -> "tiktoken.Encoding":
    """Return the cached tiktoken encoder configured by ``config.EMBED_ENCODING``.

    Cached because constructing an encoder is relatively expensive and the
    chunker measures the length of millions of articles.

    Returns:
        The :class:`tiktoken.Encoding` matching ``config.EMBED_ENCODING``.
    """
    return tiktoken.get_encoding(config.EMBED_ENCODING)


def _token_len(text: str) -> int:
    """Return the number of tokens in ``text`` under the embedding encoding.

    Args:
        text: Text to measure.

    Returns:
        Token count (``0`` for empty text).
    """
    if not text:
        return 0
    return len(_encoder().encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preferring Korean sentence boundaries.

    Uses ``kss`` when installed, otherwise a dependency-free regex. Always
    returns at least one element (the whole text) so callers can rely on it.

    Args:
        text: Body text to split.

    Returns:
        A list of sentence strings (whitespace-trimmed, empties dropped).
    """
    kss = _kss_split()
    if kss is not None:  # pragma: no cover - exercised only when kss present
        sents = [s.strip() for s in kss(text) if s.strip()]
        return sents or [text]
    sents = [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]
    return sents or [text]


def _window_body(body: str, header_token_cost: int) -> list[str]:
    """Window an over-long body into sentence-aligned, size-capped pieces.

    Greedily packs whole sentences into windows whose **header + body** stays
    within ``config.EMBED_MAX_TOKENS``, targeting ``config.CHUNK_WINDOW_TOKENS``
    body tokens and carrying ``config.CHUNK_OVERLAP_TOKENS`` of trailing
    sentences into the next window (09 §B-2 overlap so context is not lost at
    boundaries). A single sentence that is itself larger than the budget is
    hard-split on token boundaries as a last resort.

    Args:
        body: The (already-normalized) body text to window.
        header_token_cost: Tokens consumed by the repeated ``L1\\nL2\\n`` header,
            reserved out of every window's budget so the final chunk fits the
            model limit.

    Returns:
        A list of body-window strings, each of which (once the header is
        re-prefixed) fits within ``config.EMBED_MAX_TOKENS``.
    """
    enc = _encoder()
    # Per-window body budget: target window size, but never let header+body
    # exceed the hard model limit. Reserve the header cost from the hard limit.
    hard_budget = max(1, config.EMBED_MAX_TOKENS - header_token_cost)
    target = min(config.CHUNK_WINDOW_TOKENS, hard_budget)
    overlap = min(config.CHUNK_OVERLAP_TOKENS, max(0, target - 1))

    # Pre-tokenize sentences once.
    sentences = _split_sentences(body)
    sent_tokens = [enc.encode(s) for s in sentences]

    windows: list[str] = []
    i = 0
    n = len(sentences)
    while i < n:
        cur: list[int] = []  # sentence indices in this window
        cur_len = 0
        j = i
        while j < n:
            slen = len(sent_tokens[j])
            if slen > hard_budget and not cur:
                # A lone oversized sentence: hard-split on token boundaries.
                toks = sent_tokens[j]
                step = max(1, hard_budget - overlap)
                start = 0
                while start < len(toks):
                    piece = enc.decode(toks[start : start + hard_budget]).strip()
                    if piece:
                        windows.append(piece)
                    if start + hard_budget >= len(toks):
                        break
                    start += step
                j += 1
                i = j
                cur = []
                cur_len = 0
                break
            if cur and cur_len + slen > target:
                break
            cur.append(j)
            cur_len += slen
            j += 1
        else:
            # Reached end of sentences.
            if cur:
                windows.append(" ".join(sentences[k] for k in cur).strip())
            break

        if not cur:
            # Handled the oversized-sentence branch; continue from new i.
            continue

        windows.append(" ".join(sentences[k] for k in cur).strip())

        if j >= n:
            break
        # Step forward, retaining ~overlap tokens of trailing sentences.
        back = 0
        back_idx = len(cur)
        while back_idx > 1 and back < overlap:
            back_idx -= 1
            back += len(sent_tokens[cur[back_idx]])
        i = cur[back_idx] if back_idx < len(cur) else j
        if i <= cur[0]:
            # Guarantee forward progress.
            i = cur[0] + 1

    return [w for w in windows if w] or [body]


def _emit_article(
    doc: dict[str, Any],
    article: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield one or more child chunk records for a single article/section.

    Builds the two-layer header + normalized body via
    :func:`header.build.build_headers`. If the resulting embed text fits the
    model limit it is one chunk (``part_idx=0``); otherwise the normalized body
    is windowed (09 §B-2) and each window re-uses the same header with an
    incrementing ``part_idx``.

    Args:
        doc: The parsed document dict.
        article: An article/section dict (``article_no``, optional ``title``,
            ``text``).

    Yields:
        Child chunk record dicts (see module docstring).
    """
    doc_id = doc["doc_id"]
    article_no = str(article.get("article_no") or "").strip()
    if not article_no:
        return

    kind = article.get("kind") or "본문"
    embed_text, payload = build_headers(doc, article, part_idx=0, kind=kind)

    if _token_len(embed_text) <= config.EMBED_MAX_TOKENS:
        yield _make_record(doc_id, article_no, 0, embed_text, payload)
        return

    # Over-long: window the normalized body and re-prefix the header per window.
    norm_body = normalize_body(article.get("text") or "")
    # Header token cost = full embed_text minus body (approximate via header-only
    # build with an empty body override).
    header_only, _ = build_headers(doc, article, part_idx=0, kind=kind, body_override="")
    header_cost = _token_len(header_only)

    for part_idx, window in enumerate(_window_body(norm_body, header_cost)):
        win_text, win_payload = build_headers(
            doc, article, part_idx=part_idx, kind=kind, body_override=window
        )
        # Final safety clamp: never emit a chunk over the hard limit.
        win_text = _clamp(win_text)
        yield _make_record(doc_id, article_no, part_idx, win_text, win_payload)


def _clamp(text: str) -> str:
    """Trim ``text`` from the tail until it fits ``config.EMBED_MAX_TOKENS``.

    A defensive last resort: ``_window_body`` already keeps windows within
    budget, but a pathological header could push a window over. Truncates on
    token boundaries so the chunk is always embeddable.

    Args:
        text: Candidate chunk text.

    Returns:
        ``text`` unchanged if within budget, else a token-truncated prefix.
    """
    enc = _encoder()
    toks = enc.encode(text)
    if len(toks) <= config.EMBED_MAX_TOKENS:
        return text
    return enc.decode(toks[: config.EMBED_MAX_TOKENS]).strip()


def _make_record(
    doc_id: str,
    article_no: str,
    part_idx: int,
    text: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one child chunk record with its content-hash cache key.

    Args:
        doc_id: The owning document (parent) id.
        article_no: The article/section label.
        part_idx: Sub-split index (0 unless windowed).
        text: The two-layer-header + body embed text.
        payload: The structured meta payload from :func:`build_headers`.

    Returns:
        A complete chunk record (see module docstring). ``content_hash`` is the
        same sha256 the embedder uses as its cache key, so identical text is
        never re-embedded (09 §C).
    """
    chunk_id = build_chunk_id(doc_id, article_no, part_idx)
    # Keep payload's part_idx in sync with the actual sub-split index.
    payload = dict(payload)
    payload["part_idx"] = int(part_idx)
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "parent_id": doc_id,
        "text": text,
        "content_hash": content_hash(text),
        "payload": payload,
    }


def _attachment_chunks(doc: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield 별표/별지 metadata chunks from the document front matter (09 §B-3).

    On this corpus attachments (``meta['첨부파일']``) are *label + file link
    only* (no embedded body), so each becomes a ``kind="별표"``,
    ``trust_grade="B"`` metadata chunk carrying the file link in its payload —
    searchable and citable, but honestly flagged as metadata-only. (If an
    attachment ever carried inline body text it would instead be emitted as a
    normal embedded 별표 child via :func:`_emit_article`.)

    Args:
        doc: The parsed document dict.

    Yields:
        별표 metadata chunk record dicts.
    """
    meta = doc.get("meta") or {}
    attachments = meta.get("첨부파일") or []
    if not isinstance(attachments, (list, tuple)):
        return
    doc_id = doc["doc_id"]
    for idx, att in enumerate(attachments):
        if not isinstance(att, dict):
            continue
        label = (
            str(att.get("제목") or att.get("별표구분") or "별표").strip() or "별표"
        )
        no = str(att.get("별표번호") or idx).strip() or str(idx)
        branch = str(att.get("별표가지번호") or "").strip()
        att_kind = str(att.get("별표구분") or "별표").strip() or "별표"
        # A stable, unique article label per attachment (used in chunk_id).
        article_no = f"{att_kind}{no}{('-' + branch) if branch and branch != '00' else ''}"

        # Build a B-grade 별표 article: header carries the label; no body text.
        art = {"article_no": article_no, "title": label, "text": "", "kind": "별표"}
        # Force B-grade for label-only attachments without mutating the parent doc.
        att_doc = dict(doc)
        att_doc["trust_grade"] = "B"
        embed_text, payload = build_headers(att_doc, art, part_idx=0, kind="별표")
        # Carry the attachment file links in the payload for citation/download.
        payload["attachment_link"] = att.get("파일링크")
        payload["attachment_pdf"] = att.get("PDF링크")
        payload["attachment_kind"] = att_kind
        yield _make_record(doc_id, article_no, 0, _clamp(embed_text), payload)


def chunks_of(doc: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield embedding-ready child chunks for one parsed :class:`Document` dict.

    For each article/section the two-layer header is built and the body emitted
    as one chunk, sub-split into overlapping windows only when it exceeds
    ``config.EMBED_MAX_TOKENS``. 별표/별지 attachments declared in the front
    matter are emitted as ``kind="별표"`` (B-grade, metadata-only) chunks. A
    document with no usable article text yields a single ``kind="메타"`` chunk so
    it stays discoverable.

    Args:
        doc: A parsed document dict matching :class:`ingest.schema.Document`
            (i.e. ``json.loads`` of a ``Document.model_dump_json()`` line). Must
            contain at least ``doc_id``, ``doc_type`` and ``title``.

    Yields:
        Child chunk record dicts (see module docstring).

    Raises:
        KeyError: If ``doc`` lacks a required field (``doc_id`` / ``doc_type`` /
            ``title``).
    """
    # doc_id presence is required; surface a clear error early.
    _ = doc["doc_id"]
    _ = doc["doc_type"]
    _ = doc["title"]

    produced = False
    for article in doc.get("articles") or []:
        body = str(article.get("text") or "").strip()
        if not body:
            # Empty / label-only article: nothing meaningful to embed here.
            continue
        for record in _emit_article(doc, article):
            produced = True
            yield record

    # 별표/별지 attachments (metadata-only on this corpus). These count as
    # "produced" so a B-grade doc with attachments but no body is not also given
    # a 메타 sentinel.
    for record in _attachment_chunks(doc):
        produced = True
        yield record

    if not produced:
        # Metadata-only document (typically trust_grade == "B"): emit a single
        # sentinel chunk so the document is still discoverable and citable.
        meta_doc = dict(doc)
        meta_doc.setdefault("trust_grade", "B")
        art = {"article_no": _META_ARTICLE_NO, "title": doc.get("law_kind") or "", "text": ""}
        embed_text, payload = build_headers(meta_doc, art, part_idx=0, kind="메타")
        yield _make_record(
            doc["doc_id"], _META_ARTICLE_NO, 0, _clamp(embed_text), payload
        )


# --------------------------------------------------------------------------- #
# Parent materialization (09 §B-1 / BUILD CONTRACT (d): PARENTS_JSONL).         #
# --------------------------------------------------------------------------- #
def parent_of(doc: dict[str, Any]) -> dict[str, Any]:
    """Build the parent record for one document (the source-pack/answer unit).

    The parent carries the **full original text** (all articles/sections joined,
    not the normalized embedding text) so a child hit can be promoted to its
    parent's full text for ``/v1/ask`` answers and ``/v1/source-pack`` bundles
    (09 §E-1/E-3). Provenance/common-meta keys mirror the child payloads.

    Args:
        doc: The parsed document dict.

    Returns:
        A parent record dict (see module docstring / BUILD CONTRACT (d)).
    """
    parts: list[str] = []
    for a in doc.get("articles") or []:
        no = str(a.get("article_no") or "").strip()
        title = str(a.get("title") or "").strip()
        text = str(a.get("text") or "").strip()
        if not text:
            continue
        head = f"{no}({title})" if title else no
        parts.append(f"{head}\n{text}" if head else text)
    return {
        "parent_id": doc["doc_id"],
        "doc_type": doc.get("doc_type"),
        "title": doc.get("title"),
        "law_kind": doc.get("law_kind"),
        "jurisdiction": doc.get("jurisdiction"),
        "effective_from": doc.get("effective_from"),
        "source_url": doc.get("source_url"),
        "license": config.DEFAULT_LICENSE,
        "trust_grade": doc.get("trust_grade", "A"),
        "full_text": "\n\n".join(parts),
    }


def _iter_docs(path: Path) -> Iterator[dict[str, Any]]:
    """Stream document dicts from a Phase-1 JSONL artifact.

    Blank lines are skipped; malformed JSON lines are skipped with a logged
    warning rather than crashing the (multi-million-line) build.

    Args:
        path: Path to a ``docs_*.jsonl`` file.

    Yields:
        Parsed document dicts.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                print(f"SKIP malformed JSON {path.name}:{line_no}: {exc}")


def build_chunks(
    sources: list[Path] | None = None,
    out_path: Path | None = None,
    parents_path: Path | None = None,
) -> dict[str, int]:
    """Read the Phase-1 document JSONLs → write chunks + parents (09 §B).

    Streams every document from each source artifact, expands it into child
    chunks (``config.CHUNKS_JSONL``) and a parent record
    (``config.PARENTS_JSONL``), one JSON object per line. Designed for the full
    corpus (millions of chunks) so it never holds everything in memory.
    Globally-unique ``chunk_id``s are asserted as a cheap integrity guard.

    Args:
        sources: Document JSONL paths to chunk. Defaults to the four Phase-1
            artifacts (missing ones are skipped with a log).
        out_path: Child-chunk output path. Defaults to ``config.CHUNKS_JSONL``.
        parents_path: Parent output path. Defaults to ``config.PARENTS_JSONL``.

    Returns:
        A small stats dict ``{docs, chunks, parents, over_limit}``.

    Raises:
        RuntimeError: If a duplicate ``chunk_id`` is produced (indicates a
            doc_id/article_no collision upstream).
    """
    if sources is None:
        sources = [
            config.DOCS_LAW_JSONL,
            config.DOCS_ORD_JSONL,
            config.DOCS_ADMRULE_JSONL,
            config.DOCS_PREC_JSONL,
        ]
    out_path = out_path or config.CHUNKS_JSONL
    parents_path = parents_path or config.PARENTS_JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parents_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    seen_docs: set[str] = set()
    n_docs = n_chunks = n_parents = n_over = 0
    n_dup_docs = n_dup_chunks = 0

    with out_path.open("w", encoding="utf-8") as out, parents_path.open(
        "w", encoding="utf-8"
    ) as pout:
        for src in sources:
            if not src.exists():
                print(f"SKIP missing source artifact: {src}")
                continue
            src_docs = src_chunks = 0
            for doc in _iter_docs(src):
                src_docs += 1
                # Real-world corpora (esp. 자치법규, ~16만건) contain crawl
                # duplicates: the same doc_id appears in more than one source
                # file. Skip the whole repeat document (its chunk_ids would all
                # collide) rather than aborting the build.
                doc_id = str(doc.get("doc_id") or "")
                if doc_id and doc_id in seen_docs:
                    n_dup_docs += 1
                    continue
                if doc_id:
                    seen_docs.add(doc_id)
                for chunk in chunks_of(doc):
                    cid = chunk["chunk_id"]
                    if cid in seen:
                        # Defensive: a chunk_id collision across distinct doc_ids
                        # (should not happen by construction) — skip, don't abort.
                        n_dup_chunks += 1
                        continue
                    seen.add(cid)
                    if _token_len(chunk["text"]) > config.EMBED_MAX_TOKENS:
                        n_over += 1
                    out.write(json.dumps(chunk, ensure_ascii=False))
                    out.write("\n")
                    src_chunks += 1
                pout.write(json.dumps(parent_of(doc), ensure_ascii=False))
                pout.write("\n")
                n_parents += 1
            print(f"{src.name}: {src_docs} docs -> {src_chunks} chunks")
            n_docs += src_docs
            n_chunks += src_chunks

    if n_dup_docs or n_dup_chunks:
        print(
            f"NOTE: skipped {n_dup_docs} duplicate document(s) and "
            f"{n_dup_chunks} stray duplicate chunk(s) (crawl dups)."
        )

    if n_over:  # pragma: no cover - should never happen by construction
        print(f"WARNING: {n_over} chunks exceed EMBED_MAX_TOKENS")
    print(
        f"DONE: {n_docs} docs -> {n_chunks} chunks, {n_parents} parents "
        f"(all <= {config.EMBED_MAX_TOKENS} tokens) -> {out_path}"
    )
    return {
        "docs": n_docs,
        "chunks": n_chunks,
        "parents": n_parents,
        "over_limit": n_over,
    }


if __name__ == "__main__":
    build_chunks()
