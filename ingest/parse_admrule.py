"""Administrative-rule parser (Playbook 08, Task 1.3).

Parses ``03_행정규칙/{부처}/.../{규칙명}/본문.md`` files into the unified
:class:`ingest.schema.Document` model. Administrative-rule bodies use **inline**
``제N조(제목) 내용...`` article markers (not ``#####`` headers — verified on this
dataset: 0/400 sampled files used hash headers, 305/400 used inline article
markers), so the splitter is regex-based on line-leading article labels.

On-disk facts this parser depends on (verified against ``ADMRULE_DIR``):

* Each file is YAML front matter delimited by a leading ``---\n...\n---\n``
  block, followed by the Markdown body.
* Front-matter keys: ``행정규칙ID, 행정규칙명, 행정규칙종류, 소관부처명,
  발령일자, 시행일자, 본문출처, 출처, 첨부파일``.
* Body articles are inline: ``^제\\d+조(?:의\\d+)?\\s*\\(제목\\) 내용``.
  Chapters appear as ``제N장 ...`` lines and are tracked for context headers.
* Some 고시/공고-type rules have a body with no ``제N조`` structure (numbered
  lists, tables) — that body is still real text, kept as a single citable
  article (``trust_grade="A"``).
* Label-only / image-only / empty bodies carry no usable text →
  ``trust_grade="B"``, ``articles=[]`` (the Document is still emitted with its
  metadata so coverage stays honest).

Public interface (``_BUILD_CONTRACT.md`` §c)::

    def parse_all() -> Iterator[Document]: ...

Running the module as a script streams every parsed Document as one
``Document.model_dump_json()`` line to ``config.DOCS_ADMRULE_JSONL``.

Owner: builder ``parse_admrule``. Imports the Contracts-owned ``config`` and
``ingest.schema``; adds no shared state.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterator

import yaml

import config
from ingest.schema import Article, Document, build_doc_id

# --------------------------------------------------------------------------- #
# Patterns                                                                      #
# --------------------------------------------------------------------------- #

# Front matter: a leading ``---\n ... \n---`` block, then the body. The body may
# be empty, so the trailing newline after the closing fence is optional.
_FRONT_MATTER = re.compile(r"^﻿?---\n(.*?)\n---\n?(.*)$", re.S)

# Primary (inline) article splitter — matches a line that *starts* with an
# article label, optionally followed by a parenthesized title, e.g.
# ``제1조(목적) 이 지침은 ...`` or ``제4조의2  ...``. The title group is optional
# because a few articles have no parenthetical title.
_ARTICLE_INLINE = re.compile(
    r"^(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?",
    re.M,
)

# Fallback for any stray hash-header files (kept per the build contract although
# none were observed in this corpus). Same capture groups as the inline form.
_ARTICLE_HASH = re.compile(
    r"^#{3,6}\s*(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?",
    re.M,
)

# Chapter marker, e.g. ``제2장  총칙`` — used only to annotate context, never as
# a citable unit. Captured per-article for the deterministic context header.
_CHAPTER = re.compile(r"^(제\d+장(?:의\d+)?)\s*(.*?)\s*$", re.M)

# A body that is only an embedded-image placeholder (``<img id="...">``) or HTML
# scaffolding carries no readable legal text -> treat as metadata only (B-grade).
_IMG_ONLY = re.compile(r"^\s*(?:<img\b[^>]*>\s*(?:</img>)?\s*)+$", re.I | re.S)

# Minimum length (characters) below which a body is considered label-only.
_MIN_BODY_CHARS = 5

__all__ = ["parse_all", "parse_file"]


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _as_text(value: Any) -> str:
    """Coerce a front-matter scalar (str / date / int) to a stripped string.

    YAML parses unquoted dates (``2026-01-26``) to ``datetime.date`` and numeric
    ids to ``int``; downstream stages expect ISO-ish strings, so normalize here.

    Args:
        value: A front-matter value of any scalar type, or ``None``.

    Returns:
        The value as a trimmed string (``""`` for ``None``).
    """
    if value is None:
        return ""
    return str(value).strip()


def _split_front_matter(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    """Split a ``본문.md`` file into its YAML front matter and Markdown body.

    Args:
        raw: Full file contents.
        path: Source path (for error messages only).

    Returns:
        A ``(front_matter, body)`` tuple. ``front_matter`` is a dict; ``body``
        is the raw Markdown after the closing fence (may be empty).

    Raises:
        ValueError: If the file has no leading front-matter block or the block
            is not a YAML mapping.
    """
    match = _FRONT_MATTER.match(raw)
    if match is None:
        raise ValueError(f"missing YAML front matter: {path}")
    front_matter = yaml.safe_load(match.group(1))
    if not isinstance(front_matter, dict):
        raise ValueError(f"front matter is not a mapping: {path}")
    return front_matter, match.group(2)


def _chapter_at(chapters: list[tuple[int, str]], pos: int) -> str | None:
    """Return the chapter label covering byte offset ``pos`` in the body.

    Args:
        chapters: Sorted ``(start_offset, "제N장 제목")`` pairs.
        pos: Offset of an article within the body.

    Returns:
        The most recent chapter label at or before ``pos``, or ``None`` if the
        article precedes the first chapter (or the body has no chapters).
    """
    current: str | None = None
    for start, label in chapters:
        if start <= pos:
            current = label
        else:
            break
    return current


def _collect_chapters(body: str) -> list[tuple[int, str]]:
    """Build a sorted list of ``(offset, chapter_label)`` from the body.

    The label combines the chapter number and its title (e.g. ``"제1장 총칙"``)
    so the context header can place each article in its structural section.

    Args:
        body: The Markdown body of a ``본문.md`` file.

    Returns:
        Chapter occurrences in document order.
    """
    chapters: list[tuple[int, str]] = []
    for m in _CHAPTER.finditer(body):
        number = m.group(1)
        title = (m.group(2) or "").strip()
        label = f"{number} {title}".strip()
        chapters.append((m.start(), label))
    return chapters


def _extract_articles(
    body: str,
) -> tuple[list[Article], dict[str, str]]:
    """Split a body into citable :class:`Article` units.

    The inline ``제N조(...)`` form is primary; a hash-header fallback is tried
    only if no inline articles are found. When neither matches but the body
    holds real text (common for 고시-type rules with numbered lists or tables),
    the whole body is kept as a single synthetic article so it remains
    searchable.

    Args:
        body: The Markdown body (already whitespace-trimmed by the caller).

    Returns:
        A ``(articles, chapter_by_article)`` tuple. ``chapter_by_article`` maps
        each article's ``article_no`` to its enclosing chapter label (for the
        deterministic context header); articles without a chapter are omitted.
    """
    chapters = _collect_chapters(body)
    hits = list(_ARTICLE_INLINE.finditer(body))
    if not hits:
        hits = list(_ARTICLE_HASH.finditer(body))

    if not hits:
        # No article structure but the body has substance -> single article.
        return [Article(article_no="전문", title=None, text=body.strip())], {}

    articles: list[Article] = []
    chapter_by_article: dict[str, str] = {}
    seen: set[str] = set()
    for i, hit in enumerate(hits):
        start = hit.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        text = body[start:end].strip()
        article_no = hit.group(1).strip()
        title = (hit.group(2) or "").strip() or None

        # Disambiguate duplicate labels (rare; e.g. a 부칙 re-using 제1조) so
        # chunk ids stay unique downstream.
        unique_no = article_no
        suffix = 2
        while unique_no in seen:
            unique_no = f"{article_no}#{suffix}"
            suffix += 1
        seen.add(unique_no)

        if not text:
            # Heading present but no body (e.g. trailing "제5조(생략)"): keep the
            # title as the text so the unit is not empty.
            text = title or unique_no

        chapter = _chapter_at(chapters, hit.start())
        articles.append(
            Article(
                article_no=unique_no,
                title=title,
                text=text,
                chapter_path=chapter,
            )
        )
        if chapter:
            chapter_by_article[unique_no] = chapter
    return articles, chapter_by_article


def _is_label_only(body: str) -> bool:
    """Return ``True`` when a body carries no usable legal text (B-grade).

    Args:
        body: The whitespace-trimmed Markdown body.

    Returns:
        ``True`` if the body is empty, shorter than :data:`_MIN_BODY_CHARS`, or
        consists solely of ``<img>`` placeholders.
    """
    if not body or len(body) < _MIN_BODY_CHARS:
        return True
    return bool(_IMG_ONLY.match(body))


# --------------------------------------------------------------------------- #
# Public parser                                                                 #
# --------------------------------------------------------------------------- #


def parse_file(path: Path) -> Document:
    """Parse one administrative-rule ``본문.md`` into a :class:`Document`.

    Args:
        path: Path to a ``본문.md`` file under :data:`config.ADMRULE_DIR`.

    Returns:
        A :class:`Document` with ``doc_type="admrule"``. Documents whose body is
        label-only/empty get ``trust_grade="B"`` and ``articles=[]``; otherwise
        ``trust_grade="A"`` with one or more articles.

    Raises:
        ValueError: If the file lacks valid YAML front matter or a 행정규칙ID.
        OSError: If the file cannot be read.
    """
    raw = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(raw, path)
    body = unicodedata.normalize("NFC", body).strip()

    rule_id = _as_text(front_matter.get("행정규칙ID"))
    if not rule_id:
        raise ValueError(f"missing 행정규칙ID: {path}")

    title = _as_text(front_matter.get("행정규칙명")) or path.parent.name
    law_kind = _as_text(front_matter.get("행정규칙종류")) or None
    ministry = _as_text(front_matter.get("소관부처명")) or _as_text(
        front_matter.get("상위기관명")
    )
    effective_from = _as_text(front_matter.get("시행일자")) or _as_text(
        front_matter.get("발령일자")
    )
    source_url = _as_text(front_matter.get("출처")) or None

    if _is_label_only(body):
        articles: list[Article] = []
        chapter_by_article: dict[str, str] = {}
        trust_grade = "B"
    else:
        articles, chapter_by_article = _extract_articles(body)
        trust_grade = "A"

    # Preserve original-format fields for traceability and the context header
    # (jurisdiction/ministry, attachment links, chapter map).
    meta: dict[str, Any] = {
        "행정규칙ID": rule_id,
        "행정규칙일련번호": _as_text(front_matter.get("행정규칙일련번호")) or None,
        "소관부처명": ministry or None,
        "상위기관명": _as_text(front_matter.get("상위기관명")) or None,
        "발령일자": _as_text(front_matter.get("발령일자")) or None,
        "제개정구분": _as_text(front_matter.get("제개정구분")) or None,
        "본문출처": _as_text(front_matter.get("본문출처")) or None,
        "첨부파일": front_matter.get("첨부파일"),
        "chapter_by_article": chapter_by_article or None,
    }
    # Drop empties so the artifact stays compact.
    meta = {k: v for k, v in meta.items() if v not in (None, "", [], {})}

    return Document(
        doc_id=build_doc_id("admrule", rule_id),
        doc_type="admrule",
        title=title,
        jurisdiction=ministry or "행정규칙",
        law_kind=law_kind,
        effective_from=effective_from or None,
        source_url=source_url,
        trust_grade=trust_grade,
        articles=articles,
        meta=meta,
    )


def _revision_rank(doc: Document) -> tuple[int, int, int]:
    """Sort key picking the canonical file among duplicate-行政規則ID rules.

    The corpus carries the same logical rule under several files (a current
    version plus ``_YYYY-NN`` re-issuances), all sharing one 행정규칙ID. Because
    ``doc_id = ADMRULE:{행정규칙ID}`` is deterministic (Contracts rule), we must
    emit exactly **one** Document per id or chunk ids collide downstream. The
    canonical pick is the latest revision with the richest text:

    1. highest 행정규칙일련번호 (revision sequence — newest current text),
    2. then ``trust_grade="A"`` over ``"B"`` (prefer a body over metadata-only),
    3. then the longest combined article text (most complete capture).

    Args:
        doc: A parsed :class:`Document`.

    Returns:
        A descending-preference rank tuple (larger is more canonical).
    """
    seq_raw = (doc.meta or {}).get("행정규칙일련번호") or "0"
    try:
        seq = int(re.sub(r"\D", "", str(seq_raw)) or "0")
    except ValueError:  # pragma: no cover - defensive
        seq = 0
    grade = 1 if doc.trust_grade == "A" else 0
    text_len = sum(len(a.text) for a in doc.articles)
    return (seq, grade, text_len)


def parse_all() -> Iterator[Document]:
    """Stream the canonical administrative rule per id under ``ADMRULE_DIR``.

    Globs ``ADMRULE_DIR/**/본문.md`` and parses each file. Because the corpus
    stores the same logical rule under multiple files that share a single
    행정규칙ID (current + ``_YYYY-NN`` re-issuances), and ``doc_id`` is derived
    deterministically from that id, files are **deduplicated by ``doc_id``**:
    among files sharing an id, only the canonical one (latest revision / richest
    text, see :func:`_revision_rank`) is yielded — guaranteeing globally unique
    ``doc_id``/``chunk_id`` values for the embedding and Qdrant stages.

    Malformed files are skipped with a stderr log (the run never crashes), per
    the build contract. Files are processed in sorted order for reproducibility.

    Yields:
        One :class:`Document` (``doc_type="admrule"``) per unique 행정규칙ID.
    """
    paths = sorted(config.ADMRULE_DIR.glob("**/본문.md"))
    best: dict[str, Document] = {}
    order: list[str] = []
    for path in paths:
        try:
            doc = parse_file(path)
        except Exception as exc:  # noqa: BLE001 - resilience: log and continue
            print(f"SKIP {path}: {exc}", file=sys.stderr)
            continue
        existing = best.get(doc.doc_id)
        if existing is None:
            best[doc.doc_id] = doc
            order.append(doc.doc_id)
        elif _revision_rank(doc) > _revision_rank(existing):
            # Newer revision supersedes the one seen earlier; log the drop.
            print(
                f"DEDUP {doc.doc_id}: superseded earlier revision "
                f"(seq {(existing.meta or {}).get('행정규칙일련번호')} -> "
                f"{(doc.meta or {}).get('행정규칙일련번호')})",
                file=sys.stderr,
            )
            best[doc.doc_id] = doc
        else:
            print(
                f"DEDUP {doc.doc_id}: dropped duplicate file {path.name}",
                file=sys.stderr,
            )
    for doc_id in order:
        yield best[doc_id]


def _main() -> None:
    """Write every parsed Document to :data:`config.DOCS_ADMRULE_JSONL`.

    Streams to disk (never accumulates all Documents in memory) and prints a
    one-line summary (totals + A/B split) to stdout.
    """
    out_path: Path = config.DOCS_ADMRULE_JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_a = n_b = n_articles = 0
    with out_path.open("w", encoding="utf-8") as out:
        for doc in parse_all():
            out.write(doc.model_dump_json())
            out.write("\n")
            n_total += 1
            n_articles += len(doc.articles)
            if doc.trust_grade == "A":
                n_a += 1
            else:
                n_b += 1

    print(
        f"parse_admrule: wrote {n_total} documents "
        f"(A={n_a}, B={n_b}, articles={n_articles}) -> {out_path}"
    )


if __name__ == "__main__":
    _main()
