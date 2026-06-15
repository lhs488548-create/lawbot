"""National-law parser (Playbook 08, Task 1.1).

Parses the national-law corpus
(``01_국가법령/kr/{법령명}/{구분}.md``) into the unified
:class:`ingest.schema.Document` model. Each file is a *separate* document — a
single law folder may hold ``법률.md``, ``시행령.md``, ``시행규칙.md``,
``대통령령.md`` … and each is its own Document, distinguished by ``법령구분``
(reflected in the ``doc_id`` ``LAW:{법령ID}:{법령구분}``).

On-disk format (verified, see ``_BUILD_CONTRACT.md`` §(c))::

    ---
    제목: 도로교통법
    법령MST: 253527
    법령ID: '011463'
    법령구분: 법률
    소관부처:
    - 경찰청
    공포일자: 2024-01-30
    시행일자: 2024-07-03
    상태: 시행
    출처: https://www.law.go.kr/법령/도로교통법
    첨부파일: []
    ---

    # 도로교통법

    ##### 제1조 (목적)
    이 법은 …

Article splitting:
    * **Primary:** ``#####``-style headers ``제N조 (제목)`` / ``제N조의2`` /
      ``제N조`` (no title). On this dataset 5,622 / 5,673 files use this form.
    * **Fallback (~51 files):** short "폐지/명칭변경" laws with no article
      headers — their substantive lead paragraph (before ``## 부칙``) is emitted
      as a single synthetic article labelled ``본문`` so the content is still
      searchable, ``trust_grade="A"`` (text present). A genuinely empty body
      yields ``articles=[]`` and ``trust_grade="B"``.

This module exposes the contract's public generator
``parse_all() -> Iterator[Document]`` and, run as a script, streams one
``Document.model_dump_json()`` per line to ``config.DOCS_LAW_JSONL``.
The original article text is preserved verbatim; normalization (moving
``<개정 …>`` notes to payload, ``①②③`` → ``(1)(2)(3)`` …) happens later in
``header/build.py`` per 09 §B-4, not here.
"""

from __future__ import annotations

import datetime as _dt
import glob
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

# Allow ``python ingest/parse_statute.py`` (script form) as well as
# ``python -m ingest.parse_statute`` (package form) by ensuring the project
# root is importable. The contract recommends the ``-m`` form.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from ingest.schema import Article, Document, build_doc_id  # noqa: E402

# --------------------------------------------------------------------------- #
# Regexes (compiled once)                                                      #
# --------------------------------------------------------------------------- #
# Front matter: a leading ``---\n … \n---\n`` block followed by the body.
_FM_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.S)

# Primary article header: ``#####`` (3–6 hashes) + ``제N조`` / ``제N조의M`` with
# the remainder of the line captured raw as the (optional) title. The title is
# captured loosely (rest-of-line) rather than with a strict ``\(([^)]*)\)`` so
# real-world variants like double parens ``((목적))`` are handled (124 files /
# 1,500 header lines on this corpus use that form); :func:`_clean_title`
# normalizes it. Multiline so ``^`` matches each line start.
_ART_HASH_RE = re.compile(
    r"^#{3,6}\s*(제\d+조(?:의\d+)?)[ \t]*(.*?)[ \t]*$",
    re.M,
)

# Inline fallback header (header-less short laws): ``제N조(제목) 본문…`` at line
# start with no leading hashes. Here the body text follows on the SAME line, so
# the match must stop right after the ``(제목)`` parenthetical — group(2) is just
# the title (non-greedy, no nested parens expected inline) — leaving the body as
# the segment between this match's end and the next header.
_ART_INLINE_RE = re.compile(
    r"^(제\d+조(?:의\d+)?)[ \t]*\(([^)]*)\)",
    re.M,
)

# Chapter / section structural headings. National-law bodies render these as
# ``## 제N장 <제목>`` and ``### 제N절 <제목>`` (verified: 장 headings are
# consistently ``##`` level across the corpus; ~34% of files carry them). They
# are *not* citable units — they are tracked only to give each article its
# chapter_path for the L1 citation header / payload (09 §D-1). A trailing
# ``<개정 …>`` annotation on the heading is stripped from the captured label.
_CHAPTER_RE = re.compile(r"^#{1,4}\s*(제\d+장(?:의\d+)?)[ \t]*(.*?)[ \t]*$", re.M)
_SECTION_RE = re.compile(r"^#{1,4}\s*(제\d+절(?:의\d+)?)[ \t]*(.*?)[ \t]*$", re.M)
# Amendment/annotation tail on a heading title (e.g. "차마 … <개정 2018.3.27>").
_HEADING_AMEND_RE = re.compile(r"\s*[<＜].*$")

# The leading top-level ``# 제목`` title line of the body (dropped from article
# text — it duplicates the front-matter 제목).
_TITLE_LINE_RE = re.compile(r"^#\s+.*$", re.M)

# Start of the supplementary-provisions section. The lead-paragraph fallback
# captures only the substantive content *before* this marker.
_BUCHIK_RE = re.compile(r"^##\s*부칙", re.M)

# Label used for the synthetic single article in the no-header fallback.
_LEAD_ARTICLE_LABEL = "본문"


def _clean_title(raw: str | None) -> Optional[str]:
    """Normalize a raw article-title fragment into the title text or ``None``.

    The header regexes capture the remainder of the header line as the raw
    title, e.g. ``"(목적)"``, ``"((목적))"`` (double parens, 124 files), or
    ``""`` (a title-less ``제4조``). This strips one or more layers of wrapping
    parentheses and surrounding whitespace.

    Args:
        raw: The captured title fragment (possibly empty/``None``).

    Returns:
        The cleaned title, or ``None`` if there is none.
    """
    s = (raw or "").strip()
    # Peel matched wrapping () or full-width （） pairs (handles ((...))).
    while len(s) >= 2 and s[0] in "(（" and s[-1] in ")）":
        s = s[1:-1].strip()
    return s or None


def _to_str(value: Any) -> Optional[str]:
    """Coerce a front-matter value to a clean string (or ``None``).

    YAML parses dates to :class:`datetime.date` and some scalars to ``int``;
    this normalizes them to ISO-ish strings so downstream filtering is stable.
    Empty strings collapse to ``None``.
    """
    if value is None:
        return None
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()[:10]
    if isinstance(value, list):
        parts = [s for s in (_to_str(v) for v in value) if s]
        return ", ".join(parts) if parts else None
    s = str(value).strip()
    return s or None


def _first(value: Any) -> Optional[str]:
    """Return the first element of a YAML list (or the scalar) as a string."""
    if isinstance(value, list):
        for v in value:
            s = _to_str(v)
            if s:
                return s
        return None
    return _to_str(value)


def _clean_heading_title(raw: str | None) -> str:
    """Tidy a 장/절 heading title: drop a trailing ``<개정 …>`` annotation."""
    s = (raw or "").strip()
    s = _HEADING_AMEND_RE.sub("", s).strip()
    return s


def _collect_structural(body: str) -> list[tuple[int, str, str]]:
    """Return sorted ``(offset, level, label)`` for every 장/절 heading.

    ``level`` is ``"장"`` or ``"절"``; ``label`` combines the number and its
    cleaned title (e.g. ``"제2장 보행자의 통행방법"``). Used to compute each
    article's chapter_path; the headings themselves are never citable units.
    """
    marks: list[tuple[int, str, str]] = []
    for m in _CHAPTER_RE.finditer(body):
        title = _clean_heading_title(m.group(2))
        label = f"{m.group(1)} {title}".strip()
        marks.append((m.start(), "장", label))
    for m in _SECTION_RE.finditer(body):
        title = _clean_heading_title(m.group(2))
        label = f"{m.group(1)} {title}".strip()
        marks.append((m.start(), "절", label))
    marks.sort(key=lambda t: t[0])
    return marks


def _chapter_path_at(marks: list[tuple[int, str, str]], pos: int) -> Optional[str]:
    """Build the ``장 > 절`` chapter_path covering body offset ``pos``.

    Walks the ordered headings up to ``pos`` keeping the most recent 장 and the
    most recent 절 *within that 장*; a new 장 resets the active 절 so a section
    label never leaks across chapter boundaries.

    Args:
        marks: Sorted ``(offset, level, label)`` from :func:`_collect_structural`.
        pos: An article's start offset in the body.

    Returns:
        ``"제2장 …"``, ``"제4장 … > 제3절 …"``, or ``None`` if ``pos`` precedes any
        heading.
    """
    chapter: Optional[str] = None
    section: Optional[str] = None
    for start, level, label in marks:
        if start > pos:
            break
        if level == "장":
            chapter = label
            section = None  # a new chapter resets the active section
        else:  # "절"
            section = label
    if chapter and section:
        return f"{chapter} > {section}"
    return chapter or section


def _split_articles(body: str) -> list[Article]:
    """Split a law body into :class:`Article` units.

    Tries ``#####`` headers first (primary, on the full body), then — on the
    *main* body only (before ``## 부칙``) — the inline ``제N조(...)`` form, then
    a single-paragraph fallback for header-less short laws.

    Restricting the inline/lead fallbacks to the pre-부칙 main body is
    deliberate: on this corpus every header-less national-law file carries its
    only inline ``제N조`` markers inside 부칙 (supplementary provisions), never in
    the substantive body (verified: 101/101 such files). Treating those as the
    article would index the transitional 부칙 instead of the real content, so the
    lead-paragraph capture must win.

    Args:
        body: The Markdown body (everything after the front matter).

    Returns:
        A list of articles; possibly empty if the body has no substantive text.
    """
    hits = list(_ART_HASH_RE.finditer(body))
    if hits:
        return _articles_from_hits(body, hits, _collect_structural(body))

    # No ##### headers: work on the main body (before 부칙) so inline 제N조 in
    # 부칙 are not mistaken for the law's substantive articles.
    main = _BUCHIK_RE.split(body, maxsplit=1)[0]
    inline_hits = list(_ART_INLINE_RE.finditer(main))
    if inline_hits:
        articles = _articles_from_hits(main, inline_hits)
        if articles:
            return articles

    # Fallback: capture the lead paragraph (title line stripped) as one
    # synthetic article so the substantive content remains searchable.
    lead = _TITLE_LINE_RE.sub("", main, count=1).strip()
    if lead:
        return [Article(article_no=_LEAD_ARTICLE_LABEL, title=None, text=lead)]
    return []


def _articles_from_hits(
    body: str,
    hits: list[re.Match[str]],
    marks: list[tuple[int, str, str]] | None = None,
) -> list[Article]:
    """Build :class:`Article` units from ordered header matches over ``body``.

    Each article spans from one header's end to the next header's start (or the
    end of ``body`` for the last). Headers whose body is empty are skipped.

    Args:
        body: The text the ``hits`` index into.
        hits: Ordered, non-overlapping ``제N조`` header matches; ``group(1)`` is
            the article number, ``group(2)`` the optional parenthetical title.
        marks: Optional sorted 장/절 headings (from :func:`_collect_structural`)
            used to attach each article's ``chapter_path``. Omitted (``None``)
            for the inline/lead fallbacks, which carry no chapter structure.

    Returns:
        The articles, in document order (may be empty).
    """
    articles: list[Article] = []
    for i, h in enumerate(hits):
        start = h.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        text = body[start:end].strip()
        if not text:
            # Header with no body text -> nothing citable; skip.
            continue
        article_no = h.group(1).strip()
        # group(2) is the raw title fragment; absent/empty for a bare "제4조".
        title = _clean_title(h.group(2))
        # chapter_path uses the heading offset (h.start()) so the article is
        # placed in the chapter it falls under.
        chapter_path = _chapter_path_at(marks, h.start()) if marks else None
        articles.append(
            Article(
                article_no=article_no,
                title=title,
                text=text,
                chapter_path=chapter_path,
            )
        )
    return articles


def parse_file(path: str | Path) -> Optional[Document]:
    """Parse one national-law ``.md`` file into a :class:`Document`.

    Args:
        path: Path to a ``01_국가법령/kr/{법령명}/{구분}.md`` file.

    Returns:
        A :class:`Document`, or ``None`` if the file lacks a valid front-matter
        block or the identifying keys (``법령ID``, ``법령구분``) needed for a
        stable ``doc_id``. Such files are skipped+logged by :func:`parse_all`
        rather than crashing the run.

    Raises:
        Nothing for malformed input — returns ``None`` instead. Genuine I/O
        errors propagate to the caller (handled in :func:`parse_all`).
    """
    raw = Path(path).read_text(encoding="utf-8")
    m = _FM_RE.match(raw)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict):
        return None

    law_id = _to_str(fm.get("법령ID"))
    law_kind = _to_str(fm.get("법령구분"))
    if not law_id or not law_kind:
        # Without both, the LAW:{법령ID}:{법령구분} id is not well-formed.
        return None

    body = m.group(2)
    articles = _split_articles(body)
    # A=full text present, B=metadata only (no parseable article/section text).
    trust_grade = "A" if articles else "B"

    return Document(
        doc_id=build_doc_id("law", law_id, law_kind),
        doc_type="law",
        title=_to_str(fm.get("제목")) or "",
        jurisdiction="국가",
        law_kind=law_kind,
        effective_from=_to_str(fm.get("시행일자")),
        source_url=_to_str(fm.get("출처")),
        trust_grade=trust_grade,
        articles=articles,
        meta={
            "법령ID": law_id,
            "법령MST": _to_str(fm.get("법령MST")),
            "소관부처": _first(fm.get("소관부처")),
            "소관부처_전체": _to_str(fm.get("소관부처")),
            "공포일자": _to_str(fm.get("공포일자")),
            "공포번호": _to_str(fm.get("공포번호")),
            "상태": _to_str(fm.get("상태")),
            "첨부파일": fm.get("첨부파일") or [],
        },
    )


def parse_all(root: str | Path | None = None) -> Iterator[Document]:
    """Yield every national-law :class:`Document` (contract public API).

    Globs ``{root}/*/*.md`` and parses each file, streaming results so the whole
    corpus is never held in memory. Malformed/identifier-less files are skipped
    and logged to stderr (never crash the run, per contract §(c)).

    Args:
        root: Corpus root holding ``{법령명}/{구분}.md`` folders. Defaults to
            ``config.LAW_DIR`` (the contract's zero-arg form); an explicit root
            is accepted for tests against a synthetic corpus.

    Yields:
        One :class:`Document` per parseable file.
    """
    base = Path(root) if root is not None else config.LAW_DIR
    pattern = str(base / "*" / "*.md")
    n_ok = n_skip = 0
    for path in sorted(glob.glob(pattern)):
        try:
            doc = parse_file(path)
        except Exception as exc:  # noqa: BLE001 - never crash the batch run
            n_skip += 1
            print(f"SKIP {path}: {exc!r}", file=sys.stderr)
            continue
        if doc is None:
            n_skip += 1
            print(f"SKIP {path}: no valid front matter / identifiers", file=sys.stderr)
            continue
        n_ok += 1
        yield doc
    print(
        f"[parse_statute] parsed={n_ok} skipped={n_skip} "
        f"from {pattern}",
        file=sys.stderr,
    )


def _write_jsonl(out_path: Path) -> dict[str, int]:
    """Write all documents to ``out_path`` (one JSON per line). Returns stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_articles = n_grade_b = 0
    with out_path.open("w", encoding="utf-8") as out:
        for doc in parse_all():
            out.write(doc.model_dump_json() + "\n")
            n_docs += 1
            n_articles += len(doc.articles)
            if doc.trust_grade == "B":
                n_grade_b += 1
    return {"docs": n_docs, "articles": n_articles, "grade_b": n_grade_b}


def main() -> None:
    """Entry point: parse the corpus to ``config.DOCS_LAW_JSONL``."""
    stats = _write_jsonl(config.DOCS_LAW_JSONL)
    print(
        f"[parse_statute] wrote {stats['docs']} documents "
        f"({stats['articles']} articles, {stats['grade_b']} B-grade) "
        f"-> {config.DOCS_LAW_JSONL}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
