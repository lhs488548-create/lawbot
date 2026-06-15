"""Local-ordinance (자치법규) parser — Phase 1, Task 1.4.

Parses ``02_자치법규/{광역}/.../{법령명}/본문.md`` (all **18 시도**) into the
unified :class:`ingest.schema.Document` model. Each parsed file becomes one
``Document`` whose articles are the ``제N조`` units of the ordinance body.

Format (verified on disk, see ``_BUILD_CONTRACT.md`` §c):

* The file is **YAML front matter** (``---\\n...\\n---``) followed by a Markdown
  body. Front-matter keys: ``자치법규ID, 자치법규일련번호, 자치법규명,
  자치법규종류(조례·규칙), 지자체기관명, 지자체구분{광역,기초}, 공포일자,
  공포번호, 시행일자, 담당부서, 본문출처, 출처, 첨부파일``.
* The body uses ``##### 제N조 (제목)`` **headers** (NOT the inline form the 08
  playbook prose assumed; the contract's correction applies — all three
  legal-text corpora use ``#####`` headers on this dataset). An inline fallback
  (``^제N조(제목) ...``) is kept for any stray files.
* Body-less files (only a ``# 제목`` heading plus "본문은 첨부파일 또는 원문을
  참조하세요.") yield ``trust_grade="B"`` with ``articles=[]`` — metadata is
  still emitted so the document is discoverable and honestly flagged.

The article-splitting logic (:func:`parse_bonmun`) is written to be reusable by
the administrative-rule parser (Task 1.3), which consumes the identical
``본문.md`` + ``#####`` structure.

Run as a script to materialize the corpus JSONL::

    cd /home/user1/lawbot && .venv/bin/python -m ingest.parse_ordinance

writes ``config.DOCS_ORD_JSONL`` (one ``Document.model_dump_json()`` per line).

Owner: builder (parse_ordinance). Imports the Contracts-owned schema/config;
does not modify shared files.
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

import config
from ingest.schema import Article, Document, build_doc_id

# --------------------------------------------------------------------------- #
# Regexes (module-level so they compile once)                                  #
# --------------------------------------------------------------------------- #

# Front matter: a leading ``---\n ... \n---`` block, then the body. Tolerates a
# missing trailing newline after the closing fence.
_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.S)

# Primary article splitter: a ``###``..``######`` heading whose text starts with
# ``제N조`` / ``제N조의M``, optionally followed by a parenthesised title.
_ARTICLE_HEADER_RE = re.compile(
    r"^#{3,6}\s*(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?\s*$",
    re.M,
)

# Fallback for any stray inline-style file (``제N조(제목) 본문...`` at line start
# with no ``#`` prefix). Primary is the ``#####`` header above.
_ARTICLE_INLINE_RE = re.compile(
    r"^(제\d+조(?:의\d+)?)\s*\(([^)]*)\)",
    re.M,
)

# Inline article marker **anchored at the start of a header's body block**
# (``제N조(제목) 본문...``). On a sizable slice of this corpus (~17%) the source
# ``#####`` headers are shifted by one — a phantom ``##### 제0조 (목적)`` whose
# body is just a chapter line (``제1장 총칙``), with every subsequent header's
# number/title belonging to the *next* real article — while each body block
# *restates* its own true ``제N조(제목)`` inline. When present, that inline marker
# is therefore the authoritative article identity (and the ``#####`` header is
# discarded); see ``_split_articles``. ``\s*`` lets it sit on the block's first
# non-empty line after the header.
_BODY_INLINE_RE = re.compile(
    r"^\s*(제\d+조(?:의\s*\d+)?)\s*\(([^)]*)\)\s*",
)

# A standalone chapter/section heading line (``제1장 총칙``, ``제 2 절 …``). Used
# to recognize (and drop) the phantom ``제0조`` block whose only content is such a
# heading, so it is never emitted as a spurious article.
_CHAPTER_ONLY_RE = re.compile(
    r"^제\s*\d+\s*[장절관편](?:의\s*\d+)?.*$",
)

# Sentinel body used by the source when no real text exists (별표/원문 only).
_BODYLESS_MARKERS = ("본문은 첨부파일", "본문은 원문")

# Circled numerals ①..⑮ → "(1)".."(15)" annotation (09 §B-4: 병기). We *append*
# the plain form rather than replace, preserving the original glyph for fidelity.
_CIRCLED = {chr(0x2460 + i): f"({i + 1})" for i in range(20)}

# Inline ``<개정 ...>`` / ``<신설 ...>`` amendment annotations (09 §B-4): captured
# into payload-style meta and stripped from article text to cut search noise.
_AMEND_RE = re.compile(r"<(?:개정|신설|전문개정|본조신설|본조개정)[^>]*>")


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _to_iso_date(value: Any) -> Optional[str]:
    """Coerce a front-matter date value to an ``YYYY-MM-DD`` string.

    YAML parses unquoted dates (e.g. ``공포일자: 2024-01-04``) as
    :class:`datetime.date` while quoted ones stay strings; normalize both.

    Args:
        value: A ``date``/``datetime``, an ISO-ish string, or ``None``.

    Returns:
        An ISO ``YYYY-MM-DD`` string, or ``None`` when the value is empty.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()[:10]
    return str(value).strip() or None


def _normalize_text(text: str) -> tuple[str, list[str]]:
    """Normalize article body text for embedding while preserving meaning.

    Applies the 09 §B-4 rules that are safe to do at parse time: NFC Unicode
    normalization, circled-numeral co-notation, amendment-annotation extraction,
    and whitespace collapse. The *original* glyphs are kept where it matters
    (circled numerals are co-noted, not replaced); amendment notes are pulled out
    into a returned list so downstream payload can carry them.

    Args:
        text: Raw article body.

    Returns:
        A ``(clean_text, amendments)`` tuple. ``amendments`` is the list of
        ``<개정 ...>``-style notes removed from the text (may be empty).
    """
    text = unicodedata.normalize("NFC", text)

    amendments = _AMEND_RE.findall(text)
    if amendments:
        text = _AMEND_RE.sub("", text)

    if any(g in text for g in _CIRCLED):
        for glyph, plain in _CIRCLED.items():
            # Co-note: "① ..." -> "①(1) ..." keeps the original and aids search.
            text = text.replace(glyph, f"{glyph}{plain} ")

    # Collapse runs of spaces/tabs and trailing spaces, but keep line breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), amendments


def _split_articles(body: str) -> tuple[list[Article], dict[str, list[str]]]:
    """Split an ordinance/admin-rule body into :class:`Article` units.

    Uses the ``#####``-header form first; falls back to the inline form only if
    no header matches (defensive — unobserved on this corpus but cheap).

    Args:
        body: The Markdown body following the front matter.

    Returns:
        A ``(articles, amendments_by_article)`` tuple. ``amendments_by_article``
        maps each article_no to its extracted amendment notes (for meta).
    """
    hits = list(_ARTICLE_HEADER_RE.finditer(body))
    inline = False
    if not hits:
        hits = list(_ARTICLE_INLINE_RE.finditer(body))
        inline = True
    if not hits:
        return [], {}

    articles: list[Article] = []
    amendments: dict[str, list[str]] = {}
    for i, h in enumerate(hits):
        start = h.start() if inline else h.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        segment = body[start:end]

        header_no = h.group(1)
        header_title = (h.group(2) or "").strip() or None

        # Identity resolution (corpus correction): when the body block restates
        # its own ``제N조(제목)`` inline, that inline marker is authoritative and
        # the ``#####`` header is discarded — the source shifts header
        # numbers/titles by one (phantom ``제0조`` + off-by-one) on ~17% of files,
        # but always restates the true article inline. We strip the inline marker
        # so it is not duplicated in the body. (For clean files there is no inline
        # restatement and the header identity is used as-is.)
        article_no = header_no
        title = header_title
        if not inline:
            bm = _BODY_INLINE_RE.match(segment)
            if bm is not None:
                # Canonicalize "제3조의 2" -> "제3조의2" (source sometimes spaces it).
                article_no = re.sub(r"\s+", "", bm.group(1))
                title = (bm.group(2) or "").strip() or None
                segment = segment[bm.end():]
                # A few source files double-stamp the marker
                # ("제1조(목적) 제1조(목적) 본문…"); strip any further identical
                # leading restatements so the body never opens with its own label.
                while True:
                    again = _BODY_INLINE_RE.match(segment)
                    if again is None or re.sub(r"\s+", "", again.group(1)) != article_no:
                        break
                    segment = segment[again.end():]

        clean, notes = _normalize_text(segment)
        if not clean:
            # Header with no body (e.g. a bare "제13조" tail) — keep the article
            # so the citation exists, but with empty text it adds no value; skip
            # to avoid empty chunks while preserving real ones.
            continue

        # Drop a phantom ``제0조`` whose only content is a chapter heading
        # (``제1장 총칙``): there is no Article 0 — this is the source's
        # off-by-one chapter sentinel, not a citable article.
        if article_no == "제0조" and _CHAPTER_ONLY_RE.match(clean):
            continue

        articles.append(Article(article_no=article_no, title=title, text=clean))
        if notes:
            amendments[article_no] = notes
    return articles, amendments


def parse_bonmun(path: Path) -> tuple[dict[str, Any], list[Article], dict[str, list[str]]]:
    """Parse a single ``본문.md`` file into front matter + articles.

    This is the **shared** primitive for the admin-rule (Task 1.3) and ordinance
    (Task 1.4) corpora, which have identical on-disk structure.

    Args:
        path: Absolute path to a ``본문.md`` file.

    Returns:
        A ``(front_matter, articles, amendments)`` tuple where ``front_matter``
        is the parsed YAML mapping, ``articles`` the parsed units, and
        ``amendments`` the per-article amendment notes.

    Raises:
        ValueError: If the file has no parseable YAML front-matter block, or the
            front matter is not a mapping. Callers should catch this, log, and
            skip the file rather than abort the run.
    """
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        raise ValueError("missing or malformed YAML front matter")

    front_matter = yaml.safe_load(match.group(1))
    if not isinstance(front_matter, dict):
        raise ValueError("front matter is not a mapping")

    body = match.group(2)
    articles, amendments = _split_articles(body)
    return front_matter, articles, amendments


# Sentinel ``기초`` values the source uses for non-기초 (basic-municipality)
# bodies: ``_본청`` = the 시·도 (광역) head office itself; ``_교육청`` = the
# (광역) 교육청. Anything else is a real 기초 자치단체 (구/시/군).
_WIDE_AREA_SENTINELS: Final[dict[str, str]] = {
    "_본청": "광역",
    "_교육청": "교육청",
}


def _classify_gov_level(basic_area: Optional[str]) -> tuple[str, Optional[str]]:
    """Classify an ordinance into its government level + specific locality.

    The source encodes the wide-area-vs-basic distinction in the ``기초`` slot of
    ``지자체구분``: the sentinels ``_본청``/``_교육청`` mark a 광역 (시·도) body,
    while any other value is the actual 기초 자치단체 (구/시/군). This derives an
    explicit, filter-friendly ``gov_level`` and the clean ``locality`` name so
    headers and payloads can distinguish 광역 vs 기초 (a corpus requirement) and a
    caller can scope to one 기초 단체 within a 시·도.

    Args:
        basic_area: The raw ``지자체구분.기초`` value (or ``None``).

    Returns:
        A ``(gov_level, locality)`` tuple. ``gov_level`` is one of
        ``"광역"`` | ``"교육청"`` | ``"기초"``; ``locality`` is the 기초 단체 name
        for 기초 ordinances, else ``None`` (the 광역 itself is named by
        ``jurisdiction``).
    """
    if not basic_area:
        # No 기초 marker at all ⇒ treat as the 광역 (시·도) body.
        return "광역", None
    level = _WIDE_AREA_SENTINELS.get(basic_area)
    if level is not None:
        return level, None
    return "기초", basic_area


def _build_document(fm: dict[str, Any], articles: list[Article],
                    amendments: dict[str, list[str]]) -> Document:
    """Assemble a :class:`Document` from ordinance front matter + articles.

    Args:
        fm: Parsed YAML front matter of an ordinance ``본문.md``.
        articles: Parsed article units (possibly empty ⇒ B-grade).
        amendments: Per-article amendment notes for ``meta``.

    Returns:
        A fully-populated :class:`Document` with a stable ``ORD:{광역}:{ID}`` id.

    Raises:
        ValueError: If required identifying fields (``자치법규ID``,
            ``자치법규명``, 광역 jurisdiction) are missing/empty, so the caller
            can skip+log instead of emitting an unciteable record.
    """
    ord_id = str(fm.get("자치법규ID") or "").strip()
    title = str(fm.get("자치법규명") or "").strip()

    region = fm.get("지자체구분")
    wide_area = ""
    basic_area = None
    if isinstance(region, dict):
        wide_area = str(region.get("광역") or "").strip()
        basic_area = (str(region.get("기초")).strip() or None) if region.get("기초") else None

    if not ord_id or not title or not wide_area:
        raise ValueError(
            f"missing required fields (자치법규ID={ord_id!r}, "
            f"자치법규명={title!r}, 광역={wide_area!r})"
        )

    trust_grade = "A" if articles else "B"
    attachments = fm.get("첨부파일") or []
    if not isinstance(attachments, list):
        attachments = []

    gov_level, locality = _classify_gov_level(basic_area)

    meta: dict[str, Any] = {
        "자치법규ID": ord_id,
        "자치법규일련번호": str(fm.get("자치법규일련번호") or "") or None,
        "지자체기관명": str(fm.get("지자체기관명") or "") or None,
        "기초": basic_area,
        # Derived 광역/기초 classification (filter- and header-friendly):
        # gov_level ∈ {광역, 교육청, 기초}; locality = the 기초 단체 name (구/시/군)
        # or None for a 광역/교육청 body (which is named by ``jurisdiction``).
        "gov_level": gov_level,
        "locality": locality,
        "공포일자": _to_iso_date(fm.get("공포일자")),
        "공포번호": str(fm.get("공포번호") or "") or None,
        "담당부서": str(fm.get("담당부서") or "") or None,
        "본문출처": str(fm.get("본문출처") or "") or None,
        "첨부파일": attachments,
    }
    if amendments:
        meta["amendments"] = amendments

    return Document(
        doc_id=build_doc_id("ordinance", wide_area, ord_id),
        doc_type="ordinance",
        title=title,
        jurisdiction=wide_area,
        law_kind=(str(fm.get("자치법규종류") or "").strip() or None),
        effective_from=_to_iso_date(fm.get("시행일자")),
        source_url=(str(fm.get("출처") or "").strip() or None),
        trust_grade=trust_grade,
        articles=articles,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Public interface (contract §c)                                               #
# --------------------------------------------------------------------------- #


def parse_file(path: Path) -> Document:
    """Parse one ordinance ``본문.md`` into a :class:`Document`.

    Args:
        path: Absolute path to a ``본문.md`` file under ``ORDINANCE_DIR``.

    Returns:
        The normalized :class:`Document`.

    Raises:
        ValueError: On malformed front matter or missing required fields (caller
            skips+logs).
    """
    fm, articles, amendments = parse_bonmun(path)
    return _build_document(fm, articles, amendments)


def parse_all() -> Iterator[Document]:
    """Yield a :class:`Document` for every ordinance ``본문.md`` (all 18 시도).

    Streams over ``config.ORDINANCE_DIR/**/본문.md`` (so ~160k files never load
    into memory at once). Malformed files are logged to stderr and skipped — a
    single bad file never aborts the run (contract §c).

    Yields:
        One :class:`Document` per successfully-parsed ordinance.
    """
    root = config.ORDINANCE_DIR
    for path in sorted(root.rglob("본문.md")):
        try:
            yield parse_file(path)
        except Exception as exc:  # noqa: BLE001 — robust ingest, never crash run
            rel = _safe_rel(path, root)
            print(f"SKIP {rel}: {exc}", file=sys.stderr)


def _safe_rel(path: Path, root: Path) -> str:
    """Best-effort relative path for logging (never raises)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    """Write every parsed ordinance Document to ``config.DOCS_ORD_JSONL``.

    Prints a compact summary (총/조례/규칙/A/B/skipped counts, 시도 coverage) to
    stdout. No secrets are touched or logged.
    """
    out_path: Path = config.DOCS_ORD_JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_a = 0
    n_b = 0
    n_articles = 0
    by_kind: dict[str, int] = {}
    seen_regions: set[str] = set()

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
            by_kind[doc.law_kind or "(미상)"] = by_kind.get(doc.law_kind or "(미상)", 0) + 1
            seen_regions.add(doc.jurisdiction)

    print(f"docs={n_total} A(본문有)={n_a} B(메타만)={n_b} articles={n_articles}")
    print(f"law_kind={by_kind}")
    print(f"시도수={len(seen_regions)}: {sorted(seen_regions)}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
