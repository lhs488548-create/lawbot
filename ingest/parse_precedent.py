r"""Precedent corpus parser (Playbook 08, Task 1.2).

Parses Korean court precedents under ``config.PRECEDENT_DIR`` (the
``04_판례/`` corpus) into the unified :class:`ingest.schema.Document` model,
one :class:`ingest.schema.Article` per markdown ``## 섹션`` (판결요지, 판시사항,
참조조문, 판례내용, …).

On-disk format (verified against the dataset)::

    04_판례/{사건종류}/{등급}/{법원}_{선고일}_{사건번호}.md

    ---
    판례일련번호: '424370'
    사건번호: (청주)2022누50008
    사건명: 학교용지 부담금 부과처분 취소(...)
    법원명: 대전고등법원
    법원등급: 하급심
    사건종류: 선거·특별
    출처: https://www.law.go.kr/LSW/precInfoP.do?precSeq=424370
    첨부파일: []
    선고일자: 2022-05-25
    ---

    # 학교용지 부담금 부과처분 취소(...)

    ## 판결요지
    ...
    ## 판례내용
    ...

Key facts the parser depends on (surveyed on the real corpus, ~123,742 files):

- Every file is **YAML front matter + Markdown body** delimited by ``---``.
- Sections are ``^##\s+(.+)$`` headers; an optional leading ``# {사건명}`` H1 is
  ignored. Every file has at least ``## 판례내용``; section presence otherwise
  varies (판결요지/판시사항/참조조문/참조판례 are frequently absent — that is fine,
  we emit only the sections that exist).
- ``선고일자`` is **frequently empty** (``None``); when so, the date is recovered
  from the filename (``{법원}_{선고일}_{사건번호}.md``). A placeholder
  ``0000-00-00`` (in either place) is treated as *no date*.
- ``판례일련번호`` is the stable unique id ⇒ ``doc_id = PREC:{판례일련번호}``.
- Bodies are already partly de-identified with fillers (``○○○``, ``△△△`` …);
  these are normalized to ``[당사자]`` so embedding text is consistent and free
  of raw filler noise (03 비식별 정규화).

Contract interface (``_BUILD_CONTRACT.md`` §c):

    def parse_all() -> Iterator[Document]: ...

Running the module as a script streams every parsed :class:`Document` as one
``model_dump_json()`` line into :data:`config.DOCS_PREC_JSONL`. The run is
**streaming** (12만 건) — files are processed one at a time and never all held
in memory — and **never crashes** on a malformed file: such files are skipped
and logged to ``stderr``.

Owner: builder ``parse_precedent``. This module adds only itself; the shared
``config.py`` / ``ingest/schema.py`` are owned by Contracts.

Usage (WSL venv)::

    cd /home/user1/lawbot && .venv/bin/python -m ingest.parse_precedent
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


class _StringDateLoader(yaml.SafeLoader):
    """A SafeLoader that does **not** auto-convert ISO dates to ``datetime``.

    Two problems with the default resolver on this corpus:

    1. ``선고일자: 2020-01-01`` would deserialize to a :class:`datetime.date`
       object, but ``Document.effective_from`` is a plain ISO string.
    2. The placeholder ``선고일자: 0000-00-00`` makes the default loader raise
       ``ValueError: year 0 is out of range`` and abort the whole file.

    Removing the implicit ``tag:yaml.org,2002:timestamp`` resolver keeps every
    scalar a string, so dates are handled uniformly (and ``0000-00-00`` is a
    harmless string we later normalize to ``None``).
    """


# Drop only the implicit timestamp resolver; keep all other resolvers intact.
_StringDateLoader.yaml_implicit_resolvers = {
    first_char: [
        (tag, regexp)
        for tag, regexp in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for first_char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

# --------------------------------------------------------------------------- #
# Regexes                                                                      #
# --------------------------------------------------------------------------- #

# YAML front matter block at the very start of the file: ---\n...\n---\n<body>.
# Non-greedy so it stops at the *first* closing fence.
_FRONT_MATTER = re.compile(r"^﻿?---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)

# Section header: a level-2 markdown heading. The capture is the section title.
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.M)

# Filename pattern: {법원}_{선고일}_{사건번호}.md (선고일 = YYYY-MM-DD).
_FILENAME_DATE = re.compile(r"_(\d{4}-\d{2}-\d{2})_")

# Placeholder "no date" forms that appear in front matter / filenames.
_NULL_DATE = re.compile(r"^0{4}-0{2}-0{2}$")

# De-identification fillers used by the source data for masked parties / names /
# places. Normalized to a single readable token so embedding text is stable and
# we do not leak rows of raw filler characters into the vector (03 §비식별).
_DEID_FILLERS = re.compile(r"[○◯⊙]{2,}|[△▲]{2,}|[□■]{3,}|[☆★]{2,}|[×✕]{3,}")

# Collapse runs of intra-line whitespace (not newlines) introduced by masking.
_INLINE_WS = re.compile(r"[ \t]{2,}")

# Front-matter keys we surface as a doc_type-specific meta block (everything
# else from the YAML is also preserved under ``meta`` verbatim).
_META_KEYS = (
    "판례일련번호",
    "사건번호",
    "사건명",
    "법원명",
    "법원등급",
    "사건종류",
    "출처",
    "선고일자",
    "첨부파일",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _clean_date(value: Any) -> str | None:
    """Normalize a 선고일자 value to ``YYYY-MM-DD`` or ``None``.

    Treats empty values and the ``0000-00-00`` placeholder as "no date".

    Args:
        value: Raw ``선고일자`` value from YAML (may be ``None``, a date, or a
            string).

    Returns:
        An ISO date string, or ``None`` when no real date is available.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or _NULL_DATE.match(s):
        return None
    return s


def _date_from_filename(path: Path) -> str | None:
    """Recover the decision date from the ``{법원}_{date}_{번호}.md`` filename.

    Args:
        path: The precedent file path.

    Returns:
        The ``YYYY-MM-DD`` date embedded in the filename, or ``None`` when it is
        absent or the placeholder ``0000-00-00``.
    """
    m = _FILENAME_DATE.search(path.name)
    if not m:
        return None
    date = m.group(1)
    return None if _NULL_DATE.match(date) else date


def normalize_text(text: str) -> str:
    """Normalize a precedent section body for downstream embedding/storage.

    Applies the de-identification and whitespace rules from 03 (비식별 정규화):
    Unicode NFC, masked-party fillers (``○○○`` …) collapsed to ``[당사자]``,
    runs of inline whitespace collapsed, and trailing/leading blank lines
    trimmed. Newlines (paragraph structure) are preserved.

    The header builder (``header/``) may apply further normalization; this is
    the parser-level pass that removes raw masking noise and unifies encoding.

    Args:
        text: Raw section body.

    Returns:
        The normalized text. Empty input yields an empty string.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFC", text)
    out = _DEID_FILLERS.sub("[당사자]", out)
    # Collapse intra-line whitespace runs, but keep newlines.
    out = "\n".join(_INLINE_WS.sub(" ", line) for line in out.split("\n"))
    return out.strip()


def _split_sections(body: str) -> list[Article]:
    """Split a precedent body into one :class:`Article` per ``## 섹션``.

    A leading ``# {사건명}`` H1 (and any preamble before the first ``##``) is
    ignored. Sections whose body is empty after normalization are skipped so we
    never emit an empty citable unit.

    Args:
        body: The markdown body following the YAML front matter.

    Returns:
        A list of :class:`Article` (possibly empty if no usable section exists).
    """
    articles: list[Article] = []
    matches = list(_SECTION.finditer(body))
    for i, m in enumerate(matches):
        section_name = m.group(1).strip()
        if not section_name:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[start:end]
        text = normalize_text(raw)
        if not text:
            continue
        articles.append(Article(article_no=section_name, title=None, text=text))
    return articles


def parse_file(path: Path) -> Document | None:
    """Parse a single precedent ``.md`` file into a :class:`Document`.

    Args:
        path: Path to a ``04_판례/**/*.md`` file.

    Returns:
        A :class:`Document`, or ``None`` if the file is malformed (no front
        matter, no usable id). Callers should skip-and-log a ``None`` result.

    Raises:
        Nothing for ordinary content problems — these return ``None``. Only
        truly unexpected errors (e.g. unreadable file) propagate to the caller,
        which logs and continues.
    """
    raw = path.read_text(encoding="utf-8")
    m = _FRONT_MATTER.match(raw)
    if not m:
        print(f"SKIP (no front matter): {path}", file=sys.stderr)
        return None

    fm_block, body = m.group(1), m.group(2)
    try:
        fm = yaml.load(fm_block, Loader=_StringDateLoader) or {}
    except yaml.YAMLError as exc:
        print(f"SKIP (bad YAML): {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(fm, dict):
        print(f"SKIP (front matter not a mapping): {path}", file=sys.stderr)
        return None

    seq = fm.get("판례일련번호")
    seq = "" if seq is None else str(seq).strip()
    if not seq:
        print(f"SKIP (no 판례일련번호): {path}", file=sys.stderr)
        return None

    # Decision date: front matter first, then recover from the filename.
    effective = _clean_date(fm.get("선고일자")) or _date_from_filename(path)

    articles = _split_sections(body)
    # A precedent with no usable section is metadata-only (trust grade B). We
    # still emit it so the case is discoverable, but flag it honestly.
    trust_grade = "A" if articles else "B"

    title = str(fm.get("사건명") or fm.get("사건번호") or seq).strip()
    court = str(fm.get("법원명") or "").strip() or "법원미상"

    meta: dict[str, Any] = {k: fm.get(k) for k in _META_KEYS if k in fm}
    # Preserve any other front-matter fields verbatim for traceability.
    for k, v in fm.items():
        if k not in meta:
            meta[k] = v

    return Document(
        doc_id=build_doc_id("precedent", seq),
        doc_type="precedent",
        title=title,
        jurisdiction=court,
        law_kind=(str(fm.get("사건종류")).strip() if fm.get("사건종류") else None),
        effective_from=effective,
        source_url=(str(fm.get("출처")).strip() if fm.get("출처") else None),
        trust_grade=trust_grade,
        articles=articles,
        meta=meta,
    )


def _iter_precedent_files(root: Path) -> Iterator[Path]:
    """Yield every precedent ``.md`` file under ``root`` (excluding READMEs).

    Streams the directory tree with :meth:`Path.rglob` so the 12만-file corpus
    is never materialized as a list.

    Args:
        root: The precedent corpus root (``config.PRECEDENT_DIR``).

    Yields:
        File paths, in filesystem order.
    """
    for path in root.rglob("*.md"):
        if path.name == "README.md":
            continue
        yield path


def parse_all(root: Path | None = None) -> Iterator[Document]:
    """Stream every precedent :class:`Document` under ``root``.

    This is the public parser interface required by ``_BUILD_CONTRACT.md`` §c.
    Malformed files are skipped (logged to ``stderr``) and never abort the run.

    Args:
        root: Corpus root to walk. Defaults to :data:`config.PRECEDENT_DIR`.

    Yields:
        Parsed :class:`Document` instances (one per source file that has a
        usable id), one at a time (streaming).
    """
    base = root or config.PRECEDENT_DIR
    for path in _iter_precedent_files(base):
        try:
            doc = parse_file(path)
        except Exception as exc:  # never let one bad file abort 12만 files
            print(f"SKIP (error): {path}: {exc}", file=sys.stderr)
            continue
        if doc is not None:
            yield doc


def main() -> None:
    """Parse the whole corpus and write :data:`config.DOCS_PREC_JSONL`.

    Streams one :meth:`Document.model_dump_json` per line. Prints a final count
    summary (documents written, A-grade vs B-grade) to ``stderr``.
    """
    out_path = config.DOCS_PREC_JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = n_b_grade = 0
    with out_path.open("w", encoding="utf-8") as out:
        for doc in parse_all():
            out.write(doc.model_dump_json())
            out.write("\n")
            n_written += 1
            if doc.trust_grade == "B":
                n_b_grade += 1
            if n_written % 10000 == 0:
                print(f"... {n_written} precedents written", file=sys.stderr)

    print(
        f"DONE: wrote {n_written} precedents to {out_path} "
        f"({n_written - n_b_grade} A-grade, {n_b_grade} B-grade/metadata-only).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
