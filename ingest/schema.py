"""Unified ingest schema (Playbook 08, Task 1.0).

Every one of the four source formats (national law, local ordinance,
administrative rule, court precedent) is normalized into a single
:class:`Document` made of :class:`Article` units. Downstream stages (chunking,
embedding, retrieval, RAG) depend only on this schema and never on the raw
on-disk formats.

Owner: Contracts. The four parsers in ``ingest/`` must each return instances of
:class:`Document`; builders extend parsers, not this schema.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Document kind. Drives payload filters and citation formatting downstream.
DocType = Literal["law", "ordinance", "admrule", "precedent"]

# Trust grade: "A" => full original text is present, "B" => metadata only
# (e.g. an administrative rule whose body is missing). Builders must surface
# B-grade documents honestly ("metadata only") in answers.
TrustGrade = Literal["A", "B"]


class Article(BaseModel):
    """A single citable unit within a document.

    For statutes/ordinances/admin-rules this is one article ("제N조"); for
    precedents it is one section ("판결요지", "판례내용", ...).

    Attributes:
        article_no: Stable label, e.g. "제4조", "제4조의2", or a precedent
            section name such as "판결요지".
        title: Optional parenthetical article title (e.g. "목적"); often
            ``None`` for precedent sections.
        text: The article/section body text (header stripped).
        chapter_path: Optional structural location of this article within the
            document's 장/절 hierarchy, e.g. ``"제2장 차마 및 노면전차의 통행방법"``
            or ``"제4장 압수물의 처분 > 제3절 몰수물처분"``. Tracked by the
            statute/admin-rule parsers and rendered into the L1 citation header
            and payload (09 §D-1). ``None`` when the document has no chapter
            structure (or for precedents).
    """

    article_no: str
    title: Optional[str] = None
    text: str
    chapter_path: Optional[str] = None

    @field_validator("article_no")
    @classmethod
    def _non_empty_no(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("article_no must be non-empty")
        return v.strip()


class Document(BaseModel):
    """A normalized legal document.

    Attributes:
        doc_id: Stable, deterministic unique id (see ``doc_id`` rules below).
            Never derive this from a git commit hash (07 verification).
        doc_type: One of the four :data:`DocType` values.
        title: Law name / case name.
        jurisdiction: "국가" for national law, the wide-area local government
            name (e.g. "전라남도") for ordinances, the issuing ministry for
            admin-rules, or the court name for precedents.
        law_kind: Sub-kind, e.g. "법률" | "시행령" | "조례" | "훈령" | the
            precedent case category. ``None`` if unknown.
        effective_from: Enforcement date (statute) or decision date
            (precedent), kept as an ISO-ish string for stable filtering.
        source_url: Canonical source URL (law.go.kr) when available.
        trust_grade: "A" (text present) or "B" (metadata only).
        articles: The citable units. May be empty for B-grade documents.
        meta: Original-format fields preserved verbatim (ministry, attachment
            tables, internal ids, ...) for traceability.
    """

    doc_id: str
    doc_type: DocType
    title: str
    jurisdiction: str
    law_kind: Optional[str] = None
    effective_from: Optional[str] = None
    source_url: Optional[str] = None
    trust_grade: TrustGrade = "A"
    articles: list[Article] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def _well_formed_doc_id(cls, v: str) -> str:
        if not DOC_ID_RE.match(v):
            raise ValueError(
                f"doc_id {v!r} does not match the required "
                f"PREFIX:...:... convention (see build_doc_id)."
            )
        return v


# --------------------------------------------------------------------------- #
# doc_id rules (Playbook 08, Task 1.0)                                          #
#   law      : LAW:{법령ID}:{법령구분}        e.g. LAW:014565:법률              #
#   ordinance: ORD:{광역}:{자치법규ID}        e.g. ORD:경상북도:2206415         #
#   admrule  : ADMRULE:{행정규칙ID}           e.g. ADMRULE:58423                #
#   precedent: PREC:{판례일련번호}            e.g. PREC:424370                  #
# --------------------------------------------------------------------------- #

# Allow Korean and word characters in the variable segments; the prefix is
# fixed to the four known corpora.
DOC_ID_RE = re.compile(r"^(LAW|ORD|ADMRULE|PREC):.+")

_PREFIX: dict[DocType, str] = {
    "law": "LAW",
    "ordinance": "ORD",
    "admrule": "ADMRULE",
    "precedent": "PREC",
}


def build_doc_id(doc_type: DocType, *parts: str) -> str:
    """Build a stable ``doc_id`` from its identifying parts.

    Args:
        doc_type: The document kind, selecting the id prefix.
        *parts: The identifying segments, in the order defined by the rules
            above (e.g. for ``law``: 법령ID, 법령구분).

    Returns:
        A colon-joined id such as ``"LAW:014565:법률"``.

    Raises:
        ValueError: If ``doc_type`` is unknown or any part is empty.
    """
    if doc_type not in _PREFIX:
        raise ValueError(f"Unknown doc_type: {doc_type!r}")
    cleaned: list[str] = []
    for p in parts:
        s = "" if p is None else str(p).strip()
        if not s:
            raise ValueError(
                f"Empty doc_id part for doc_type={doc_type!r}; parts={parts!r}"
            )
        cleaned.append(s)
    if not cleaned:
        raise ValueError(f"doc_id for {doc_type!r} requires at least one part")
    return ":".join([_PREFIX[doc_type], *cleaned])


# --------------------------------------------------------------------------- #
# Parent / child linkage (09 §B-1)                                              #
#   child  = one article / precedent section  -> the search & embedding unit    #
#   parent = the whole law / precedent        -> the generation & pack unit      #
# A child chunk's ``parent_id`` is exactly its owning ``doc_id`` (the law /      #
# precedent). The child's globally-unique ``chunk_id`` is                        #
# ``{doc_id}#{article_no}#{part_idx}``. Keeping both helpers here (Contracts)    #
# guarantees every builder forms identical ids.                                  #
# --------------------------------------------------------------------------- #


def build_chunk_id(doc_id: str, article_no: str, part_idx: int = 0) -> str:
    """Build the globally-unique child chunk id.

    Args:
        doc_id: The owning document (parent) id, e.g. ``"LAW:014565:법률"``.
        article_no: The article/section label, e.g. ``"제4조"`` or ``"판결요지"``.
        part_idx: Sub-split index (0 unless the article exceeded the token limit
            and was windowed; see 09 §B-2).

    Returns:
        ``"{doc_id}#{article_no}#{part_idx}"``.

    Raises:
        ValueError: If ``doc_id`` or ``article_no`` is empty.
    """
    did = (doc_id or "").strip()
    ano = (article_no or "").strip()
    if not did or not ano:
        raise ValueError(
            f"build_chunk_id requires non-empty doc_id/article_no; "
            f"got doc_id={doc_id!r}, article_no={article_no!r}"
        )
    return f"{did}#{ano}#{int(part_idx)}"


def parent_id_of(chunk_id: str) -> str:
    """Return the parent (document) id for a child ``chunk_id``.

    The parent id is the substring before the first ``'#'`` separator, i.e. the
    owning ``doc_id``. Used to promote a child hit to its parent's full text.

    Args:
        chunk_id: A child chunk id produced by :func:`build_chunk_id`.

    Returns:
        The owning ``doc_id``.
    """
    return chunk_id.split("#", 1)[0]


__all__ = [
    "DocType",
    "TrustGrade",
    "Article",
    "Document",
    "build_doc_id",
    "build_chunk_id",
    "parent_id_of",
    "DOC_ID_RE",
]
