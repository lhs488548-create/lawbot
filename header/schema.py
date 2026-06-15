"""Header schema — the single source of truth for header governance (09 §D-3).

This module is the **one place** that declares, per ``doc_type``:

* which L1 (citation-header) fields are *required* — :data:`HEADER_REQUIRED`;
* which payload keys every chunk must carry — :data:`PAYLOAD_REQUIRED`;
* the closed set of ``kind`` values and ``trust_grade`` values.

``header/build.py`` produces headers/payloads to satisfy this schema and
``header/validate.py`` enforces it over every chunk (in the ingest pipeline and
in ``pytest``). Keeping the definition here — and importing it from both the
builder and the validator — guarantees the rules cannot drift apart (the whole
point of the "헤더 거버넌스 = 강제 스키마 + 검증기 + 테스트" requirement).

The L1 / L2 / body layout this schema governs (09 §D-1)::

    [법령] 도로교통법 > 제2장 > 제17조(자동차등의 속도) · 시행 2026-04-02 · 소관 경찰청   <- L1 인용헤더
    이 조문은 '도로교통법'(법률)의 일부로, …를 규정한다.                                   <- L2 맥락헤더
    (정규화된 조문 본문)                                                                    <- 본문

Owner: header builder. Imports nothing project-specific except the shared
``ingest.schema.DocType`` so the doc-type set has one definition.
"""

from __future__ import annotations

from typing import Final, get_args

from ingest.schema import DocType

# --------------------------------------------------------------------------- #
# L1 (citation header) required fields, per doc_type.                          #
# --------------------------------------------------------------------------- #
# These are the *semantic* fields the L1 line must encode for a citation to be
# self-sufficient (09 §D-3(a): "L1 필수필드(법령명/식별자·조문번호·시행일/선고일)").
# The validator checks the corresponding payload values are present AND that the
# rendered L1 string actually contains them — so a malformed/empty header line
# fails even if the payload happens to be complete.
#
# For legal texts (law/ordinance/admrule): law name + article number + effective
# date. For precedents the L1 cites by court + decision date + section (the case
# *name* is long and not part of the citation crumb), so the required identifiers
# differ — hence the per-doc_type mapping. Every field listed here is both a
# payload key AND rendered into the L1 line, so the validator can confirm it by
# substring (see header.validate._l1_tokens).
HEADER_REQUIRED: Final[dict[str, list[str]]] = {
    "law": ["title", "article_no", "effective_from"],
    "ordinance": ["title", "article_no", "effective_from"],
    "admrule": ["title", "article_no", "effective_from"],
    # precedent: 법원(jurisdiction) + 선고일(effective_from) + 섹션(article_no).
    "precedent": ["jurisdiction", "article_no", "effective_from"],
}

# --------------------------------------------------------------------------- #
# Payload (structured meta) required keys (09 §D-2) — the filter + citation     #
# surface that must exist on EVERY chunk regardless of doc_type.               #
# --------------------------------------------------------------------------- #
# The four retrieval *filter* keys are mandatory and non-null wherever the data
# allows; the remaining provenance keys must be present (value may legitimately
# be None for, e.g., a missing source_url, but the KEY must exist so downstream
# code can rely on it).
PAYLOAD_FILTER_KEYS: Final[tuple[str, ...]] = (
    "doc_type",
    "jurisdiction",
    "law_kind",
    "effective_from",
)

# Of the filter keys, these must be non-null on EVERY chunk of EVERY doc_type.
# (``effective_from`` is deliberately *not* here: see FILTER_KEY_NULLABLE.)
PAYLOAD_FILTER_KEYS_STRICT: Final[tuple[str, ...]] = (
    "doc_type",
    "jurisdiction",
    "law_kind",
)

# doc_type → filter keys whose value may legitimately be ``None`` for that type
# (the KEY must still exist; only the non-null *value* requirement is relaxed).
#
# ``effective_from`` (= 선고일자) is genuinely absent for a real ~10% of court
# precedents: the source 선고일자 is empty and the filename carries the
# placeholder ``0000-00-00``. Such a case is still fully citable (법원 + 사건번호
# + 섹션) and must remain discoverable, so a null 선고일 is *allowed data*, not a
# header defect — forcing it non-null would either fail the build (09 §D-3) or
# silently drop ~12,000 precedents. Legal texts (법령/자치법규/행정규칙) always
# carry a 시행일, so for those ``effective_from`` stays strictly non-null.
# ``as_of_date`` filtering (09 §E-1) treats a null effective_from as
# "effective over all time" (no lower bound), which is the safe default for an
# undated precedent.
FILTER_KEY_NULLABLE: Final[dict[str, frozenset[str]]] = {
    "precedent": frozenset({"effective_from"}),
}

PAYLOAD_REQUIRED: Final[tuple[str, ...]] = (
    # filter keys (09 §E-1 metadata pre-filter)
    "doc_type",
    "jurisdiction",
    "law_kind",
    "effective_from",
    # identity / linkage (09 §B-1 parent/child)
    "doc_id",
    "parent_id",
    "article_no",
    "part_idx",
    # citation / provenance (09 §A common meta + §D-2)
    "title",
    "trust_grade",
    "source_url",
    "license",
    "kind",
)

# --------------------------------------------------------------------------- #
# Closed value sets.                                                            #
# --------------------------------------------------------------------------- #
# Allowed doc_type values — derived from the shared Literal so there is exactly
# one definition of the corpus set across the codebase.
DOC_TYPES: Final[tuple[str, ...]] = tuple(get_args(DocType))

# Chunk kind: "본문" = real article/section body; "별표" = attached table/form
# with its own body; "메타" = metadata-only (B-grade, label-only) sentinel chunk.
KINDS: Final[tuple[str, ...]] = ("본문", "별표", "메타")

# Trust grade mirrors ingest.schema.TrustGrade ("A" text present, "B" meta only).
TRUST_GRADES: Final[tuple[str, ...]] = ("A", "B")

# Layout constant: the two-layer header is two lines, body follows. The validator
# splits the embed text into at most these many leading header lines.
N_HEADER_LINES: Final[int] = 2


__all__ = [
    "HEADER_REQUIRED",
    "PAYLOAD_FILTER_KEYS",
    "PAYLOAD_FILTER_KEYS_STRICT",
    "FILTER_KEY_NULLABLE",
    "PAYLOAD_REQUIRED",
    "DOC_TYPES",
    "KINDS",
    "TRUST_GRADES",
    "N_HEADER_LINES",
]
