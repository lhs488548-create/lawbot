"""Unit tests for the chunker (``embed/chunk.py``) — 09 §B / BUILD CONTRACT (d).

These tests are **offline** (no OpenAI / Qdrant calls): chunking and header
building are deterministic and free. The header-governance gate is exercised by
running :func:`header.validate.validate_chunk` over every produced chunk and
asserting **zero** errors (09 §D-3: a header regression fails the build).

Run from the project root with the WSL venv::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_chunk.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from embed import chunk
from header.validate import validate_chunk, validate_file
from ingest.schema import parent_id_of


# --------------------------------------------------------------------------- #
# Fixtures — small, representative documents for each corpus (no I/O).          #
# --------------------------------------------------------------------------- #
def _law_doc() -> dict:
    """A national-law document with two articles (one with a circled-항 body)."""
    return {
        "doc_id": "LAW:000325:법률",
        "doc_type": "law",
        "title": "항공우주산업개발 촉진법",
        "jurisdiction": "국가",
        "law_kind": "법률",
        "effective_from": "2024-07-31",
        "source_url": "https://www.law.go.kr/법령/항공우주산업개발촉진법",
        "trust_grade": "A",
        "articles": [
            {"article_no": "제1조", "title": "목적", "text": "이 법은 …을 목적으로 한다."},
            {
                "article_no": "제3조",
                "title": "기본계획의 수립",
                "text": (
                    "**①** 정부는 기본계획을 수립하여야 한다. "
                    "<개정 2007.4.27, 2013.4.5>\n"
                    "**②** 정부는 시행계획을 수립하여 시행하여야 한다."
                ),
            },
        ],
        "meta": {"소관부처": ["우주항공청"], "법령ID": "000325"},
    }


def _precedent_doc() -> dict:
    """A precedent with two sections, one containing an anonymized party glyph."""
    return {
        "doc_id": "PREC:424370",
        "doc_type": "precedent",
        "title": "학교용지 부담금 부과처분 취소",
        "jurisdiction": "대전고등법원",
        "law_kind": "선거·특별",
        "effective_from": "2022-05-25",
        "source_url": "https://www.law.go.kr/LSW/precInfoP.do?precSeq=424370",
        "trust_grade": "A",
        "articles": [
            {"article_no": "판결요지", "title": None, "text": "학교용지부담금은 … 위법함."},
            {"article_no": "판례내용", "title": None, "text": "원고 ○○○과 피고 사이의 …"},
        ],
        "meta": {"사건번호": "(청주)2022누50008", "사건종류": "선거·특별"},
    }


def _admrule_with_attachment() -> dict:
    """An admin rule whose only content is a label-only 별지 attachment."""
    return {
        "doc_id": "ADMRULE:58423",
        "doc_type": "admrule",
        "title": "중소벤처기업부 인사관리 규정",
        "jurisdiction": "중소벤처기업부",
        "law_kind": "훈령",
        "effective_from": "2026-01-26",
        "source_url": "https://www.law.go.kr/행정규칙/중소벤처기업부인사관리규정",
        "trust_grade": "A",
        "articles": [
            {"article_no": "제1조", "title": "목적", "text": "이 규정은 인사관리에 관한 사항을 정한다."}
        ],
        "meta": {
            "소관부처명": "중소벤처기업부",
            "첨부파일": [
                {
                    "별표번호": "0001",
                    "별표가지번호": "00",
                    "별표구분": "별지",
                    "제목": "심의결정서",
                    "파일링크": "https://www.law.go.kr/LSW/flDownload.do?flSeq=161210737",
                    "PDF링크": "https://www.law.go.kr/LSW/flDownload.do?flSeq=161210901",
                }
            ],
        },
    }


def _bgrade_doc() -> dict:
    """A metadata-only (B-grade) document with no article bodies."""
    return {
        "doc_id": "ADMRULE:99999",
        "doc_type": "admrule",
        "title": "본문결측 규칙",
        "jurisdiction": "테스트부",
        "law_kind": "고시",
        "effective_from": "2025-01-01",
        "source_url": None,
        "trust_grade": "B",
        "articles": [],
        "meta": {"소관부처명": "테스트부"},
    }


# --------------------------------------------------------------------------- #
# Structure & contract-shape tests.                                            #
# --------------------------------------------------------------------------- #
def test_law_chunk_shape_and_header():
    chunks = list(chunk.chunks_of(_law_doc()))
    assert len(chunks) == 2  # one chunk per article, neither over the limit
    c = chunks[0]
    # Required top-level keys (BUILD CONTRACT (d)).
    for key in ("chunk_id", "doc_id", "parent_id", "text", "content_hash", "payload"):
        assert key in c, f"missing top-level key {key}"
    assert c["doc_id"] == "LAW:000325:법률"
    assert c["parent_id"] == c["doc_id"] == parent_id_of(c["chunk_id"])
    assert c["chunk_id"] == "LAW:000325:법률#제1조#0"
    # content_hash is a 64-char sha256 hex of the text.
    assert len(c["content_hash"]) == 64
    # Two-layer header: L1 then L2 then body.
    l1, l2, body = c["text"].split("\n", 2)
    assert l1.startswith("[법령]")
    assert "항공우주산업개발 촉진법" in l1
    assert "제1조" in l1
    assert "2024-07-31" in l1  # effective_from in L1 (validator requirement)
    assert l2.strip()  # non-empty L2 context line
    assert body.strip()


def test_normalization_applied_to_embed_text():
    chunks = list(chunk.chunks_of(_law_doc()))
    # 제3조 body had **①**/**②** and a <개정 …> annotation.
    art3 = next(c for c in chunks if "제3조" in c["chunk_id"])
    text = art3["text"]
    # Circled markers normalized to (1)/(2); amendment annotation stripped.
    assert "(1)" in text and "(2)" in text
    assert "<개정" not in text
    assert "①" not in text and "②" not in text
    # Amendment preserved in payload, not lost.
    assert any("개정" in a for a in art3["payload"].get("amendments", []))


def test_precedent_party_anonymization_and_header():
    chunks = list(chunk.chunks_of(_precedent_doc()))
    assert len(chunks) == 2
    content = next(c for c in chunks if "판례내용" in c["chunk_id"])
    assert "○○○" not in content["text"]
    assert "[당사자]" in content["text"]
    l1 = content["text"].split("\n", 1)[0]
    assert l1.startswith("[판례]")
    assert "대전고등법원" in l1
    assert "2022-05-25" in l1


def test_attachment_emits_bgrade_byeolpyo_chunk():
    chunks = list(chunk.chunks_of(_admrule_with_attachment()))
    # One real article + one 별표 metadata chunk.
    assert len(chunks) == 2
    byeolpyo = [c for c in chunks if c["payload"]["kind"] == "별표"]
    assert len(byeolpyo) == 1
    bp = byeolpyo[0]
    assert bp["payload"]["trust_grade"] == "B"
    assert bp["payload"]["attachment_link"].startswith("https://")
    assert bp["parent_id"] == "ADMRULE:58423"


def test_bgrade_document_yields_single_meta_chunk():
    chunks = list(chunk.chunks_of(_bgrade_doc()))
    assert len(chunks) == 1
    c = chunks[0]
    assert c["payload"]["kind"] == "메타"
    assert c["payload"]["trust_grade"] == "B"
    assert c["chunk_id"] == "ADMRULE:99999#메타#0"


def test_chunk_ids_are_unique_across_corpora():
    seen: set[str] = set()
    for doc in (_law_doc(), _precedent_doc(), _admrule_with_attachment(), _bgrade_doc()):
        for c in chunk.chunks_of(doc):
            assert c["chunk_id"] not in seen, f"dup {c['chunk_id']}"
            seen.add(c["chunk_id"])


# --------------------------------------------------------------------------- #
# Second-split (09 §B-2): over-long article -> overlapping windows.            #
# --------------------------------------------------------------------------- #
def _long_precedent(n_sentences: int = 4000) -> dict:
    """A precedent section deliberately exceeding EMBED_MAX_TOKENS tokens."""
    body = " ".join(f"이것은 {i}번째 문장이다." for i in range(n_sentences))
    return {
        "doc_id": "PREC:111111",
        "doc_type": "precedent",
        "title": "장문 판례",
        "jurisdiction": "대법원",
        "law_kind": "민사",
        "effective_from": "2020-01-01",
        "source_url": "https://example.test/prec",
        "trust_grade": "A",
        "articles": [{"article_no": "판례내용", "title": None, "text": body}],
        "meta": {"사건번호": "2020다1", "사건종류": "민사"},
    }


def test_overlong_article_is_windowed_and_within_limit():
    doc = _long_precedent()
    enc = chunk._encoder()
    # Sanity: the single section really is over the limit before splitting.
    assert len(enc.encode(doc["articles"][0]["text"])) > config.EMBED_MAX_TOKENS

    chunks = list(chunk.chunks_of(doc))
    assert len(chunks) > 1, "over-long section must be split into >1 window"
    # Every window fits the model limit; part_idx increments; ids stay unique.
    part_indices = []
    ids = set()
    for c in chunks:
        assert len(enc.encode(c["text"])) <= config.EMBED_MAX_TOKENS
        assert c["parent_id"] == "PREC:111111"
        part_indices.append(c["payload"]["part_idx"])
        assert c["chunk_id"] not in ids
        ids.add(c["chunk_id"])
    assert part_indices == list(range(len(chunks)))  # 0,1,2,...


# --------------------------------------------------------------------------- #
# Header governance gate (09 §D-3): every produced chunk must validate.        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "doc",
    [_law_doc(), _precedent_doc(), _admrule_with_attachment(), _bgrade_doc()],
    ids=["law", "precedent", "admrule+별표", "bgrade"],
)
def test_every_chunk_passes_header_validator(doc):
    for c in chunk.chunks_of(doc):
        errors = validate_chunk(c)
        assert errors == [], f"{c['chunk_id']}: {errors}"


# --------------------------------------------------------------------------- #
# build_chunks() end-to-end over a temp artifact (file I/O, still offline).    #
# --------------------------------------------------------------------------- #
def test_build_chunks_writes_chunks_and_parents(tmp_path: Path):
    src = tmp_path / "docs_law.jsonl"
    with src.open("w", encoding="utf-8") as fh:
        for doc in (_law_doc(), _precedent_doc()):
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    out = tmp_path / "chunks.jsonl"
    parents = tmp_path / "parents.jsonl"
    stats = chunk.build_chunks(sources=[src], out_path=out, parents_path=parents)

    assert stats["docs"] == 2
    assert stats["chunks"] >= 4
    assert stats["parents"] == 2
    assert stats["over_limit"] == 0

    # Governance gate over the produced file: zero bad chunks.
    report = validate_file(out)
    assert report["n"] == stats["chunks"]
    assert report["n_bad"] == 0, report["error_counts"]

    # Parents carry full original text (not the normalized embedding text).
    plines = [json.loads(l) for l in parents.read_text(encoding="utf-8").splitlines()]
    assert {p["parent_id"] for p in plines} == {"LAW:000325:법률", "PREC:424370"}
    law_parent = next(p for p in plines if p["parent_id"] == "LAW:000325:법률")
    assert "목적" in law_parent["full_text"]
    assert law_parent["full_text"]  # non-empty


def test_duplicate_chunk_id_raises(tmp_path: Path):
    # Two identical docs -> identical chunk_ids -> must be rejected.
    src = tmp_path / "docs.jsonl"
    line = json.dumps(_law_doc(), ensure_ascii=False)
    src.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Duplicate chunk_id"):
        chunk.build_chunks(
            sources=[src],
            out_path=tmp_path / "c.jsonl",
            parents_path=tmp_path / "p.jsonl",
        )
