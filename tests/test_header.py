"""Unit tests for the deterministic header module (header/) — 09 §D, §F.

Covers, with **no network and no OpenAI calls** (cost rule):

* L1 / L2 header generation for **every** doc_type (law, ordinance, admrule,
  precedent), including the required-field content of L1.
* Body normalization (09 §B-4): circled numerals, amendment stripping,
  precedent de-identification, NFC, whitespace.
* Payload completeness against the single schema definition.
* The governance validator: a well-formed chunk passes (0 errors), and each
  injected defect (missing L1 field, empty L2, missing payload key, bad kind,
  parent/doc_id mismatch) is detected.
* ``validate_file`` over a sample file reports 0 bad rows for good chunks and
  the exact bad count for mixed input.

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_header.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from header import build, schema, validate

# --------------------------------------------------------------------------- #
# Sample parsed Documents (one per corpus), shaped like                        #
# ingest.schema.Document.model_dump().                                          #
# --------------------------------------------------------------------------- #

LAW_DOC = {
    "doc_id": "LAW:002049:대통령령",
    "doc_type": "law",
    "title": "도로교통법",
    "jurisdiction": "국가",
    "law_kind": "법률",
    "effective_from": "2026-04-02",
    "source_url": "https://www.law.go.kr/법령/도로교통법",
    "trust_grade": "A",
    "articles": [
        {
            "article_no": "제17조",
            "title": "자동차등의 속도",
            "text": "① 자동차등의 속도는 다음과 같다. <개정 2016. 1. 22.> ② 시·도경찰청장은…",
            "chapter": "제2장",
        }
    ],
    "meta": {"소관부처": ["경찰청"], "법령ID": "002049"},
}

ORD_DOC = {
    "doc_id": "ORD:경상북도:2206415",
    "doc_type": "ordinance",
    "title": "영주시 주차장 무료개방 지원조례 시행규칙",
    "jurisdiction": "경상북도",
    "law_kind": "규칙",
    "effective_from": "2021-03-22",
    "source_url": "https://www.law.go.kr/자치법규/영주시...",
    "trust_grade": "A",
    "articles": [
        {"article_no": "제1조", "title": "목적", "text": "이 규칙은 …목적으로 한다."}
    ],
    "meta": {"지자체기관명": "경상북도 영주시"},
}

ADM_DOC = {
    "doc_id": "ADMRULE:58423",
    "doc_type": "admrule",
    "title": "중소벤처기업부 인사관리 규정",
    "jurisdiction": "중소벤처기업부",
    "law_kind": "훈령",
    "effective_from": "2026-01-26",
    "source_url": "https://www.law.go.kr/행정규칙/중소벤처기업부인사관리규정",
    "trust_grade": "A",
    "articles": [
        {"article_no": "제3조", "title": "적용범위", "text": "이 규정은 …적용한다."}
    ],
    "meta": {"소관부처명": "중소벤처기업부"},
}

PREC_DOC = {
    "doc_id": "PREC:424370",
    "doc_type": "precedent",
    "title": "학교용지 부담금 부과처분 취소",
    "jurisdiction": "대전고등법원",
    "law_kind": "선거·특별",
    "effective_from": "2022-05-25",
    "source_url": "https://www.law.go.kr/LSW/precInfoP.do?precSeq=424370",
    "trust_grade": "A",
    "articles": [
        {
            "article_no": "판결요지",
            "title": None,
            "text": "원고 ○○○ 와 피고 ○○○ 사이의 …재량권을 일탈·남용한 것임.",
        }
    ],
    "meta": {"사건번호": "2022누50008", "사건종류": "선거·특별", "법원명": "대전고등법원"},
}

ALL_DOCS = [LAW_DOC, ORD_DOC, ADM_DOC, PREC_DOC]


def _good_chunk(doc: dict, part_idx: int = 0) -> dict:
    """Produce a contract-shaped chunk from a doc's first article via the builder."""
    article = doc["articles"][0]
    text, payload = build.build_headers(doc, article, part_idx=part_idx)
    return {
        "chunk_id": f"{doc['doc_id']}#{article['article_no']}#{part_idx}",
        "doc_id": doc["doc_id"],
        "parent_id": doc["doc_id"],
        "text": text,
        "content_hash": "deadbeef",
        "payload": payload,
    }


# --------------------------------------------------------------------------- #
# build_headers — structure & layout                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("doc", ALL_DOCS, ids=lambda d: d["doc_type"])
def test_embed_text_has_two_header_lines_then_body(doc):
    text, _ = build.build_headers(doc, doc["articles"][0])
    lines = text.split("\n", schema.N_HEADER_LINES)
    assert len(lines) == 3, "expected L1, L2, body"
    assert lines[0].strip(), "L1 must be non-empty"
    assert lines[1].strip(), "L2 must be non-empty"
    assert lines[2].strip(), "body must be non-empty"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=lambda d: d["doc_type"])
def test_l1_contains_required_fields(doc):
    l1 = build.build_l1(doc, doc["articles"][0])
    for field in schema.HEADER_REQUIRED[doc["doc_type"]]:
        if field == "article_no":
            expected = doc["articles"][0]["article_no"]
        else:
            expected = doc.get(field)
        assert str(expected) in l1, f"L1 for {doc['doc_type']} missing {field}={expected!r}: {l1}"


def test_l1_law_format():
    l1 = build.build_l1(LAW_DOC, LAW_DOC["articles"][0])
    assert l1.startswith("[법령] 도로교통법 > 제2장 > 제17조(자동차등의 속도)")
    assert "시행 2026-04-02" in l1
    assert "소관 경찰청" in l1


def test_l1_precedent_format():
    l1 = build.build_l1(PREC_DOC, PREC_DOC["articles"][0])
    assert l1.startswith("[판례] 대전고등법원 2022-05-25 2022누50008")
    assert "선거·특별" in l1
    assert "판결요지" in l1


def test_l2_is_rule_based_nonempty_sentence():
    # L2 is generated by template, never an LLM; deterministic and non-empty.
    for doc in ALL_DOCS:
        l2 = build.build_l2(doc, doc["articles"][0])
        assert l2 and l2 == build.build_l2(doc, doc["articles"][0])  # deterministic
        assert doc["title"] in l2 or (doc["meta"].get("사건번호", "") in l2)


# --------------------------------------------------------------------------- #
# normalize_body — 09 §B-4                                                      #
# --------------------------------------------------------------------------- #


def test_normalize_circled_numerals():
    assert "(1)" in build.normalize_body("① 첫째")
    assert "(3)" in build.normalize_body("③ 셋째")


def test_normalize_strips_amendments_into_payload():
    body = "본문 내용 <개정 2016. 1. 22.> 계속"
    normalized = build.normalize_body(body)
    assert "개정" not in normalized
    assert build.extract_amendments(body) == ["<개정 2016. 1. 22.>"]


def test_normalize_deidentifies_parties():
    assert "[당사자]" in build.normalize_body("원고 ○○○ 와 피고 ㅇㅇㅇ")


def test_payload_carries_amendments_for_law():
    _, payload = build.build_headers(LAW_DOC, LAW_DOC["articles"][0])
    assert payload.get("amendments") == ["<개정 2016. 1. 22.>"]


# --------------------------------------------------------------------------- #
# payload completeness                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("doc", ALL_DOCS, ids=lambda d: d["doc_type"])
def test_payload_has_all_required_keys(doc):
    _, payload = build.build_headers(doc, doc["articles"][0])
    for key in schema.PAYLOAD_REQUIRED:
        assert key in payload, f"payload missing {key} for {doc['doc_type']}"
    for key in schema.PAYLOAD_FILTER_KEYS:
        assert payload[key], f"filter key {key} empty for {doc['doc_type']}"
    assert payload["parent_id"] == doc["doc_id"]
    assert payload["license"] == __import__("config").DEFAULT_LICENSE


# --------------------------------------------------------------------------- #
# validate_chunk — good path + each defect                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("doc", ALL_DOCS, ids=lambda d: d["doc_type"])
def test_good_chunk_validates(doc):
    assert validate.validate_chunk(_good_chunk(doc)) == []


def test_detect_missing_l1_field():
    chunk = _good_chunk(LAW_DOC)
    l1, _, rest = chunk["text"].partition("\n")
    chunk["text"] = "[법령] 도로교통법\n" + rest  # drop article_no + date from L1
    errs = validate.validate_chunk(chunk)
    assert any("L1 missing required field" in e for e in errs)


def test_detect_empty_l2():
    chunk = _good_chunk(LAW_DOC)
    l1, _, body = chunk["text"].split("\n", 2)
    chunk["text"] = f"{l1}\n\n{body}"  # blank L2
    errs = validate.validate_chunk(chunk)
    assert any("L2" in e for e in errs)


def test_detect_missing_payload_key():
    chunk = _good_chunk(LAW_DOC)
    del chunk["payload"]["license"]
    errs = validate.validate_chunk(chunk)
    assert any("license" in e for e in errs)


def test_detect_empty_filter_key():
    chunk = _good_chunk(LAW_DOC)
    chunk["payload"]["jurisdiction"] = None
    errs = validate.validate_chunk(chunk)
    assert any("jurisdiction" in e for e in errs)


def test_precedent_allows_null_effective_from():
    """A precedent with no 선고일자 is valid data (FILTER_KEY_NULLABLE), not a
    governance failure — ~10% of the real corpus has an empty/0000-00-00 date but
    is still citable by 법원+사건번호+섹션. The key must stay present; only the
    non-null *value* requirement is relaxed for precedents."""
    doc = dict(PREC_DOC)
    doc["effective_from"] = None
    chunk = _good_chunk(doc)
    assert chunk["payload"]["effective_from"] is None  # key present, value null
    assert validate.validate_chunk(chunk) == []


def test_law_still_requires_effective_from():
    """The relaxation is precedent-only: legal texts always carry a 시행일, so a
    null effective_from on a 법령 must still fail (strictness preserved)."""
    chunk = _good_chunk(LAW_DOC)
    chunk["payload"]["effective_from"] = None
    errs = validate.validate_chunk(chunk)
    assert any("effective_from" in e for e in errs)


def test_effective_from_key_must_exist_even_for_precedent():
    """Nullable != optional: the effective_from KEY must always be present, even
    for a precedent, so downstream as_of_date filtering can rely on it."""
    chunk = _good_chunk(PREC_DOC)
    del chunk["payload"]["effective_from"]
    errs = validate.validate_chunk(chunk)
    assert any("effective_from" in e for e in errs)


def test_detect_bad_kind():
    chunk = _good_chunk(LAW_DOC)
    chunk["payload"]["kind"] = "garbage"
    errs = validate.validate_chunk(chunk)
    assert any("kind" in e for e in errs)


def test_detect_parent_id_mismatch():
    chunk = _good_chunk(LAW_DOC)
    chunk["payload"]["parent_id"] = "LAW:999:법률"
    errs = validate.validate_chunk(chunk)
    assert any("parent_id" in e for e in errs)


def test_detect_bad_doc_type():
    chunk = _good_chunk(LAW_DOC)
    chunk["payload"]["doc_type"] = "statute"
    errs = validate.validate_chunk(chunk)
    assert any("doc_type" in e for e in errs)


# --------------------------------------------------------------------------- #
# validate_file — governance gate over a sample file                           #
# --------------------------------------------------------------------------- #


def test_validate_file_all_good(tmp_path: Path):
    f = tmp_path / "chunks_good.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for doc in ALL_DOCS:
            fh.write(json.dumps(_good_chunk(doc), ensure_ascii=False) + "\n")
    report = validate.validate_file(f)
    assert report["n"] == 4
    assert report["n_bad"] == 0, report["errors"]
    assert set(report["by_doc_type"]) == {"law", "ordinance", "admrule", "precedent"}


def test_validate_file_counts_bad(tmp_path: Path):
    f = tmp_path / "chunks_mixed.jsonl"
    good = _good_chunk(LAW_DOC)
    bad = _good_chunk(PREC_DOC)
    del bad["payload"]["title"]  # one defect
    with f.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(good, ensure_ascii=False) + "\n")
        fh.write(json.dumps(bad, ensure_ascii=False) + "\n")
    report = validate.validate_file(f)
    assert report["n"] == 2
    assert report["n_bad"] == 1


def test_validate_file_missing_path():
    report = validate.validate_file(Path("/nonexistent/chunks.jsonl"))
    assert report["n"] == 0 and report["n_bad"] == 0
    assert "note" in report
