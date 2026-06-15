"""Unit + light-integration tests for the precedent parser (Task 1.2).

Covers the contract interface (``_BUILD_CONTRACT.md`` §c) and the Task-1.2 DoD:

- ``parse_all`` yields :class:`ingest.schema.Document` instances (never dicts).
- 사건번호 / 법원 / 선고일 are populated; the doc_id rule ``PREC:{판례일련번호}``
  holds; sections become :class:`Article` units.
- Robustness: a missing ``선고일자`` is recovered from the filename; the
  ``0000-00-00`` placeholder becomes ``None``; a section-less / malformed file
  does not crash the run (skip-and-log), and a section-less file is emitted as
  trust grade ``B``.
- De-identification fillers (``○○○`` …) are normalized to ``[당사자]``.
- A spot-check parses real corpus files when present (skipped in their absence).

No network and **no OpenAI calls** are made here (cost rule: 0 OpenAI calls).

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_parse_precedent.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

import config
from ingest import parse_precedent as pp
from ingest.schema import Article, Document

# --------------------------------------------------------------------------- #
# Fixtures: synthetic precedent files                                          #
# --------------------------------------------------------------------------- #

_FULL = """---
판례일련번호: '424370'
사건번호: (청주)2022누50008
사건명: 학교용지 부담금 부과처분 취소
법원명: 대전고등법원
법원등급: 하급심
사건종류: 선거·특별
출처: https://www.law.go.kr/LSW/precInfoP.do?precSeq=424370
첨부파일: []
선고일자: 2022-05-25
---

# 학교용지 부담금 부과처분 취소

## 판결요지

원고 ○○○ 와 피고  ○○ 사이의   쟁점은 다음과 같다.

## 판례내용

【주문】 제1심판결을 취소한다.
"""

# Missing 선고일자 in front matter, recoverable from the filename.
_NO_DATE_FM = """---
판례일련번호: '85839'
사건번호: 4283형상72
사건명: 국가보안법위반
법원명: 대법원
법원등급: 대법원
사건종류: 형사
출처: https://www.law.go.kr/LSW/precInfoP.do?precSeq=85839
첨부파일: []
선고일자:
---

# 국가보안법위반

## 판시사항

연속범 일부에 대한 위법과 상고이유.
"""

# Placeholder date both in FM and filename ⇒ effective_from is None.
_NULL_DATE = """---
판례일련번호: '999001'
사건번호: 2021구합73089
사건명: 처분취소
법원명: 서울행정법원
법원등급: 하급심
사건종류: 일반행정
출처: https://www.law.go.kr/x
첨부파일: []
선고일자: 0000-00-00
---

## 판례내용

본문.
"""

# Section-less / metadata-only body ⇒ trust grade B, articles == [].
_META_ONLY = """---
판례일련번호: '999002'
사건번호: 2020다1
사건명: 손해배상
법원명: 대법원
법원등급: 대법원
사건종류: 민사
출처: https://www.law.go.kr/y
첨부파일: []
선고일자: 2020-01-01
---

# 손해배상
"""

_MALFORMED = "no front matter here\n## 판례내용\n본문"


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Write a small synthetic precedent corpus and return its root."""
    root = tmp_path / "04_판례"
    sub = root / "선거·특별" / "하급심"
    sub.mkdir(parents=True)
    (sub / "대전고등법원_2022-05-25_2022누50008.md").write_text(_FULL, encoding="utf-8")
    (root / "형사" / "대법원").mkdir(parents=True)
    (root / "형사" / "대법원" / "대법원_1950-03-20_4283형상72.md").write_text(
        _NO_DATE_FM, encoding="utf-8"
    )
    g = root / "일반행정" / "하급심"
    g.mkdir(parents=True)
    (g / "서울행정법원_0000-00-00_2021구합73089.md").write_text(_NULL_DATE, encoding="utf-8")
    (root / "민사" / "대법원").mkdir(parents=True)
    (root / "민사" / "대법원" / "대법원_2020-01-01_2020다1.md").write_text(
        _META_ONLY, encoding="utf-8"
    )
    (sub / "broken.md").write_text(_MALFORMED, encoding="utf-8")
    (root / "README.md").write_text("# readme", encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# parse_file                                                                   #
# --------------------------------------------------------------------------- #


def test_full_document_fields(corpus: Path) -> None:
    path = corpus / "선거·특별" / "하급심" / "대전고등법원_2022-05-25_2022누50008.md"
    doc = pp.parse_file(path)
    assert isinstance(doc, Document)
    assert doc.doc_id == "PREC:424370"
    assert doc.doc_type == "precedent"
    assert doc.title == "학교용지 부담금 부과처분 취소"
    assert doc.jurisdiction == "대전고등법원"
    assert doc.law_kind == "선거·특별"
    assert doc.effective_from == "2022-05-25"
    assert doc.source_url and doc.source_url.startswith("https://")
    assert doc.trust_grade == "A"
    # Sections -> Articles, in document order.
    assert [a.article_no for a in doc.articles] == ["판결요지", "판례내용"]
    assert all(isinstance(a, Article) for a in doc.articles)
    # 사건번호 preserved in meta verbatim.
    assert doc.meta["사건번호"] == "(청주)2022누50008"


def test_deidentification_normalized(corpus: Path) -> None:
    path = corpus / "선거·특별" / "하급심" / "대전고등법원_2022-05-25_2022누50008.md"
    doc = pp.parse_file(path)
    body = next(a.text for a in doc.articles if a.article_no == "판결요지")
    assert "○" not in body
    assert "[당사자]" in body
    # Inline whitespace runs collapsed.
    assert "  " not in body


def test_date_recovered_from_filename(corpus: Path) -> None:
    path = corpus / "형사" / "대법원" / "대법원_1950-03-20_4283형상72.md"
    doc = pp.parse_file(path)
    assert doc.effective_from == "1950-03-20"  # FM 선고일자 empty -> from filename


def test_null_placeholder_date_is_none(corpus: Path) -> None:
    path = corpus / "일반행정" / "하급심" / "서울행정법원_0000-00-00_2021구합73089.md"
    doc = pp.parse_file(path)
    assert doc.effective_from is None


def test_metadata_only_is_grade_b(corpus: Path) -> None:
    path = corpus / "민사" / "대법원" / "대법원_2020-01-01_2020다1.md"
    doc = pp.parse_file(path)
    assert doc.trust_grade == "B"
    assert doc.articles == []


def test_malformed_returns_none(corpus: Path) -> None:
    path = corpus / "선거·특별" / "하급심" / "broken.md"
    assert pp.parse_file(path) is None


# --------------------------------------------------------------------------- #
# parse_all (streaming, robustness)                                            #
# --------------------------------------------------------------------------- #


def test_parse_all_streams_documents_and_skips_bad(corpus: Path) -> None:
    docs = list(pp.parse_all(corpus))
    # 4 valid files (README + the malformed one are skipped).
    assert len(docs) == 4
    assert all(isinstance(d, Document) for d in docs)
    ids = {d.doc_id for d in docs}
    assert ids == {"PREC:424370", "PREC:85839", "PREC:999001", "PREC:999002"}
    # README is never yielded.
    assert all(d.doc_id.startswith("PREC:") for d in docs)


def test_parse_all_is_lazy(corpus: Path) -> None:
    gen = pp.parse_all(corpus)
    first = next(gen)  # should not require exhausting the tree
    assert isinstance(first, Document)


# --------------------------------------------------------------------------- #
# helper-level                                                                 #
# --------------------------------------------------------------------------- #


def test_normalize_text_collapses_fillers() -> None:
    out = pp.normalize_text("피고 ○○○ 및 △△△ 와   □□□")
    assert "○" not in out and "△" not in out and "□" not in out
    assert out.count("[당사자]") == 3
    assert "   " not in out


# --------------------------------------------------------------------------- #
# DoD spot-check against the real corpus (skipped if absent)                   #
# --------------------------------------------------------------------------- #


def test_real_corpus_spotcheck() -> None:
    root = config.PRECEDENT_DIR
    if not root.exists():
        pytest.skip("real precedent corpus not present")
    sample = list(itertools.islice(pp.parse_all(root), 5))
    if not sample:
        pytest.skip("no precedent files found in corpus")
    for doc in sample:
        assert isinstance(doc, Document)
        assert doc.doc_id.startswith("PREC:")
        assert doc.jurisdiction  # 법원명 populated
        assert doc.meta.get("사건번호")  # case number present
        # Sections present ⇒ grade A with at least one citable article.
        if doc.trust_grade == "A":
            assert doc.articles
