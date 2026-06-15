"""Unit tests for ``ingest.parse_ordinance`` (Phase 1, Task 1.4).

Self-contained: builds tiny synthetic ``본문.md`` fixtures in a tmp dir so the
suite runs in milliseconds and needs no network / OpenAI call. A single
``slow``-style spot check against the real corpus is included but skipped unless
the data root exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from ingest.parse_ordinance import (
    _normalize_text,
    _split_articles,
    parse_bonmun,
    parse_file,
)
from ingest.schema import Document

# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #

_A_GRADE = """---
자치법규ID: '2170373'
자치법규일련번호: '1263750'
자치법규명: '동해시 지역축제 육성 및 지원에 관한 조례'
자치법규종류: '조례'
지자체기관명: '강원특별자치도 동해시'
지자체구분:
  광역: '강원특별자치도'
  기초: '동해시'
공포일자: 2016-11-11
공포번호: '1873'
시행일자: '2016-11-11'
출처: 'https://www.law.go.kr/자치법규/동해시지역축제육성및지원에관한조례'
첨부파일: []
---

# 동해시 지역축제 육성 및 지원에 관한 조례

##### 제1조 (목적)

이 조례는 축제의 육성을 목적으로 한다.<개정 2016.10.31>

##### 제2조 (정의)

① 이 조례에서 "축제"란 행사를 말한다.
② 두 번째 항이다.

##### 제3조의2 (특례)

특례 조문 본문.
"""

_B_GRADE = """---
자치법규ID: '2191948'
자치법규명: '경주시 아동급식 지원 조례'
자치법규종류: '조례'
지자체구분:
  광역: '경상북도'
  기초: '경주시'
시행일자: '2020-01-01'
출처: 'https://www.law.go.kr/x'
첨부파일: []
---

# 경주시 아동급식 지원 조례

본문은 첨부파일 또는 원문을 참조하세요.
"""

_RULE_KIND = """---
자치법규ID: '9001'
자치법규명: '울릉군 금고지정 및 운영에 관한 규칙'
자치법규종류: '규칙'
지자체구분:
  광역: '경상북도'
  기초: '울릉군'
시행일자: '2021-03-03'
출처: 'https://www.law.go.kr/y'
첨부파일: []
---

# 울릉군 금고지정 및 운영에 관한 규칙

##### 제1조 (목적)

규칙의 목적.
"""

_MISSING_FIELDS = """---
자치법규명: '식별자 없는 조례'
자치법규종류: '조례'
지자체구분:
  광역: '서울특별시'
시행일자: '2020-01-01'
---

##### 제1조 (목적)

본문.
"""

_NO_FRONT_MATTER = "그냥 텍스트, front matter 없음.\n##### 제1조 (목적)\n본문.\n"

# A real-corpus defect (~17% of files): the ``#####`` headers are shifted by one
# — a phantom ``제0조`` whose body is just a chapter line (``제1장 총칙``), every
# later header's number/title actually belongs to the *next* article — while each
# body block restates its true ``제N조(제목)`` inline. The parser must trust the
# inline marker, drop the phantom 제0조, and end up with correctly-numbered
# articles. Includes a spaced ``제3조의 2`` gaji and a mangled ``제302조`` header.
_SHIFTED_HEADERS = """---
자치법규ID: '7777'
자치법규명: '테스트시 복무 조례'
자치법규종류: '조례'
지자체구분:
  광역: '경상북도'
  기초: '테스트시'
시행일자: '2020-01-01'
출처: 'https://www.law.go.kr/z'
첨부파일: []
---

# 테스트시 복무 조례

##### 제0조 (목적)

제1장 총칙

##### 제1조 (복무선서)

제1조(목적) 이 조례는 목적을 규정한다.

##### 제2조 (책임완수)

제2조(복무선서) 공무원은 선서를 한다.

##### 제302조 (근무기강확립)

제3조의 2(비밀엄수) 비밀을 지킨다.
"""

# 광역(시·도 본청) ordinance: 기초 sentinel ``_본청`` ⇒ gov_level 광역, no locality.
_WIDE_AREA = """---
자치법규ID: '8001'
자치법규명: '부산광역시 금고 지정 및 운영 조례'
자치법규종류: '조례'
지자체기관명: '부산광역시'
지자체구분:
  광역: '부산광역시'
  기초: '_본청'
시행일자: '2020-05-27'
출처: 'https://www.law.go.kr/w'
첨부파일: []
---

# 부산광역시 금고 지정 및 운영 조례

##### 제1조 (목적)

이 조례는 금고 지정에 관한 사항을 규정한다.
"""

# 교육청 ordinance: 기초 sentinel ``_교육청`` ⇒ gov_level 교육청.
_EDU_OFFICE = """---
자치법규ID: '8002'
자치법규명: '서울특별시 각급학교 인정도서 인정수수료 징수 규칙'
자치법규종류: '규칙'
지자체기관명: '서울특별시교육청'
지자체구분:
  광역: '서울특별시'
  기초: '_교육청'
시행일자: '2021-01-01'
출처: 'https://www.law.go.kr/e'
첨부파일: []
---

# 서울특별시 각급학교 인정도서 인정수수료 징수 규칙

##### 제1조 (목적)

이 규칙은 인정수수료 징수에 관한 사항을 규정한다.
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name / "본문.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# parse_file — happy paths                                                      #
# --------------------------------------------------------------------------- #


def test_a_grade_full_parse(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "a", _A_GRADE))
    assert isinstance(doc, Document)
    assert doc.doc_id == "ORD:강원특별자치도:2170373"
    assert doc.doc_type == "ordinance"
    assert doc.jurisdiction == "강원특별자치도"
    assert doc.law_kind == "조례"
    assert doc.title == "동해시 지역축제 육성 및 지원에 관한 조례"
    assert doc.effective_from == "2016-11-11"
    assert doc.source_url and doc.source_url.startswith("https://")
    assert doc.trust_grade == "A"


def test_article_numbers_titles_and_gaji(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "a", _A_GRADE))
    nos = [a.article_no for a in doc.articles]
    assert nos == ["제1조", "제2조", "제3조의2"]
    titles = [a.title for a in doc.articles]
    assert titles == ["목적", "정의", "특례"]


def test_date_coercion_for_unquoted_yaml_date(tmp_path: Path) -> None:
    # 공포일자 is an unquoted YAML date -> must serialize to ISO string in meta.
    doc = parse_file(_write(tmp_path, "a", _A_GRADE))
    assert doc.meta["공포일자"] == "2016-11-11"
    assert isinstance(doc.meta["공포일자"], str)


def test_amendment_extracted_and_stripped(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "a", _A_GRADE))
    art1 = doc.articles[0]
    assert "<개정" not in art1.text
    assert doc.meta["amendments"]["제1조"] == ["<개정 2016.10.31>"]


def test_circled_numerals_co_noted(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "a", _A_GRADE))
    art2 = doc.articles[1]
    # original glyph preserved AND plain co-notation added.
    assert "①" in art2.text and "(1)" in art2.text
    assert "②" in art2.text and "(2)" in art2.text


def test_rule_kind(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "r", _RULE_KIND))
    assert doc.law_kind == "규칙"
    assert doc.doc_id == "ORD:경상북도:9001"
    assert doc.trust_grade == "A"


# --------------------------------------------------------------------------- #
# B-grade & error handling                                                      #
# --------------------------------------------------------------------------- #


def test_b_grade_bodyless(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "b", _B_GRADE))
    assert doc.trust_grade == "B"
    assert doc.articles == []
    assert doc.doc_id == "ORD:경상북도:2191948"
    # metadata still present so the doc is discoverable.
    assert doc.title == "경주시 아동급식 지원 조례"
    assert doc.effective_from == "2020-01-01"


def test_missing_required_fields_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_file(_write(tmp_path, "m", _MISSING_FIELDS))


def test_no_front_matter_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_file(_write(tmp_path, "n", _NO_FRONT_MATTER))


# --------------------------------------------------------------------------- #
# Corpus correction: shifted ``#####`` headers / phantom 제0조 / inline restate  #
# --------------------------------------------------------------------------- #


def test_shifted_headers_use_inline_identity(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "s", _SHIFTED_HEADERS))
    nos = [a.article_no for a in doc.articles]
    titles = [a.title for a in doc.articles]
    # Phantom 제0조 (chapter-only body) dropped; inline markers are authoritative.
    assert nos == ["제1조", "제2조", "제3조의2"]
    assert titles == ["목적", "복무선서", "비밀엄수"]
    assert "제0조" not in nos


def test_shifted_headers_strip_inline_marker_from_body(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "s", _SHIFTED_HEADERS))
    # The body must not re-open with its own "제N조(제목)" restatement.
    for a in doc.articles:
        assert not a.text.lstrip().startswith(a.article_no)
    assert doc.articles[0].text == "이 조례는 목적을 규정한다."


def test_phantom_je0_chapter_only_dropped(tmp_path: Path) -> None:
    # A bare phantom 제0조 whose only content is a chapter line is never an article.
    body = "##### 제0조 (목적)\n제1장 총칙\n##### 제1조 (목적)\n제1조(목적) 진짜 본문.\n"
    arts, _ = _split_articles(body)
    assert [a.article_no for a in arts] == ["제1조"]
    assert arts[0].text == "진짜 본문."


# --------------------------------------------------------------------------- #
# 광역/기초/교육청 classification (gov_level / locality)                          #
# --------------------------------------------------------------------------- #


def test_gov_level_basic_municipality(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "a", _A_GRADE))  # 강원특별자치도 동해시
    assert doc.meta["gov_level"] == "기초"
    assert doc.meta["locality"] == "동해시"


def test_gov_level_wide_area_bonchung(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "w", _WIDE_AREA))  # 부산 _본청
    assert doc.meta["gov_level"] == "광역"
    assert doc.meta["locality"] is None
    assert doc.jurisdiction == "부산광역시"


def test_gov_level_education_office(tmp_path: Path) -> None:
    doc = parse_file(_write(tmp_path, "e", _EDU_OFFICE))  # 서울 _교육청
    assert doc.meta["gov_level"] == "교육청"
    assert doc.meta["locality"] is None


def test_gov_level_in_header_and_payload(tmp_path: Path) -> None:
    from embed.chunk import chunks_of

    doc = parse_file(_write(tmp_path, "a", _A_GRADE)).model_dump()
    chunk = next(iter(chunks_of(doc)))
    # L1 citation header carries the explicit level label.
    assert "(기초)" in chunk["text"].splitlines()[0]
    # Payload carries filter-friendly gov_level + locality.
    assert chunk["payload"]["gov_level"] == "기초"
    assert chunk["payload"]["locality"] == "동해시"
    # jurisdiction (시·도) remains the indexed pre-filter key.
    assert chunk["payload"]["jurisdiction"] == "강원특별자치도"


# --------------------------------------------------------------------------- #
# Lower-level helpers                                                           #
# --------------------------------------------------------------------------- #


def test_split_articles_header_form() -> None:
    body = "##### 제1조 (목적)\n본문1.\n##### 제2조\n본문2.\n"
    arts, _ = _split_articles(body)
    assert [a.article_no for a in arts] == ["제1조", "제2조"]
    assert arts[1].title is None  # title is optional


def test_split_articles_inline_fallback() -> None:
    body = "제1조(목적) 인라인 본문 텍스트입니다.\n제2조(정의) 두번째 인라인.\n"
    arts, _ = _split_articles(body)
    assert [a.article_no for a in arts] == ["제1조", "제2조"]


def test_normalize_text_collapses_and_nfc() -> None:
    clean, notes = _normalize_text("가   나\t다  <개정 2020>")
    assert "  " not in clean
    assert notes == ["<개정 2020>"]


def test_parse_bonmun_returns_fm_dict(tmp_path: Path) -> None:
    fm, arts, _ = parse_bonmun(_write(tmp_path, "a", _A_GRADE))
    assert isinstance(fm, dict)
    assert fm["자치법규ID"] == "2170373"
    assert len(arts) == 3


# --------------------------------------------------------------------------- #
# Real-corpus spot check (skipped if data root absent)                          #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not config.ORDINANCE_DIR.exists(),
    reason="ordinance source corpus not available",
)
def test_real_corpus_spot_check() -> None:
    files = list(config.ORDINANCE_DIR.rglob("본문.md"))
    assert files, "expected ordinance 본문.md files on disk"
    ok = 0
    for p in files[:50]:
        doc = parse_file(p)
        assert doc.doc_id.startswith("ORD:")
        assert doc.jurisdiction
        ok += 1
    assert ok == 50
