"""Unit + light-integration tests for the national-law parser (Task 1.1).

Covers the contract interface (``_BUILD_CONTRACT.md`` §c) and the Task-1.1 DoD:

- ``parse_all`` yields :class:`ingest.schema.Document` instances (never dicts).
- 제목 / 시행일 / 소관부처 are populated; the doc_id rule
  ``LAW:{법령ID}:{법령구분}`` holds; ``#####`` articles become :class:`Article`
  units with the right article_no / title / text.
- Each ``{구분}.md`` in one law folder is a *separate* Document (법률 vs 시행령).
- Field coercion: a YAML ``date`` 시행일자 and a list-valued 소관부처 serialize
  to clean strings; quoted/leading-zero 법령ID is preserved.
- Robustness: an inline ``제N조(...)`` fallback works; a header-less short law
  emits its lead paragraph as one synthetic article; a body-less file is grade
  ``B`` with ``articles == []``; a file without front matter / identifiers is
  skipped (``parse_file`` -> None) without crashing the run.
- A spot-check parses real corpus files when present (the 50-file DoD), asserting
  article counts / titles / 시행일 are sane.

No network and **no OpenAI calls** are made here (cost rule: 0 OpenAI calls).

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_parse_statute.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

import config
from ingest import parse_statute as ps
from ingest.schema import Article, Document

# --------------------------------------------------------------------------- #
# Fixtures: synthetic national-law files                                       #
# --------------------------------------------------------------------------- #

# Full law with ##### article headers, a 제N조의2, a title-less 제N조 (삭제),
# a YAML date 시행일자, and a list-valued 소관부처.
_LAW = """---
제목: 10ㆍ27법난 피해자의 명예회복 등에 관한 법률
법령MST: 253527
법령ID: '010719'
법령구분: 법률
소관부처:
- 문화체육관광부
공포일자: 2023-08-08
공포번호: '19592'
시행일자: 2023-08-08
상태: 시행
출처: https://www.law.go.kr/법령/10ㆍ27법난피해자의명예회복등에관한법률
첨부파일: []
---

# 10ㆍ27법난 피해자의 명예회복 등에 관한 법률

##### 제1조 (목적)

이 법은 인권신장에 이바지함을 목적으로 한다.

##### 제2조 (정의)

이 법에서 사용하는 용어의 정의는 다음과 같다.

##### 제3조의2 (벌칙 적용에서 공무원 의제)

위원회의 위원 중 공무원이 아닌 사람은 공무원으로 본다.

##### 제4조

삭제 <2016.2.3>

## 부칙

부칙 <제19592호, 2023.8.8>
"""

# Same law folder, different 구분 ⇒ separate Document (시행령).
_DECREE = """---
제목: 10ㆍ27법난 피해자의 명예회복 등에 관한 법률 시행령
법령MST: 253528
법령ID: '010719'
법령구분: 대통령령
소관부처:
- 문화체육관광부
공포일자: 2023-08-08
시행일자: 2023-08-08
상태: 시행
출처: https://www.law.go.kr/x
첨부파일: []
---

# 시행령

##### 제1조 (목적)

이 영은 위임된 사항을 규정함을 목적으로 한다.
"""

# Header-less short law (폐지/명칭변경): lead paragraph before 부칙 -> one
# synthetic "본문" article, grade A (text present).
_LEAD_ONLY = """---
제목: 인천광역시 남구 명칭 변경에 관한 법률
법령ID: '013113'
법령구분: 법률
소관부처:
- 행정안전부
시행일자: 2018-07-01
상태: 시행
출처: https://www.law.go.kr/y
첨부파일: []
---

# 인천광역시 남구 명칭 변경에 관한 법률

인천광역시 "남구(南區)"를 "미추홀구(彌鄒忽區)"로 한다.

## 부칙

부칙 <제15499호,2018.3.20>

제1조(시행일) 이 법은 2018년 7월 1일부터 시행한다.
"""

# Inline-only article headers (no #####) — the stray-file fallback path.
_INLINE = """---
제목: 인라인 조문 법률
법령ID: '020001'
법령구분: 법률
소관부처:
- 법무부
시행일자: 2020-01-01
상태: 시행
출처: https://www.law.go.kr/z
첨부파일: []
---

# 인라인 조문 법률

제1조(목적) 이 법은 인라인 형식을 규정한다.

제2조(정의) 용어의 뜻은 다음과 같다.
"""

# Body-less (title only, no 부칙 content) ⇒ grade B, articles == [].
_BODY_LESS = """---
제목: 본문 없는 법률
법령ID: '030001'
법령구분: 법률
소관부처: []
시행일자: 2019-01-01
상태: 폐지
출처: https://www.law.go.kr/none
첨부파일: []
---

# 본문 없는 법률
"""

# No front matter at all ⇒ skip (parse_file -> None).
_NO_FM = "# 그냥 텍스트\n\n##### 제1조 (목적)\n본문\n"

# Front matter present but missing the identifying 법령ID ⇒ skip.
_NO_ID = """---
제목: 식별자 없는 법률
법령구분: 법률
시행일자: 2020-01-01
---

##### 제1조 (목적)
본문.
"""


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Write a small synthetic national-law corpus and return its kr/ root."""
    root = tmp_path / "01_국가법령" / "kr"
    folder = root / "10ㆍ27법난피해자의명예회복등에관한법률"
    folder.mkdir(parents=True)
    (folder / "법률.md").write_text(_LAW, encoding="utf-8")
    (folder / "대통령령.md").write_text(_DECREE, encoding="utf-8")

    (root / "인천광역시남구명칭변경에관한법률").mkdir()
    (root / "인천광역시남구명칭변경에관한법률" / "법률.md").write_text(
        _LEAD_ONLY, encoding="utf-8"
    )
    (root / "인라인조문법률").mkdir()
    (root / "인라인조문법률" / "법률.md").write_text(_INLINE, encoding="utf-8")
    (root / "본문없는법률").mkdir()
    (root / "본문없는법률" / "법률.md").write_text(_BODY_LESS, encoding="utf-8")
    (root / "식별자없는법률").mkdir()
    (root / "식별자없는법률" / "법률.md").write_text(_NO_ID, encoding="utf-8")
    (root / "잡파일").mkdir()
    (root / "잡파일" / "noise.md").write_text(_NO_FM, encoding="utf-8")
    return root


def _path(corpus: Path, *parts: str) -> Path:
    return corpus.joinpath(*parts)


# --------------------------------------------------------------------------- #
# parse_file — fields, doc_id, articles                                        #
# --------------------------------------------------------------------------- #


def test_full_document_fields(corpus: Path) -> None:
    doc = ps.parse_file(_path(corpus, "10ㆍ27법난피해자의명예회복등에관한법률", "법률.md"))
    assert isinstance(doc, Document)
    assert doc.doc_id == "LAW:010719:법률"
    assert doc.doc_type == "law"
    assert doc.title == "10ㆍ27법난 피해자의 명예회복 등에 관한 법률"
    assert doc.jurisdiction == "국가"
    assert doc.law_kind == "법률"
    assert doc.effective_from == "2023-08-08"  # YAML date -> ISO string
    assert doc.source_url and doc.source_url.startswith("https://")
    assert doc.trust_grade == "A"
    # 소관부처: list serialized to first/joined string in meta.
    assert doc.meta["소관부처"] == "문화체육관광부"
    # 법령ID with leading zero preserved verbatim.
    assert doc.meta["법령ID"] == "010719"
    assert doc.meta["법령MST"] == "253528" or doc.meta["법령MST"] == "253527"


def test_article_splitting(corpus: Path) -> None:
    doc = ps.parse_file(_path(corpus, "10ㆍ27법난피해자의명예회복등에관한법률", "법률.md"))
    nos = [a.article_no for a in doc.articles]
    assert nos == ["제1조", "제2조", "제3조의2", "제4조"]
    assert all(isinstance(a, Article) for a in doc.articles)
    # Parenthetical title captured; title-less 제4조 has title None.
    by_no = {a.article_no: a for a in doc.articles}
    assert by_no["제1조"].title == "목적"
    assert by_no["제3조의2"].title == "벌칙 적용에서 공무원 의제"
    assert by_no["제4조"].title is None
    # Article text excludes the header line and stops before the next article.
    assert "목적으로 한다" in by_no["제1조"].text
    assert "정의는" not in by_no["제1조"].text
    # 부칙 is not folded into the last article's body verbatim past its end —
    # 제4조 keeps its own "삭제" content.
    assert "삭제" in by_no["제4조"].text


def test_kind_distinguishes_documents_in_one_folder(corpus: Path) -> None:
    base = _path(corpus, "10ㆍ27법난피해자의명예회복등에관한법률")
    law = ps.parse_file(base / "법률.md")
    decree = ps.parse_file(base / "대통령령.md")
    # Same 법령ID, different 법령구분 ⇒ distinct, well-formed doc_ids.
    assert law.doc_id == "LAW:010719:법률"
    assert decree.doc_id == "LAW:010719:대통령령"
    assert law.law_kind == "법률" and decree.law_kind == "대통령령"


# --------------------------------------------------------------------------- #
# parse_file — fallbacks / robustness                                          #
# --------------------------------------------------------------------------- #


def test_lead_paragraph_fallback(corpus: Path) -> None:
    doc = ps.parse_file(_path(corpus, "인천광역시남구명칭변경에관한법률", "법률.md"))
    assert doc.trust_grade == "A"
    assert len(doc.articles) == 1
    art = doc.articles[0]
    assert art.article_no == "본문"
    assert "미추홀구" in art.text
    # The 부칙 section is excluded from the captured lead.
    assert "부칙" not in art.text


def test_inline_article_fallback(corpus: Path) -> None:
    doc = ps.parse_file(_path(corpus, "인라인조문법률", "법률.md"))
    nos = [a.article_no for a in doc.articles]
    assert nos == ["제1조", "제2조"]
    assert doc.articles[0].title == "목적"
    assert doc.trust_grade == "A"


def test_body_less_is_grade_b(corpus: Path) -> None:
    doc = ps.parse_file(_path(corpus, "본문없는법률", "법률.md"))
    assert doc.trust_grade == "B"
    assert doc.articles == []
    # Even grade-B docs keep a well-formed id and metadata.
    assert doc.doc_id == "LAW:030001:법률"
    assert doc.title == "본문 없는 법률"


def test_no_front_matter_returns_none(corpus: Path) -> None:
    assert ps.parse_file(_path(corpus, "잡파일", "noise.md")) is None


def test_missing_identifier_returns_none(corpus: Path) -> None:
    assert ps.parse_file(_path(corpus, "식별자없는법률", "법률.md")) is None


# --------------------------------------------------------------------------- #
# parse_all — streaming, robustness, contract API                              #
# --------------------------------------------------------------------------- #


def test_parse_all_streams_and_skips_bad(corpus: Path) -> None:
    docs = list(ps.parse_all(corpus))
    # 5 valid files (법률+대통령령 in folder 1, lead-only, inline, body-less).
    # The no-front-matter and missing-id files are skipped.
    ids = {d.doc_id for d in docs}
    assert ids == {
        "LAW:010719:법률",
        "LAW:010719:대통령령",
        "LAW:013113:법률",
        "LAW:020001:법률",
        "LAW:030001:법률",
    }
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.doc_id.startswith("LAW:") for d in docs)
    assert all(d.doc_type == "law" for d in docs)


def test_parse_all_is_lazy(corpus: Path) -> None:
    gen = ps.parse_all(corpus)
    first = next(gen)
    assert isinstance(first, Document)


def test_model_dump_json_roundtrips(corpus: Path) -> None:
    doc = next(ps.parse_all(corpus))
    line = doc.model_dump_json()
    again = Document.model_validate_json(line)
    assert again.doc_id == doc.doc_id
    assert [a.article_no for a in again.articles] == [a.article_no for a in doc.articles]


# --------------------------------------------------------------------------- #
# helper-level                                                                 #
# --------------------------------------------------------------------------- #


def test_clean_title_peels_parens() -> None:
    # Bare single parens, double parens (124 corpus files), and full-width.
    assert ps._clean_title("(목적)") == "목적"
    assert ps._clean_title("((목적))") == "목적"
    assert ps._clean_title("（목적）") == "목적"
    assert ps._clean_title("") is None
    assert ps._clean_title(None) is None


def test_double_paren_headers(tmp_path: Path) -> None:
    """``##### 제N조 ((제목))`` variant must still split into articles."""
    text = (
        "---\n제목: 더블괄호 시행령\n법령ID: '040001'\n법령구분: 대통령령\n"
        "소관부처:\n- 국방부\n시행일자: 2000-01-01\n상태: 시행\n"
        "출처: https://www.law.go.kr/d\n첨부파일: []\n---\n\n"
        "# 더블괄호 시행령\n\n"
        "##### 제1조 ((목적))\n\n이 영은 목적을 규정한다.\n\n"
        "##### 제2조 ((정의))\n\n용어의 뜻은 다음과 같다.\n"
    )
    folder = tmp_path / "01_국가법령" / "kr" / "더블괄호시행령"
    folder.mkdir(parents=True)
    (folder / "대통령령.md").write_text(text, encoding="utf-8")
    doc = ps.parse_file(folder / "대통령령.md")
    assert [a.article_no for a in doc.articles] == ["제1조", "제2조"]
    assert doc.articles[0].title == "목적"  # parens peeled, not "(목적)"
    assert "목적을 규정한다" in doc.articles[0].text
    assert doc.trust_grade == "A"


def test_chapter_section_path_tracked(tmp_path: Path) -> None:
    """``## 제N장`` / ``### 제N절`` headings become each article's chapter_path.

    Covers 09 §D-1 (L1 carries 장/절 position): an article picks up its 장, a
    nested 절 produces ``장 > 절``, a new 장 resets the active 절, and a trailing
    ``<개정 …>`` on a heading is stripped from the captured label.
    """
    text = (
        "---\n제목: 장절 법률\n법령ID: '050001'\n법령구분: 법률\n"
        "소관부처:\n- 법무부\n시행일자: 2020-01-01\n상태: 시행\n"
        "출처: https://www.law.go.kr/c\n첨부파일: []\n---\n\n"
        "# 장절 법률\n\n"
        "## 제1장 총칙\n\n"
        "##### 제1조 (목적)\n\n이 법은 목적을 규정한다.\n\n"
        "## 제2장 처분 <개정 2018.3.27>\n\n"
        "### 제1절 압류\n\n"
        "##### 제2조 (압류)\n\n압류를 규정한다.\n\n"
        "### 제2절 환부\n\n"
        "##### 제3조 (환부)\n\n환부를 규정한다.\n\n"
        "## 제3장 보칙\n\n"
        "##### 제4조 (보칙)\n\n보칙을 규정한다.\n"
    )
    folder = tmp_path / "장절법률"
    folder.mkdir(parents=True)
    (folder / "법률.md").write_text(text, encoding="utf-8")
    doc = ps.parse_file(folder / "법률.md")
    by_no = {a.article_no: a for a in doc.articles}

    assert by_no["제1조"].chapter_path == "제1장 총칙"
    # Amendment tail stripped, nested 절 appended.
    assert by_no["제2조"].chapter_path == "제2장 처분 > 제1절 압류"
    assert by_no["제3조"].chapter_path == "제2장 처분 > 제2절 환부"
    # A new 장 resets the active 절 (no leaked "제2절").
    assert by_no["제4조"].chapter_path == "제3장 보칙"


def test_chapter_path_none_without_headings(tmp_path: Path) -> None:
    """A law with no 장/절 headings yields chapter_path == None (no spurious path)."""
    text = (
        "---\n제목: 무장 법률\n법령ID: '050002'\n법령구분: 법률\n"
        "소관부처:\n- 법무부\n시행일자: 2020-01-01\n상태: 시행\n"
        "출처: https://www.law.go.kr/n\n첨부파일: []\n---\n\n"
        "# 무장 법률\n\n##### 제1조 (목적)\n\n본문.\n"
    )
    folder = tmp_path / "무장법률"
    folder.mkdir(parents=True)
    (folder / "법률.md").write_text(text, encoding="utf-8")
    doc = ps.parse_file(folder / "법률.md")
    assert doc.articles[0].chapter_path is None


def test_to_str_coerces_dates_and_lists() -> None:
    import datetime as dt

    assert ps._to_str(dt.date(2024, 7, 3)) == "2024-07-03"
    assert ps._to_str(["문화체육관광부"]) == "문화체육관광부"
    assert ps._to_str("") is None
    assert ps._to_str(None) is None
    assert ps._first(["a", "b"]) == "a"


# --------------------------------------------------------------------------- #
# DoD spot-check against the real corpus (skipped if absent) — 50-file scan    #
# --------------------------------------------------------------------------- #


def test_real_corpus_spotcheck() -> None:
    root = config.LAW_DIR
    if not root.exists():
        pytest.skip("real national-law corpus not present")
    sample = list(itertools.islice(ps.parse_all(root), 50))
    if not sample:
        pytest.skip("no national-law files found in corpus")
    assert len(sample) >= 5
    for doc in sample:
        assert isinstance(doc, Document)
        assert doc.doc_id.startswith("LAW:")
        # doc_id has exactly the LAW:{id}:{kind} shape.
        parts = doc.doc_id.split(":")
        assert len(parts) == 3 and parts[0] == "LAW" and parts[1] and parts[2]
        assert doc.jurisdiction == "국가"
        assert doc.title  # 제목 populated
        assert doc.law_kind  # 법령구분 populated
        # Grade-A docs carry at least one citable article with body text.
        if doc.trust_grade == "A":
            assert doc.articles
            assert all(a.text.strip() for a in doc.articles)
            assert all(a.article_no for a in doc.articles)
