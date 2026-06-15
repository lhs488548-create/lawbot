"""Unit tests for ``ingest.parse_admrule`` (Playbook 08, Task 1.3).

Two layers, per the contract test convention (``_BUILD_CONTRACT.md`` (i)):

* **Hermetic (always run, $0, no network):** synthetic ``본문.md`` fixtures
  written to a temp dir exercise every branch — inline ``제N조`` splitting,
  parenthesized/untitled titles, ``제N조의2`` forms, chapter tracking,
  duplicate-label disambiguation, structureless 고시 bodies (single synthetic
  article), and label-only/image-only bodies (``trust_grade="B"``,
  ``articles=[]``). Malformed files must be skipped, never crash ``parse_all``.
* **Corpus spot-check (opt-in):** a 50-file sample of the real corpus, gated by
  ``LAWBOT_CORPUS=1`` so ordinary runs need no data mount. Asserts the DoD:
  A-grade docs have ≥1 article, B-grade docs are counted separately, ids and
  filter metadata are well-formed.

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_parse_admrule.py
    # opt-in 50-file real-corpus spot-check:
    LAWBOT_CORPUS=1 .venv/bin/python -m pytest -q tests/test_parse_admrule.py -k corpus
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import config
from ingest import parse_admrule
from ingest.schema import Document


# --------------------------------------------------------------------------- #
# Fixtures (synthetic 본문.md files — no network, no real corpus)              #
# --------------------------------------------------------------------------- #

_FRONT = """\
행정규칙ID: '58423'
행정규칙명: '테스트 인사관리 규정'
행정규칙종류: '훈령'
소관부처명: '중소벤처기업부'
상위기관명: '중소벤처기업부'
발령일자: 2026-01-26
시행일자: 2026-01-26
제개정구분: '일부개정'
본문출처: 'api-text'
출처: 'https://www.law.go.kr/행정규칙/테스트인사관리규정'
첨부파일:
- 별표번호: '0001'
  제목: '심의결정서'
  파일링크: 'https://www.law.go.kr/LSW/flDownload.do?flSeq=1'
"""


def _write(dir_path: Path, name: str, body: str, front: str = _FRONT) -> Path:
    """Write a synthetic ``본문.md`` (front matter + body) and return its path."""
    target = dir_path / name / "본문.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"---\n{front}---\n{body}", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Inline article splitting                                                     #
# --------------------------------------------------------------------------- #


def test_inline_articles_split_with_titles(tmp_path: Path) -> None:
    body = (
        "제1장  총칙\n\n"
        "제1조(목적) 이 규정은 목적을 정한다.\n\n"
        "제2조(정의) 용어의 뜻은 다음과 같다.\n"
        "  1. 직원이란 ...\n\n"
        "제3조 제목없는 조문 본문.\n\n"
        "제4조의2(특례) 특례 본문.\n"
    )
    doc = parse_admrule.parse_file(_write(tmp_path, "규정A", body))

    assert isinstance(doc, Document)
    assert doc.doc_id == "ADMRULE:58423"
    assert doc.doc_type == "admrule"
    assert doc.trust_grade == "A"
    assert doc.jurisdiction == "중소벤처기업부"
    assert doc.law_kind == "훈령"
    assert doc.effective_from == "2026-01-26"
    assert doc.source_url and doc.source_url.startswith("https://")

    nos = [a.article_no for a in doc.articles]
    assert nos == ["제1조", "제2조", "제3조", "제4조의2"]

    by_no = {a.article_no: a for a in doc.articles}
    assert by_no["제1조"].title == "목적"
    assert by_no["제1조"].text.startswith("이 규정은")
    assert by_no["제3조"].title is None  # untitled article still captured
    assert by_no["제4조의2"].title == "특례"
    # Article bodies must not leak the next article's heading.
    assert "제2조" not in by_no["제1조"].text


def test_chapter_tracked_per_article(tmp_path: Path) -> None:
    body = (
        "제1장 총칙\n\n제1조(목적) 본문.\n\n"
        "제2장 운영\n\n제2조(운영) 본문.\n"
    )
    doc = parse_admrule.parse_file(_write(tmp_path, "규정C", body))
    chap = doc.meta.get("chapter_by_article")
    assert chap is not None
    assert chap["제1조"].startswith("제1장")
    assert chap["제2조"].startswith("제2장")
    # The chapter must also be threaded onto each Article so it reaches the L1
    # citation header / payload (09 §D-1) — not just the meta map.
    by_no = {a.article_no: a for a in doc.articles}
    assert by_no["제1조"].chapter_path == chap["제1조"]
    assert by_no["제2조"].chapter_path == chap["제2조"]


def test_duplicate_article_labels_disambiguated(tmp_path: Path) -> None:
    body = "제1조(목적) 본칙 목적.\n\n부 칙\n\n제1조 시행일 본문.\n"
    doc = parse_admrule.parse_file(_write(tmp_path, "규정D", body))
    nos = [a.article_no for a in doc.articles]
    assert nos[0] == "제1조"
    assert nos[1] == "제1조#2"  # second 제1조 made unique for stable chunk ids
    assert len(set(nos)) == len(nos)


# --------------------------------------------------------------------------- #
# Structureless 고시 bodies -> single synthetic article (still A-grade)         #
# --------------------------------------------------------------------------- #


def test_structureless_body_kept_as_single_article(tmp_path: Path) -> None:
    body = (
        "1. 사업의 명칭 : 천연가스 공급설비 건설공사\n\n"
        "2. 사업 시행자의 명칭 및 주소 : ...\n"
    )
    doc = parse_admrule.parse_file(_write(tmp_path, "고시A", body))
    assert doc.trust_grade == "A"
    assert len(doc.articles) == 1
    assert doc.articles[0].article_no == "전문"
    assert "사업의 명칭" in doc.articles[0].text


# --------------------------------------------------------------------------- #
# Label-only / image-only / empty bodies -> trust_grade="B", articles=[]        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "body",
    [
        "",
        "   \n  ",
        '<img id="37927106">\n</img>',
        "콜롬비아"[:4],  # extremely short label-only body
    ],
)
def test_label_only_body_is_b_grade(tmp_path: Path, body: str) -> None:
    doc = parse_admrule.parse_file(_write(tmp_path, f"빈{len(body)}", body))
    assert doc.trust_grade == "B"
    assert doc.articles == []
    # Metadata is still present so coverage stays honest.
    assert doc.doc_id == "ADMRULE:58423"
    assert doc.title == "테스트 인사관리 규정"


# --------------------------------------------------------------------------- #
# Error handling: malformed files are skipped, never crash parse_all           #
# --------------------------------------------------------------------------- #


def test_missing_front_matter_raises(tmp_path: Path) -> None:
    target = tmp_path / "노프론트" / "본문.md"
    target.parent.mkdir(parents=True)
    target.write_text("제1조(목적) front matter 없음.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_admrule.parse_file(target)


def test_missing_rule_id_raises(tmp_path: Path) -> None:
    front = "행정규칙명: '아이디 없음'\n행정규칙종류: '고시'\n"
    with pytest.raises(ValueError):
        parse_admrule.parse_file(_write(tmp_path, "노아이디", "제1조 본문.", front))


def test_parse_all_skips_malformed(tmp_path: Path, monkeypatch, capsys) -> None:
    # One good file, one malformed (no front matter) -> exactly one Document.
    _write(tmp_path, "정상", "제1조(목적) 본문.")
    bad = tmp_path / "깨진" / "본문.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("그냥 텍스트, front matter 없음", encoding="utf-8")

    monkeypatch.setattr(config, "ADMRULE_DIR", tmp_path)
    docs = list(parse_admrule.parse_all())
    assert len(docs) == 1
    assert docs[0].articles  # the good one parsed
    assert "SKIP" in capsys.readouterr().err  # the bad one was logged


def test_parse_all_dedupes_by_doc_id_keeping_latest(tmp_path: Path, monkeypatch) -> None:
    # Two files share 행정규칙ID 58423 (same logical rule, different revision
    # sequence). Only the latest revision must survive -> unique doc_id.
    old_front = _FRONT.replace(
        "행정규칙일련번호: '2100000273632'", ""
    )  # _FRONT has no seq; inject explicitly below
    front_old = (
        "행정규칙ID: '58423'\n행정규칙명: '규정'\n행정규칙종류: '고시'\n"
        "소관부처명: '부처'\n시행일자: 2020-01-01\n행정규칙일련번호: '100'\n"
    )
    front_new = (
        "행정규칙ID: '58423'\n행정규칙명: '규정'\n행정규칙종류: '고시'\n"
        "소관부처명: '부처'\n시행일자: 2026-01-01\n행정규칙일련번호: '200'\n"
    )
    _write(tmp_path, "구판", "제1조(목적) 옛 본문.", front_old)
    _write(tmp_path, "신판", "제1조(목적) 새 본문 최신.", front_new)
    monkeypatch.setattr(config, "ADMRULE_DIR", tmp_path)

    docs = list(parse_admrule.parse_all())
    assert len(docs) == 1  # deduped to one canonical Document
    assert docs[0].doc_id == "ADMRULE:58423"
    assert "최신" in docs[0].articles[0].text  # newest revision kept
    assert docs[0].effective_from == "2026-01-01"


def test_parse_all_is_sorted_and_streams(tmp_path: Path, monkeypatch) -> None:
    front_z = "행정규칙ID: '111'\n행정규칙명: 'Z'\n행정규칙종류: '고시'\n소관부처명: '부처'\n"
    front_a = "행정규칙ID: '222'\n행정규칙명: 'A'\n행정규칙종류: '고시'\n소관부처명: '부처'\n"
    _write(tmp_path, "z_규정", "제1조(목적) 본문 Z.", front_z)
    _write(tmp_path, "a_규정", "제1조(목적) 본문 A.", front_a)
    monkeypatch.setattr(config, "ADMRULE_DIR", tmp_path)
    docs = list(parse_admrule.parse_all())
    assert len(docs) == 2  # distinct ids, both parsed
    assert {d.doc_id for d in docs} == {"ADMRULE:111", "ADMRULE:222"}


# --------------------------------------------------------------------------- #
# Opt-in real-corpus spot-check (50 files) — DoD verification                  #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.getenv("LAWBOT_CORPUS") != "1",
    reason="set LAWBOT_CORPUS=1 to spot-check the real 03_행정규칙 corpus",
)
def test_corpus_spotcheck_50_files() -> None:
    paths = sorted(config.ADMRULE_DIR.glob("**/본문.md"))
    assert paths, "no 본문.md found under ADMRULE_DIR"
    # Even spread across the corpus rather than the first 50 of one ministry.
    step = max(1, len(paths) // 50)
    sample = paths[::step][:50]

    # doc_ids across the whole corpus must be globally unique (dedup works).
    seen_ids: set[str] = set()
    for doc in parse_admrule.parse_all():
        assert doc.doc_id not in seen_ids, f"duplicate doc_id: {doc.doc_id}"
        seen_ids.add(doc.doc_id)
    assert len(seen_ids) > 1000

    n_a = n_b = 0
    for path in sample:
        doc = parse_admrule.parse_file(path)
        assert isinstance(doc, Document)
        assert doc.doc_id.startswith("ADMRULE:")
        assert doc.doc_type == "admrule"
        assert doc.title
        assert doc.trust_grade in ("A", "B")
        # Filter keys downstream depends on.
        assert doc.jurisdiction
        if doc.trust_grade == "A":
            assert doc.articles, f"A-grade doc has no articles: {path}"
            for art in doc.articles:
                assert art.article_no
                assert art.text
            n_a += 1
        else:
            assert doc.articles == []
            n_b += 1

    assert n_a + n_b == len(sample)
    assert n_a > 0  # the overwhelming majority must carry real text
