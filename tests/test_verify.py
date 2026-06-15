"""Unit tests for the Citation Firewall (``search/verify.py``, 09 §E-2).

No live network and no OpenAI calls: the law.go.kr HTTP layer (``_law_get``) and
the DB-existence helpers are monkeypatched, so every verdict path
(현행 일치 / 오인용 / 폐지 / 허위사건 / API 미응답 / as_of) is exercised
deterministically and for free. A single *opt-in* live law.go.kr smoke test is
gated behind the ``LAWBOT_LIVE`` env var so it never runs in normal CI.

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_verify.py
"""

from __future__ import annotations

import json
import os

import pytest

from search import verify


# --------------------------------------------------------------------------- #
# Fake law.go.kr responses                                                     #
# --------------------------------------------------------------------------- #
def _fake_law_search_hit() -> dict:
    return {
        "LawSearch": {
            "law": [
                {
                    "현행연혁코드": "현행",
                    "법령일련번호": "281875",
                    "법령명한글": "도로교통법",
                    "법령구분명": "법률",
                    "시행일자": "20260402",
                    "법령상세링크": (
                        "/DRF/lawService.do?OC=SECRET&target=law&MST=281875&type=HTML"
                    ),
                }
            ]
        }
    }


def _fake_law_service(article_no: str, title: str = "자동차등의 속도") -> dict:
    return {
        "법령": {
            "조문": {
                "조문단위": [
                    {
                        "조문번호": article_no,
                        "조문제목": title,
                        "조문시행일자": "20260402",
                        "조문내용": f"제{article_no}조({title}) ...",
                    }
                ]
            }
        }
    }


def _fake_prec_search(case_no: str) -> dict:
    return {
        "PrecSearch": {
            "prec": [
                {
                    "사건번호": case_no,
                    "법원명": "대법원",
                    "선고일자": "2024.10.08",
                    "판례일련번호": "241797",
                    "판례상세링크": (
                        "/DRF/lawService.do?OC=SECRET&target=prec&ID=241797&type=HTML"
                    ),
                }
            ]
        }
    }


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """Default: corpus DB has nothing (isolates the API-side logic)."""
    monkeypatch.setattr(verify, "_db_check_statute", lambda t, a: None)
    monkeypatch.setattr(verify, "_db_check_precedent", lambda c: None)


# --------------------------------------------------------------------------- #
# Pure-logic units                                                             #
# --------------------------------------------------------------------------- #
def test_offline_selftest_passes():
    assert verify._selftest() == 0


@pytest.mark.parametrize(
    "label,key,jo",
    [("제17조", "17", "001700"), ("제4조의2", "4의2", "000402"), ("17", "17", "001700")],
)
def test_article_and_jo_parsing(label, key, jo):
    assert verify._article_number(label) == key
    assert verify._jo_param(label) == jo


def test_strip_oc_never_leaks_token():
    url = "/DRF/lawService.do?OC=FAKEOC9999&target=law&MST=1&type=HTML"
    safe = verify._strip_oc(url)
    assert "OC=" not in safe and "FAKEOC9999" not in safe
    assert safe.startswith("https://www.law.go.kr")


def test_extract_citation_classification():
    assert verify._extract_citation({"law_name": "민법", "article_no": "제4조"})[0] == "statute"
    assert verify._extract_citation({"사건번호": "2022도1401"})[0] == "precedent"
    with pytest.raises(ValueError):
        verify._extract_citation({"foo": "bar"})


# --------------------------------------------------------------------------- #
# Statute verdicts (API mocked)                                               #
# --------------------------------------------------------------------------- #
def test_statute_current_article_match(monkeypatch):
    def fake_get(path, params):
        if path == verify._SEARCH_PATH:
            return _fake_law_search_hit()
        return _fake_law_service("17")

    monkeypatch.setattr(verify, "_law_get", fake_get)
    r = verify.verify_citation({"law_name": "도로교통법", "article_no": "제17조"})
    assert r["api_match"] is True
    assert r["current"] is True
    assert r["effective_from"] == "2026-04-02"
    assert "OC=" not in json.dumps(r, ensure_ascii=False)
    # DB empty (fixture) so overall verified is False, but the API confirmed it.
    assert r["db_match"] is False


def test_statute_misquoted_article_detected(monkeypatch):
    def fake_get(path, params):
        if path == verify._SEARCH_PATH:
            return _fake_law_search_hit()
        return _fake_law_service("17")  # law has 제17조, citation asks 제999조

    monkeypatch.setattr(verify, "_law_get", fake_get)
    r = verify.verify_citation({"law_name": "도로교통법", "article_no": "제999조"})
    assert r["api_match"] is False
    assert r["verified"] is False
    assert "오인용" in r["note"] or "없음" in r["note"]


def test_statute_unknown_law_detected(monkeypatch):
    monkeypatch.setattr(
        verify, "_law_get", lambda p, q: {"LawSearch": {"law": []}}
    )
    r = verify.verify_citation({"law_name": "존재하지않는법", "article_no": "제1조"})
    assert r["api_match"] is False
    assert r["verified"] is False


def test_api_unavailable_degrades_to_db_only(monkeypatch):
    def boom(path, params):
        raise RuntimeError("network down")

    monkeypatch.setattr(verify, "_law_get", boom)
    # DB has the article (with text) => verified True on DB alone, api_match None.
    monkeypatch.setattr(
        verify,
        "_db_check_statute",
        lambda t, a: {"text": "본문", "trust_grade": "A", "effective_from": "2024-01-01"},
    )
    r = verify.verify_citation({"law_name": "민법", "article_no": "제4조"})
    assert r["api_match"] is None
    assert r["db_match"] is True
    assert r["verified"] is True
    assert r["trust_grade"] == "A"
    assert "DB-only" in r["note"]


# --------------------------------------------------------------------------- #
# Precedent verdicts (API mocked)                                             #
# --------------------------------------------------------------------------- #
def test_precedent_exact_match(monkeypatch):
    monkeypatch.setattr(
        verify, "_law_get", lambda p, q: _fake_prec_search("2024도10062")
    )
    r = verify.verify_citation({"사건번호": "2024도10062"})
    assert r["api_match"] is True
    assert r["effective_from"] == "2024-10-08"


def test_precedent_fuzzy_nonmatch_is_rejected(monkeypatch):
    # law.go.kr returns a *different* case (fuzzy search). Must NOT be accepted.
    monkeypatch.setattr(
        verify, "_law_get", lambda p, q: _fake_prec_search("2024도10062")
    )
    r = verify.verify_citation({"사건번호": "2022도1401"})
    assert r["api_match"] is False
    assert "허위사건" in r["note"] or "없음" in r["note"]


# --------------------------------------------------------------------------- #
# as_of_date point-in-time validity                                           #
# --------------------------------------------------------------------------- #
def test_as_of_before_effective_is_invalid(monkeypatch):
    def fake_get(path, params):
        if path == verify._SEARCH_PATH:
            return _fake_law_search_hit()  # 시행일자 2026-04-02
        return _fake_law_service("17")

    monkeypatch.setattr(verify, "_law_get", fake_get)
    monkeypatch.setattr(
        verify,
        "_db_check_statute",
        lambda t, a: {"text": "x", "trust_grade": "A", "effective_from": "2026-04-02"},
    )
    r = verify.verify_citation(
        {"law_name": "도로교통법", "article_no": "제17조"}, as_of_date="2025-01-01"
    )
    assert r["verified"] is False
    assert "미시행" in r["note"]


def test_batch_preserves_order_and_handles_malformed(monkeypatch):
    monkeypatch.setattr(verify, "_law_get", lambda p, q: {"LawSearch": {"law": []}})
    out = verify.verify_citations(
        [{"law_name": "X", "article_no": "제1조"}, {"foo": "bar"}]
    )
    assert len(out) == 2
    assert out[1]["verified"] is False and "형식 오류" in out[1]["note"]


# --------------------------------------------------------------------------- #
# Opt-in live smoke test (one real law.go.kr call)                            #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.getenv("LAWBOT_LIVE"), reason="set LAWBOT_LIVE=1 to hit law.go.kr"
)
def test_live_real_law_go_kr():
    r = verify.verify_citation({"law_name": "도로교통법", "article_no": "제17조"})
    assert r["api_match"] is True
    assert r["current"] is True
    assert "OC=" not in json.dumps(r, ensure_ascii=False)
