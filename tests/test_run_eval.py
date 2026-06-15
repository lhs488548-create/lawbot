"""Unit tests for the golden-set evaluation harness (eval/run_eval.py).

These are offline tests: they use a fake backend so no network / OpenAI / Qdrant
call is made. They verify golden-set integrity, the matching/ranking logic, the
answer-pass citation & grounding audit, and aggregate metric math.
"""

from __future__ import annotations

import json

import pytest

from eval.run_eval import (
    Backend,
    GoldenItem,
    Report,
    _article_in_rows,
    _first_match_rank,
    _row_matches,
    _score_answer,
    evaluate,
    load_golden,
)


# --------------------------------------------------------------------------- #
# Golden-set integrity                                                         #
# --------------------------------------------------------------------------- #
def test_golden_set_loads_and_is_well_formed() -> None:
    items = load_golden()
    assert len(items) == 20, "contract: golden_set has 20 questions"
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids)), "ids must be unique"
    # All four corpora must be represented (coverage honesty).
    doc_types = {i.doc_type for i in items}
    assert {"law", "precedent", "admrule", "ordinance"} <= doc_types
    for it in items:
        assert it.query.strip(), f"{it.id}: empty query"


# --------------------------------------------------------------------------- #
# Matching / ranking                                                           #
# --------------------------------------------------------------------------- #
def _item(**kw) -> GoldenItem:
    base = dict(
        id="x",
        query="q",
        doc_type="law",
        expect_title_contains=[],
        expect_article=None,
        expect_keywords=[],
        as_of_date=None,
    )
    base.update(kw)
    return GoldenItem(**base)


def test_row_matches_requires_doc_type() -> None:
    item = _item(doc_type="law")
    assert _row_matches(item, {"doc_type": "law", "title": "민법", "text": "..."})
    assert not _row_matches(item, {"doc_type": "precedent", "title": "민법", "text": "..."})


def test_row_matches_title_and_keyword() -> None:
    item = _item(expect_title_contains=["민법"], expect_keywords=["성년"])
    assert _row_matches(item, {"doc_type": "law", "title": "민법", "text": "19세 성년"})
    # right title, missing keyword -> no match
    assert not _row_matches(item, {"doc_type": "law", "title": "민법", "text": "혼인"})
    # keyword present but wrong title -> no match
    assert not _row_matches(item, {"doc_type": "law", "title": "형법", "text": "성년"})


def test_first_match_rank_and_article() -> None:
    item = _item(expect_keywords=["속도"], expect_article="제17조")
    rows = [
        {"doc_type": "law", "title": "A", "text": "무관"},
        {"doc_type": "law", "title": "도로교통법", "text": "최고 속도", "article_no": "제17조"},
    ]
    assert _first_match_rank(item, rows) == 2
    assert _article_in_rows(item, rows) is True
    # no article expectation -> None
    assert _article_in_rows(_item(), rows) is None


# --------------------------------------------------------------------------- #
# Answer-pass audit                                                            #
# --------------------------------------------------------------------------- #
def test_score_answer_citation_ok_and_grounded() -> None:
    item = _item()
    answer = {
        "answer": "성년은 19세이다 [1].",
        "citations": [{"source_id": "LAW:1:법률#제4조#0"}],
        "used_context": [{"source_id": "LAW:1:법률#제4조#0"}],
    }
    n, cit_ok, grounded = _score_answer(item, answer, [])
    assert (n, cit_ok, grounded) == (1, True, True)


def test_score_answer_detects_unverifiable_citation() -> None:
    item = _item()
    answer = {
        "answer": "...",
        "citations": [{"source_id": "FAKE:hallucinated"}],
        "used_context": [{"source_id": "LAW:1:법률#제4조#0"}],
    }
    _, cit_ok, _ = _score_answer(item, answer, [])
    assert cit_ok is False


def test_score_answer_empty_citation_but_says_no_basis_is_grounded() -> None:
    item = _item()
    answer = {"answer": "검색결과만으로는 근거 불충분합니다.", "citations": [], "used_context": []}
    n, cit_ok, grounded = _score_answer(item, answer, [])
    assert n == 0
    assert cit_ok is True  # no citation cannot be wrong
    assert grounded is True  # explicit 근거 불충분


def test_score_answer_empty_citation_without_disclaimer_is_ungrounded() -> None:
    item = _item()
    answer = {"answer": "성년은 20세입니다.", "citations": [], "used_context": []}
    _, _, grounded = _score_answer(item, answer, [])
    assert grounded is False


# --------------------------------------------------------------------------- #
# Aggregation via a fake backend (no network)                                  #
# --------------------------------------------------------------------------- #
class _FakeBackend(Backend):
    """Returns canned rows/answers keyed by question id."""

    def __init__(self, rows_by_id, answers_by_id=None) -> None:
        self._rows = rows_by_id
        self._answers = answers_by_id or {}

    def search(self, item, k):
        return self._rows.get(item.id, [])

    def ask(self, item, k):
        return self._answers.get(item.id)


def test_evaluate_aggregates_metrics() -> None:
    items = [
        _item(id="q1", expect_keywords=["성년"]),
        _item(id="q2", expect_keywords=["속도"]),
    ]
    rows = {
        "q1": [{"doc_type": "law", "title": "민법", "text": "성년", "article_no": "제4조"}],
        "q2": [{"doc_type": "law", "title": "X", "text": "무관"}],  # miss
    }
    report = evaluate(_FakeBackend(rows), items, k=8, ask=False, ask_limit=0)
    assert report.n == 2
    assert report.hit_at_k == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)  # q1 rank1 -> 1.0, q2 -> 0
    assert report.citation_accuracy is None  # no ask pass


def test_evaluate_respects_ask_limit() -> None:
    items = [_item(id=f"q{i}", expect_keywords=["x"]) for i in range(5)]
    rows = {it.id: [{"doc_type": "law", "title": "t", "text": "x"}] for it in items}
    answers = {
        it.id: {"answer": "a", "citations": [{"source_id": "S"}], "used_context": [{"source_id": "S"}]}
        for it in items
    }
    report = evaluate(_FakeBackend(rows, answers), items, k=8, ask=True, ask_limit=2)
    asked = [i for i in report.items if i.asked]
    assert len(asked) == 2  # cost cap honored
    assert report.citation_accuracy == pytest.approx(1.0)
    assert report.grounding == pytest.approx(1.0)


def test_evaluate_survives_backend_error() -> None:
    class _Boom(Backend):
        def search(self, item, k):
            raise RuntimeError("qdrant down")

        def ask(self, item, k):
            return None

    items = [_item(id="q1")]
    report = evaluate(_Boom(), items, k=8, ask=False, ask_limit=0)
    assert report.items[0].error is not None
    assert report.n == 0  # errored items excluded from scoring denominator
