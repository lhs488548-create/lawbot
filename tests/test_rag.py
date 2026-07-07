"""Unit tests for ``search.rag`` (RAG ask + citation post-verification).

Two layers, per the contract test convention (`_BUILD_CONTRACT.md` (i)):

* **Pure / mocked (always run, $0, no network):** the citation post-verifier,
  context construction, the empty-/low-score grounding gates, ``as_of_date``
  forwarding, and the full ``ask`` pipeline with the OpenAI client and the
  retriever mocked. These cover the anti-hallucination control end to end
  without spending a token.
* **Sanctioned live (opt-in):** exactly **one** real GPT call exercising
  Structured Outputs + citation verification, gated behind ``LAWBOT_LIVE=1`` so
  ordinary ``pytest`` runs cost nothing and need no network (cost rule: at most
  1–2 real OpenAI calls total across the suite).

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_rag.py
    # opt-in single real GPT call:
    LAWBOT_LIVE=1 .venv/bin/python -m pytest -q tests/test_rag.py -k live
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest

import config
from search import rag


# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class FakeHit:
    """Minimal stand-in for a retriever ``Hit`` (``.id``/``.score``/``.payload``)."""

    id: str
    score: float = 0.9
    payload: dict[str, Any] = field(default_factory=dict)


def _hit(
    hid: str,
    *,
    doc_type: str = "law",
    title: str = "도로교통법",
    article_no: str = "제17조",
    url: str = "https://www.law.go.kr/x",
    trust: str = "A",
    text: str = "자동차등의 속도는 ...",
    score: float = 0.9,
) -> FakeHit:
    return FakeHit(
        id=hid,
        score=score,
        payload={
            "doc_type": doc_type,
            "title": title,
            "article_no": article_no,
            "source_url": url,
            "trust_grade": trust,
            "text": text,
        },
    )


# --------------------------------------------------------------------------- #
# build_context                                                               #
# --------------------------------------------------------------------------- #
def test_build_context_numbers_blocks_and_indexes_by_source_id() -> None:
    hits = [_hit("id-a"), _hit("id-b", title="민법", article_no="제4조")]
    context, index = rag.build_context(hits)

    assert "[1] id=id-a" in context
    assert "[2] id=id-b" in context
    # source_index keyed by the (stringified) hit id, with authoritative meta.
    assert set(index) == {"id-a", "id-b"}
    assert index["id-b"]["title"] == "민법"
    assert index["id-b"]["location"] == "제4조"


def test_build_context_flags_b_grade_metadata_only() -> None:
    hits = [_hit("id-b", trust="B", text="")]
    context, index = rag.build_context(hits)
    assert "B등급" in context  # honest uncertainty surfacing
    assert index["id-b"]["trust_grade"] == "B"


def test_build_context_truncates_long_bodies() -> None:
    long_text = "가" * 5000
    context, _ = rag.build_context([_hit("id-a", text=long_text)])
    assert "…(생략)" in context
    assert len(context) < 5000  # truncated well below the raw length


# --------------------------------------------------------------------------- #
# verify_citations — the always-on anti-hallucination control                 #
# --------------------------------------------------------------------------- #
def _index_from(hits: list[FakeHit]) -> dict[str, dict[str, Any]]:
    return rag.build_context(hits)[1]


def test_verify_drops_hallucinated_source_id() -> None:
    index = _index_from([_hit("real-1")])
    cites = [
        {"source_id": "real-1", "title": "X", "location": "제1조"},
        {"source_id": "ghost-99", "title": "지어낸 법", "location": "제999조"},
    ]
    out = rag.verify_citations(cites, index)
    ids = [c["source_id"] for c in out]
    assert ids == ["real-1"]  # the fabricated citation is removed


def test_verify_overrides_model_metadata_with_retrieved_truth() -> None:
    # The model echoes a WRONG title/url; the verifier must replace them with the
    # authoritative retrieved metadata (model can't alter a citation's identity).
    index = _index_from([_hit("real-1", title="도로교통법", url="https://real")])
    cites = [{"source_id": "real-1", "title": "엉뚱한법", "source_url": "https://fake"}]
    out = rag.verify_citations(cites, index)
    assert out[0]["title"] == "도로교통법"
    assert out[0]["source_url"] == "https://real"


def test_verify_keeps_model_location_when_present() -> None:
    index = _index_from([_hit("real-1", article_no="제17조")])
    cites = [{"source_id": "real-1", "location": "제17조 제1항"}]
    out = rag.verify_citations(cites, index)
    assert out[0]["location"] == "제17조 제1항"  # finer pin-cite preserved


def test_verify_dedups_by_source_id_preserving_order() -> None:
    index = _index_from([_hit("a"), _hit("b")])
    cites = [
        {"source_id": "b"},
        {"source_id": "a"},
        {"source_id": "b"},  # duplicate
    ]
    out = rag.verify_citations(cites, index)
    assert [c["source_id"] for c in out] == ["b", "a"]


@pytest.mark.parametrize(
    "bad",
    [
        [],  # nothing
        [{"title": "no id"}],  # missing source_id
        [{"source_id": ""}],  # empty source_id
        ["not-a-dict"],  # wrong type
        [{"source_id": "unknown"}],  # not in context
    ],
)
def test_verify_discards_invalid_citations(bad: list[Any]) -> None:
    index = _index_from([_hit("known")])
    assert rag.verify_citations(bad, index) == []


# --------------------------------------------------------------------------- #
# ask — grounding gates (no model call expected)                              #
# --------------------------------------------------------------------------- #
def test_ask_rejects_blank_query() -> None:
    with pytest.raises(ValueError):
        rag.ask("   ")


def test_ask_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        rag.ask("질문", k=0)


def test_ask_empty_retrieval_returns_grounded_no_citations(monkeypatch) -> None:
    """No hits ⇒ honest '근거 불충분', no model call, no citations."""
    monkeypatch.setattr("search.retriever.search", lambda *a, **k: [])

    def _boom(*a: Any, **k: Any) -> Any:  # the model must NOT be called
        raise AssertionError("model should not be called when retrieval is empty")

    monkeypatch.setattr(rag, "_generate", _boom)

    res = rag.ask("아주 생소한 질의")
    assert res["citations"] == []
    assert res["ai_generated"] is True
    assert res["disclaimer"] == config.ANSWER_DISCLAIMER
    assert "불충분" in res["answer"]


def test_ask_low_score_returns_grounded_no_citations(monkeypatch) -> None:
    """Best hit below MIN_RETRIEVAL_SCORE ⇒ no model call, no fabrication."""
    low = config.MIN_RETRIEVAL_SCORE - 0.05
    monkeypatch.setattr(
        "search.retriever.search", lambda *a, **k: [_hit("x", score=low)]
    )

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("model should not be called below the score floor")

    monkeypatch.setattr(rag, "_generate", _boom)

    res = rag.ask("관련도 낮은 질의")
    assert res["citations"] == []
    assert "불충분" in res["answer"]


# --------------------------------------------------------------------------- #
# ask — full pipeline with mocked model (verifies wiring + citation firewall) #
# --------------------------------------------------------------------------- #
def test_ask_full_pipeline_mocked_model_verifies_citations(monkeypatch) -> None:
    hits = [_hit("good-1", score=0.8), _hit("good-2", score=0.7)]
    monkeypatch.setattr("search.retriever.search", lambda *a, **k: hits)

    # Model returns one valid citation and one fabricated one — the latter must
    # be filtered by the post-verifier.
    fake_parsed = {
        "answer": "자동차 속도는 도로교통법 제17조에 따른다 [1].",
        "citations": [
            {"source_id": "good-1", "title": "echoed", "location": "제17조"},
            {"source_id": "fabricated", "title": "허위", "location": "제999조"},
        ],
    }
    captured: dict[str, Any] = {}

    def _fake_generate(query: str, context: str, model: str) -> dict[str, Any]:
        captured["context"] = context
        captured["model"] = model
        return fake_parsed

    monkeypatch.setattr(rag, "_generate", _fake_generate)

    res = rag.ask("자동차 속도 제한은?", k=2)

    assert res["answer"].startswith("자동차 속도")
    # Only the verifiable citation survives; metadata is the retrieved truth.
    assert [c["source_id"] for c in res["citations"]] == ["good-1"]
    assert res["citations"][0]["title"] == "도로교통법"
    assert res["citations"][0]["doc_type"] == "law"
    assert res["model"] == config.GEN_MODEL
    assert res["ai_generated"] is True
    assert len(res["used_context"]) == 2
    # Context handed to the model is numbered and id-tagged.
    assert "[1] id=good-1" in captured["context"]


def test_ask_forwards_as_of_date_when_retriever_supports_it(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def _search(query: str, k: int = 8, flt=None, as_of_date=None):
        seen["as_of_date"] = as_of_date
        return [_hit("g", score=0.8)]

    monkeypatch.setattr("search.retriever.search", _search)
    monkeypatch.setattr(
        rag, "_generate", lambda q, c, m: {"answer": "ok", "citations": []}
    )

    rag.ask("질의", as_of_date="2025-01-01")
    assert seen["as_of_date"] == "2025-01-01"


def test_call_search_skips_as_of_date_when_unsupported() -> None:
    """Graceful degradation: a retriever without as_of_date is still callable."""

    def _legacy_search(query: str, k: int = 8, flt=None):  # no as_of_date kw
        return ["hit"]

    out = rag._call_search(_legacy_search, "q", 8, None, "2025-01-01")
    assert out == ["hit"]  # did not crash; kw was dropped


# --------------------------------------------------------------------------- #
# Sanctioned live test — ONE real GPT call (opt-in)                           #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("LAWBOT_LIVE") != "1",
    reason="live GPT call disabled (set LAWBOT_LIVE=1 to run; cost: 1 call)",
)
def test_live_citation_verification_single_gpt_call(monkeypatch) -> None:
    """One real ``gpt`` call through the full ask pipeline.

    Asserts the production invariants on real output: a non-empty grounded
    answer, and that **every** returned citation's ``source_id`` is one that was
    actually retrieved (the citation firewall holds end to end). The retriever is
    mocked so this test needs no Qdrant and makes exactly one OpenAI request.
    """
    hits = [
        _hit(
            "live-1",
            doc_type="law",
            title="도로교통법",
            article_no="제17조",
            text="제17조(자동차등의 속도) 자동차등의 운전자는 법정 최고속도를 준수하여야 한다.",
            score=0.85,
        )
    ]
    monkeypatch.setattr("search.retriever.search", lambda *a, **k: hits)

    res = rag.ask("자동차 운전자의 속도 준수 의무는 어느 조문에 있나?", k=1)

    assert res["answer"].strip(), "answer must be non-empty"
    assert res["ai_generated"] is True
    assert res["disclaimer"] == config.ANSWER_DISCLAIMER
    retrieved_ids = {h.id for h in hits}
    for c in res["citations"]:
        assert c["source_id"] in retrieved_ids, (
            f"citation source_id {c['source_id']!r} was not retrieved — "
            "citation firewall breached"
        )


# --------------------------------------------------------------------------- #
# Trust signal (P5) — per-citation flag + aggregate trust_score                #
# --------------------------------------------------------------------------- #
def test_aggregate_trust_scores() -> None:
    assert rag._aggregate_trust([]) == 0
    assert rag._aggregate_trust([{"trust_flag": "green"}, {"trust_flag": "green"}]) == 100
    assert rag._aggregate_trust([{"trust_flag": "green"}, {"trust_flag": "yellow"}]) == 80


def test_verify_citations_attaches_trust_signal() -> None:
    idx = _index_from([_hit("a-1", trust="A")])
    out = rag.verify_citations([{"source_id": "a-1"}], idx)
    assert out[0]["trust_flag"] == "green"  # full text held
    assert out[0]["status"] in ("현행", "미상")
    assert "effective_from" in out[0]


def test_verify_citations_b_grade_is_yellow() -> None:
    idx = _index_from([_hit("b-1", trust="B")])  # metadata-only
    out = rag.verify_citations([{"source_id": "b-1"}], idx)
    assert out[0]["trust_flag"] == "yellow"
