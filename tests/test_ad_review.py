"""Unit tests for ``search.ad_review`` (the ``/v1/ad-review`` backend).

Per the contract test convention (`_BUILD_CONTRACT.md` (i)):

* **Pure / mocked (always run, $0, no network):** PDF + text extraction, claim
  decomposition wiring, the per-claim RAG context builder, the two-stage
  citation firewall (context drop + law.go.kr 2nd-pass annotation), and the full
  ``review`` pipeline with the OpenAI client + retriever + verify mocked. A real
  ``reportlab``-generated PDF exercises the ``pdfplumber``/``pypdf`` extractor.
* **Sanctioned live (opt-in):** at most **one** real GPT call through the full
  pipeline, gated behind ``LAWBOT_LIVE=1`` so ordinary ``pytest`` runs cost
  nothing and need no network (cost rule: ≤1–2 real OpenAI calls total).

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_ad_review.py
    # opt-in single real GPT call:
    LAWBOT_LIVE=1 .venv/bin/python -m pytest -q tests/test_ad_review.py -k live
"""

from __future__ import annotations

import io
import os
from typing import Any

import pytest

import config
from search import ad_review


# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #
class _FakeHit:
    """Minimal retriever ``Hit`` stand-in (``.id``/``.score``/``.payload``)."""

    def __init__(self, hid: str, payload: dict[str, Any], score: float = 0.8) -> None:
        self.id = hid
        self.score = score
        self.payload = payload


def _ad_law_hit() -> _FakeHit:
    return _FakeHit(
        "ctx-1",
        {
            "doc_type": "law",
            "title": "표시·광고의 공정화에 관한 법률",
            "article_no": "제3조",
            "source_url": "https://www.law.go.kr/x",
            "trust_grade": "A",
            "text": "사업자등은 부당한 표시·광고 행위를 하여서는 아니 된다 …",
        },
    )


def _make_pdf(text_lines: list[str]) -> bytes:
    """Render a tiny single-page PDF (ASCII) for extraction tests."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in text_lines:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# extract_text                                                                #
# --------------------------------------------------------------------------- #
def test_extract_text_uses_raw_text_verbatim() -> None:
    out = ad_review.extract_text(text="  광고 문구  ", file_bytes=None, filename=None)
    assert out["source"] == "text"
    assert out["text"] == "광고 문구"
    assert out["n_chars"] == len("광고 문구")


def test_extract_text_requires_some_input() -> None:
    with pytest.raises(ValueError):
        ad_review.extract_text(text="   ", file_bytes=None, filename=None)


def test_extract_text_non_pdf_blob_is_decoded() -> None:
    out = ad_review.extract_text(
        text=None, file_bytes="안녕하세요".encode("utf-8"), filename="ad.txt"
    )
    assert out["source"] == "file"
    assert "안녕하세요" in out["text"]


def test_extract_text_truncates_huge_input() -> None:
    big = "가" * (ad_review._MAX_INPUT_CHARS + 500)
    out = ad_review.extract_text(text=big, file_bytes=None, filename=None)
    assert out["truncated"] is True
    assert out["n_chars"] == ad_review._MAX_INPUT_CHARS


def test_extract_pdf_real_reportlab_pdf() -> None:
    """A real generated PDF round-trips through pdfplumber/pypdf extraction."""
    pdf = _make_pdf(["MIRACLE CREAM removes 100% of wrinkles", "Best in the world"])
    out = ad_review.extract_text(text=None, file_bytes=pdf, filename="ad.pdf")
    assert out["source"] == "pdf"
    assert out["n_pages"] == 1
    assert "MIRACLE CREAM" in out["text"]
    assert "wrinkles" in out["text"]


def test_extract_pdf_detects_pdf_by_magic_without_extension() -> None:
    pdf = _make_pdf(["No extension but real PDF magic bytes"])
    out = ad_review.extract_text(text=None, file_bytes=pdf, filename="upload.bin")
    assert out["source"] == "pdf"
    assert "real PDF magic" in out["text"]


def test_extract_pdf_image_only_raises() -> None:
    """A PDF with no extractable text layer is a clear 422-worthy error."""
    # A valid but text-free PDF (blank page).
    blank = _make_pdf([])
    with pytest.raises(ValueError):
        ad_review.extract_text(text=None, file_bytes=blank, filename="scan.pdf")


# --------------------------------------------------------------------------- #
# RAG context builder                                                         #
# --------------------------------------------------------------------------- #
def test_retrieve_for_claims_numbers_and_indexes_blocks() -> None:
    hits = [_ad_law_hit()]
    context, index = ad_review._retrieve_for_claims(
        [{"text": "주름이 사라집니다", "claim_type": "효능"}],
        search_fn=lambda *a, **k: hits,
    )
    assert "[1] id=ctx-1" in context
    assert set(index) == {"ctx-1"}
    assert index["ctx-1"]["title"] == "표시·광고의 공정화에 관한 법률"
    assert index["ctx-1"]["location"] == "제3조"


def test_retrieve_for_claims_dedups_across_passes() -> None:
    """The same hit returned by scoped + supplementary passes appears once."""
    hits = [_ad_law_hit()]
    context, index = ad_review._retrieve_for_claims(
        [{"text": "최고의 제품", "claim_type": "최상급"}],
        search_fn=lambda *a, **k: hits,  # both passes return the same hit
    )
    assert context.count("id=ctx-1") == 1
    assert len(index) == 1


def test_retrieve_for_claims_survives_retriever_errors() -> None:
    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("qdrant down")

    context, index = ad_review._retrieve_for_claims(
        [{"text": "x", "claim_type": "y"}], search_fn=_boom
    )
    assert context == "" and index == {}


# --------------------------------------------------------------------------- #
# Citation firewall — context pass                                            #
# --------------------------------------------------------------------------- #
def _index_from(hits: list[_FakeHit]) -> dict[str, dict[str, Any]]:
    return ad_review._retrieve_for_claims(
        [{"text": "q", "claim_type": "t"}], search_fn=lambda *a, **k: hits
    )[1]


def test_context_firewall_drops_hallucinated_source_id() -> None:
    index = _index_from([_ad_law_hit()])
    out = ad_review._verify_against_context(
        [
            {"source_id": "ctx-1", "title": "echoed", "location": "제3조"},
            {"source_id": "ghost", "title": "지어낸법", "location": "제999조"},
        ],
        index,
    )
    assert [c["source_id"] for c in out] == ["ctx-1"]


def test_context_firewall_overrides_model_metadata() -> None:
    index = _index_from([_ad_law_hit()])
    out = ad_review._verify_against_context(
        [{"source_id": "ctx-1", "title": "엉뚱한법", "source_url": "https://fake"}], index
    )
    assert out[0]["title"] == "표시·광고의 공정화에 관한 법률"
    assert out[0]["source_url"] == "https://www.law.go.kr/x"
    assert out[0]["db_verified"] is True
    assert out[0]["api_verified"] is None  # not yet 2nd-verified


def test_context_firewall_keeps_finer_model_location() -> None:
    index = _index_from([_ad_law_hit()])
    out = ad_review._verify_against_context(
        [{"source_id": "ctx-1", "location": "제3조 제1항"}], index
    )
    assert out[0]["location"] == "제3조 제1항"


# --------------------------------------------------------------------------- #
# Citation firewall — law.go.kr 2nd pass                                      #
# --------------------------------------------------------------------------- #
def test_law_verify_annotates_and_adopts_authoritative_url() -> None:
    index = _index_from([_ad_law_hit()])
    cits = ad_review._verify_against_context([{"source_id": "ctx-1"}], index)
    seen: dict[str, Any] = {}

    def _verify(citation: dict[str, Any], as_of_date: str | None = None) -> dict[str, Any]:
        seen.update(citation)
        return {
            "api_match": True,
            "current": True,
            "source_url": "https://www.law.go.kr/authoritative",
            "note": "law.go.kr 현행 조문 일치: 제3조.",
        }

    ad_review._law_verify(cits, _verify)
    assert seen["law_name"] == "표시·광고의 공정화에 관한 법률"
    assert seen["article_no"] == "제3조"
    assert cits[0]["api_verified"] is True
    assert cits[0]["current"] is True
    assert cits[0]["source_url"] == "https://www.law.go.kr/authoritative"


def test_law_verify_downgrades_on_api_mismatch() -> None:
    index = _index_from([_ad_law_hit()])
    cits = ad_review._verify_against_context([{"source_id": "ctx-1"}], index)
    ad_review._law_verify(
        cits, lambda c, **k: {"api_match": False, "current": False, "note": "오인용"}
    )
    assert cits[0]["api_verified"] is False
    assert cits[0]["trust_grade"] == "B"


def test_law_verify_skips_when_no_verify_fn() -> None:
    index = _index_from([_ad_law_hit()])
    cits = ad_review._verify_against_context([{"source_id": "ctx-1"}], index)
    ad_review._law_verify(cits, None)  # must not raise
    assert cits[0]["api_verified"] is None


# --------------------------------------------------------------------------- #
# Full review() pipeline (mocked LLM + retriever + verify)                     #
# --------------------------------------------------------------------------- #
def _patch_pipeline(monkeypatch, *, decompose: dict, judge: dict, verify_fn) -> list[str]:
    """Wire deterministic fakes for the two LLM calls, retriever, and verify."""
    monkeypatch.setattr("search.retriever.search", lambda *a, **k: [_ad_law_hit()])

    seq: list[str] = []

    def _fake_structured(*, system: str, user: str, schema: dict, model: str | None = None):
        name = schema.get("name")
        seq.append(name)
        return decompose if name == "ad_claims" else judge

    monkeypatch.setattr(ad_review, "_structured_call", _fake_structured)
    monkeypatch.setattr(ad_review, "_resolve_verify_fn", lambda: verify_fn)
    return seq


def test_review_full_pipeline_two_llm_calls_and_firewall(monkeypatch) -> None:
    decompose = {
        "product_type": "화장품",
        "claims": [{"text": "주름이 100% 사라집니다", "claim_type": "효능효과"}],
    }
    judge = {
        "summary": "효능 과장으로 표시광고법 위반 소지가 있습니다 [1].",
        "issues": [
            {
                "claim": "주름이 100% 사라집니다",
                "verdict": "위반소지",
                "severity": "high",
                "rationale": "객관적 근거 없는 절대적 효능 표현 [1].",
                "law_basis": "표시광고법 제3조",
                "citations": [
                    {"source_id": "ctx-1", "title": "x", "location": "제3조", "source_url": "x"},
                    {"source_id": "ghost", "title": "허위", "location": "제9조", "source_url": "x"},
                ],
                "suggested_fix": "임상 결과 범위로 한정하십시오.",
            }
        ],
        "corrected_copy": "임상 결과에 따라 주름 개선에 도움을 줄 수 있습니다.",
    }

    def _verify(c: dict[str, Any], **k: Any) -> dict[str, Any]:
        return {"api_match": True, "current": True, "source_url": None, "note": "현행 일치."}

    seq = _patch_pipeline(monkeypatch, decompose=decompose, judge=judge, verify_fn=_verify)
    res = ad_review.review(text="주름이 100% 사라집니다! 단 한 번으로 완벽하게.")

    assert seq == ["ad_claims", "ad_review"]  # exactly two LLM calls
    assert res["claims_reviewed"] == 1
    assert res["ai_generated"] is True
    assert res["disclaimer"] == config.ANSWER_DISCLAIMER
    assert res["corrected_copy"]
    # Issue preserved; hallucinated citation dropped; survivor 2nd-verified.
    issue = res["issues"][0]
    assert issue["verdict"] == "위반소지"
    assert issue["severity"] == "high"
    assert [c["source_id"] for c in issue["citations"]] == ["ctx-1"]
    assert issue["citations"][0]["api_verified"] is True
    # Union of citations across issues, de-duplicated.
    assert [c["source_id"] for c in res["citations"]] == ["ctx-1"]


def test_review_coerces_bad_verdict_and_severity(monkeypatch) -> None:
    decompose = {"product_type": "식품", "claims": [{"text": "암 예방", "claim_type": "효능"}]}
    judge = {
        "summary": "s",
        "issues": [
            {
                "claim": "암 예방",
                "verdict": "TOTALLY_BOGUS",
                "severity": "catastrophic",
                "rationale": "r",
                "law_basis": "b",
                "citations": [],
                "suggested_fix": "",
            }
        ],
        "corrected_copy": "c",
    }
    _patch_pipeline(monkeypatch, decompose=decompose, judge=judge, verify_fn=None)
    res = ad_review.review(text="이 식품은 암을 예방합니다")
    assert res["issues"][0]["verdict"] == "확인필요"  # unknown verdict coerced
    assert res["issues"][0]["severity"] == "none"  # unknown severity coerced


def test_review_falls_back_when_no_claims(monkeypatch) -> None:
    """Empty claim decomposition still retrieves + judges on the whole ad."""
    captured: dict[str, Any] = {}

    monkeypatch.setattr("search.retriever.search", lambda *a, **k: [_ad_law_hit()])

    def _fake_structured(*, system: str, user: str, schema: dict, model: str | None = None):
        if schema.get("name") == "ad_claims":
            return {"product_type": "", "claims": []}
        captured["judge_called"] = True
        return {"summary": "s", "issues": [], "corrected_copy": ""}

    monkeypatch.setattr(ad_review, "_structured_call", _fake_structured)
    monkeypatch.setattr(ad_review, "_resolve_verify_fn", lambda: None)

    res = ad_review.review(text="평범한 광고 문구")
    assert captured.get("judge_called") is True
    assert res["claims_reviewed"] == 0


def test_review_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        ad_review.review(text=None, file_bytes=None, filename=None)


def test_module_exposes_review_callable_for_api() -> None:
    """api.main resolves the backend by ``hasattr(mod, 'review')``."""
    assert callable(getattr(ad_review, "review", None))


# --------------------------------------------------------------------------- #
# Sanctioned live test — ONE real GPT call (opt-in)                           #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("LAWBOT_LIVE") != "1",
    reason="live GPT call disabled (set LAWBOT_LIVE=1 to run; cost: ≤2 calls)",
)
def test_live_review_single_real_pipeline(monkeypatch) -> None:
    """Real GPT calls through the full pipeline (retriever + law API mocked).

    Asserts production invariants on real output: the AI notice is present, every
    returned citation's ``source_id`` was actually retrieved (firewall holds),
    and a corrected rewrite is produced. The retriever and law.go.kr verifier are
    mocked so the test needs no Qdrant/network beyond the OpenAI requests.
    """
    monkeypatch.setattr("search.retriever.search", lambda *a, **k: [_ad_law_hit()])
    monkeypatch.setattr(
        ad_review,
        "_resolve_verify_fn",
        lambda: (lambda c, **k: {"api_match": None, "current": None, "note": "DB-only"}),
    )

    res = ad_review.review(
        text="이 화장품을 바르면 주름이 100% 완전히 사라지고, 세계 최고의 효과를 보장합니다!"
    )

    assert res["ai_generated"] is True
    assert res["disclaimer"] == config.ANSWER_DISCLAIMER
    retrieved = {"ctx-1"}
    for issue in res["issues"]:
        for c in issue["citations"]:
            assert c["source_id"] in retrieved, "citation firewall breached"
    assert isinstance(res["summary"], str) and res["summary"].strip()
