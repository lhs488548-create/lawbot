"""Unit tests for ``search.source_pack`` (09 §E-3, ``/v1/source-pack``).

These tests are **offline and free**: the retriever's ``search`` is monkeypatched
to return canned hits and the parents index is stubbed, so no OpenAI or Qdrant
call is made (contract (i): tests must not require live network beyond the
sanctioned 1–2 OpenAI calls, of which source-pack needs zero).

Run from the project root so imports resolve::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_source_pack.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import config
from search import retriever as retriever_mod
from search import source_pack


@dataclass
class FakeHit:
    """Minimal stand-in for a retriever ``Hit`` (``.id``, ``.score``, ``.payload``)."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def _law_hit(article_no: str, score: float, *, eff: str = "2013-07-01") -> FakeHit:
    return FakeHit(
        id=f"h-{article_no}",
        score=score,
        payload={
            "chunk_id": f"LAW:000001:법률#{article_no}#0",
            "parent_id": "LAW:000001:법률",
            "doc_type": "law",
            "title": "민법",
            "law_kind": "법률",
            "jurisdiction": "국가",
            "article_no": article_no,
            "effective_from": eff,
            "source_url": "https://law.go.kr/민법",
            "trust_grade": "A",
            "text": f"[민법 {article_no}] 본문 {article_no}.",
        },
    )


def _prec_hit(score: float, *, eff: str = "2020-05-14") -> FakeHit:
    return FakeHit(
        id="h-prec",
        score=score,
        payload={
            "chunk_id": "PREC:424370#판결요지#0",
            "parent_id": "PREC:424370",
            "doc_type": "precedent",
            "title": "손해배상청구",
            "law_kind": "민사",
            "jurisdiction": "대법원",
            "article_no": "판결요지",
            "effective_from": eff,
            "source_url": "https://law.go.kr/prec",
            "trust_grade": "A",
            "text": "[대법원 2020다12345 판결요지] 손해배상 책임 인정.",
        },
    )


@pytest.fixture
def stub_retrieval(monkeypatch: pytest.MonkeyPatch):
    """Install a canned ``retriever.search`` and a stubbed parents index.

    Yields a setter taking the list of hits each ``search`` call should return.
    민법 has a materialized parent (full_text); the precedent has none so it
    exercises the child-text fallback path.
    """
    state: dict[str, list[FakeHit]] = {"hits": []}

    def fake_search(query: str, k: int = config.DEFAULT_TOP_K, flt: Any = None):
        out = state["hits"]
        if flt and "doc_type" in flt:
            out = [h for h in out if h.payload.get("doc_type") == flt["doc_type"]]
        return out[:k]

    def stub_parents() -> dict[str, Any]:
        return {
            "LAW:000001:법률": {
                "parent_id": "LAW:000001:법률",
                "doc_type": "law",
                "title": "민법",
                "law_kind": "법률",
                "jurisdiction": "국가",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "license": config.DEFAULT_LICENSE,
                "trust_grade": "A",
                "full_text": "제4조(성년) 사람은 19세로 성년에 이른다.\n\n제5조 …",
            }
        }

    monkeypatch.setattr(retriever_mod, "search", fake_search)
    # Force the source_pack to *not* find retriever.get_parent so the JSONL/stub
    # path is used deterministically.
    monkeypatch.setattr(retriever_mod, "get_parent", None, raising=False)
    source_pack._retriever_get_parent.cache_clear()
    monkeypatch.setattr(source_pack, "_parents_index", stub_parents)

    def _set(hits: list[FakeHit]) -> None:
        state["hits"] = hits

    yield _set
    source_pack._retriever_get_parent.cache_clear()


def test_build_promotes_children_to_distinct_parents(stub_retrieval):
    stub_retrieval([_law_hit("제4조", 0.82), _law_hit("제5조", 0.61), _prec_hit(0.55)])
    pack = source_pack.build("성년 나이와 손해배상", k=8)

    assert set(pack) == {"markdown", "sources", "as_of_date"}
    # Two distinct parents (민법 collapses its two child hits), best-score first.
    ids = [s["doc_id"] for s in pack["sources"]]
    assert ids == ["LAW:000001:법률", "PREC:424370"]


def test_parent_fulltext_vs_child_fallback(stub_retrieval):
    stub_retrieval([_law_hit("제4조", 0.82), _prec_hit(0.55)])
    pack = source_pack.build("질의", k=8)

    by_id = {s["doc_id"]: s for s in pack["sources"]}
    # 민법 resolves from the materialized parent full_text.
    assert by_id["LAW:000001:법률"]["text_origin"] == "parent"
    # The precedent has no parent record -> reconstructed from child text.
    assert by_id["PREC:424370"]["text_origin"] == "child"
    # The bundle markdown contains the law's full article text.
    assert "사람은 19세로 성년에 이른다" in pack["markdown"]


def test_common_meta_present_on_every_source(stub_retrieval):
    stub_retrieval([_law_hit("제4조", 0.82), _prec_hit(0.55)])
    pack = source_pack.build("질의", k=8)

    for s in pack["sources"]:
        for key in ("trust_grade", "source_url", "license", "effective_from"):
            assert key in s, f"missing common-meta key {key!r}"
        # License always defaults to the configured public-domain notice.
        assert s["license"]


def test_as_of_date_excludes_future_sources(stub_retrieval):
    # Precedent decided 2025 must be excluded for an as_of of 2020.
    stub_retrieval([_law_hit("제4조", 0.82), _prec_hit(0.55, eff="2025-05-14")])
    pack = source_pack.build("질의", k=8, as_of_date="2020-01-01")

    ids = [s["doc_id"] for s in pack["sources"]]
    assert "PREC:424370" not in ids
    assert "LAW:000001:법률" in ids
    assert pack["as_of_date"] == "2020-01-01"
    # Echoed onto each source's common meta too.
    assert all(s["as_of_date"] == "2020-01-01" for s in pack["sources"])


def test_doc_type_filter_forwarded(stub_retrieval):
    stub_retrieval([_law_hit("제4조", 0.82), _prec_hit(0.55)])
    pack = source_pack.build("질의", k=8, filter={"doc_type": "precedent"})

    assert [s["doc_id"] for s in pack["sources"]] == ["PREC:424370"]


def test_parent_cap_enforced(stub_retrieval, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_PACK_MAX_PARENTS", 1)
    stub_retrieval([_law_hit("제4조", 0.82), _prec_hit(0.55)])
    pack = source_pack.build("질의", k=8)

    assert len(pack["sources"]) == 1
    assert pack["sources"][0]["doc_id"] == "LAW:000001:법률"


def test_empty_retrieval_returns_no_sources(stub_retrieval):
    stub_retrieval([])
    pack = source_pack.build("존재하지 않는 질의", k=8)

    assert pack["sources"] == []
    assert "소스 팩" in pack["markdown"]


def test_blank_query_rejected(stub_retrieval):
    with pytest.raises(ValueError):
        source_pack.build("   ")


def test_invalid_k_rejected(stub_retrieval):
    with pytest.raises(ValueError):
        source_pack.build("질의", k=0)


def test_markdown_flags_child_origin(stub_retrieval):
    stub_retrieval([_prec_hit(0.55)])
    pack = source_pack.build("질의", k=8)
    # Honest flag that only partial (child) text was available.
    assert "전체 원문 미확보" in pack["markdown"]
