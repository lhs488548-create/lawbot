"""Unit tests for ``search.retriever`` (Playbook 08 Task 3.1; 09 §E-1).

These tests are fully offline and free:

* The **FAISS** path runs against an in-memory ``_DummyFlatIP`` index injected
  via ``retriever.set_index`` — no Docker, no network, no cloud.
* The **OpenAI** embedding call is replaced by a deterministic fake via
  monkeypatching ``embed_query`` — **zero real OpenAI calls** (cost rule: tests
  must not bill the account).

Coverage: filter construction (allow-list, match-any, ``parent_id``), the
``as_of_date`` point-in-time current-law filter, ranking, ``Hit`` accessors
(incl. the ``parent_id`` fallback derived from ``chunk_id``), the server-reject
client-side ``as_of_date`` fallback, and ``get_parent`` parent-promotion.
"""

from __future__ import annotations

import json

import pytest

import config
from search import retriever
from search.retriever import Hit, build_filter, get_parent, search

# --------------------------------------------------------------------------- #
# Fixtures: an in-memory Qdrant seeded with predictable rows                   #
# --------------------------------------------------------------------------- #
_DIM = 4

# (vec, chunk_id, payload-without-chunk_id). The 민법 row deliberately omits
# ``parent_id`` to exercise the chunk_id-derived fallback in ``Hit.parent_id``.
_FIXTURES = [
    (
        [1.0, 0.0, 0.0, 0.0],
        "LAW:000001:법률#제4조#0",
        {
            "text": "[민법 제4조 성년] 사람은 19세로 성년에 이르게 된다.",
            "doc_type": "law",
            "title": "민법",
            "jurisdiction": "국가",
            "law_kind": "법률",
            "article_no": "제4조",
            "effective_from": "2013-07-01",
            "source_url": "https://law.go.kr/민법",
            "trust_grade": "A",
        },
    ),
    (
        [0.0, 1.0, 0.0, 0.0],
        "ORD:전라남도:2200001#제2조#0",
        {
            "text": "[전라남도 ○○ 조례 제2조] 정의 규정.",
            "doc_type": "ordinance",
            "title": "전라남도 ○○ 조례",
            "jurisdiction": "전라남도",
            "law_kind": "조례",
            "article_no": "제2조",
            "effective_from": "2022-01-01",
            "source_url": "https://law.go.kr/ord",
            "trust_grade": "A",
            "parent_id": "ORD:전라남도:2200001",
        },
    ),
    (
        [0.0, 0.0, 1.0, 0.0],
        "PREC:424370#판결요지#0",
        {
            "text": "[대법원 2020다12345 판결요지] 손해배상 책임 인정.",
            "doc_type": "precedent",
            "title": "손해배상청구",
            "jurisdiction": "대법원",
            "law_kind": "민사",
            "article_no": "판결요지",
            "effective_from": "2020-05-14",
            "source_url": "https://law.go.kr/prec",
            "trust_grade": "A",
            "parent_id": "PREC:424370",
        },
    ),
    (
        [0.95, 0.05, 0.0, 0.0],
        "LAW:000099:법률#제1조#0",
        {
            "text": "[미래법 제1조] 2030년 시행 예정 조문.",
            "doc_type": "law",
            "title": "미래법",
            "jurisdiction": "국가",
            "law_kind": "법률",
            "article_no": "제1조",
            "effective_from": "2030-01-01",
            "source_url": "https://law.go.kr/future",
            "trust_grade": "A",
            "parent_id": "LAW:000099:법률",
        },
    ),
]


@pytest.fixture()
def seeded_qdrant(monkeypatch):
    """Inject an in-memory FAISS (``_DummyFlatIP``) index seeded with ``_FIXTURES``.

    Mirrors ``embed.faiss_index.load_index()``'s return shape (row-aligned metas,
    row ``i`` ↔ ``metas[i]``) and stubs embedding so no OpenAI call is made. The
    fixture name is kept for back-compat with the existing test signatures.
    """
    index = retriever._DummyFlatIP([vec for vec, _cid, _pl in _FIXTURES])
    metas = [
        {
            "chunk_id": cid,
            "doc_id": cid.split("#", 1)[0],
            "parent_id": payload.get("parent_id"),
            "text": payload["text"],
            "payload": dict(payload),
        }
        for _vec, cid, payload in _FIXTURES
    ]
    retriever.set_index(index=index, metas=metas)
    # Deterministic fake embedding on the 민법 axis -> 민법 ranks first. No
    # OpenAI call is ever made.
    monkeypatch.setattr(
        retriever, "embed_query", lambda _q: retriever._l2_normalize([1.0, 0.0, 0.0, 0.0])
    )
    yield
    # Drop the injected index so other tests don't reuse this in-memory store.
    retriever.reset_index_cache()


# --------------------------------------------------------------------------- #
# build_filter                                                                 #
# --------------------------------------------------------------------------- #
def test_build_filter_none_returns_none():
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_build_filter_rejects_unknown_key():
    with pytest.raises(ValueError):
        build_filter({"nope": "x"})


def test_build_filter_rejects_empty_value():
    with pytest.raises(ValueError):
        build_filter({"doc_type": ""})
    with pytest.raises(ValueError):
        build_filter({"doc_type": None})
    with pytest.raises(ValueError):
        build_filter({"doc_type": []})


def test_build_filter_scalar_and_match_any():
    # FAISS era: build_filter returns a row-predicate, not a Qdrant Filter.
    flt = build_filter({"doc_type": "law", "law_kind": ["법률", "시행령"]})
    assert callable(flt)
    assert flt({"doc_type": "law", "law_kind": "법률"})
    assert flt({"doc_type": "law", "law_kind": "시행령"})  # match-any
    assert not flt({"doc_type": "law", "law_kind": "부령"})  # match-any miss
    assert not flt({"doc_type": "ordinance", "law_kind": "법률"})  # scalar miss


def test_build_filter_parent_id_allowed():
    flt = build_filter({"parent_id": "LAW:000001:법률"})
    assert callable(flt)
    assert flt({"parent_id": "LAW:000001:법률"})
    assert not flt({"parent_id": "LAW:000002:법률"})


def test_build_filter_as_of_date_adds_range():
    flt = build_filter(None, as_of_date="2025-01-01")
    assert callable(flt)
    assert flt({"effective_from": "2013-07-01"})  # in force by 2025 -> kept
    assert not flt({"effective_from": "2030-01-01"})  # future -> excluded


@pytest.mark.parametrize("bad", ["2025/01/01", "2025-1-1", "20250101", "not-a-date", ""])
def test_build_filter_rejects_bad_as_of_date(bad):
    with pytest.raises(ValueError):
        build_filter(None, as_of_date=bad)


# --------------------------------------------------------------------------- #
# Hit accessors                                                               #
# --------------------------------------------------------------------------- #
def test_hit_parent_id_prefers_payload():
    h = Hit(id="x", score=1.0, payload={"parent_id": "LAW:1:법률", "chunk_id": "LAW:1:법률#제2조#0"})
    assert h.parent_id == "LAW:1:법률"
    assert h.doc_id == "LAW:1:법률"


def test_hit_parent_id_falls_back_to_chunk_id():
    h = Hit(id="x", score=1.0, payload={"chunk_id": "LAW:9:시행령#제3조#1"})
    assert h.parent_id == "LAW:9:시행령"


def test_hit_parent_id_none_when_unknown():
    assert Hit(id="x", score=1.0, payload={}).parent_id is None


def test_hit_effective_from_normalizes_blank():
    assert Hit(id="x", score=1.0, payload={"effective_from": ""}).effective_from is None
    assert Hit(id="x", score=1.0, payload={"effective_from": "2024-01-01"}).effective_from == "2024-01-01"


def test_hit_trust_grade_default_a():
    assert Hit(id="x", score=1.0, payload={}).trust_grade == "A"
    assert Hit(id="x", score=1.0, payload={"trust_grade": "B"}).trust_grade == "B"


# --------------------------------------------------------------------------- #
# search: ranking + metadata filters                                          #
# --------------------------------------------------------------------------- #
def test_search_rejects_blank_query(seeded_qdrant):
    with pytest.raises(ValueError):
        search("   ")


def test_search_ranks_law_first_and_populates_accessors(seeded_qdrant):
    hits = search("성년 나이", k=4)
    assert hits, "expected hits"
    top = hits[0]
    assert top.doc_type == "law"
    assert top.title == "민법"
    assert top.location() == "제4조"
    # parent_id derived from chunk_id (payload key intentionally absent).
    assert top.parent_id == "LAW:000001:법률"
    assert -1.0001 <= top.score <= 1.0001


def test_search_doc_type_filter(seeded_qdrant):
    hits = search("아무 질의", k=4, flt={"doc_type": "ordinance"})
    assert [h.doc_type for h in hits] == ["ordinance"]


def test_search_non_matching_filter_returns_empty(seeded_qdrant):
    assert search("아무 질의", k=4, flt={"jurisdiction": "서울특별시"}) == []


def test_search_match_any_filter(seeded_qdrant):
    hits = search("아무 질의", k=4, flt={"doc_type": ["ordinance", "precedent"]})
    assert {h.doc_type for h in hits} == {"ordinance", "precedent"}


def test_search_parent_id_filter_scopes(seeded_qdrant):
    hits = search("아무 질의", k=4, flt={"parent_id": "PREC:424370"})
    assert [h.parent_id for h in hits] == ["PREC:424370"]


# --------------------------------------------------------------------------- #
# search: as_of_date point-in-time filter                                      #
# --------------------------------------------------------------------------- #
def test_as_of_date_drops_future_law(seeded_qdrant):
    hits = search("성년 나이", k=4, as_of_date="2025-01-01")
    titles = {h.title for h in hits}
    assert "미래법" not in titles  # effective 2030 -> excluded at 2025
    assert "민법" in titles  # effective 2013 -> kept


def test_as_of_date_before_everything_empty(seeded_qdrant):
    assert search("성년 나이", k=4, as_of_date="2000-01-01") == []


def test_as_of_date_combined_with_filter(seeded_qdrant):
    hits = search("성년 나이", k=4, flt={"doc_type": "law"}, as_of_date="2025-01-01")
    assert {h.title for h in hits} == {"민법"}


def test_search_rejects_bad_as_of_date(seeded_qdrant):
    with pytest.raises(ValueError):
        search("질의", as_of_date="2025/01/01")


# NOTE: the old ``test_as_of_date_client_side_fallback`` (Qdrant server-side
# DatetimeRange rejection → client-side fallback) was removed in the FAISS port:
# FAISS applies the as_of cut as a Python row-predicate always, so there is no
# server-side range to reject. The cut itself is covered by the as_of tests above.


# --------------------------------------------------------------------------- #
# get_parent (parent-promotion)                                               #
# --------------------------------------------------------------------------- #
def test_get_parent_loads_record(tmp_path, monkeypatch):
    parents = tmp_path / "parents.jsonl"
    parents.write_text(
        json.dumps(
            {"parent_id": "LAW:000001:법률", "doc_type": "law", "title": "민법",
             "full_text": "[민법] 제4조(성년) …"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PARENTS_JSONL", parents)
    retriever.reset_parents_cache()
    rec = get_parent("LAW:000001:법률")
    assert rec is not None
    assert rec["title"] == "민법"
    assert "full_text" in rec
    assert get_parent("LAW:nope") is None
    assert get_parent("") is None
    retriever.reset_parents_cache()


def test_get_parent_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PARENTS_JSONL", tmp_path / "absent.jsonl")
    retriever.reset_parents_cache()
    assert get_parent("LAW:000001:법률") is None
    retriever.reset_parents_cache()
