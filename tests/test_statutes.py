"""Unit tests for ``search.statutes`` (unified /v1/statutes/search).

These tests run **fully offline** — the shared retriever's ``search`` is
monkeypatched with a deterministic stub, so no OpenAI embedding call and no live
Qdrant are required (test convention, contract §(i): at most 1–2 real OpenAI
calls *total* across the suite; this module makes none).

Run from the project root::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_statutes.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import config
from search import retriever, statutes


# --------------------------------------------------------------------------- #
# Fixtures: a tiny in-memory corpus and a stub retriever                       #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeHit:
    """A minimal stand-in for ``search.retriever.Hit`` (``.id/.score/.payload``)."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def _corpus() -> list[_FakeHit]:
    """Four ranked rows spanning law / precedent / future ordinance / date-less law."""
    return [
        _FakeHit(
            id="uuid-law-1",
            score=0.91,
            payload={
                "chunk_id": "LAW:000001:법률#제4조#0",
                "doc_id": "LAW:000001:법률",
                "parent_id": "LAW:000001:법률",
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
        _FakeHit(
            id="uuid-prec-1",
            score=0.77,
            payload={
                "chunk_id": "PREC:424370#판결요지#0",
                "doc_id": "PREC:424370",
                "text": "[대법원 2020다12345 판결요지] 손해배상 책임 인정.",
                "doc_type": "precedent",
                "title": "손해배상청구",
                "jurisdiction": "대법원",
                "law_kind": "민사",
                "article_no": "판결요지",
                "effective_from": "2020-05-14",
                "source_url": "https://law.go.kr/prec",
                "trust_grade": "A",
            },
        ),
        _FakeHit(
            id="uuid-ord-1",
            score=0.63,
            payload={
                "chunk_id": "ORD:전라남도:2200001#제2조#0",
                "doc_id": "ORD:전라남도:2200001",
                "text": "[전라남도 미래 조례 제2조] 2030년 시행 정의 규정.",
                "doc_type": "ordinance",
                "title": "전라남도 미래 조례",
                "jurisdiction": "전라남도",
                "law_kind": "조례",
                "article_no": "제2조",
                "effective_from": "2030-01-01",
                "source_url": "https://law.go.kr/ord",
                "trust_grade": "A",
                # No "license" key -> must default to config.DEFAULT_LICENSE.
            },
        ),
        _FakeHit(
            id="uuid-law-2",
            score=0.55,
            payload={
                "chunk_id": "LAW:000002:법률#제1조#0",
                "doc_id": "LAW:000002:법률",
                "text": "[연혁미상법 제1조] 시행일 메타 결측 조문.",
                "doc_type": "law",
                "title": "연혁미상법",
                "jurisdiction": "국가",
                "law_kind": "법률",
                "article_no": "제1조",
                # effective_from intentionally absent.
                "source_url": "https://law.go.kr/none",
                "trust_grade": "B",
            },
        ),
    ]


def _apply_flt(corpus: list[_FakeHit], flt: dict[str, Any] | None) -> list[_FakeHit]:
    """Filter the corpus by a flat ``{key: value|[values]}`` mapping (match-any)."""
    if not flt:
        return list(corpus)

    def _ok(hit: _FakeHit) -> bool:
        for key, value in flt.items():
            want = value if isinstance(value, (list, tuple, set)) else [value]
            if hit.payload.get(key) not in {str(v) for v in want}:
                return False
        return True

    return [h for h in corpus if _ok(h)]


@pytest.fixture
def stub_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``retriever.search`` with a deterministic, network-free stub.

    The stub mirrors the **production** retriever surface used by
    ``statutes_search``: a flat ``flt`` mapping (validated against the real
    ``build_filter`` so bad keys still raise ``ValueError``), ``k`` as a fetch
    cap, and a native ``as_of_date`` parameter applying the retriever's
    *safe current-law* policy — a row whose ``effective_from`` is missing or
    later than ``as_of_date`` is excluded. Because the stub exposes
    ``as_of_date``, ``statutes._retriever_supports_as_of()`` reports ``True`` and
    the date cut is exercised via delegation, exactly as in production.
    """
    corpus = _corpus()

    def _fake_search(
        query: str,
        k: int = config.DEFAULT_TOP_K,
        flt: dict[str, Any] | None = None,
        as_of_date: str | None = None,
    ) -> list[_FakeHit]:
        if not (query or "").strip():
            raise ValueError("query must be a non-empty string")
        # Reuse the real filter validator so unknown keys are rejected exactly
        # as the production retriever would reject them.
        retriever.build_filter(flt)
        rows = _apply_flt(corpus, flt)
        if as_of_date:
            cut = as_of_date[:10]
            rows = [
                h
                for h in rows
                if (ef := h.payload.get("effective_from")) and str(ef)[:10] <= cut
            ]
        return rows[: max(1, int(k))]

    monkeypatch.setattr(retriever, "search", _fake_search)
    # statutes imported the module object, so patching the attribute is enough.


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #
def test_row_shape_and_common_meta(stub_retriever: None) -> None:
    rows = statutes.statutes_search("성년 나이", k=4)
    assert rows, "expected results"
    top = rows[0]
    expected_keys = {
        "doc_id",
        "doc_type",
        "title",
        "article_no",
        "score",
        "text",
        "trust_grade",
        "source_url",
        "license",
        "as_of_date",
        "effective_from",
    }
    assert expected_keys <= set(top)
    assert top["doc_id"] == "LAW:000001:법률"
    assert top["doc_type"] == "law"
    assert top["article_no"] == "제4조"
    assert isinstance(top["score"], float)
    assert top["as_of_date"] is None  # not requested


def test_license_defaults_when_payload_missing_it(stub_retriever: None) -> None:
    rows = statutes.statutes_search("아무 질의", k=8)
    ord_row = next(r for r in rows if r["doc_type"] == "ordinance")
    assert ord_row["license"] == config.DEFAULT_LICENSE


def test_doc_id_falls_back_to_chunk_id() -> None:
    # Pure helper check: no doc_id/parent_id -> derive from chunk_id prefix.
    hit = _FakeHit(id="pt", score=0.1, payload={"chunk_id": "ADMRULE:58423#제3조#0"})
    assert statutes._doc_id_of(hit, hit.payload) == "ADMRULE:58423"


@pytest.mark.parametrize(
    ("effective_from", "as_of", "expected"),
    [
        ("2013-07-01", "2025-12-31", True),  # in force before
        ("2025-12-31", "2025-12-31", True),  # exactly on the date
        ("2030-01-01", "2025-12-31", False),  # not yet effective
        (None, "2025-12-31", False),  # unknown date -> excluded (safe policy)
        ("", "2025-12-31", False),  # blank date -> excluded
        ("2026-04-02T00:00:00", "2026-04-02", True),  # stray time component ok
    ],
)
def test_is_effective_on(effective_from: Any, as_of: str, expected: bool) -> None:
    assert statutes._is_effective_on(effective_from, as_of) is expected


def test_doc_type_filter(stub_retriever: None) -> None:
    rows = statutes.statutes_search("질의", k=8, filter={"doc_type": "law"})
    assert {r["doc_type"] for r in rows} == {"law"}


def test_as_of_date_excludes_future_and_dateless(stub_retriever: None) -> None:
    rows = statutes.statutes_search("질의", k=8, as_of_date="2025-12-31")
    ids = {r["doc_id"] for r in rows}
    # Safe current-law policy: both the 2030 ordinance and the date-less law are
    # excluded; the in-force 2013 law remains.
    assert "ORD:전라남도:2200001" not in ids
    assert "LAW:000002:법률" not in ids
    assert "LAW:000001:법률" in ids
    # as_of echoed onto every row.
    assert all(r["as_of_date"] == "2025-12-31" for r in rows)


def test_as_of_date_fallback_when_retriever_lacks_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the retriever has no ``as_of_date`` param, this module filters itself.

    Covers the forward/backward-compatibility branch: the stub here mimics an
    *older* retriever (no ``as_of_date``), so ``statutes_search`` must apply the
    point-in-time cut locally — with the same safe policy (future + date-less
    rows excluded).
    """
    corpus = _corpus()

    def _old_search(
        query: str,
        k: int = config.DEFAULT_TOP_K,
        flt: dict[str, Any] | None = None,
    ) -> list[_FakeHit]:
        retriever.build_filter(flt)
        return _apply_flt(corpus, flt)[: max(1, int(k))]

    monkeypatch.setattr(retriever, "search", _old_search)
    assert not statutes._retriever_supports_as_of()

    rows = statutes.statutes_search("질의", k=8, as_of_date="2025-12-31")
    ids = {r["doc_id"] for r in rows}
    assert "ORD:전라남도:2200001" not in ids  # future, excluded
    assert "LAW:000002:법률" not in ids  # date-less, excluded by safe policy
    assert "LAW:000001:법률" in ids  # in force
    assert all(r["as_of_date"] == "2025-12-31" for r in rows)


def test_as_of_date_includes_when_in_range(stub_retriever: None) -> None:
    rows = statutes.statutes_search("질의", k=8, as_of_date="2030-06-01")
    assert "ORD:전라남도:2200001" in {r["doc_id"] for r in rows}


def test_pagination_is_disjoint_and_complete(stub_retriever: None) -> None:
    page1 = statutes.statutes_search("질의", k=2, offset=0)
    page2 = statutes.statutes_search("질의", k=2, offset=2)
    ids1 = [r["doc_id"] for r in page1]
    ids2 = [r["doc_id"] for r in page2]
    assert len(ids1) == 2
    assert not (set(ids1) & set(ids2))
    assert len(set(ids1) | set(ids2)) == 4


def test_offset_past_end_returns_empty(stub_retriever: None) -> None:
    assert statutes.statutes_search("질의", k=4, offset=100) == []


def test_ordering_is_deterministic(stub_retriever: None) -> None:
    a = statutes.statutes_search("질의", k=4)
    b = statutes.statutes_search("질의", k=4)
    assert [r["doc_id"] for r in a] == [r["doc_id"] for r in b]
    # Descending score.
    scores = [r["score"] for r in a]
    assert scores == sorted(scores, reverse=True)


def test_page_size_is_capped(stub_retriever: None) -> None:
    # k above MAX_PAGE_SIZE must not raise and must not exceed the corpus.
    rows = statutes.statutes_search("질의", k=10_000)
    assert len(rows) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "   "},
        {"query": "q", "as_of_date": "2026/01/01"},
        {"query": "q", "as_of_date": "not-a-date"},
        {"query": "q", "offset": -1},
        {"query": "q", "filter": {"unknown_key": "x"}},
    ],
)
def test_input_validation(stub_retriever: None, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        statutes.statutes_search(**kwargs)


def test_empty_as_of_date_is_treated_as_none(stub_retriever: None) -> None:
    rows = statutes.statutes_search("질의", k=8, as_of_date="")
    # Empty string is normalized to None -> no point-in-time filtering applied.
    assert "ORD:전라남도:2200001" in {r["doc_id"] for r in rows}
    assert all(r["as_of_date"] is None for r in rows)
