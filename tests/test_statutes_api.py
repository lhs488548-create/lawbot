"""TestClient integration for ``POST /v1/statutes/search``.

Exercises the real FastAPI app (``api.main``) wired to the real
``search.statutes.statutes_search``, with only the **lowest** layer — the shared
retriever's ``search`` — stubbed so the test is offline (no OpenAI embedding, no
live Qdrant). This proves the contract end to end:

    HTTP request -> api.main.v1_statutes_search -> search.statutes.statutes_search
                 -> search.retriever.search (stubbed) -> common-meta rows

The ``/v1/statutes/search`` endpoint uses ``optional_key`` auth (anonymous reads
allowed, IP-rate-limited), so no API key is needed here.

Run from the project root::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_statutes_api.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import config
from search import retriever


@dataclass
class _FakeHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def _rows() -> list[_FakeHit]:
    return [
        _FakeHit(
            id="uuid-law-1",
            score=0.88,
            payload={
                "chunk_id": "LAW:014565:법률#제4조#0",
                "doc_id": "LAW:014565:법률",
                "parent_id": "LAW:014565:법률",
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
            score=0.71,
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
    ]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """A TestClient over the real app with only the retriever's search stubbed."""
    # Import lazily so the module-level wiring picks up our environment.
    from fastapi.testclient import TestClient

    from api import main as api_main

    def _fake_search(
        query: str,
        k: int = config.DEFAULT_TOP_K,
        flt: dict[str, Any] | None = None,
        as_of_date: str | None = None,
    ) -> list[_FakeHit]:
        if not (query or "").strip():
            raise ValueError("query must be a non-empty string")
        retriever.build_filter(flt)  # real validation of filter keys
        rows = _rows()
        if flt:
            rows = [
                h
                for h in rows
                if all(h.payload.get(key) == val for key, val in flt.items())
            ]
        if as_of_date:
            cut = as_of_date[:10]
            rows = [
                h
                for h in rows
                if (ef := h.payload.get("effective_from")) and str(ef)[:10] <= cut
            ]
        return rows[: max(1, int(k))]

    monkeypatch.setattr(retriever, "search", _fake_search)

    # Ensure the optional ``search.statutes`` backend is loaded (it is imported
    # dynamically at app construction; force-attach in case import order varied).
    if getattr(api_main.backends, "statutes", None) is None:
        import search.statutes as statutes_mod

        api_main.backends.statutes = statutes_mod

    # Disable slowapi rate limiting for these tests. We are verifying the
    # *statutes search* code path (HTTP -> statutes_search -> retriever), not the
    # rate-limit middleware; toggling ``limiter.enabled`` is slowapi's supported
    # way to bypass limit evaluation in tests and keeps this suite independent of
    # the (collaborator-owned) limiter wiring in ``api.keys``/``api.main``.
    try:
        from api import keys as keys_module

        monkeypatch.setattr(keys_module.limiter, "enabled", False)
        if getattr(api_main.app.state, "limiter", None) is not None:
            monkeypatch.setattr(api_main.app.state.limiter, "enabled", False)
    except Exception:  # pragma: no cover - defensive; limiter is expected to exist
        pass

    return TestClient(api_main.app)


def test_statutes_search_endpoint_returns_common_meta(client) -> None:
    resp = client.post(
        "/v1/statutes/search",
        json={"query": "성년 나이", "k": 5},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "results" in body
    assert body["as_of_date"] is None
    assert body["results"], "expected at least one result row"
    top = body["results"][0]
    for key in (
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
    ):
        assert key in top, f"missing meta key {key}"
    assert top["doc_id"] == "LAW:014565:법률"
    assert top["license"] == config.DEFAULT_LICENSE


def test_statutes_search_filter_and_as_of(client) -> None:
    resp = client.post(
        "/v1/statutes/search",
        json={"query": "질의", "k": 8, "filter": {"doc_type": "law"}, "as_of_date": "2026-04-02"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {r["doc_type"] for r in body["results"]} == {"law"}
    assert body["as_of_date"] == "2026-04-02"
    assert all(r["as_of_date"] == "2026-04-02" for r in body["results"])


def test_statutes_search_bad_as_of_date_is_422(client) -> None:
    resp = client.post(
        "/v1/statutes/search",
        json={"query": "질의", "as_of_date": "2026/04/02"},
    )
    # statutes_search raises ValueError -> endpoint maps to 422.
    assert resp.status_code == 422, resp.text


def test_statutes_search_blank_query_is_422(client) -> None:
    # Pydantic min_length=1 rejects empties before reaching the backend.
    resp = client.post("/v1/statutes/search", json={"query": ""})
    assert resp.status_code == 422
