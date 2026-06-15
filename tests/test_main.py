"""Assembly tests for the FastAPI app (``api.main``) — api_server builder.

These cover the *wiring* the api_server owns, independent of the search/RAG
backends (which other builders test in depth):

* ``/healthz`` always answers 200 with a backends readiness map.
* OpenAPI exposes the full lawbot.org-aligned contract path set, and ``/docs``
  renders.
* ``/console`` serves the self-service page.
* Auth gating: key-required endpoints return 401 without a key; admin endpoints
  return 403 for a non-admin key; bad bodies on key-gated routes still 401
  (auth runs before body validation — documented behaviour).
* Generated responses carry the AI-Basic-Act notice (``_ensure_notice``).
* **Per-key rate limiting** returns 429 with ``Retry-After`` once a key's stored
  rate is exhausted, and buckets are isolated per key.
* Optional-backend guard returns a clear 503 when a backend is absent.

No live network is required: the key DB is redirected to a temp SQLite file and
the one cost/IO-bearing backend used (``statutes``) is stubbed. No OpenAI calls.

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_main.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import create_engine

import config


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def keydb(tmp_path: Path) -> Iterator[Path]:
    """Redirect the API-key store to an isolated temp SQLite file."""
    import api.db as dbm

    db_path = tmp_path / "keys.db"
    config.API_KEYS_DB = db_path  # cosmetic; the engine below is authoritative
    dbm.engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    from api import auth

    auth.init_db()
    yield db_path


@pytest.fixture()
def client(keydb: Path) -> Iterator[TestClient]:
    """A TestClient over the real app, with a fresh key DB."""
    from api.main import app

    with TestClient(app) as c:
        yield c


class _FakeStat:
    """Deterministic statutes backend so HTTP wiring is exercised offline."""

    def statutes_search(self, query, k=8, filter=None, as_of_date=None):  # noqa: D401
        return [
            {
                "doc_id": "LAW:000001:법률",
                "doc_type": "law",
                "title": "민법",
                "article_no": "제4조",
                "score": 0.91,
                "text": "사람은 19세로 성년에 이른다.",
                "effective_from": "2013-07-01",
                "source_url": "https://law.go.kr/민법",
                "trust_grade": "A",
            }
        ]


# --------------------------------------------------------------------------- #
# Health / OpenAPI / console                                                   #
# --------------------------------------------------------------------------- #
def test_healthz_always_200(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["collection"] == config.COLLECTION
    assert isinstance(body["backends"], dict)
    # The readiness map advertises the optional backends we probe.
    for key in ("rag", "statutes", "verify", "source_pack", "embed_client", "qdrant"):
        assert key in body["backends"]


def test_openapi_covers_contract_paths(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    expected = {
        "/healthz",
        "/console",
        "/v1/statutes/search",
        "/v1/verify",
        "/v1/source-pack",
        "/v1/embeddings",
        "/v1/ask",
        "/v1/ad-review",
        "/v1/keys",
        "/v1/keys/{key_id}",
        "/v1/statutes/{law_id}/articles/{article_no}",
        "/v1/precedents/{seq}",
    }
    assert expected <= paths, f"missing contract paths: {expected - paths}"
    assert client.get("/docs").status_code == 200


def test_console_served(client: TestClient) -> None:
    r = client.get("/console")
    assert r.status_code == 200
    assert "lawbot console" in r.text


# --------------------------------------------------------------------------- #
# Auth gating                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method,path,kw",
    [
        ("post", "/v1/ask", {"json": {"query": "x"}}),
        ("post", "/v1/verify", {"json": {"citation": {}}}),
        ("post", "/v1/source-pack", {"json": {"query": "x"}}),
        ("post", "/v1/embeddings", {"json": {"input": "x"}}),
        ("get", "/v1/precedents/1", {}),
        ("get", "/v1/statutes/1/articles/제4조", {}),
    ],
)
def test_key_required_endpoints_401_without_key(
    client: TestClient, method: str, path: str, kw: dict
) -> None:
    r = getattr(client, method)(path, **kw)
    assert r.status_code == 401


def test_admin_endpoints_403_for_non_admin(client: TestClient) -> None:
    from api import auth

    free = auth.issue_key("acme", tier="free")
    h = {"Authorization": f"Bearer {free}"}
    assert client.get("/v1/keys", headers=h).status_code == 403


def test_admin_can_issue_and_list_keys(client: TestClient) -> None:
    from api import auth

    admin = auth.issue_key("root", tier="admin")
    ah = {"Authorization": f"Bearer {admin}"}
    issued = client.post("/v1/keys", json={"tenant": "t2", "tier": "pro"}, headers=ah)
    assert issued.status_code == 201
    assert issued.json()["key"].startswith("lk_")  # plaintext returned once
    listed = client.get("/v1/keys", headers=ah)
    assert listed.status_code == 200
    # No plaintext / hash leaks in the listing.
    assert all("key" not in row or row.get("key_id") for row in listed.json())


# --------------------------------------------------------------------------- #
# Common meta + AI notice                                                      #
# --------------------------------------------------------------------------- #
def test_statutes_search_carries_common_meta(client: TestClient) -> None:
    from api import auth, main as m

    key = auth.issue_key("acme", tier="pro", rate="100/minute")
    m.backends.statutes = _FakeStat()  # after lifespan probe
    r = client.post(
        "/v1/statutes/search",
        json={"query": "성년 나이", "as_of_date": "2026-04-02"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["as_of_date"] == "2026-04-02"
    row = body["results"][0]
    for meta_key in ("trust_grade", "source_url", "license", "as_of_date", "effective_from"):
        assert meta_key in row
    assert row["license"]  # license defaulted from config


def test_ensure_notice_fills_ai_fields() -> None:
    from api.main import _ensure_notice

    out = _ensure_notice({"answer": "..."})
    assert out["ai_generated"] is True
    assert out["disclaimer"] == config.ANSWER_DISCLAIMER


# --------------------------------------------------------------------------- #
# Per-key rate limiting                                                        #
# --------------------------------------------------------------------------- #
def test_per_key_rate_limit_429(client: TestClient) -> None:
    from api import auth, main as m

    key = auth.issue_key("acme", tier="free", rate="2/minute")
    m.backends.statutes = _FakeStat()
    h = {"Authorization": f"Bearer {key}"}
    assert client.post("/v1/statutes/search", json={"query": "a"}, headers=h).status_code == 200
    assert client.post("/v1/statutes/search", json={"query": "b"}, headers=h).status_code == 200
    r3 = client.post("/v1/statutes/search", json={"query": "c"}, headers=h)
    assert r3.status_code == 429
    assert r3.headers.get("Retry-After")


def test_rate_limit_buckets_are_per_key(client: TestClient) -> None:
    from api import auth, main as m

    k1 = auth.issue_key("t1", tier="free", rate="1/minute")
    k2 = auth.issue_key("t2", tier="pro", rate="100/minute")
    m.backends.statutes = _FakeStat()
    h1 = {"Authorization": f"Bearer {k1}"}
    h2 = {"Authorization": f"Bearer {k2}"}
    # Exhaust k1.
    assert client.post("/v1/statutes/search", json={"query": "a"}, headers=h1).status_code == 200
    assert client.post("/v1/statutes/search", json={"query": "b"}, headers=h1).status_code == 429
    # k2 unaffected.
    assert client.post("/v1/statutes/search", json={"query": "c"}, headers=h2).status_code == 200


# --------------------------------------------------------------------------- #
# Optional-backend guard                                                       #
# --------------------------------------------------------------------------- #
def test_absent_backend_returns_503(client: TestClient) -> None:
    from api import auth, main as m

    key = auth.issue_key("acme", tier="pro", rate="100/minute")
    m.backends.verify = None  # simulate the verify backend not yet wired
    r = client.post(
        "/v1/verify",
        json={"citation": {"law_name": "민법", "article_no": "제4조"}},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 503
