"""Unit + light-integration tests for the multi-tenant API-key surface.

Covers the contract-mandated lifecycle (BUILD_CONTRACT §f / the multitenant
task brief): **issue → authenticate → rate-limit (429) → revoke**, plus the
security invariants (no plaintext stored, hash-only persistence, tier gating,
metering).

Isolation: the key store is SQLite bound at import time to
``config.API_KEYS_DB``. To avoid touching the real store and to give each test
session a clean slate, we repoint ``config.API_KEYS_DB`` at a temp file and
rebuild the engine **before** importing ``api.db`` / ``api.auth`` / ``api.keys``.

No network and no OpenAI calls are made here (cost rule: 0 OpenAI calls).

Run::

    cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests/test_auth_keys.py
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine


# --------------------------------------------------------------------------- #
# Isolated key-store fixture (rebind the engine to a per-test temp SQLite DB)  #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def authmods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Yield (db, auth, keys) bound to a fresh temp SQLite store.

    The engine in :mod:`api.db` is created at import time from
    ``config.API_KEYS_DB``. Reloading the module would re-register the
    ``ApiKey`` table on the shared SQLModel metadata (an error), so instead we
    point ``config.API_KEYS_DB`` at a per-test temp file and **rebind the
    module-level engine in place**. ``auth``/``keys`` reference ``db._ENGINE``
    (via ``session_scope``) at call time, so the rebind takes effect without a
    reload. Each test thus gets an empty, isolated key store.
    """
    import config

    db = importlib.import_module("api.db")
    auth = importlib.import_module("api.auth")
    keys = importlib.import_module("api.keys")

    db_path = tmp_path / "test_keys.db"
    monkeypatch.setattr(config, "API_KEYS_DB", db_path, raising=False)

    fresh_engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(db, "_ENGINE", fresh_engine, raising=True)

    db.init_db()
    assert db_path.exists(), "schema should have been created in the temp store"
    yield db, auth, keys


# --------------------------------------------------------------------------- #
# 1. issue → verify (authenticate)                                             #
# --------------------------------------------------------------------------- #
def test_issue_returns_plaintext_once_and_hashes_storage(authmods):
    db, auth, _keys = authmods
    raw = auth.issue_key(tenant="acme", tier="pro")

    assert raw.startswith("lk_"), "raw key must carry the lk_ prefix"
    # The plaintext must never be persisted: only its sha256 hash is stored.
    with db.session_scope() as s:
        from sqlmodel import select

        rows = s.exec(select(db.ApiKey)).all()
        assert len(rows) == 1
        stored = rows[0]
        assert stored.key_hash == db.hash_key(raw)
        assert raw not in stored.key_hash
        assert stored.tenant == "acme"
        assert stored.tier == "pro"
        # pro tier default rate applied.
        assert stored.rate == db.DEFAULT_RATE_BY_TIER["pro"]


def test_verify_authenticates_and_meters_usage(authmods):
    _db, auth, _keys = authmods
    raw = auth.issue_key(tenant="t1", tier="free")

    p1 = auth.verify(raw)
    assert p1 is not None
    assert p1["tenant"] == "t1"
    assert p1["tier"] == "free"
    assert p1["usage"] == 1  # usage incremented on first auth

    p2 = auth.verify(raw)
    assert p2 is not None and p2["usage"] == 2  # metering accumulates


def test_verify_rejects_unknown_and_malformed_keys(authmods):
    _db, auth, _keys = authmods
    assert auth.verify("lk_does-not-exist") is None
    assert auth.verify("not-a-key") is None  # missing lk_ prefix
    assert auth.verify("") is None
    assert auth.verify(None) is None  # type: ignore[arg-type]


def test_record_tokens_meters_llm_usage(authmods):
    db, auth, _keys = authmods
    raw = auth.issue_key(tenant="t1", tier="pro")
    principal = auth.verify(raw)
    auth.record_tokens(principal["key_id"], 1500)
    auth.record_tokens(principal["key_id"], 500)

    info = next(k for k in auth.list_keys() if k["key_id"] == principal["key_id"])
    assert info["tokens"] == 2000
    # Unknown key id is a silent no-op (never breaks a successful answer).
    auth.record_tokens("ffffffffffff", 999)  # no exception


def test_issue_validates_tenant_and_tier(authmods):
    _db, auth, _keys = authmods
    with pytest.raises(ValueError):
        auth.issue_key(tenant="", tier="free")
    with pytest.raises(ValueError):
        auth.issue_key(tenant="t1", tier="superuser")


# --------------------------------------------------------------------------- #
# 2. list (no plaintext) + revoke                                             #
# --------------------------------------------------------------------------- #
def test_list_keys_never_exposes_secrets(authmods):
    _db, auth, _keys = authmods
    raw = auth.issue_key(tenant="t1", tier="enterprise")
    listing = auth.list_keys()
    assert len(listing) == 1
    item = listing[0]
    # No plaintext, no hash leaked through the listing surface.
    assert "key" not in item
    assert "key_hash" not in item
    assert raw not in str(item)
    assert set(item) >= {
        "key_id", "tenant", "tier", "rate", "usage", "tokens",
        "revoked", "created_at",
    }


def test_revoke_blocks_subsequent_auth(authmods):
    _db, auth, _keys = authmods
    raw = auth.issue_key(tenant="t1", tier="free")
    principal = auth.verify(raw)
    assert principal is not None

    assert auth.revoke(principal["key_id"]) is True
    # A revoked key no longer authenticates.
    assert auth.verify(raw) is None
    # Revocation is idempotent and reflected in the listing.
    assert auth.revoke(principal["key_id"]) is True
    assert auth.list_keys()[0]["revoked"] is True
    # Unknown id -> False.
    assert auth.revoke("zzzzzzzzzzzz") is False
    assert auth.revoke("") is False


# --------------------------------------------------------------------------- #
# 3. /v1/keys router: admin gate (401/403) + HTTP issue/list/revoke           #
# --------------------------------------------------------------------------- #
def _app_with_keys_router(keys) -> FastAPI:
    app = FastAPI()
    keys.setup_rate_limiting(app)
    app.include_router(keys.router)
    return app


def test_keys_router_requires_admin(authmods):
    _db, auth, keys = authmods
    client = TestClient(_app_with_keys_router(keys))

    # No key -> 401.
    r = client.post("/v1/keys", json={"tenant": "x"})
    assert r.status_code == 401

    # Non-admin key -> 403.
    free_raw = auth.issue_key(tenant="t1", tier="free")
    r = client.post(
        "/v1/keys",
        json={"tenant": "x"},
        headers={"Authorization": f"Bearer {free_raw}"},
    )
    assert r.status_code == 403


def test_keys_router_full_admin_lifecycle(authmods):
    _db, auth, keys = authmods
    admin_raw = auth.issue_key(tenant="root", tier="admin")
    admin_hdr = {"Authorization": f"Bearer {admin_raw}"}
    client = TestClient(_app_with_keys_router(keys))

    # Issue a tenant key via the admin endpoint; plaintext returned once.
    r = client.post(
        "/v1/keys", json={"tenant": "tenantA", "tier": "pro"}, headers=admin_hdr
    )
    assert r.status_code == 201
    body = r.json()
    new_raw = body["key"]
    new_id = body["key_id"]
    assert new_raw.startswith("lk_")
    assert body["tenant"] == "tenantA" and body["tier"] == "pro"

    # The issued tenant key authenticates against the backend.
    assert auth.verify(new_raw) is not None

    # List shows it (admin + tenantA), never plaintext.
    r = client.get("/v1/keys", headers=admin_hdr)
    assert r.status_code == 200
    listing = r.json()
    assert any(k["key_id"] == new_id for k in listing)
    assert new_raw not in r.text

    # Revoke it.
    r = client.delete(f"/v1/keys/{new_id}", headers=admin_hdr)
    assert r.status_code == 200 and r.json()["revoked"] is True
    assert auth.verify(new_raw) is None

    # Revoking an unknown id -> 404.
    r = client.delete("/v1/keys/zzzzzzzzzzzz", headers=admin_hdr)
    assert r.status_code == 404

    # Invalid tier -> 422.
    r = client.post(
        "/v1/keys", json={"tenant": "x", "tier": "root"}, headers=admin_hdr
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# 4. Rate limiting -> 429 (the per-key limiter from api.keys)                  #
# --------------------------------------------------------------------------- #
def test_rate_limit_returns_429_with_retry_after(authmods):
    """A protected route limited to 2/minute returns 429 on the 3rd call.

    Exercises ``api.keys.limiter`` + ``per_key_rate`` + ``setup_rate_limiting``
    end-to-end through a FastAPI app, bucketed per authenticated key.
    """
    _db, auth, keys = authmods

    app = FastAPI()
    keys.setup_rate_limiting(app)

    @app.get("/limited")
    @keys.limiter.limit("2/minute")
    def limited(request: Request, principal: dict = Depends(auth.require_key)):
        return {"ok": True, "tenant": principal["tenant"]}

    raw = auth.issue_key(tenant="rl-tenant", tier="free")
    hdr = {"Authorization": f"Bearer {raw}"}
    client = TestClient(app)

    assert client.get("/limited", headers=hdr).status_code == 200
    assert client.get("/limited", headers=hdr).status_code == 200
    blocked = client.get("/limited", headers=hdr)
    # The 3rd call within the window is rate-limited (the contract-critical
    # 발급→인증→429→폐기 path).
    assert blocked.status_code == 429
    # 429 carries a Retry-After header (contract §e), because the limiter is
    # constructed with headers_enabled + retry_after="delta-seconds".
    assert "retry-after" in {h.lower() for h in blocked.headers}
    # A second tenant has its own independent bucket (per-key isolation): its
    # first call is not blocked by the first tenant's exhausted budget.
    raw2 = auth.issue_key(tenant="rl-tenant-2", tier="free")
    other = client.get(
        "/limited", headers={"Authorization": f"Bearer {raw2}"}
    )
    assert other.status_code == 200


def test_require_admin_dependency_rejects_non_admin(authmods):
    """The FastAPI ``require_admin`` dependency: 401 anon, 403 non-admin, 200 admin."""
    _db, auth, keys = authmods

    app = FastAPI()

    @app.get("/admin-only")
    def admin_only(principal: dict = Depends(auth.require_admin)):
        return {"tenant": principal["tenant"]}

    client = TestClient(app)
    assert client.get("/admin-only").status_code == 401

    free_raw = auth.issue_key(tenant="t1", tier="free")
    r = client.get(
        "/admin-only", headers={"Authorization": f"Bearer {free_raw}"}
    )
    assert r.status_code == 403

    admin_raw = auth.issue_key(tenant="root", tier="admin")
    r = client.get(
        "/admin-only", headers={"Authorization": f"Bearer {admin_raw}"}
    )
    assert r.status_code == 200 and r.json()["tenant"] == "root"
