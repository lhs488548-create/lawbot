"""Multi-tenant API-key issuance, verification, and FastAPI auth dependencies.

This module is the public surface the rest of the service depends on. It
implements the Contracts interface (BUILD_CONTRACT §f)::

    init_db() -> None
    issue_key(tenant, tier="free", rate=None) -> str   # raw key, shown once
    verify(raw_key) -> dict | None                      # increments usage
    list_keys() -> list[dict]                           # never returns plaintext
    revoke(key_id) -> bool

and adds FastAPI dependencies used by the routers:

    require_key   -> any active key (per-key metered)
    require_admin -> an active ``admin``-tier key (gates /v1/keys)

Security properties:

* Plaintext keys are returned exactly once from :func:`issue_key`; only the
  SHA-256 hash is persisted (see :mod:`api.db`).
* Verification is revocation-aware and atomically increments the per-key usage
  counter so metering cannot be lost between read and write.
* No function ever logs, prints, or returns a stored plaintext key.

Owner: multitenant builder (new module).
"""

from __future__ import annotations

import secrets
from typing import Final, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import select

from api import db
from api.db import (
    DEFAULT_RATE_BY_TIER,
    KEY_PREFIX,
    VALID_TIERS,
    ApiKey,
    hash_key,
)

# Number of URL-safe bytes in the random token body of each key.
_TOKEN_BYTES: Final[int] = 24

# FastAPI security scheme: Authorization: Bearer <key>. auto_error=False so we
# can return our own 401 shape and also support the admin gate cleanly.
_bearer = HTTPBearer(auto_error=False, description="API key as 'Bearer lk_...'")


# --------------------------------------------------------------------------- #
# Contracts interface (§f)                                                     #
# --------------------------------------------------------------------------- #


def init_db() -> None:
    """Initialise the key store (create tables if needed). Idempotent."""
    db.init_db()


def _resolve_rate(tier: str, rate: Optional[str]) -> str:
    """Return the effective rate spec for ``tier``, honouring an explicit override."""
    if rate:
        return rate
    return DEFAULT_RATE_BY_TIER.get(tier, DEFAULT_RATE_BY_TIER["free"])


def issue_key(tenant: str, tier: str = "free", rate: Optional[str] = None) -> str:
    """Issue a new API key for ``tenant`` and return the raw key **once**.

    Args:
        tenant: Tenant / customer id (multi-tenant isolation key). Required,
            non-empty.
        tier: One of ``free|pro|enterprise|admin``. Defaults to ``free``.
        rate: Optional slowapi rate spec (``"<count>/<period>"``). When omitted
            a tier default is applied.

    Returns:
        The plaintext key ``lk_<token>``. This is the only time it is available;
        only its hash is stored.

    Raises:
        ValueError: If ``tenant`` is blank or ``tier`` is not recognised.
    """
    tenant = (tenant or "").strip()
    if not tenant:
        raise ValueError("tenant must be a non-empty string")
    if tier not in VALID_TIERS:
        raise ValueError(
            f"tier must be one of {sorted(VALID_TIERS)}, got {tier!r}"
        )

    raw_key = f"{KEY_PREFIX}{secrets.token_urlsafe(_TOKEN_BYTES)}"
    digest = hash_key(raw_key)
    row = ApiKey(
        key_hash=digest,
        key_id=digest[:12],
        tenant=tenant,
        tier=tier,
        rate=_resolve_rate(tier, rate),
    )
    with db.session_scope() as session:
        session.add(row)
    return raw_key


def verify(raw_key: str) -> Optional[dict]:
    """Verify a presented key, increment its usage, and return its principal.

    Args:
        raw_key: The plaintext ``lk_...`` key from the ``Authorization`` header.

    Returns:
        ``{"tenant", "tier", "rate", "usage", "key_id"}`` for an active key, or
        ``None`` if the key is unknown or revoked. The returned ``usage`` value
        reflects the count *after* this request has been recorded.

    Notes:
        The usage increment and the read happen inside one transaction so the
        per-key meter never loses an increment under concurrency.
    """
    if not raw_key or not raw_key.startswith(KEY_PREFIX):
        return None
    digest = hash_key(raw_key)
    with db.session_scope() as session:
        row = session.get(ApiKey, digest)
        if row is None or row.revoked:
            return None
        row.usage += 1
        session.add(row)
        # Snapshot fields before the session closes / row expires.
        principal = {
            "tenant": row.tenant,
            "tier": row.tier,
            "rate": row.rate,
            "usage": row.usage,
            "key_id": row.key_id,
        }
    return principal


def record_tokens(key_id: str, tokens: int) -> None:
    """Add ``tokens`` to a key's cumulative LLM-token meter (best effort).

    Used by cost-bearing endpoints (``/v1/ask``, ``/v1/ad-review``) to attribute
    token usage to the calling tenant. Silently no-ops for unknown keys so a
    metering failure never breaks a successful answer.

    Args:
        key_id: Public key id (``key_hash[:12]``) from :func:`verify`.
        tokens: Non-negative token count to add.
    """
    if tokens <= 0:
        return
    with db.session_scope() as session:
        row = db.get_by_id(session, key_id)
        if row is not None:
            row.tokens += int(tokens)
            session.add(row)


def list_keys() -> list[dict]:
    """Return metadata for all keys. Never includes plaintext or the hash.

    Returns:
        A list of dicts with ``key_id, tenant, tier, rate, usage, tokens,
        revoked, created_at`` ordered by creation time (newest first).
    """
    with db.session_scope() as session:
        rows = session.exec(
            select(ApiKey).order_by(ApiKey.created_at.desc())  # type: ignore[attr-defined]
        ).all()
        return [
            {
                "key_id": r.key_id,
                "tenant": r.tenant,
                "tier": r.tier,
                "rate": r.rate,
                "usage": r.usage,
                "tokens": r.tokens,
                "revoked": bool(r.revoked),
                "created_at": r.created_at,
            }
            for r in rows
        ]


def revoke(key_id: str) -> bool:
    """Revoke a key by its public id or unique id-prefix.

    Args:
        key_id: The ``key_id`` returned by :func:`list_keys` (or a unique
            prefix of it). Revocation is idempotent.

    Returns:
        ``True`` if a matching key was found (and is now revoked), ``False`` if
        no unique match exists.
    """
    key_id = (key_id or "").strip()
    if not key_id:
        return False
    with db.session_scope() as session:
        row = db.get_by_id(session, key_id)
        if row is None:
            return False
        row.revoked = 1
        session.add(row)
    return True


def bootstrap_admin(tenant: str = "root") -> Optional[str]:
    """Create the first ``admin`` key if no admin key exists yet.

    Self-service tenants cannot mint their own admin keys (that would defeat the
    gate), so the very first admin key is bootstrapped out-of-band — typically
    once, at deploy time, from a trusted shell::

        python -m api.auth   # prints the raw admin key ONCE

    Returns:
        The raw admin key if one was just created, else ``None`` (an admin key
        already existed; nothing is printed/returned).
    """
    init_db()
    with db.session_scope() as session:
        existing = session.exec(
            select(ApiKey).where(ApiKey.tier == "admin", ApiKey.revoked == 0)
        ).first()
        if existing is not None:
            return None
    return issue_key(tenant=tenant, tier="admin")


# --------------------------------------------------------------------------- #
# FastAPI dependencies                                                         #
# --------------------------------------------------------------------------- #


def _extract_principal(
    credentials: Optional[HTTPAuthorizationCredentials],
) -> dict:
    """Validate bearer credentials and return the principal, or raise 401."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required (Authorization: Bearer lk_...)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    principal = verify(credentials.credentials)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """FastAPI dependency: require any active API key.

    On success, stashes the principal on ``request.state.principal`` so the
    rate limiter (keyed by tenant+tier) and metering can read it, and returns
    the principal dict.
    """
    principal = _extract_principal(credentials)
    request.state.principal = principal
    return principal


def require_admin(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """FastAPI dependency: require an active ``admin``-tier key.

    Gates the ``/v1/keys`` management endpoints. Returns 401 for missing/invalid
    keys and 403 for valid-but-non-admin keys.
    """
    principal = _extract_principal(credentials)
    if principal["tier"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin tier required for key management",
        )
    request.state.principal = principal
    return principal


if __name__ == "__main__":  # pragma: no cover - operational bootstrap helper
    # One-shot admin bootstrap. Prints the raw admin key exactly once. This is
    # the only sanctioned place a raw key is printed, and only at operator
    # request from a trusted shell.
    _raw = bootstrap_admin()
    if _raw is None:
        print("An active admin key already exists; no new key issued.")
    else:
        print("ADMIN KEY (store securely — shown once):")
        print(_raw)
