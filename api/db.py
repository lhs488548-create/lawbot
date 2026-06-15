"""Persistence layer for the multi-tenant API-key store.

This module owns the SQLModel/SQLite schema and the low-level session plumbing
used by :mod:`api.auth` (issue/verify/list/revoke) and :mod:`api.keys` (the
``/v1/keys`` management router). Everything above this layer speaks in terms of
the public functions in :mod:`api.auth`; only this file knows about SQL.

Design notes (production-minded, MVP storage):

* **Never store plaintext keys.** The primary key is ``sha256(raw_key)``; the
  raw ``lk_...`` token is shown to the caller exactly once at issue time and is
  not recoverable afterwards.
* The store is SQLite (``config.API_KEYS_DB``) for the MVP. Because the rest of
  the codebase only depends on the :mod:`api.auth` function surface, swapping in
  Postgres later is a matter of changing :data:`_ENGINE` (e.g. a Postgres URL)
  without touching callers.
* SQLite is opened with ``check_same_thread=False`` so it can be shared across
  FastAPI worker threads; usage-counter writes are serialised per-statement by
  SQLite's own locking, which is sufficient for the MVP metering granularity.

Owner: multitenant builder. This is a new module; it does not modify any
Contracts-owned file.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Final, Optional

from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

import config

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Recognised tenant tiers. ``admin`` gates the ``/v1/keys`` endpoints.
VALID_TIERS: Final[frozenset[str]] = frozenset({"free", "pro", "enterprise", "admin"})

#: Default slowapi rate spec per tier (slowapi "<count>/<period>" syntax).
DEFAULT_RATE_BY_TIER: Final[dict[str, str]] = {
    "free": "30/minute",
    "pro": "120/minute",
    "enterprise": "600/minute",
    "admin": "240/minute",
}

#: Prefix for every raw key. The token body is ``secrets.token_urlsafe(24)``.
KEY_PREFIX: Final[str] = "lk_"


def utcnow_iso() -> str:
    """Return the current UTC time as a stable ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest used as the storage key.

    Args:
        raw_key: The plaintext ``lk_...`` API key as presented by a client.

    Returns:
        Lower-case hex SHA-256 digest. This is what is persisted; the plaintext
        is never stored.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #


class ApiKey(SQLModel, table=True):
    """A single issued API key (one row per key).

    The plaintext key is never stored; ``key_hash`` is the SHA-256 of the raw
    token and serves as the primary key. ``usage`` is a cumulative request
    counter incremented on every successful authentication; ``tokens`` meters
    cumulative LLM token consumption attributed to the key.
    """

    __tablename__ = "api_keys"

    key_hash: str = Field(primary_key=True, description="sha256(raw_key)")
    tenant: str = Field(index=True, description="tenant / customer id")
    tier: str = Field(default="free", description="free|pro|enterprise|admin")
    rate: str = Field(default="30/minute", description="slowapi rate spec")
    usage: int = Field(default=0, description="cumulative request count")
    tokens: int = Field(default=0, description="cumulative metered LLM tokens")
    revoked: int = Field(default=0, description="0=active, 1=revoked")
    created_at: str = Field(default_factory=utcnow_iso, description="ISO timestamp")
    #: Stable, non-secret identifier for management/DELETE (first 12 hex chars of
    #: the hash). Stored explicitly so it can be indexed and matched by prefix.
    key_id: str = Field(index=True, description="public id = key_hash[:12]")


# --------------------------------------------------------------------------- #
# Engine / session                                                            #
# --------------------------------------------------------------------------- #


def _build_engine() -> Engine:
    """Create the SQLite engine for the configured key store path."""
    url = f"sqlite:///{config.API_KEYS_DB}"
    # check_same_thread=False: the engine is shared across FastAPI threads.
    return create_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False},
    )


_ENGINE: Final[Engine] = _build_engine()


def get_engine() -> Engine:
    """Return the process-wide SQLModel engine (handy for tests/overrides)."""
    return _ENGINE


def init_db() -> None:
    """Create the ``api_keys`` table if it does not yet exist.

    Idempotent: safe to call at every process start. Does not drop or migrate
    existing data.
    """
    SQLModel.metadata.create_all(_ENGINE)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a transactional :class:`~sqlmodel.Session`.

    Commits on success and rolls back on exception, then always closes. Use for
    any write; reads may also use it for a consistent snapshot.
    """
    session = Session(_ENGINE)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_by_id(session: Session, key_id: str) -> Optional[ApiKey]:
    """Look up an active-or-revoked key by its public ``key_id`` prefix.

    Args:
        session: An open SQLModel session.
        key_id: The public id (``key_hash[:12]``) or a longer unique prefix.

    Returns:
        The matching :class:`ApiKey` row, or ``None`` if no/ambiguous match.
    """
    from sqlmodel import select

    # Exact match on the stored 12-char id first (the common case).
    row = session.get(ApiKey, _hash_for_id_lookup(session, key_id))
    if row is not None:
        return row
    # Fall back to prefix match on key_id (lets callers pass a longer prefix).
    rows = session.exec(
        select(ApiKey).where(ApiKey.key_id.startswith(key_id))  # type: ignore[attr-defined]
    ).all()
    return rows[0] if len(rows) == 1 else None


def _hash_for_id_lookup(session: Session, key_id: str) -> str:
    """Resolve a public ``key_id`` to its primary-key hash, if uniquely known.

    Returns a sentinel that never matches when the id cannot be resolved, so the
    caller's ``session.get`` simply misses and the prefix fallback runs.
    """
    from sqlmodel import select

    row = session.exec(
        select(ApiKey).where(ApiKey.key_id == key_id)
    ).first()
    return row.key_hash if row is not None else "\x00no-such-key"
