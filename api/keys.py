"""Self-service API-key management router + per-key rate limiting.

Two responsibilities live here:

1. **Rate limiting** (:data:`limiter`, :func:`rate_limit_key`,
   :func:`per_key_rate`): a process-wide slowapi
   :class:`~slowapi.Limiter` whose limit *and* bucket key are both derived from
   the authenticated principal, so each tenant/tier gets its own per-key budget
   (with an IP fallback for anonymous routes). ``api.main`` mounts the limiter
   and the 429 handler.

2. **The ``/v1/keys`` router**: the external self-service / admin flow to issue
   (POST), list (GET), and revoke (DELETE) keys. All three are gated by an
   ``admin``-tier key via :func:`api.auth.require_admin`. The plaintext key is
   returned exactly once, on POST.

Owner: multitenant builder (new module). Mounted by ``api.main`` (not modified
here — ``main`` imports ``router``, ``limiter``, and ``setup_rate_limiting``).
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from api import auth
from api.db import VALID_TIERS

# --------------------------------------------------------------------------- #
# Rate limiting                                                               #
# --------------------------------------------------------------------------- #


def rate_limit_key(request: Request) -> str:
    """Return the rate-limit bucket key for ``request``.

    Authenticated requests are bucketed per **key** (``tenant:tier:key_id``) so
    one tenant's traffic never consumes another's budget and each issued key has
    its own counter. Anonymous requests fall back to the client IP.
    """
    principal = getattr(request.state, "principal", None)
    if principal:
        return f"{principal['tenant']}:{principal['tier']}:{principal['key_id']}"
    return get_remote_address(request)


def per_key_rate(request: Request) -> str:
    """Return the slowapi limit string for the authenticated principal.

    slowapi evaluates this callable per request, so the limit honours the
    ``rate`` stored on the specific key (e.g. ``"30/minute"`` for free,
    ``"600/minute"`` for enterprise). Anonymous requests get a conservative
    default IP limit.
    """
    principal = getattr(request.state, "principal", None)
    if principal and principal.get("rate"):
        return principal["rate"]
    return "20/minute"


#: Process-wide limiter. The default key function buckets per authenticated key
#: (IP fallback). ``Retry-After`` on 429 is added by
#: :func:`rate_limit_exceeded_handler` (not by ``headers_enabled``, which would
#: force every limited view to declare a ``response`` parameter). ``api.main``
#: registers ``limiter`` on the app state and wires the handler via
#: :func:`setup_rate_limiting`.
limiter: Limiter = Limiter(key_func=rate_limit_key)


def rate_limit_exceeded_handler(request: Request, exc):
    """429 handler that always sets ``Retry-After`` (contract §e).

    slowapi's built-in handler only emits ``Retry-After``/``X-RateLimit-*`` when
    ``headers_enabled`` is set on the limiter, which in turn requires every
    limited view to declare a ``response: Response`` parameter. To keep views
    simple while still honouring the contract's ``Retry-After`` requirement on
    rate-limited responses, we set the header here explicitly from the window of
    the exceeded limit.

    Args:
        request: The incoming request (unused but required by the handler API).
        exc: The :class:`slowapi.errors.RateLimitExceeded` instance.

    Returns:
        A ``429`` JSON response carrying a ``Retry-After`` (delta-seconds) header.
    """
    from starlette.responses import JSONResponse

    # The exceeded limit's window in seconds (e.g. 60 for "2/minute").
    try:
        retry_after = int(exc.limit.limit.get_expiry())
    except Exception:  # pragma: no cover - defensive fallback
        retry_after = 60
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.limit.limit}"},
        headers={"Retry-After": str(retry_after)},
    )


def setup_rate_limiting(app) -> None:
    """Attach the limiter and a 429 handler (with ``Retry-After``) to ``app``.

    Args:
        app: The FastAPI application instance.
    """
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)


# --------------------------------------------------------------------------- #
# Request / response models                                                   #
# --------------------------------------------------------------------------- #


class IssueKeyRequest(BaseModel):
    """Body for ``POST /v1/keys``."""

    tenant: str = Field(..., min_length=1, description="Tenant / customer id")
    tier: str = Field(
        default="free",
        description="Access tier: free | pro | enterprise | admin",
    )
    rate: Optional[str] = Field(
        default=None,
        description="Optional slowapi rate spec, e.g. '60/minute'. "
        "Defaults to a per-tier value when omitted.",
    )


class IssueKeyResponse(BaseModel):
    """Response for ``POST /v1/keys`` — carries the plaintext key once."""

    key: str = Field(..., description="Plaintext API key (shown only once)")
    key_id: str = Field(..., description="Stable public id for management/DELETE")
    tenant: str
    tier: str
    rate: str


class KeyInfo(BaseModel):
    """A single key's non-secret metadata (``GET /v1/keys`` item)."""

    key_id: str
    tenant: str
    tier: str
    rate: str
    usage: int
    tokens: int
    revoked: bool
    created_at: str


class RevokeResponse(BaseModel):
    """Response for ``DELETE /v1/keys/{key_id}``."""

    revoked: bool
    key_id: str


# --------------------------------------------------------------------------- #
# Router                                                                       #
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/v1/keys", tags=["keys"])


@router.post(
    "",
    response_model=IssueKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Issue a new API key (admin only) — plaintext returned once",
)
def create_key(
    body: IssueKeyRequest,
    _admin: Annotated[dict, Depends(auth.require_admin)],
) -> IssueKeyResponse:
    """Issue a new key for a tenant and return the plaintext **once**.

    Requires an ``admin``-tier key. The returned ``key`` is the only time the
    plaintext is available; only its hash is stored.

    Raises:
        HTTPException: 422 if ``tier`` is invalid.
    """
    if body.tier not in VALID_TIERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"tier must be one of {sorted(VALID_TIERS)}",
        )
    try:
        raw = auth.issue_key(tenant=body.tenant, tier=body.tier, rate=body.rate)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    # Recover the just-created row's public metadata without exposing the hash.
    from api.db import hash_key

    key_id = hash_key(raw)[:12]
    # The effective rate may have been defaulted by tier; read it back.
    info = next((k for k in auth.list_keys() if k["key_id"] == key_id), None)
    effective_rate = info["rate"] if info else (body.rate or "")
    return IssueKeyResponse(
        key=raw,
        key_id=key_id,
        tenant=body.tenant,
        tier=body.tier,
        rate=effective_rate,
    )


@router.get(
    "",
    response_model=list[KeyInfo],
    summary="List all issued keys (admin only) — no plaintext",
)
def get_keys(_admin: Annotated[dict, Depends(auth.require_admin)]) -> list[KeyInfo]:
    """Return non-secret metadata for every issued key (newest first)."""
    return [KeyInfo(**k) for k in auth.list_keys()]


@router.delete(
    "/{key_id}",
    response_model=RevokeResponse,
    summary="Revoke a key by its public id (admin only)",
)
def delete_key(
    key_id: Annotated[str, Path(description="Public key id (key_hash prefix)")],
    _admin: Annotated[dict, Depends(auth.require_admin)],
) -> RevokeResponse:
    """Revoke a key. Idempotent.

    Raises:
        HTTPException: 404 if no key matches ``key_id``.
    """
    ok = auth.revoke(key_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No key found for id {key_id!r}",
        )
    return RevokeResponse(revoked=True, key_id=key_id)
