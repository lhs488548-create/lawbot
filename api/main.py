"""lawbot FastAPI application — production assembly (Playbook 08 Task 4.2).

Production Korean-legal **data-infra** API for practising lawyers (expert mode),
delivered as a multi-tenant cloud service. This module is the api_server
builder's only file; it *assembles* the public HTTP surface defined in
``_BUILD_CONTRACT.md`` section (e), aligned to lawbot.org (09 §A):

    GET    /healthz                                       (no auth)
    GET    /console                                       (page; actions use key)
    POST   /v1/statutes/search                            (key; anon IP-limited)
    POST   /v1/verify                                     (key — Citation Firewall)
    POST   /v1/source-pack                                (key — citable bundle)
    POST   /v1/embeddings                                 (key — OpenAI-compatible)
    POST   /v1/ask                                        (key required — LLM cost)
    POST   /v1/ad-review                                  (key; multipart — LLM)
    GET    /v1/statutes/{law_id}/articles/{article_no}    (key)
    GET    /v1/precedents/{seq}                           (key)
    POST   /v1/keys  ·  GET /v1/keys  ·  DELETE /v1/keys/{key_id}   (admin key)

Assembly notes
--------------
* **Shared, already-built collaborators are imported directly** and are owned by
  sibling builders: ``api.auth`` / ``api.db`` / ``api.keys`` (multi-tenant keys +
  per-key slowapi rate limiting), ``search.rag`` (``/v1/ask``),
  ``search.retriever`` (``Hit`` shape).
* **Optional collaborators that may not be built yet** (``search.statutes``,
  ``search.verify``, ``search.source_pack``, ``embed.embed_client``,
  ``search.ad_review`` / ``api.ad_review``) are resolved **lazily** at startup
  and recorded in :data:`backends`. A call to an endpoint whose backend is not
  yet wired returns a clear ``503`` instead of crashing the process, so the
  service still boots for ``/healthz``, ``/docs`` and integration wiring.
* **Common response meta** ``{trust_grade, source_url, license, as_of_date,
  effective_from}`` and an optional ``as_of_date`` request field are threaded
  through every search/verify/source-pack/ask response (09 §A).
* **Secrets** are never logged or returned. Auth verifies an opaque ``Bearer``
  token via ``api.auth`` (SHA-256 hash lookup), never the raw key.
* **AI Basic Act**: ``/v1/ask`` and ``/v1/ad-review`` always carry ``disclaimer``
  + ``ai_generated: true`` (defensively re-asserted by :func:`_ensure_notice`).

Run::

    cd /home/user1/lawbot && .venv/bin/python -m uvicorn api.main:app \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import contextvars
import json
import logging
from contextlib import asynccontextmanager
from queue import Queue
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import config

logger = logging.getLogger("lawbot.api")

# Static assets (the multi-tenant console UI) live in ``web/``.
_WEB_DIR: Path = Path(__file__).resolve().parent.parent / "web"


# --------------------------------------------------------------------------- #
# OpenAPI metadata                                                            #
# --------------------------------------------------------------------------- #
TAGS_METADATA = [
    {"name": "health", "description": "Liveness / readiness probes (no auth)."},
    {
        "name": "statutes",
        "description": (
            "Unified dense retrieval over 국가법령·자치법규·행정규칙·판례 with "
            "article/section precision and `as_of_date` point-in-time scoping "
            "(no LLM)."
        ),
    },
    {
        "name": "verify",
        "description": (
            "Citation Firewall — checks an AI citation against the DB and "
            "law.go.kr OpenAPI (현행·문구·시점 대조)."
        ),
    },
    {
        "name": "source-pack",
        "description": "Deterministic LLM-citable markdown bundle of relevant originals.",
    },
    {
        "name": "embeddings",
        "description": "OpenAI-compatible embeddings (internally forced to the index model).",
    },
    {
        "name": "qa",
        "description": (
            "Grounded question answering. Answers cite **only** retrieved "
            "originals; citations are post-verified (anti-hallucination)."
        ),
    },
    {
        "name": "review",
        "description": "Expert-mode draft/document review (issue spotting, lawyer audience).",
    },
    {"name": "documents", "description": "Direct lookup of a statute article or precedent by id."},
    {"name": "keys", "description": "Multi-tenant API-key issuance / listing / revocation (admin)."},
    {"name": "console", "description": "Self-service multi-tenant console (keys · usage · sources)."},
]

DESCRIPTION = """\
**lawbot** — production Korean-legal **data-infra** API for practising lawyers.

Core data-infra surface (lawbot.org-aligned, no generation cost):
`POST /v1/statutes/search`, `POST /v1/verify` (Citation Firewall),
`POST /v1/source-pack`, `POST /v1/embeddings`.

Lawyer add-ons (LLM cost): `POST /v1/ask` (grounded, **post-verified citations**)
and `POST /v1/ad-review` (expert issue-spotting on a draft).

* Every search/verify/source-pack/ask response carries the common meta
  `{trust_grade, source_url, license, as_of_date, effective_from}` and accepts an
  optional `as_of_date` (ISO `YYYY-MM-DD`) for point-in-time current-law lookup.
* Multi-tenant: authenticate with `Authorization: Bearer lk_…`; per-key rate
  limits; admin keys manage tenant keys at `/v1/keys`.
* Coverage is honestly bounded (헌재 결정·법령해석례 등 일부 미포함); every generated
  answer carries an `ai_generated` flag and a provenance disclaimer.
"""


# --------------------------------------------------------------------------- #
# Optional-collaborator probing (sibling builders may not all be present yet)  #
# --------------------------------------------------------------------------- #
class _Backends:
    """Lazily-resolved references to *optional* sibling builder modules.

    The auth/keys stack is a hard dependency (imported at module load); the
    search-layer data-infra modules below may still be under construction, so
    they are probed once at startup and surfaced via ``/healthz`` rather than
    crashing the process. Each attribute is ``None`` until imported.
    """

    def __init__(self) -> None:
        self.rag: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.statutes: Optional[Any] = None
        self.verify: Optional[Any] = None
        self.source_pack: Optional[Any] = None
        self.embed_client: Optional[Any] = None
        self.ad_review: Optional[Callable[..., dict]] = None
        self.qdrant_ok: bool = False
        self.qdrant_points: Optional[int] = None
        self.errors: dict[str, str] = {}

    def _try_import(self, name: str) -> Optional[Any]:
        try:
            return import_module(name)
        except Exception as exc:  # ImportError or downstream config error
            self.errors[name] = type(exc).__name__
            logger.warning("optional backend %s unavailable: %s", name, type(exc).__name__)
            return None

    def probe(self) -> None:
        """Resolve every optional collaborator once; never raises."""
        self.rag = self._try_import("search.rag")
        self.retriever = self._try_import("search.retriever")
        self.statutes = self._try_import("search.statutes")
        self.verify = self._try_import("search.verify")
        self.source_pack = self._try_import("search.source_pack")
        self.embed_client = self._try_import("embed.embed_client")

        # The ad-review handler may live in either module; it must expose review().
        for candidate in ("search.ad_review", "api.ad_review"):
            mod = self._try_import(candidate)
            if mod is not None and hasattr(mod, "review"):
                self.ad_review = mod.review
                break

        self._probe_qdrant()

    def _probe_qdrant(self) -> None:
        """Best-effort check that the Qdrant collection is reachable."""
        try:
            # Reuse the retriever's shared client (server-first, local-path
            # fallback). Opening a second client here would deadlock the
            # embedded on-disk Qdrant, which allows only one handle per process.
            if self.retriever is not None and hasattr(self.retriever, "get_qdrant_client"):
                client = self.retriever.get_qdrant_client()
            else:
                from embed import upsert_qdrant

                client, _ = upsert_qdrant.get_client()
            info = client.count(collection_name=config.COLLECTION, exact=False)
            self.qdrant_points = int(info.count)
            self.qdrant_ok = True
        except Exception as exc:
            self.qdrant_ok = False
            self.qdrant_points = None
            self.errors["qdrant"] = type(exc).__name__
            logger.warning("qdrant probe failed: %s", type(exc).__name__)


backends = _Backends()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the key DB and probe optional backends on startup.

    Probing is non-fatal: an unavailable optional backend is recorded and
    surfaced via ``/healthz`` and ``503`` responses, so the service still boots
    for health-checking, OpenAPI inspection, and admin key management.
    """
    logger.info("lawbot API starting up")
    try:
        from api import auth

        auth.init_db()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("auth.init_db failed: %s", type(exc).__name__)
        backends.errors["api.auth.init_db"] = type(exc).__name__
    backends.probe()
    if backends.errors:
        logger.warning("startup with degraded optional backends: %s", sorted(backends.errors))
    else:
        logger.info("all backends ready (qdrant points=%s)", backends.qdrant_points)
    yield
    logger.info("lawbot API shutting down")


app = FastAPI(
    title="lawbot API",
    version="0.1.0",
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    contact={"name": "lawbot", "url": "https://law.go.kr"},
    license_info={"name": "Source originals: 공공저작물 (저작권법 §7 취지)"},
)

# CORS — permissive for the MVP API surface; tighten allow_origins per tenant
# domain before public launch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# --------------------------------------------------------------------------- #
# Multi-tenant auth + per-key rate limiting (owned by the multitenant builder)  #
# --------------------------------------------------------------------------- #
# ``api.auth`` provides the request dependencies (``require_key`` /
# ``require_admin``), ``api.keys`` provides the ``/v1/keys`` router plus the
# slowapi limiter and its 429 handler. We mount them rather than re-implementing.
from api import auth  # noqa: E402  (after app/CORS so middleware order is clear)
from api import keys as keys_module  # noqa: E402

keys_module.setup_rate_limiting(app)
app.include_router(keys_module.router)

# --------------------------------------------------------------------------- #
# Per-key dynamic rate limiting                                                #
# --------------------------------------------------------------------------- #
# ``keys_module.per_key_rate(request)`` cannot be used directly as a slowapi
# dynamic-limit provider here: this slowapi version only forwards a request to a
# provider whose parameter is literally named ``key`` (it then passes
# ``key_function(request)``), and calls any other provider with **no** arguments.
# We therefore build our own limiter whose ``key_function`` both (a) returns the
# per-key bucket (``tenant:tier:key_id``, IP fallback — same scheme as
# ``keys_module.rate_limit_key``) and (b) stashes the principal's stored ``rate``
# in a contextvar. slowapi calls ``key_function(request)`` and the provider in the
# *same* call/thread during the limit check, so the provider reliably reads back
# the right per-key rate. This honours each key's stored ``rate`` (e.g.
# ``"30/minute"`` free vs ``"600/minute"`` enterprise) with an IP fallback for
# anonymous reads, exactly as the contract requires.
from slowapi import Limiter  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

_RATE_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "lawbot_rate", default="20/minute"
)


def _rate_bucket(request: Request) -> str:
    """Limiter key: per-key bucket (IP fallback) + record the key's rate.

    Runs inside slowapi's limit check, in the same thread/context as the limit
    provider, so the ``rate`` it records is the one the provider then reads.
    """
    principal = getattr(request.state, "principal", None)
    if principal:
        rate = principal.get("rate")
        if rate:
            _RATE_CTX.set(rate)
        return f"{principal['tenant']}:{principal['tier']}:{principal['key_id']}"
    _RATE_CTX.set("20/minute")  # anonymous default
    return get_remote_address(request)


def _dynamic_rate(key: str) -> str:  # noqa: ARG001 - slowapi passes the bucket key
    """slowapi dynamic-limit provider: the per-key rate recorded by the key func.

    The parameter is named ``key`` so slowapi treats this as a request-aware
    provider (and supplies the bucket key, which we ignore); the actual rate
    comes from :data:`_RATE_CTX`, set by :func:`_rate_bucket` moments earlier in
    the same limit check.
    """
    return _RATE_CTX.get()


# Replace the app limiter with one bucketed by our key func. ``setup_rate_limiting``
# already installed the middleware + 429 handler against ``keys_module.limiter``;
# we swap the active limiter on the app state so route decorators below and the
# middleware agree.
limiter = Limiter(key_func=_rate_bucket)
app.state.limiter = limiter


# --------------------------------------------------------------------------- #
# Request / response models                                                    #
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    """Body for retrieval endpoints (``/v1/statutes/search`` · ``/v1/source-pack``)."""

    query: str = Field(..., min_length=1, max_length=4000, description="자연어 법률 질의")
    filter: Optional[dict[str, str]] = Field(
        default=None,
        description='페이로드 필터 (예: {"doc_type": "law"}, {"jurisdiction": "전라남도"})',
    )
    k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=50, description="검색 top-K")
    as_of_date: Optional[str] = Field(
        default=None,
        description="시점조회 (ISO YYYY-MM-DD). effective_from <= as_of_date 행으로 제한.",
        examples=["2026-04-02"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "임대차 계약 갱신요구권의 행사 기간은?",
                    "filter": {"doc_type": "law"},
                    "k": 8,
                    "as_of_date": "2026-04-02",
                }
            ]
        }
    }


class AskRequest(QueryRequest):
    """Body for ``/v1/ask`` (adds an optional generation-model override)."""

    model: Optional[str] = Field(
        default=None,
        description="생성 모델 override (예: 어려운 질의에 한해 강한 모델). 기본=config.GEN_MODEL.",
    )
    stream: bool = Field(
        default=False,
        description=(
            "true이면 답변을 SSE(text/event-stream)로 토큰 단위 스트리밍 (09 §E-4.4). "
            "이벤트: meta → token* → done(검증된 citations 포함)."
        ),
    )


class CitationModel(BaseModel):
    """A single citation in a verify request / an answer."""

    law_name: Optional[str] = Field(default=None, description="법령명 (또는 title)")
    title: Optional[str] = None
    article_no: Optional[str] = Field(default=None, description="조문번호, 예) 제17조")
    case_no: Optional[str] = Field(default=None, description="사건번호")
    사건번호: Optional[str] = None
    source_id: Optional[str] = None

    model_config = {"extra": "allow"}


class VerifyRequest(BaseModel):
    """Body for ``/v1/verify`` — one citation or a batch."""

    citation: Optional[dict[str, Any]] = Field(
        default=None, description="단일 인용 {law_name?/title?, article_no?} 또는 {case_no?}"
    )
    citations: Optional[list[dict[str, Any]]] = Field(
        default=None, description="여러 인용 배열 (citation과 둘 중 하나)"
    )
    as_of_date: Optional[str] = Field(default=None, description="시점 유효성 검증 기준일 (YYYY-MM-DD)")


class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible body for ``/v1/embeddings``."""

    input: str | list[str] = Field(..., description="임베딩할 텍스트 또는 텍스트 배열")
    model: Optional[str] = Field(
        default=None,
        description="무시되고 항상 config.EMBED_MODEL 사용 (단일 컬렉션 모델 일치 보장).",
    )


class HealthResponse(BaseModel):
    """``/healthz`` payload."""

    ok: bool
    collection: str
    points: Optional[int] = None
    backends: dict[str, bool]


# --------------------------------------------------------------------------- #
# Auth dependencies (thin wrappers over api.auth so route signatures are local) #
# --------------------------------------------------------------------------- #
def require_key(principal: dict = Depends(auth.require_key)) -> dict:
    """Require any active API key (delegates to ``api.auth.require_key``).

    ``api.auth.require_key`` has already stashed the principal on
    ``request.state.principal``; the rate limiter's key function reads it to
    bucket and rate-limit per key.
    """
    return principal


def optional_key(request: Request) -> Optional[dict]:
    """Accept anonymous calls but attach the principal when a valid key is present.

    Anonymous requests fall through to the per-IP rate-limit bucket
    (``api.keys.rate_limit_key``). Invalid (but present) keys are treated as
    anonymous rather than rejected, so public read endpoints stay reachable.
    """
    header = request.headers.get("authorization")
    if not header:
        return None
    token = header.strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer ") :].strip()
    if not token:
        return None
    try:
        principal = auth.verify(token)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("auth.verify error: %s", type(exc).__name__)
        return None
    if principal:
        request.state.principal = principal
    return principal


def require_admin(principal: dict = Depends(auth.require_admin)) -> dict:
    """Require an ``admin``-tier key (delegates to ``api.auth.require_admin``)."""
    return principal


# --------------------------------------------------------------------------- #
# Optional-backend guards                                                      #
# --------------------------------------------------------------------------- #
def _require(attr: str, func: str, label: str) -> Any:
    """Return an optional backend module exposing ``func``, or raise 503."""
    mod = getattr(backends, attr, None)
    if mod is None or not hasattr(mod, func):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{label} backend not available yet.",
        )
    return mod


# --------------------------------------------------------------------------- #
# Exception handlers                                                          #
# --------------------------------------------------------------------------- #
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Uniform JSON error envelope; preserves auth / Retry-After headers."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"status": exc.status_code, "detail": exc.detail}},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: log the *type* (never the payload/PII) and return a 500."""
    logger.error("unhandled error on %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"status": 500, "detail": "Internal server error."}},
    )


# --------------------------------------------------------------------------- #
# Common-meta helpers                                                          #
# --------------------------------------------------------------------------- #
_COMMON_META_KEYS = ("trust_grade", "source_url", "license", "as_of_date", "effective_from")


def _common_meta(row: dict[str, Any], as_of_date: Optional[str]) -> dict[str, Any]:
    """Attach the common response meta (09 §A) to a result row, filling defaults.

    Existing values on ``row`` win; missing keys default to license / the request
    ``as_of_date`` / ``None`` so every row exposes the full meta surface.
    """
    out = dict(row)
    out.setdefault("license", config.DEFAULT_LICENSE)
    out.setdefault("as_of_date", as_of_date)
    for key in _COMMON_META_KEYS:
        out.setdefault(key, None if key != "license" else config.DEFAULT_LICENSE)
    return out


def _ensure_notice(result: Any) -> dict:
    """Guarantee the AI-Basic-Act notice fields on a generated response.

    The RAG / ad-review backends set these already; this is a defensive
    belt-and-suspenders so the notice is never silently missing.
    """
    if not isinstance(result, dict):
        result = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    result.setdefault("ai_generated", True)
    result.setdefault("disclaimer", config.ANSWER_DISCLAIMER)
    return result


# --------------------------------------------------------------------------- #
# Document lookup helpers (statute article / precedent by id — from artifacts)  #
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path) -> Any:
    """Yield parsed JSON objects from a JSONL artifact, skipping bad lines."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def _normalize_article_no(value: str) -> str:
    """Normalise an article id to the stored ``제N조`` form.

    Accepts ``"4"``, ``"제4조"``, ``"4조"``, ``"제4조의2"`` etc.
    """
    v = (value or "").strip()
    if not v or v.startswith("제"):
        return v
    core = v[:-1] if v.endswith("조") else v
    return f"제{core}조"


def _find_statute_article(law_id: str, article_no: str) -> Optional[dict]:
    """Locate one statute article in the parsed law artifact, or ``None``."""
    target = _normalize_article_no(article_no)
    prefix = f"LAW:{law_id}:"
    for doc in _iter_jsonl(config.DOCS_LAW_JSONL):
        if not str(doc.get("doc_id", "")).startswith(prefix):
            continue
        for art in doc.get("articles", []):
            if art.get("article_no") == target:
                return {
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title"),
                    "law_kind": doc.get("law_kind"),
                    "article_no": art.get("article_no"),
                    "article_title": art.get("title"),
                    "text": art.get("text"),
                    "source_url": doc.get("source_url"),
                    "effective_from": doc.get("effective_from"),
                    "trust_grade": doc.get("trust_grade", "A"),
                    "license": config.DEFAULT_LICENSE,
                }
    return None


def _find_precedent(seq: str) -> Optional[dict]:
    """Locate one precedent by 판례일련번호 in the parsed precedent artifact."""
    target_id = f"PREC:{seq}"
    for doc in _iter_jsonl(config.DOCS_PREC_JSONL):
        if str(doc.get("doc_id")) != target_id:
            continue
        meta = doc.get("meta", {}) or {}
        return {
            "doc_id": doc["doc_id"],
            "사건번호": meta.get("사건번호") or doc.get("title"),
            "사건명": doc.get("title"),
            "법원명": doc.get("jurisdiction"),
            "선고일자": doc.get("effective_from"),
            "사건종류": doc.get("law_kind"),
            "sections": [
                {"article_no": a.get("article_no"), "text": a.get("text")}
                for a in doc.get("articles", [])
            ],
            "source_url": doc.get("source_url"),
            "trust_grade": doc.get("trust_grade", "A"),
            "license": config.DEFAULT_LICENSE,
        }
    return None


# --------------------------------------------------------------------------- #
# Endpoints — health                                                          #
# --------------------------------------------------------------------------- #
@app.get("/healthz", response_model=HealthResponse, tags=["health"], summary="Liveness/readiness probe")
def healthz() -> HealthResponse:
    """Return service liveness plus per-backend readiness.

    Always answers 200 so external load-balancer health checks pass even while
    individual optional backends (Qdrant, RAG, verify, …) are still warming up.
    """
    return HealthResponse(
        ok=True,
        collection=config.COLLECTION,
        points=backends.qdrant_points,
        backends={
            "rag": backends.rag is not None,
            "retriever": backends.retriever is not None,
            "statutes": backends.statutes is not None,
            "verify": backends.verify is not None,
            "source_pack": backends.source_pack is not None,
            "embed_client": backends.embed_client is not None,
            "ad_review": backends.ad_review is not None,
            "qdrant": backends.qdrant_ok,
        },
    )


# --------------------------------------------------------------------------- #
# Endpoints — core data-infra (lawbot.org-aligned, no generation cost)         #
# --------------------------------------------------------------------------- #
@app.post(
    "/v1/statutes/search",
    tags=["statutes"],
    summary="Unified statute+precedent search (no LLM)",
)
@limiter.limit(_dynamic_rate)
def v1_statutes_search(
    request: Request,
    body: QueryRequest,
    principal: Optional[dict] = Depends(optional_key),
) -> dict:
    """Unified dense law+precedent search with article/section precision.

    Each result row carries the common meta ``{trust_grade, source_url, license,
    as_of_date, effective_from}`` plus ``{doc_id, doc_type, title, article_no,
    score, text}``. Pure retrieval — no generation cost. Anonymous reads are
    IP-rate-limited; a key raises the limit.
    """
    statutes = _require("statutes", "statutes_search", "Statute search")
    try:
        rows = statutes.statutes_search(
            body.query, k=body.k, filter=body.filter, as_of_date=body.as_of_date
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:
        logger.error("statutes_search failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Retrieval backend error.") from exc
    results = [_common_meta(dict(r), body.as_of_date) for r in rows]
    return {"results": results, "as_of_date": body.as_of_date}


@app.post("/v1/verify", tags=["verify"], summary="Citation Firewall — verify an AI citation")
@limiter.limit(_dynamic_rate)
def v1_verify(
    request: Request, body: VerifyRequest, principal: dict = Depends(require_key)
) -> dict:
    """Verify one or more citations against the DB and law.go.kr (현행·시점 대조).

    Provide ``citation`` (single) **or** ``citations`` (batch). Each result is
    ``{verified, trust_grade, current, source_url, effective_from, as_of_date,
    note, db_match, api_match}``. Detects 폐지/오인용/허위사건. The OC token is
    never returned or logged.
    """
    verify_mod = _require("verify", "verify_citation", "Citation-verification")
    items = body.citations if body.citations is not None else (
        [body.citation] if body.citation is not None else []
    )
    if not items:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Provide a 'citation' object or a non-empty 'citations' array.",
        )
    results: list[dict] = []
    for cit in items:
        try:
            res = verify_mod.verify_citation(cit, as_of_date=body.as_of_date)
        except Exception as exc:
            logger.error("verify_citation failed: %s", type(exc).__name__)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Verification backend error.") from exc
        results.append(_common_meta(dict(res), body.as_of_date))
    return {"results": results, "as_of_date": body.as_of_date}


@app.post("/v1/source-pack", tags=["source-pack"], summary="LLM-citable markdown bundle")
@limiter.limit(_dynamic_rate)
def v1_source_pack(
    request: Request, body: QueryRequest, principal: dict = Depends(require_key)
) -> dict:
    """Assemble relevant originals into an LLM-citable markdown bundle.

    Retrieves child hits, promotes to parents, and emits ``{markdown, sources:
    [...common meta], as_of_date}``. Deterministic assembly — no generation cost.
    """
    sp = _require("source_pack", "build", "Source-pack")
    try:
        pack = sp.build(body.query, k=body.k, filter=body.filter, as_of_date=body.as_of_date)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:
        logger.error("source_pack.build failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Source-pack backend error.") from exc
    out = dict(pack)
    out.setdefault("as_of_date", body.as_of_date)
    out["sources"] = [_common_meta(dict(s), body.as_of_date) for s in out.get("sources", [])]
    return out


@app.post("/v1/embeddings", tags=["embeddings"], summary="OpenAI-compatible embeddings")
@limiter.limit(_dynamic_rate)
def v1_embeddings(
    request: Request, body: EmbeddingsRequest, principal: dict = Depends(require_key)
) -> dict:
    """Return OpenAI-shape embeddings, internally forced to the index model.

    ``model`` in the request is ignored; the service always uses
    ``config.EMBED_MODEL``/``EMBED_DIM`` so external vectors stay compatible with
    the indexed collection.
    """
    ec = _require("embed_client", "embed_texts", "Embeddings")
    texts = [body.input] if isinstance(body.input, str) else list(body.input)
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "'input' must be a non-empty string or string list."
        )
    try:
        vectors = ec.embed_texts(texts)
    except Exception as exc:
        logger.error("embed_texts failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Embedding backend error.") from exc
    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
    return {
        "object": "list",
        "data": data,
        "model": config.EMBED_MODEL,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# --------------------------------------------------------------------------- #
# Endpoints — lawyer add-ons (LLM cost)                                        #
# --------------------------------------------------------------------------- #
def _sse_event(payload: dict[str, Any]) -> str:
    """Serialize one event dict as a single SSE ``data:`` frame (09 §E-4.4)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _ask_sse(rag: Any, query: str, kwargs: dict[str, Any]) -> Any:
    """Async SSE generator: drive ``rag.ask_stream`` off the event loop.

    The RAG streaming generator is blocking (network I/O to OpenAI/Qdrant), so we
    run it in the default threadpool and hand events back over a thread-safe
    queue, awaiting each one without blocking the loop. A sentinel marks the end;
    backend errors are surfaced as a terminal ``error`` SSE event rather than
    tearing down the response mid-stream.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    q: "Queue[Any]" = Queue()
    _DONE = object()

    def _pump() -> None:
        try:
            try:
                gen = rag.ask_stream(query, **kwargs)
            except TypeError:
                # Backend that predates the as_of_date kwarg.
                kw = {k: v for k, v in kwargs.items() if k != "as_of_date"}
                gen = rag.ask_stream(query, **kw)
            for event in gen:
                q.put(event)
        except Exception as exc:  # surface as an SSE error event, not a crash
            logger.error("rag.ask_stream failed: %s", type(exc).__name__)
            q.put({"type": "error", "detail": "Upstream model/retrieval error."})
        finally:
            q.put(_DONE)

    loop.run_in_executor(None, _pump)

    while True:
        event = await loop.run_in_executor(None, q.get)
        if event is _DONE:
            break
        yield _sse_event(event)


@app.post("/v1/ask", tags=["qa"], summary="Grounded, cited answer (LLM; optional SSE stream)")
@limiter.limit(_dynamic_rate)
async def v1_ask(
    request: Request, body: AskRequest, principal: dict = Depends(require_key)
):
    """Answer a legal query strictly from retrieved originals, with citations.

    Requires a valid API key (the call incurs LLM cost). Citations are
    post-verified by the RAG backend (hallucinated citations dropped); the
    response carries ``disclaimer`` + ``ai_generated`` (AI Basic Act).

    Set ``stream: true`` to receive a low-latency **SSE** (``text/event-stream``)
    response (09 §E-4.4): ``meta`` first, then ``token`` deltas, then a terminal
    ``done`` event carrying the verified citations. The default JSON path is
    unchanged.
    """
    rag = _require("rag", "ask", "RAG")
    kwargs: dict[str, Any] = {"k": body.k, "flt": body.filter, "as_of_date": body.as_of_date}
    if body.model:
        kwargs["model"] = body.model

    # --- SSE streaming path (token-by-token, first token <1s target) -------- #
    if body.stream:
        if not hasattr(rag, "ask_stream"):
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Streaming backend not available yet."
            )
        return StreamingResponse(
            _ask_sse(rag, body.query, kwargs),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Default JSON path -------------------------------------------------- #
    try:
        result = rag.ask(body.query, **kwargs)
    except TypeError:
        # Backend that predates the as_of_date kwarg.
        kw = {k: v for k, v in kwargs.items() if k != "as_of_date"}
        result = rag.ask(body.query, **kw)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:
        logger.error("rag.ask failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Upstream model/retrieval error.") from exc
    return _ensure_notice(result)


@app.post("/v1/ad-review", tags=["review"], summary="Expert-mode draft review (LLM)")
@limiter.limit(_dynamic_rate)
async def v1_ad_review(
    request: Request,
    principal: dict = Depends(require_key),
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    question: Optional[str] = Form(default=None),
) -> dict:
    """Review a draft document (PDF upload or raw text) against retrieved law.

    Provide **either** a ``file`` (PDF/text) **or** a ``text`` field; an optional
    ``question`` focuses the review. Returns an expert issue-spotting analysis
    with verified citations, plus ``disclaimer`` + ``ai_generated``.
    """
    if backends.ad_review is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Document-review backend not available yet."
        )
    if file is None and not (text and text.strip()):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Provide a 'file' upload or a non-empty 'text' field.",
        )
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None
    if file is not None:
        file_bytes = await file.read()
        filename = file.filename
    try:
        result = backends.ad_review(
            text=text, file_bytes=file_bytes, filename=filename, question=question
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ad_review failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Document-review backend error.") from exc
    return _ensure_notice(result)


# --------------------------------------------------------------------------- #
# Endpoints — direct document lookup                                          #
# --------------------------------------------------------------------------- #
@app.get(
    "/v1/statutes/{law_id}/articles/{article_no}",
    tags=["documents"],
    summary="Fetch one statute article",
)
def v1_statute_article(
    law_id: str, article_no: str, principal: dict = Depends(require_key)
) -> dict:
    """Return a single statute article by 법령ID and article number.

    ``article_no`` may be URL-encoded ``제4조`` or a bare number (``4``). Returns
    404 when the article is not found or the law artifact is absent.
    """
    found = _find_statute_article(law_id, article_no)
    if not found:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Statute article not found: 법령ID={law_id}, {article_no}.",
        )
    return found


@app.get("/v1/precedents/{seq}", tags=["documents"], summary="Fetch one precedent")
def v1_precedent(seq: str, principal: dict = Depends(require_key)) -> dict:
    """Return a single precedent by 판례일련번호, with its sections."""
    found = _find_precedent(seq)
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Precedent not found: 판례일련번호={seq}.")
    return found


# --------------------------------------------------------------------------- #
# Console (multi-tenant self-service page; actions use the key from the page)  #
# --------------------------------------------------------------------------- #
@app.get("/console", response_class=HTMLResponse, tags=["console"], summary="Self-service console")
def console() -> HTMLResponse:
    """Serve the minimal multi-tenant console (keys · usage · sources · sync).

    Served from ``web/console.html`` when present, else a built-in fallback page.
    The page is unauthenticated; the actions inside it call ``/v1/keys`` with an
    admin key the operator pastes in (never stored server-side).
    """
    page = _WEB_DIR / "console.html"
    if page.is_file():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse(_CONSOLE_FALLBACK)


_CONSOLE_FALLBACK = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>lawbot console</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
code{background:#f3f3f3;padding:.1rem .3rem;border-radius:3px}</style></head>
<body><h1>lawbot console</h1>
<p>멀티테넌트 API 키 셀프서비스. 관리(<code>admin</code>) 키로 발급/조회/폐기합니다.</p>
<ul>
<li><code>POST /v1/keys</code> — 테넌트 키 발급 (관리자, 평문 1회 반환)</li>
<li><code>GET /v1/keys</code> — 키 목록·usage (관리자, 평문 미노출)</li>
<li><code>DELETE /v1/keys/{key_id}</code> — 폐기 (관리자)</li>
</ul>
<p>전체 API 문서: <a href="/docs">/docs</a> · 스키마: <a href="/openapi.json">/openapi.json</a></p>
<p style="color:#666">데이터 커버리지: 국가법령·판례·행정규칙·전국 자치법규 (헌재 결정·법령해석례 등 일부 미포함).</p>
</body></html>"""


@app.get("/chat", response_class=HTMLResponse, tags=["console"], summary="Chat + PDF-review page")
def chat() -> HTMLResponse:
    """Serve the single-page chat UI (SSE-streamed ``/v1/ask`` + ``/v1/ad-review``).

    Served from ``web/chat.html`` when present, else a built-in fallback. The page
    is unauthenticated; the user pastes/issues a tenant key that the in-page JS
    sends as ``Authorization: Bearer`` on each API call (never stored server-side).
    """
    page = _WEB_DIR / "chat.html"
    if page.is_file():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse(_CHAT_FALLBACK)


_CHAT_FALLBACK = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>lawbot 채팅</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
code{background:#f3f3f3;padding:.1rem .3rem;border-radius:3px}</style></head>
<body><h1>lawbot 채팅</h1>
<p>채팅 UI(<code>web/chat.html</code>)를 찾지 못했습니다. API는 직접 호출할 수 있습니다.</p>
<ul>
<li><code>POST /v1/ask</code> — 근거·인용 답변 (SSE 스트리밍 지원, 키 필요)</li>
<li><code>POST /v1/ad-review</code> — PDF/문서 광고심사 (멀티파트, 키 필요)</li>
<li><code>POST /v1/keys</code> — 테넌트 키 발급 (관리자)</li>
</ul>
<p>전체 API 문서: <a href="/docs">/docs</a></p>
</body></html>"""


__all__ = ["app"]
