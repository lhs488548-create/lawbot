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
import threading
import time
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
    {
        "name": "AI 답변",
        "description": "AI 법률 Q&A — 검색된 원문에만 근거해 답하고, 인용은 사후검증합니다(환각 인용 제거). 응답에 면책고지·AI 생성 표시 포함.",
    },
    {
        "name": "의료광고 검토",
        "description": "의료광고 PDF·문서 검토 — 위반/주의 항목 적발과 근거 인용을 제시합니다. 응답에 면책고지·AI 생성 표시 포함.",
    },
    {
        "name": "검색·검증",
        "description": "법령·판례 검색, 인용 검증(Citation Firewall), 근거 마크다운 번들 — LLM을 쓰지 않는 데이터 인프라 기능.",
    },
    {
        "name": "관리",
        "description": "멀티테넌트 API 키 발급·조회·폐기(관리자 키) 및 서비스 상태확인.",
    },
]

DESCRIPTION = """\
# lawbot API

**한국 법령·판례 RAG API — 근거기반 답변·인용검증 (+의료광고 검토)**

전국 법령·행정규칙·판례(자치법규 제외) 원문을 검색·인용하여 법률 질문에
답하고, 인용을 검증하며, 의료광고 문안 검토(버티컬)를 제공합니다. 모든 AI
답변에는 면책고지(`disclaimer`)와 AI 생성 표시(`ai_generated: true`)가 함께
제공됩니다.

---

## 인증 방법

`/healthz` 를 제외한 대부분의 엔드포인트는 발급된 API 키가 필요합니다.
요청 헤더에 아래 형식으로 키를 넣어 호출하세요.

```http
Authorization: Bearer <발급키>
```

**Swagger(`/docs`)에서 테스트하는 법**

1. 이 페이지 우상단 **Authorize** 버튼을 클릭합니다.
2. 입력칸에 발급키(예: `lk_xxxxxxxx`)를 그대로 붙여넣고 **Authorize** → **Close**.
3. 이제 각 엔드포인트의 **Try it out** 으로 실제 호출을 테스트할 수 있습니다.

> 키 발급은 관리자(`admin`) 키로 `POST /v1/keys` 에서 합니다(아래 "관리" 그룹 참고).

---

## 핵심 기능 3가지

| 기능 | 엔드포인트 | 설명 |
| --- | --- | --- |
| ① AI 법률 Q&A | `POST /v1/ask` | 질문을 한국 법령·판례 원문에 근거해 답변(인용 사후검증) |
| ② 의료광고 검토 | `POST /v1/ad-review` | 광고 PDF·문안의 위반/주의 항목 적발 + 근거 인용 |
| ③ 법령·판례 검색 | `POST /v1/statutes/search` | LLM 없이 관련 조문·판례를 정밀 검색 |

### ① AI 법률 Q&A — `POST /v1/ask`

요청
```json
{ "query": "무면허 의료행위 처벌은?" }
```
응답 요지
```json
{
  "answer": "의료법 제27조 위반 시 …",
  "citations": [{ "law_name": "의료법", "article_no": "제27조" }],
  "disclaimer": "본 답변은 참고용이며 …",
  "ai_generated": true
}
```

### ② 의료광고 검토 — `POST /v1/ad-review`

`multipart/form-data` 로 광고 **PDF 파일**(`file`) 또는 **텍스트**(`text`)를 보냅니다.
응답 요지
```json
{
  "issues": [{ "severity": "위반", "note": "치료 효과 보장 표현 …" }],
  "citations": [{ "law_name": "의료법", "article_no": "제56조" }],
  "disclaimer": "…",
  "ai_generated": true
}
```

### ③ 법령·판례 검색 — `POST /v1/statutes/search`

요청
```json
{ "query": "의료광고 과장 금지", "k": 5 }
```
응답 요지
```json
{ "results": [{ "title": "의료법", "article_no": "제56조", "score": 0.82, "text": "…" }] }
```

---

## 참고

* 검색·검증·근거번들·답변 응답에는 공통 메타
  `{trust_grade, source_url, license, as_of_date, effective_from}` 가 포함되며,
  선택 입력 `as_of_date`(ISO `YYYY-MM-DD`)로 **시점조회**(현행 기준일)를 지정할 수 있습니다.
* 데이터 커버리지는 전국 법령·행정규칙·판례(자치법규/조례 제외)이며,
  일부 자료(자치법규·헌재 결정·법령해석례 등)는 포함되지 않을 수 있습니다.
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
        self.index_ok: bool = False
        self.index_ntotal: Optional[int] = None
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

        self._probe_index()

    def _probe_index(self) -> None:
        """Confirm the active FAISS index file is present + readable.

        Cheap and non-blocking: the index is memory-mapped only to read its
        ``ntotal`` header (no full load into RAM — the retriever loads it lazily
        on the first query). ``index_ok`` gates ``/healthz`` so a load-balancer
        sees the container as live as soon as the index file is in place.
        """
        try:
            import faiss  # local import: keep api importable without faiss

            path = (
                config.FULL_FAISS_INDEX
                if config.ACTIVE_INDEX == "full"
                else config.FAISS_INDEX
            )
            if not path.exists():
                self.index_ok = False
                self.index_ntotal = None
                self.errors["index"] = "FileNotFound"
                logger.warning("index probe: %s not found", path)
                return
            try:
                idx = faiss.read_index(str(path), faiss.IO_FLAG_MMAP)
                self.index_ntotal = int(idx.ntotal)
            except Exception:
                self.index_ntotal = None  # file present but ntotal unreadable
            self.index_ok = True
        except Exception as exc:
            self.index_ok = False
            self.index_ntotal = None
            self.errors["index"] = type(exc).__name__
            logger.warning("index probe failed: %s", type(exc).__name__)


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
    # Pre-warm the FAISS index off the request path so the first user query does
    # not pay the lazy cold-load (~46s for the full index). Non-blocking: /healthz
    # reports live immediately while this runs in a daemon thread. get_index() is
    # internally locked, so a concurrent first request shares the same load.
    if config.PREWARM_INDEX and backends.retriever is not None:
        import threading

        def _prewarm_index() -> None:
            try:
                logger.info("index pre-warm: loading %s index...", config.ACTIVE_INDEX)
                backends.retriever.get_index()
                logger.info("index pre-warm: complete")
            except Exception as exc:  # pragma: no cover - best-effort warm
                logger.warning("index pre-warm failed: %s", type(exc).__name__)

        threading.Thread(target=_prewarm_index, name="prewarm-index", daemon=True).start()
    if backends.errors:
        logger.warning("startup with degraded optional backends: %s", sorted(backends.errors))
    else:
        logger.info("all backends ready (index ntotal=%s)", backends.index_ntotal)
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
# Lightweight in-process metrics (observability for a single-server deploy).  #
# No external deps; reset on restart. Exposed read-only at GET /metrics.      #
# --------------------------------------------------------------------------- #
class _Metrics:
    """Thread-safe request counters + latency, keyed by route template."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.monotonic()
        self.total = 0
        self.by_path: dict[str, int] = {}
        self.by_status: dict[str, int] = {}  # "2xx"/"4xx"/"5xx"
        self.latency_sum = 0.0
        self.llm_tokens = 0  # cumulative LLM tokens (answer generation)

    def record(self, path: str, status_code: int, dt: float) -> None:
        cls = f"{status_code // 100}xx"
        with self._lock:
            self.total += 1
            self.by_path[path] = self.by_path.get(path, 0) + 1
            self.by_status[cls] = self.by_status.get(cls, 0) + 1
            self.latency_sum += dt

    def add_tokens(self, n: int) -> None:
        if n > 0:
            with self._lock:
                self.llm_tokens += int(n)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            up = time.monotonic() - self.started
            avg = (self.latency_sum / self.total) if self.total else 0.0
            return {
                "uptime_seconds": round(up, 1),
                "requests_total": self.total,
                "requests_by_status": dict(self.by_status),
                "requests_by_path": dict(
                    sorted(self.by_path.items(), key=lambda kv: -kv[1])
                ),
                "avg_latency_ms": round(avg * 1000, 1),
                "llm_tokens_total": self.llm_tokens,
            }


metrics = _Metrics()


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Time every request and bucket it by matched route template + status."""
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        metrics.record(request.url.path, 500, time.monotonic() - start)
        raise
    route = request.scope.get("route")
    path = getattr(route, "path", None) or request.url.path
    metrics.record(path, response.status_code, time.monotonic() - start)
    return response


def _record_cost(result: Any, principal: Optional[dict]) -> None:
    """Best-effort cost meter: add LLM tokens to /metrics + the caller's key.

    ``result`` is an ask/ad-review result carrying ``usage.total_tokens``. The
    demo principal (key_id == 'demo') is metered globally but not billed to a key.
    """
    try:
        usage = result.get("usage") if isinstance(result, dict) else None
        tok = int((usage or {}).get("total_tokens", 0) or 0)
        if tok <= 0:
            return
        metrics.add_tokens(tok)
        kid = (principal or {}).get("key_id")
        if kid and kid != "demo":
            auth.record_tokens(kid, tok)
            daily_usage.add(kid, tok)
    except Exception:  # pragma: no cover - metering must never break a response
        pass


class _DailyUsage:
    """In-process per-key daily token counter (single-server cost-cap guard).

    Resets at process restart and at date rollover. For multi-replica deploys,
    move this to a shared store (Redis/DB) so the cap holds across instances.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._d: dict[str, list] = {}  # key_id -> [date_iso, tokens_today]

    @staticmethod
    def _today() -> str:
        import datetime

        return datetime.date.today().isoformat()

    def over_cap(self, key_id: str, cap: int) -> bool:
        if cap <= 0:  # 0 = unlimited
            return False
        with self._lock:
            rec = self._d.get(key_id)
            if not rec or rec[0] != self._today():
                return False
            return rec[1] >= cap

    def add(self, key_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        today = self._today()
        with self._lock:
            rec = self._d.get(key_id)
            if not rec or rec[0] != today:
                self._d[key_id] = [today, int(tokens)]
            else:
                rec[1] += int(tokens)


daily_usage = _DailyUsage()


def _enforce_daily_cap(principal: Optional[dict]) -> None:
    """Reject (429) if this key already hit its tier's daily token cap. Demo and
    capless tiers are skipped. Called BEFORE the LLM spend on cost endpoints."""
    kid = (principal or {}).get("key_id")
    if not kid or kid == "demo":
        return
    cap = config.DAILY_TOKEN_CAP_BY_TIER.get((principal or {}).get("tier", "free"), 0)
    if daily_usage.over_cap(kid, cap):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"일일 토큰 한도({cap:,})를 초과했습니다. 내일 다시 시도하거나 상위 티어를 이용하세요.",
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
                    "query": "의료광고 과장 금지",
                    "filter": {"doc_type": "law"},
                    "k": 5,
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
            "true이면 답변을 SSE(text/event-stream)로 토큰 단위 스트리밍. "
            "이벤트: meta → token* → done(검증된 citations 포함)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "무면허 의료행위 처벌은?",
                    "k": 8,
                }
            ]
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"citation": {"law_name": "의료법", "article_no": "제56조"}}
            ]
        }
    }


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


def demo_or_require_key(request: Request) -> dict:
    """Use a valid API key if present; else allow anonymous access in demo mode.

    Powers the no-login public test page: with ``config.DEMO_MODE`` on, a keyless
    call returns a synthetic free-tier principal that is NOT stashed on
    ``request.state`` — so the rate limiter buckets it per-IP (demo abuse guard)
    rather than into a single shared key bucket. With demo off, behaves exactly
    like ``require_key`` (401 without a valid key) — production consumers unchanged.
    """
    principal = optional_key(request)
    if principal:
        return principal
    if config.DEMO_MODE:
        return {"demo": True, "tier": "free", "key_id": "demo"}
    raise HTTPException(
        status_code=401,
        detail="API 키가 필요합니다 (Authorization: Bearer <키>).",
    )


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
@app.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["관리"],
    summary="상태확인 (인증 불필요)",
)
def healthz() -> HealthResponse:
    """서비스 가동 여부와 백엔드별 준비 상태를 반환합니다.

    개별 백엔드(FAISS 인덱스·RAG·검증 등)가 준비 중이어도 항상 200을 반환하므로
    로드밸런서 헬스체크에 사용할 수 있습니다. **인증이 필요 없습니다.**

    ``points``는 적재된 FAISS 인덱스의 벡터 수(``ntotal``)입니다.

    응답 예시::

        {"ok": true, "collection": "lawbot", "points": 1238122,
         "backends": {"rag": true, "retriever": true, "faiss": true, ...}}
    """
    return HealthResponse(
        ok=True,
        collection=config.COLLECTION,
        points=backends.index_ntotal,
        backends={
            "rag": backends.rag is not None,
            "retriever": backends.retriever is not None,
            "statutes": backends.statutes is not None,
            "verify": backends.verify is not None,
            "source_pack": backends.source_pack is not None,
            "embed_client": backends.embed_client is not None,
            "ad_review": backends.ad_review is not None,
            "faiss": backends.index_ok,
        },
    )


@app.get(
    "/metrics",
    tags=["관리"],
    summary="운영 메트릭 (인증 불필요)",
)
def metrics_endpoint() -> dict:
    """요청 수·상태코드 분포·평균 지연·인덱스 적재·캐시 적중 등 운영 관측 지표.

    프로세스 내 카운터(재시작 시 초기화)로, 단일 서버 운영 가시성을 제공합니다.
    민감정보(쿼리 본문·키)는 포함하지 않습니다. 공개 운영 시 내부망 제한을 권장합니다.
    """
    snap = metrics.snapshot()
    snap["index"] = {"ntotal": backends.index_ntotal, "ready": backends.index_ok}
    caches: dict[str, Any] = {}
    try:
        from search import verify as _v

        caches["law_api_cache_size"] = len(_v._LAW_CACHE)
    except Exception:
        pass
    try:
        from search import retriever as _r

        ci = _r._llm_rewrite.cache_info()
        caches["query_rewrite_cache"] = {
            "hits": ci.hits,
            "misses": ci.misses,
            "size": ci.currsize,
        }
    except Exception:
        pass
    snap["caches"] = caches
    return snap


# --------------------------------------------------------------------------- #
# Endpoints — core data-infra (lawbot.org-aligned, no generation cost)         #
# --------------------------------------------------------------------------- #
@app.post(
    "/v1/statutes/search",
    tags=["검색·검증"],
    summary="법령·판례 검색 (LLM 미사용)",
)
@limiter.limit(_dynamic_rate)
def v1_statutes_search(
    request: Request,
    body: QueryRequest,
    principal: Optional[dict] = Depends(optional_key),
) -> dict:
    """자연어 질의로 의료 법령·판례 원문을 정밀 검색합니다(LLM 미사용).

    **언제 쓰나** — AI 답변 없이 관련 조문·판례 원문만 빠르게 찾을 때.

    각 결과 행에는 공통 메타 ``{trust_grade, source_url, license, as_of_date,
    effective_from}`` 와 ``{doc_id, doc_type, title, article_no, score, text}`` 가
    함께 담깁니다. ``k`` 로 검색 개수, ``filter`` 로 문서유형/관할을 좁힐 수 있습니다.

    요청 예시::

        {"query": "의료광고 과장 금지", "k": 5}

    응답 예시::

        {"results": [{"title": "의료법", "article_no": "제56조",
                      "score": 0.82, "text": "..."}],
         "as_of_date": null}
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


@app.post(
    "/v1/verify",
    tags=["검색·검증"],
    summary="인용 검증 (Citation Firewall)",
)
@limiter.limit(_dynamic_rate)
def v1_verify(
    request: Request, body: VerifyRequest, principal: dict = Depends(demo_or_require_key)
) -> dict:
    """인용(법령 조문·판례)이 실재·현행인지 DB와 law.go.kr로 대조 검증합니다.

    **언제 쓰나** — AI나 외부 문서가 제시한 인용이 진짜인지, 폐지/오인용/허위사건은
    아닌지 확인할 때.

    단일 인용은 ``citation``, 여러 건은 ``citations`` 배열로 보냅니다. 각 결과는
    ``{verified, trust_grade, current, source_url, effective_from, as_of_date,
    note, db_match, api_match}`` 형태입니다.

    요청 예시::

        {"citation": {"law_name": "의료법", "article_no": "제56조"}}

    응답 예시::

        {"results": [{"verified": true, "current": true,
                      "law_name": "의료법", "article_no": "제56조"}]}
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


@app.post(
    "/v1/source-pack",
    tags=["검색·검증"],
    summary="근거 마크다운 번들",
)
@limiter.limit(_dynamic_rate)
def v1_source_pack(
    request: Request, body: QueryRequest, principal: dict = Depends(require_key)
) -> dict:
    """질의에 관련된 원문들을 인용 가능한 마크다운 번들로 묶어 반환합니다.

    **언제 쓰나** — 외부 LLM에 그대로 넣어 인용시킬 근거 묶음이 필요할 때(LLM 미사용,
    결과는 결정적).

    ``{markdown, sources: [...공통 메타], as_of_date}`` 를 반환합니다.

    요청 예시::

        {"query": "의료광고 사전심의 대상", "k": 5}
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


@app.post(
    "/v1/embeddings",
    tags=["검색·검증"],
    summary="임베딩 (내부 호환용)",
    include_in_schema=False,
)
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


@app.post(
    "/v1/ask",
    tags=["AI 답변"],
    summary="AI 법률 Q&A (근거·인용 답변)",
)
@limiter.limit(_dynamic_rate)
async def v1_ask(
    request: Request, body: AskRequest, principal: dict = Depends(demo_or_require_key)
):
    """법률 질의를 검색된 한국 법령·판례 원문에만 근거해 답변합니다.

    **언제 쓰나** — 근거와 인용이 달린 AI 답변이 필요할 때.

    유효한 API 키가 필요합니다(LLM 비용 발생). 인용은 RAG 백엔드가 사후검증하여
    환각 인용을 제거하며, 응답에는 면책고지(``disclaimer``)와 AI 생성 표시
    (``ai_generated``)가 포함됩니다.

    요청 예시::

        {"query": "무면허 의료행위 처벌은?"}

    응답 예시::

        {"answer": "의료법 제27조 위반 시 ...",
         "citations": [{"law_name": "의료법", "article_no": "제27조"}],
         "disclaimer": "...", "ai_generated": true}

    ``stream: true`` 로 보내면 저지연 **SSE**(``text/event-stream``)로 받습니다:
    ``meta`` → ``token`` 델타들 → 검증된 인용을 담은 ``done`` 이벤트. 기본은 JSON 응답입니다.
    """
    rag = _require("rag", "ask", "RAG")
    _enforce_daily_cap(principal)  # 일일 토큰 상한 초과 시 LLM 호출 전 429
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
    _record_cost(result, principal)
    return _ensure_notice(result)


@app.post(
    "/v1/ad-review",
    tags=["의료광고 검토"],
    summary="의료광고 PDF·문서 검토 (multipart)",
)
@limiter.limit(_dynamic_rate)
async def v1_ad_review(
    request: Request,
    principal: dict = Depends(demo_or_require_key),
    file: Optional[UploadFile] = File(
        default=None, description="검토할 광고 PDF 또는 텍스트 파일 (또는 text 필드 사용)"
    ),
    text: Optional[str] = Form(
        default=None, description="파일 대신 직접 입력하는 광고 문안 (file 또는 text 중 하나 필수)"
    ),
    question: Optional[str] = Form(
        default=None, description="검토 초점을 좁히는 선택 질문 (예: '효과 보장 표현만 봐줘')"
    ),
) -> dict:
    """의료광고 문안(PDF 업로드 또는 텍스트)을 검색된 법령에 비추어 검토합니다.

    **언제 쓰나** — 광고 카피·전단·홈페이지 문안의 의료법 위반 소지를 점검할 때.

    **요청 형식: ``multipart/form-data``** — 아래 중 하나를 보냅니다.

    * ``file`` : 광고 PDF(또는 텍스트) 파일 업로드, 또는
    * ``text`` : 광고 문안 텍스트를 직접 입력.
    * ``question`` (선택): 검토 초점을 좁히는 질문.

    위반/주의 항목(``issues``)과 근거 인용(``citations``)을 담은 전문가 검토 결과를
    반환하며, 면책고지(``disclaimer``)와 AI 생성 표시(``ai_generated``)가 포함됩니다.

    응답 예시::

        {"issues": [{"severity": "위반", "note": "치료 효과 보장 표현 ..."}],
         "citations": [{"law_name": "의료법", "article_no": "제56조"}],
         "disclaimer": "...", "ai_generated": true}
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
        _max = config.MAX_UPLOAD_BYTES
        file_bytes = await file.read(_max + 1)  # cap memory: read at most _max+1
        if len(file_bytes) > _max:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"업로드 파일이 너무 큽니다(최대 {_max // (1024 * 1024)}MB).",
            )
        filename = file.filename
    _enforce_daily_cap(principal)  # 일일 토큰 상한 초과 시 LLM 호출 전 429
    try:
        result = backends.ad_review(
            text=text, file_bytes=file_bytes, filename=filename, question=question
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ad_review failed: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Document-review backend error.") from exc
    _record_cost(result, principal)
    return _ensure_notice(result)


# --------------------------------------------------------------------------- #
# Endpoints — direct document lookup                                          #
# --------------------------------------------------------------------------- #
@app.get(
    "/v1/statutes/{law_id}/articles/{article_no}",
    tags=["검색·검증"],
    summary="조문 단건 조회",
    include_in_schema=False,
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


@app.get(
    "/v1/precedents/{seq}",
    tags=["검색·검증"],
    summary="판례 단건 조회",
    include_in_schema=False,
)
def v1_precedent(seq: str, principal: dict = Depends(require_key)) -> dict:
    """Return a single precedent by 판례일련번호, with its sections."""
    found = _find_precedent(seq)
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Precedent not found: 판례일련번호={seq}.")
    return found


# --------------------------------------------------------------------------- #
# Console (multi-tenant self-service page; actions use the key from the page)  #
# --------------------------------------------------------------------------- #
@app.get(
    "/console",
    response_class=HTMLResponse,
    summary="Self-service console",
    include_in_schema=False,
)
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


@app.get(
    "/chat",
    response_class=HTMLResponse,
    summary="Chat + PDF-review page",
    include_in_schema=False,
)
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


# --------------------------------------------------------------------------- #
# Public web pages: landing (/) · dashboard (/dashboard) · features (/features) #
# Served from web/*.html when present, else a shared minimal fallback. All are  #
# unauthenticated static pages; the interactive calls live on /chat + /v1/*.    #
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, summary="Landing page", include_in_schema=False)
def landing() -> HTMLResponse:
    """Serve the marketing/landing page (``web/index.html``)."""
    page = _WEB_DIR / "index.html"
    if page.is_file():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse(_LANDING_FALLBACK)


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Dashboard hub",
    include_in_schema=False,
)
def dashboard() -> HTMLResponse:
    """Serve the dashboard hub (``web/dashboard.html``)."""
    page = _WEB_DIR / "dashboard.html"
    if page.is_file():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse(_LANDING_FALLBACK)


@app.get(
    "/features",
    response_class=HTMLResponse,
    summary="Features page",
    include_in_schema=False,
)
def features() -> HTMLResponse:
    """Serve the features overview (``web/features.html``)."""
    page = _WEB_DIR / "features.html"
    if page.is_file():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse(_LANDING_FALLBACK)


_LANDING_FALLBACK = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>lawbot</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}</style></head>
<body><h1>lawbot — 한국 법률, 근거로 답하는 AI</h1>
<p>전체 법령·판례·행정규칙 원문에 근거해 답하고 인용을 검증합니다.</p>
<p><a href="/chat">법률 질문 시작</a> · <a href="/dashboard">대시보드</a> · <a href="/features">기능</a> · <a href="/docs">API 문서</a></p>
</body></html>"""


__all__ = ["app"]
