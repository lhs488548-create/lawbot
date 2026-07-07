"""Central configuration for lawbot.

All tunable constants live here so that switching the embedding model, the
generation model, the vector store, or the data root requires editing exactly
one file. Secrets (API keys) are *never* hard-coded — they are read from the
process environment, which is populated from a git-ignored ``.env`` file.

Owner: Contracts. Builders import from this module; they must not redefine
these constants in their own modules.

Usage (WSL venv, Python 3.12)::

    cd /home/user1/lawbot && .venv/bin/python -c "import config; print(config.EMBED_MODEL)"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load .env from the project root (the directory containing this file) so that
# the configuration is independent of the current working directory.
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    """Return a mandatory environment variable or raise a clear error.

    Args:
        name: Environment variable name.

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is missing or empty. The error message
            never includes the value, only the key name.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Add it to {_PROJECT_ROOT / '.env'} (never commit secrets)."
        )
    return value


def _get_int(name: str, default: int) -> int:
    """Read an integer environment override, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Environment variable {name!r} must be an integer.") from exc


# --------------------------------------------------------------------------- #
# Secrets (read from .env only — do not log, print, or commit)                #
# --------------------------------------------------------------------------- #
OPENAI_API_KEY: Final[str] = _require("OPENAI_API_KEY")
# Optional: present only when targeting Qdrant Cloud; None for local Docker.
QDRANT_API_KEY: Final[str | None] = os.getenv("QDRANT_API_KEY") or None
# Optional: law.go.kr OpenAPI OC, used by freshness/refresh tooling only.
LAW_OC: Final[str | None] = os.getenv("LAW_OC") or None
# Citation Firewall caches law.go.kr responses this many seconds (current law
# does not change intra-hour). Cuts repeat-verify latency + API load/rate-limit.
# 0 disables the cache. Env: LAWBOT_LAW_CACHE_TTL.
LAW_CACHE_TTL: Final[int] = _get_int("LAWBOT_LAW_CACHE_TTL", 3600)

# --------------------------------------------------------------------------- #
# Embedding model (one place to switch small <-> large)                       #
# --------------------------------------------------------------------------- #
# text-embedding-3-small natively returns 1536 dims, but OpenAI supports the
# ``dimensions`` request parameter (Matryoshka) to return a shortened, still-
# normalizable vector. This build pins **512** dims (smaller index, faster
# brute-force FAISS search, ample recall for the medical sub-corpus) and treats
# 512 as an *unbreakable invariant*: EMBED_DIM == EMBED_DIMENSIONS == the FAISS
# index vector size, always.
#
# IMPORTANT — env-override policy: 512 is fixed in code and is NOT read from the
# environment, so a stale ``EMBED_DIM=1536`` left in a developer ``.env`` can no
# longer silently corrupt the index. (A legacy ``EMBED_DIM`` that disagrees is
# detected and rejected below rather than honored.)
EMBED_MODEL: Final[str] = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM: Final[int] = 512
# Value passed to OpenAI's embeddings ``dimensions=`` parameter. Identical to
# EMBED_DIM by construction so the requested and validated dimensionality can
# never drift.
EMBED_DIMENSIONS: Final[int] = EMBED_DIM

# Guard: a legacy ``EMBED_DIM`` override may still linger in a developer ``.env``
# (e.g. ``EMBED_DIM=1536`` from the pre-FAISS build). The pinned 512 invariant
# always wins — we never honor the stale value — but we warn once so the operator
# removes it. (A hard raise is avoided here so importing ``config`` never breaks
# unrelated modules; the FAISS builders additionally assert dim==512 at build.)
_legacy_embed_dim = os.getenv("EMBED_DIM")
if _legacy_embed_dim not in (None, "", str(EMBED_DIM)):
    import warnings

    warnings.warn(
        f"Ignoring stale EMBED_DIM={_legacy_embed_dim!r} from the environment/.env; "
        f"this build pins EMBED_DIM={EMBED_DIM} (text-embedding-3-small @ "
        f"dimensions=512). Remove the EMBED_DIM line from .env to silence this.",
        RuntimeWarning,
        stacklevel=2,
    )

EMBED_MAX_TOKENS: Final[int] = _get_int("EMBED_MAX_TOKENS", 8191)
# tiktoken encoding used to measure chunk length (OpenAI embeddings use cl100k).
EMBED_ENCODING: Final[str] = os.getenv("EMBED_ENCODING", "cl100k_base")

# --------------------------------------------------------------------------- #
# Generation model (one place to switch mini <-> full)                        #
# --------------------------------------------------------------------------- #
GEN_MODEL: Final[str] = os.getenv("GEN_MODEL", "gpt-5-mini")
# Optional stronger fallback for hard queries (Phase 5 escalation).
GEN_MODEL_FALLBACK: Final[str] = os.getenv("GEN_MODEL_FALLBACK", "gpt-5")

# Reasoning effort for gpt-5 / o-series generation models. These models run a
# hidden reasoning phase BEFORE emitting any answer token; at the default
# "medium" that phase dominates answer latency (measured on this build:
# gpt-5-mini medium 7.6s vs minimal 1.6s for the SAME answer). "low" keeps light
# reasoning for legal synthesis while cutting most of that dead time; "minimal"
# is fastest. An EMPTY value disables the parameter entirely — required for
# non-reasoning models (e.g. gpt-4o-mini) which reject reasoning_effort with a
# 400. Env: GEN_REASONING_EFFORT.
GEN_REASONING_EFFORT: Final[str] = os.getenv("GEN_REASONING_EFFORT", "low").strip().lower()
# Effort for the colloquial→legal query REWRITE call (search.retriever._llm_rewrite).
# Rewrite is a light keyword-extraction task, so it defaults to "minimal" (faster /
# cheaper than the grounded answer generation). Env: GEN_REWRITE_EFFORT.
GEN_REWRITE_EFFORT: Final[str] = os.getenv("GEN_REWRITE_EFFORT", "minimal").strip().lower()


def reasoning_effort_kwargs(model: str, effort: str | None = None) -> dict[str, str]:
    """Return ``{"reasoning_effort": ...}`` for reasoning-capable models, else ``{}``.

    gpt-5 / o-series models accept ``reasoning_effort`` on
    ``chat.completions.create``; non-reasoning models (e.g. gpt-4o-mini) reject
    it with a 400. The parameter is therefore attached only when the model name
    indicates a reasoning model AND the resolved effort is non-empty. ``effort``
    overrides :data:`GEN_REASONING_EFFORT` for a specific call (e.g. the rewrite
    passes the lighter :data:`GEN_REWRITE_EFFORT`). Call sites splat the result:
    ``create(..., **reasoning_effort_kwargs(model))``.
    """
    eff = GEN_REASONING_EFFORT if effort is None else effort
    if eff and str(model).lower().startswith(("gpt-5", "o1", "o3", "o4")):
        return {"reasoning_effort": eff}
    return {}

# --------------------------------------------------------------------------- #
# Vector store (Qdrant)                                                       #
# --------------------------------------------------------------------------- #
QDRANT_URL: Final[str] = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION: Final[str] = os.getenv("COLLECTION", "lawbot")
# Retrieval defaults.
DEFAULT_TOP_K: Final[int] = _get_int("DEFAULT_TOP_K", 8)

# --------------------------------------------------------------------------- #
# Data roots (source corpora on disk)                                         #
# --------------------------------------------------------------------------- #
# DATA_ROOT points at the directory holding the four corpora. It is given as a
# POSIX path (WSL view). Windows tools access the same files via the UNC path
# \\wsl.localhost\Ubuntu\home\user1\... (see _BUILD_CONTRACT.md path bridge).
DATA_ROOT: Final[Path] = Path(
    os.getenv("DATA_ROOT", "/home/user1/체크/NEW2/원천데이터")
)

# Per-corpus roots. Builders glob within these.
LAW_DIR: Final[Path] = DATA_ROOT / "01_국가법령" / "kr"
ORDINANCE_DIR: Final[Path] = DATA_ROOT / "02_자치법규"
ADMRULE_DIR: Final[Path] = DATA_ROOT / "03_행정규칙"
PRECEDENT_DIR: Final[Path] = DATA_ROOT / "04_판례"

# --------------------------------------------------------------------------- #
# Artifacts (intermediate build outputs, git-ignored)                         #
# --------------------------------------------------------------------------- #
ARTIFACTS_DIR: Final[Path] = _PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Phase 1 parser outputs (one JSONL of Document records per corpus).
DOCS_LAW_JSONL: Final[Path] = ARTIFACTS_DIR / "docs_law.jsonl"
DOCS_PREC_JSONL: Final[Path] = ARTIFACTS_DIR / "docs_prec.jsonl"
DOCS_ADMRULE_JSONL: Final[Path] = ARTIFACTS_DIR / "docs_admrule.jsonl"
DOCS_ORD_JSONL: Final[Path] = ARTIFACTS_DIR / "docs_ord.jsonl"

# Phase 2 outputs.
CHUNKS_JSONL: Final[Path] = ARTIFACTS_DIR / "chunks.jsonl"
EMBEDDINGS_JSONL: Final[Path] = ARTIFACTS_DIR / "embeddings.jsonl"

# Content-hash embedding cache (09 §C: "content hash 캐시 — 같은 텍스트
# 재임베딩 금지"). A sidecar JSONL mapping ``content_hash -> vector`` so that
# re-running the build never re-embeds (and never re-bills) unchanged text. The
# hash is sha256 of the *normalized embedding text* (header + body). Git-ignored.
EMBED_CACHE_JSONL: Final[Path] = ARTIFACTS_DIR / "embed_cache.jsonl"

# --------------------------------------------------------------------------- #
# Parent / child two-layer chunking (09 §B-1)                                  #
# --------------------------------------------------------------------------- #
# child = the article / precedent-section (search & embedding unit).
# parent = the whole law / precedent (generation & source-pack unit).
# The chunk record's payload carries ``parent_id`` (== doc_id for MVP) so a
# child hit can be promoted to its parent's full text for answers/source-packs.
# Parents are materialized as a sidecar lookup keyed by parent_id.
PARENTS_JSONL: Final[Path] = ARTIFACTS_DIR / "parents.jsonl"

# --------------------------------------------------------------------------- #
# Medical sub-corpus + FAISS store (의료관련 빌드)                              #
# --------------------------------------------------------------------------- #
# The medical build materializes a filtered sub-corpus and a self-contained
# FAISS index under ``artifacts/의료관련/``. ``chunks_with_vectors.jsonl`` is the
# canonical, store-agnostic artifact (one record = chunk meta + its L2-normalized
# 512-d vector); both the FAISS index and the sqlite-vec export are built from it
# (see ``docs/_FAISS_BUILD_CONTRACT.md``). Windows tools reach the same files via
# the UNC path \\wsl.localhost\Ubuntu\home\user1\lawbot\artifacts\의료관련.
MED_DIR: Final[Path] = ARTIFACTS_DIR / "의료관련"
FAISS_DIR: Final[Path] = MED_DIR / "faiss"
# Canonical store-agnostic artifact: {chunk_id, doc_id, parent_id, text, payload,
# vector[512]} per line. FAISS and sqlite-vec are both loaded from this.
CHUNKS_VEC_JSONL: Final[Path] = MED_DIR / "chunks_with_vectors.jsonl"
# IndexFlatIP over L2-normalized 512-d vectors (inner product == cosine).
FAISS_INDEX: Final[Path] = FAISS_DIR / "index.faiss"
# Row-aligned sidecar: FAISS row i -> {chunk_id, doc_id, parent_id, text, payload}.
FAISS_META: Final[Path] = FAISS_DIR / "meta.jsonl"

# --------------------------------------------------------------------------- #
# Full-corpus index (전체 코퍼스, 저RAM 디스크-메타 서빙)                        #
# --------------------------------------------------------------------------- #
# Built by ``embed.build_full_index`` (memory-safe streaming join; B-grade
# excluded). Served with ON-DISK meta (per-row offset lookup) so the ~3GB
# meta.jsonl is NOT held in RAM — only the ~2.5GB flat index is resident, which
# fits a small (8GB) box. Selected via env LAWBOT_INDEX=full; default "medical"
# keeps the small in-RAM index (rollback-safe, unchanged behavior).
FULL_FAISS_DIR: Final[Path] = ARTIFACTS_DIR / "full_index"
FULL_FAISS_INDEX: Final[Path] = FULL_FAISS_DIR / "index.faiss"
FULL_FAISS_META: Final[Path] = FULL_FAISS_DIR / "meta.jsonl"
# "medical" (default; small in-RAM index) | "full" (full corpus; disk-backed meta)
ACTIVE_INDEX: Final[str] = os.getenv("LAWBOT_INDEX", "medical").strip().lower()

# Pre-warm the FAISS index at process startup (api.main lifespan) in a background
# thread, so the first user query does not pay the lazy cold-load (measured: full
# index ~46s in Docker; medical ~0.7s). /healthz still reports live immediately;
# the warm runs off the request path. Env: LAWBOT_PREWARM (default on). Set 0 to
# disable (e.g. fast dev restarts where lazy-load-on-first-query is acceptable).
PREWARM_INDEX: Final[bool] = os.getenv("LAWBOT_PREWARM", "1") == "1"

# Hybrid retrieval: BM25 (SQLite FTS5 unicode61 + prefix queries) fused with the
# dense FAISS ranks via RRF. BM25 fixes dense's weak spots — colloquial phrasing
# and exact law-name/article lookup ("민법 제4조"). Built by ``embed.build_bm25``.
# Off by default (env LAWBOT_HYBRID=1) so A/B measurement is explicit; the gate
# (MIN_RETRIEVAL_SCORE) still reads the true dense cosine, not the RRF score.
BM25_DB: Final[Path] = FULL_FAISS_DIR / "bm25.sqlite"
HYBRID_SEARCH: Final[bool] = os.getenv("LAWBOT_HYBRID", "0") == "1"
RRF_K0: Final[int] = _get_int("RRF_K0", 60)
HYBRID_CANDIDATES: Final[int] = _get_int("HYBRID_CANDIDATES", 50)
# Weighted RRF for the TARGETED hybrid: BM25 is fused only for citation/law-name
# queries (see retriever._is_citation_like), where it is precise, so a strong
# weight (2.0) lets the exact cited article win without touching colloquial
# queries. Tuned on the golden set: dense 61.3% -> 64.5% Hit@5, MRR 0.467->0.516,
# Article-hit 33%->67%, zero regression on formal queries.
RRF_W_BM25: Final[float] = float(os.getenv("RRF_W_BM25", "2.0"))

# Query normalization: expand colloquial terms to their legal equivalents
# (월급→임금, 집주인→임대인 …) before embedding + BM25, closing the
# colloquial↔statute vocabulary gap that neither dense nor BM25 bridges alone.
# Free (dictionary-based; no per-query LLM call). Env LAWBOT_QNORM=1.
QUERY_NORM: Final[bool] = os.getenv("LAWBOT_QNORM", "0") == "1"

# LLM query rewrite: turn colloquial questions ("월급 떼였는데 어떻게 받아요?")
# into legal search terms ("근로기준법 임금 체불 지급") before embed + BM25. This is
# the lever for the colloquial↔statute gap that dictionary normalization couldn't
# close. Adds one (cheap, cached) LLM call per *distinct* query at serve time —
# a per-query cost, so it is opt-in (env LAWBOT_REWRITE=1).
QUERY_REWRITE: Final[bool] = os.getenv("LAWBOT_REWRITE", "0") == "1"

# Demo mode: allow keyless (anonymous) access to /v1/ask and /v1/ad-review so the
# public test page works without login. Anonymous calls are IP rate-limited
# (free tier). Off by default — production API consumers still need a key.
DEMO_MODE: Final[bool] = os.getenv("LAWBOT_DEMO", "0") == "1"

# Max upload size for /v1/ad-review (memory-exhaustion DoS guard, audit). The
# endpoint reads at most this many bytes and returns 413 beyond it.
MAX_UPLOAD_BYTES: Final[int] = _get_int("LAWBOT_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)

# Per-key DAILY LLM-token cap by tier (cost-blowup guard, audit P7). 0 = no cap.
# Tracked in-process (single-server); resets at restart and at date rollover.
# For multi-replica, move to a shared store (Redis/DB).
DAILY_TOKEN_CAP_BY_TIER: Final[dict[str, int]] = {
    "free": _get_int("LAWBOT_DAILY_CAP_FREE", 500_000),
    "pro": _get_int("LAWBOT_DAILY_CAP_PRO", 5_000_000),
    "enterprise": _get_int("LAWBOT_DAILY_CAP_ENTERPRISE", 0),
    "admin": 0,
}

# Second-pass split window/overlap (09 §B-2). Only applied to chunks exceeding
# EMBED_MAX_TOKENS (mostly long precedent sections).
CHUNK_WINDOW_TOKENS: Final[int] = _get_int("CHUNK_WINDOW_TOKENS", 1000)
CHUNK_OVERLAP_TOKENS: Final[int] = _get_int("CHUNK_OVERLAP_TOKENS", 200)

# --------------------------------------------------------------------------- #
# Provenance / licensing (09 §A: common response meta)                         #
# --------------------------------------------------------------------------- #
# Every API response carries {trust_grade, source_url, license, as_of_date,
# effective_from}. The corpus body is Korean public-domain legal text (저작권법
# §7: statutes/rulings are not copyrightable); attribution to law.go.kr is kept.
DEFAULT_LICENSE: Final[str] = os.getenv(
    "DEFAULT_LICENSE",
    "공공저작물(저작권법 §7) · 출처표기: law.go.kr",
)

# law.go.kr OpenAPI base (Citation Firewall /v1/verify, freshness checks). The
# OC token is LAW_OC above; never embed it in logs/artifacts.
LAW_API_BASE: Final[str] = os.getenv("LAW_API_BASE", "https://www.law.go.kr/DRF")

# --------------------------------------------------------------------------- #
# Search / RAG knobs (09 §E)                                                   #
# --------------------------------------------------------------------------- #
# When a child hit is promoted, how many parents to include in a source-pack.
SOURCE_PACK_MAX_PARENTS: Final[int] = _get_int("SOURCE_PACK_MAX_PARENTS", 6)
# Minimum dense score below which retrieval is treated as "근거 불충분"
# (insufficient grounding) and RAG returns an empty-citation honest answer.
MIN_RETRIEVAL_SCORE: Final[float] = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.20"))

# Demo-embedding safety cap (09 §G: "데모 임베딩은 ≤2만 청크"). Builders that
# embed a subset for end-to-end smoke tests must not exceed this without an
# explicit human cost-gate confirmation.
DEMO_MAX_CHUNKS: Final[int] = _get_int("DEMO_MAX_CHUNKS", 20000)

# --------------------------------------------------------------------------- #
# API key store (multi-tenant) — Phase 4                                      #
# --------------------------------------------------------------------------- #
# MVP store is SQLite; production swaps to Postgres/Redis behind the same auth
# interface. Path is git-ignored (*.db). Honors the ``API_KEYS_DB`` env var so the
# container/compose can point it at a writable mounted volume (e.g.
# ``/app/data/lawbot_keys.db``); defaults to the project root for local dev.
API_KEYS_DB: Final[Path] = Path(os.getenv("API_KEYS_DB") or (_PROJECT_ROOT / "lawbot_keys.db"))

# Standard disclaimer appended to every /v1/ask answer. Expert (lawyer) mode:
# this is a provenance/AI-generation notice, NOT a consumer-style refusal.
ANSWER_DISCLAIMER: Final[str] = (
    "본 답변은 인덱싱된 한국 법령·행정규칙·판례 원문을 근거로 "
    "AI가 생성한 정보이며, 인용된 출처를 직접 확인하시기 바랍니다. "
    "데이터 커버리지: 국가법령·행정규칙·판례(자치법규/조례·헌재 결정·법령해석례 등 일부 미포함)."
)


__all__ = [
    "OPENAI_API_KEY",
    "QDRANT_API_KEY",
    "LAW_OC",
    "LAW_CACHE_TTL",
    "EMBED_MODEL",
    "EMBED_DIM",
    "EMBED_DIMENSIONS",
    "EMBED_MAX_TOKENS",
    "EMBED_ENCODING",
    "GEN_MODEL",
    "GEN_MODEL_FALLBACK",
    "GEN_REASONING_EFFORT",
    "GEN_REWRITE_EFFORT",
    "reasoning_effort_kwargs",
    "QDRANT_URL",
    "COLLECTION",
    "DEFAULT_TOP_K",
    "DATA_ROOT",
    "LAW_DIR",
    "ORDINANCE_DIR",
    "ADMRULE_DIR",
    "PRECEDENT_DIR",
    "ARTIFACTS_DIR",
    "DOCS_LAW_JSONL",
    "DOCS_PREC_JSONL",
    "DOCS_ADMRULE_JSONL",
    "DOCS_ORD_JSONL",
    "CHUNKS_JSONL",
    "EMBEDDINGS_JSONL",
    "EMBED_CACHE_JSONL",
    "PARENTS_JSONL",
    "MED_DIR",
    "FAISS_DIR",
    "CHUNKS_VEC_JSONL",
    "FAISS_INDEX",
    "FAISS_META",
    "FULL_FAISS_DIR",
    "FULL_FAISS_INDEX",
    "FULL_FAISS_META",
    "ACTIVE_INDEX",
    "PREWARM_INDEX",
    "BM25_DB",
    "HYBRID_SEARCH",
    "RRF_K0",
    "HYBRID_CANDIDATES",
    "RRF_W_BM25",
    "QUERY_NORM",
    "QUERY_REWRITE",
    "DEMO_MODE",
    "MAX_UPLOAD_BYTES",
    "DAILY_TOKEN_CAP_BY_TIER",
    "CHUNK_WINDOW_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "DEFAULT_LICENSE",
    "LAW_API_BASE",
    "SOURCE_PACK_MAX_PARENTS",
    "MIN_RETRIEVAL_SCORE",
    "DEMO_MAX_CHUNKS",
    "API_KEYS_DB",
    "ANSWER_DISCLAIMER",
]
