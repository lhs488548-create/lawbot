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

# --------------------------------------------------------------------------- #
# Embedding model (one place to switch small <-> large)                       #
# --------------------------------------------------------------------------- #
# small => 1536 dims, large => 3072 dims. EMBED_DIM MUST match the model and
# the Qdrant collection's vector size.
EMBED_MODEL: Final[str] = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM: Final[int] = _get_int("EMBED_DIM", 1536)
EMBED_MAX_TOKENS: Final[int] = _get_int("EMBED_MAX_TOKENS", 8191)
# tiktoken encoding used to measure chunk length (OpenAI embeddings use cl100k).
EMBED_ENCODING: Final[str] = os.getenv("EMBED_ENCODING", "cl100k_base")

# --------------------------------------------------------------------------- #
# Generation model (one place to switch mini <-> full)                        #
# --------------------------------------------------------------------------- #
GEN_MODEL: Final[str] = os.getenv("GEN_MODEL", "gpt-4o-mini")
# Optional stronger fallback for hard queries (Phase 5 escalation).
GEN_MODEL_FALLBACK: Final[str] = os.getenv("GEN_MODEL_FALLBACK", "gpt-4o")

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
# interface. Path is git-ignored (*.db).
API_KEYS_DB: Final[Path] = _PROJECT_ROOT / "lawbot_keys.db"

# Standard disclaimer appended to every /v1/ask answer. Expert (lawyer) mode:
# this is a provenance/AI-generation notice, NOT a consumer-style refusal.
ANSWER_DISCLAIMER: Final[str] = (
    "본 답변은 인덱싱된 한국 법령·판례·행정규칙·자치법규 원문을 근거로 "
    "AI가 생성한 정보이며, 인용된 출처를 직접 확인하시기 바랍니다. "
    "데이터 커버리지: 국가법령·판례·행정규칙·전국 자치법규(헌재 결정·법령해석례 등 일부 미포함)."
)


__all__ = [
    "OPENAI_API_KEY",
    "QDRANT_API_KEY",
    "LAW_OC",
    "EMBED_MODEL",
    "EMBED_DIM",
    "EMBED_MAX_TOKENS",
    "EMBED_ENCODING",
    "GEN_MODEL",
    "GEN_MODEL_FALLBACK",
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
