"""OpenAI embedding wrapper + content-hash cache (09 §C).

A thin, production-grade wrapper around the OpenAI embeddings endpoint, shared
by the retriever, the small/sync indexing path, and the OpenAI-compatible
``/v1/embeddings`` API surface. It enforces the project's cost rules:

* **Only** ``config.EMBED_MODEL`` (``text-embedding-3-small``) is used — the
  ``large`` model is forbidden (budget rule, 09 §C / §G).
* Every OpenAI embeddings request passes ``dimensions=config.EMBED_DIMENSIONS``
  (=512, Matryoshka truncation). Every returned vector has
  ``len == config.EMBED_DIM`` (=512); a mismatch raises immediately.
* **Content-hash cache (mandatory, 09 §C):** the cache key is the SHA-256 of the
  *normalized embedding text*. Text already embedded (same hash) is never
  re-embedded and never re-billed. The cache is a git-ignored sidecar JSONL at
  ``config.EMBED_CACHE_JSONL`` mapping ``content_hash -> vector``.
* All network calls are wrapped with ``tenacity`` exponential-backoff retry.

Public interface (BUILD CONTRACT (d))::

    def embed_texts(texts: list[str]) -> list[list[float]]: ...
    def embed_batch(texts: list[str]) -> list[list[float]]: ...
    def content_hash(text: str) -> str: ...
    def cached_embed(items: list[dict]) -> dict[str, list[float]]: ...

Owner: embed builder. Imports shared constants from ``config`` and never
redefines them.

Run a tiny self-check (1 real OpenAI call, two sentences) to verify dim=512::

    cd /home/user1/lawbot && .venv/bin/python -m embed.embed_client --selfcheck
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Iterable, Iterator

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import config

# --------------------------------------------------------------------------- #
# Tunables                                                                     #
# --------------------------------------------------------------------------- #
# OpenAI's embeddings endpoint accepts up to 2048 inputs per request. We batch
# the sync path at this size to minimize round-trips while staying within limit.
_SYNC_BATCH_SIZE: Final[int] = 2048

# Retry policy for transient OpenAI errors (rate limits, 5xx, connection drops).
# Bounded attempts with jittered exponential backoff so a flaky network does not
# stall the build indefinitely.
_RETRY_ATTEMPTS: Final[int] = 6
_RETRY_MAX_WAIT: Final[float] = 30.0


# --------------------------------------------------------------------------- #
# OpenAI client (lazy singleton)                                              #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Return a process-wide cached OpenAI client.

    The API key is read from ``config.OPENAI_API_KEY`` (which itself reads it
    from the git-ignored ``.env``). The key is never logged or printed.

    Returns:
        A configured :class:`openai.OpenAI` client.
    """
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _is_retryable(exc: BaseException) -> bool:
    """Return ``True`` for transient OpenAI/network errors worth retrying.

    We retry rate-limit, connection, timeout, and 5xx server errors, but not
    client errors such as authentication or malformed-request (those would never
    succeed on retry and must surface immediately).

    Args:
        exc: The raised exception.

    Returns:
        Whether the call should be retried.
    """
    # Imported lazily so the module imports even if the SDK layout shifts.
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except Exception:  # pragma: no cover - defensive
        return False
    return isinstance(
        exc,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    )


_retry = retry(
    reraise=True,
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=wait_random_exponential(multiplier=1, max=_RETRY_MAX_WAIT),
    retry=retry_if_exception(_is_retryable),
)


_ENC = None


def _token_encoder():
    """Return a cached tiktoken encoder for request-size budgeting."""
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


# --------------------------------------------------------------------------- #
# Normalization + content hashing (cache key, 09 §C)                          #
# --------------------------------------------------------------------------- #
def _normalize_for_hash(text: str) -> str:
    """Normalize text so logically-identical inputs share one cache key.

    The cache key is the SHA-256 of the *normalized* embedding text (09 §C).
    Normalization is Unicode **NFC** + stripped surrounding whitespace, matching
    the normalization applied to the embedding text itself, so the same content
    is never re-embedded merely because of an encoding or trailing-whitespace
    difference. Internal text is otherwise left intact (the header/body text the
    chunker built is already the embedding text).

    Args:
        text: Raw embedding text.

    Returns:
        The NFC-normalized, stripped text.
    """
    return unicodedata.normalize("NFC", text or "").strip()


def content_hash(text: str) -> str:
    """Return the SHA-256 content hash used as the embedding cache key.

    Args:
        text: The embedding text (header + body). Normalized (NFC, stripped)
            before hashing so trivially-different encodings collide correctly.

    Returns:
        A 64-character lowercase hex SHA-256 digest.
    """
    normalized = _normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Content-hash cache (sidecar JSONL: {content_hash, vector})                   #
# --------------------------------------------------------------------------- #
_CACHE_LOCK: Final[threading.Lock] = threading.Lock()


def _validate_dim(vector: list[float]) -> list[float]:
    """Assert a vector matches ``config.EMBED_DIM`` and return it.

    Args:
        vector: An embedding vector.

    Returns:
        The same vector.

    Raises:
        ValueError: If the dimensionality does not match ``config.EMBED_DIM``.
    """
    if len(vector) != config.EMBED_DIM:
        raise ValueError(
            f"Embedding dim mismatch: got {len(vector)}, "
            f"expected {config.EMBED_DIM} (model={config.EMBED_MODEL}). "
            f"Check EMBED_MODEL/EMBED_DIM alignment in config.py."
        )
    return vector


def load_cache(path: Path | None = None) -> dict[str, list[float]]:
    """Load the content-hash → vector cache from its JSONL sidecar.

    Malformed or dimension-mismatched lines are skipped (the cache is an
    optimization, never a correctness dependency). Later lines win on duplicate
    hashes, which lets the cache be append-only.

    Args:
        path: Cache file path. Defaults to ``config.EMBED_CACHE_JSONL``.

    Returns:
        Mapping from ``content_hash`` to its embedding vector.
    """
    path = path or config.EMBED_CACHE_JSONL
    cache: dict[str, list[float]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                h = rec["content_hash"]
                vec = rec["vector"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if isinstance(vec, list) and len(vec) == config.EMBED_DIM:
                cache[h] = vec
    return cache


def append_cache(
    new_items: Iterable[tuple[str, list[float]]],
    path: Path | None = None,
) -> int:
    """Append new ``(content_hash, vector)`` pairs to the cache JSONL.

    Append-only and process-safe within this interpreter (guarded by a lock).
    Vectors are dimension-checked before being written.

    Args:
        new_items: Iterable of ``(content_hash, vector)`` pairs to persist.
        path: Cache file path. Defaults to ``config.EMBED_CACHE_JSONL``.

    Returns:
        The number of records appended.
    """
    path = path or config.EMBED_CACHE_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with _CACHE_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            for h, vec in new_items:
                _validate_dim(vec)
                fh.write(json.dumps({"content_hash": h, "vector": vec}))
                fh.write("\n")
                written += 1
    return written


# --------------------------------------------------------------------------- #
# Core sync embedding (uncached) — the single OpenAI touch-point              #
# --------------------------------------------------------------------------- #
#: OpenAI embeddings hard cap is 300_000 tokens per request; stay safely under.
_MAX_TOKENS_PER_REQUEST = 280_000


def _batched(seq: list[str], size: int) -> Iterator[list[str]]:
    """Yield batches bounded by BOTH item count (``size``) and token budget.

    OpenAI's embeddings endpoint rejects any single request exceeding
    300_000 tokens (``max_tokens_per_request``). Slicing purely by item count
    blows past that when inputs are long (e.g. precedent sections), so we also
    cap each batch at ``_MAX_TOKENS_PER_REQUEST`` cumulative tokens. A lone
    item longer than the budget is still emitted on its own (the chunker keeps
    every chunk under the 8_191-token model limit, so one item always fits).
    """
    try:
        enc = _token_encoder()
    except Exception:  # pragma: no cover - tiktoken always available here
        enc = None
    batch: list[str] = []
    tok = 0
    for s in seq:
        n = len(enc.encode(s)) if enc is not None else max(1, len(s) // 3)
        if batch and (len(batch) >= size or tok + n > _MAX_TOKENS_PER_REQUEST):
            yield batch
            batch, tok = [], 0
        batch.append(s)
        tok += n
    if batch:
        yield batch


@_retry
def _embed_request(inputs: list[str]) -> list[list[float]]:
    """Issue one OpenAI embeddings request (retry-wrapped).

    Args:
        inputs: 1..=2048 non-empty strings to embed in a single call.

    Returns:
        Vectors aligned to ``inputs`` order (OpenAI guarantees index alignment;
        we additionally sort by ``index`` defensively).

    Raises:
        ValueError: If any returned vector has the wrong dimensionality.
    """
    # ``dimensions=512`` (Matryoshka) is MANDATORY: text-embedding-3-small returns
    # 1536d natively, but this build pins 512 (config.EMBED_DIMENSIONS) so the
    # vectors match the FAISS IndexFlatIP(512). _validate_dim then asserts 512.
    resp = _client().embeddings.create(
        model=config.EMBED_MODEL,
        input=inputs,
        dimensions=config.EMBED_DIMENSIONS,
    )
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [_validate_dim(list(d.embedding)) for d in ordered]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts synchronously, batched at <=2048 per request.

    This is the sync / small / incremental path (09 §C). It does **not** consult
    the content-hash cache (use :func:`cached_embed` for cache-aware embedding);
    it always calls OpenAI for every input. Empty input returns an empty list.

    Args:
        texts: Texts to embed. Each must be a non-empty string under the model's
            token limit (the chunker guarantees the limit; callers embedding raw
            queries pass short strings).

    Returns:
        One ``config.EMBED_DIM``-length vector per input, in input order.

    Raises:
        ValueError: If ``texts`` contains a non-string or empty element, or a
            returned vector has the wrong dimensionality.
    """
    if not texts:
        return []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"texts[{i}] must be a non-empty string")
    out: list[list[float]] = []
    for batch in _batched(texts, _SYNC_BATCH_SIZE):
        out.extend(_embed_request(batch))
    return out


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Alias for :func:`embed_texts` (batched synchronous form).

    Provided for the contract's ``embed_batch`` name. For the asynchronous,
    cost-gated *OpenAI Batch API* path (full-corpus, 50% cheaper), see
    :mod:`embed.embed_batch`.

    Args:
        texts: Texts to embed.

    Returns:
        One vector per input, in input order.
    """
    return embed_texts(texts)


# --------------------------------------------------------------------------- #
# Cache-aware embedding                                                        #
# --------------------------------------------------------------------------- #
def cached_embed(items: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Embed items, reusing the content-hash cache to avoid re-billing (09 §C).

    Only cache-*misses* are sent to OpenAI; freshly computed vectors are appended
    to the cache so a later run re-uses them. Identical texts within the same
    call are de-duplicated by their hash, so duplicate content is embedded once.

    Args:
        items: A list of dicts, each with at least ``chunk_id`` and ``text``.
            An optional ``content_hash`` is trusted if present; otherwise it is
            computed from ``text``.

    Returns:
        Mapping ``chunk_id -> vector``. Every input ``chunk_id`` is present.

    Raises:
        KeyError: If an item lacks ``chunk_id`` or ``text``.
        ValueError: If a returned vector has the wrong dimensionality.
    """
    if not items:
        return {}

    cache = load_cache()

    # Map each unique content_hash to the text and the chunk_ids needing it.
    hash_to_text: dict[str, str] = {}
    hash_to_ids: dict[str, list[str]] = {}
    id_to_hash: dict[str, str] = {}
    for it in items:
        cid = it["chunk_id"]
        text = it["text"]
        h = it.get("content_hash") or content_hash(text)
        id_to_hash[cid] = h
        hash_to_text.setdefault(h, text)
        hash_to_ids.setdefault(h, []).append(cid)

    # Determine which hashes are missing from the cache.
    missing_hashes = [h for h in hash_to_text if h not in cache]
    if missing_hashes:
        miss_texts = [hash_to_text[h] for h in missing_hashes]
        vectors = embed_texts(miss_texts)
        fresh = list(zip(missing_hashes, vectors))
        for h, vec in fresh:
            cache[h] = vec
        append_cache(fresh)

    return {cid: cache[id_to_hash[cid]] for cid in id_to_hash}


# --------------------------------------------------------------------------- #
# Self-check (1 real OpenAI call — sanctioned by the test convention)         #
# --------------------------------------------------------------------------- #
def _selfcheck() -> int:
    """Embed two sentences with a real call and verify dim=EMBED_DIM (512).

    Returns:
        Process exit code (0 on success, 1 on failure). Prints a concise report;
        never prints the API key.
    """
    sentences = [
        "민법 제4조는 사람은 19세로 성년에 이른다고 규정한다.",
        "도로교통법은 운전자의 속도와 통행 방법을 규율한다.",
    ]
    vecs = embed_texts(sentences)
    ok = (
        len(vecs) == 2
        and all(len(v) == config.EMBED_DIM for v in vecs)
        and vecs[0] != vecs[1]
    )
    print(
        f"selfcheck: model={config.EMBED_MODEL} n={len(vecs)} "
        f"dim={len(vecs[0]) if vecs else 'NA'} expected={config.EMBED_DIM} "
        f"distinct={vecs[0] != vecs[1] if vecs else 'NA'} -> {'OK' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        raise SystemExit(_selfcheck())
    print(
        "embed_client: import this module; run with --selfcheck for a "
        "1-call dim verification."
    )
