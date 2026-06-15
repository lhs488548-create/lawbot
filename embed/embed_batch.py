"""OpenAI Batch API embedding (full-corpus, cost-gated) — 09 §C / §G, Task 2.2.

The cheap (50% off, async, 24h window) path for embedding the whole corpus. It
reads ``config.CHUNKS_JSONL``, skips chunks whose ``content_hash`` is already in
the content-hash cache (``config.EMBED_CACHE_JSONL``), and embeds only the
misses. The flow is **estimate → human confirm → submit → poll → collect**:

* :func:`build_batch_input` — write sharded ``batch_in*.jsonl`` (cache-misses
  only), each shard <= 50,000 lines / <= 200 MB (OpenAI Batch limits).
* :func:`estimate_cost` — total tokens + estimated USD + cached/miss counts.
* :func:`submit` — **refuses unless ``confirm=True``** (mandatory cost gate, 09
  §G). Full-corpus embedding is **never auto-run**. Demo runs cap at
  ``config.DEMO_MAX_CHUNKS``.
* :func:`collect` — poll batches, download outputs, write
  ``config.EMBEDDINGS_JSONL`` (``{chunk_id, vector}``) and update the cache.

Hard rules honored: only ``config.EMBED_MODEL`` (small, 1536d); output dim must
equal ``config.EMBED_DIM``; the API key is never printed/logged/written to an
artifact; same content is never re-embedded (cache).

Owner: embed builder. Run as a script (estimate is always safe; submit needs an
explicit ``--confirm`` flag)::

    cd /home/user1/lawbot && .venv/bin/python -m embed.embed_batch estimate
    cd /home/user1/lawbot && .venv/bin/python -m embed.embed_batch submit --confirm --max-chunks 20000
    cd /home/user1/lawbot && .venv/bin/python -m embed.embed_batch collect <batch_id> [<batch_id> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Iterator

import tiktoken

import config
from embed.embed_client import (
    append_cache,
    content_hash,
    load_cache,
    _client,
    _validate_dim,
)

# --------------------------------------------------------------------------- #
# Batch limits + pricing (09 §C/§G)                                           #
# --------------------------------------------------------------------------- #
# OpenAI Batch hard limits per input file.
_MAX_LINES_PER_SHARD: Final[int] = 50_000
_MAX_BYTES_PER_SHARD: Final[int] = 190 * 1024 * 1024  # 190MB headroom under 200MB

# text-embedding-3-small list price (USD per 1M tokens). The Batch API applies a
# 50% discount. This is used for the *estimate only*; the real bill comes from
# OpenAI usage. The price is read via ``_price_per_1m`` (env-overridable with
# ``EMBED_PRICE_PER_1M``) so pricing changes need no code edit.
_DEFAULT_PRICE_PER_1M: Final[float] = 0.02
_BATCH_DISCOUNT: Final[float] = 0.5

# Poll cadence when waiting for batches to finish (seconds).
_POLL_INTERVAL_SEC: Final[float] = 30.0

_BATCH_IN_PREFIX: Final[str] = "batch_in"
_BATCH_MANIFEST: Final[Path] = config.ARTIFACTS_DIR / "batch_manifest.json"


@lru_cache(maxsize=1)
def _encoder() -> "tiktoken.Encoding":
    """Return the cached tiktoken encoder for ``config.EMBED_ENCODING``."""
    return tiktoken.get_encoding(config.EMBED_ENCODING)


def _price_per_1m() -> float:
    """Return the per-1M-token list price (env-overridable)."""
    import os

    raw = os.getenv("EMBED_PRICE_PER_1M")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return _DEFAULT_PRICE_PER_1M


# --------------------------------------------------------------------------- #
# Reading chunks                                                              #
# --------------------------------------------------------------------------- #
def _iter_chunks(
    path: Path | None = None,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream chunk records from ``config.CHUNKS_JSONL``.

    Each yielded chunk is augmented with a ``content_hash`` (computed from its
    ``text`` if absent) so downstream stages share one cache key.

    Args:
        path: Chunks JSONL path. Defaults to ``config.CHUNKS_JSONL``.
        limit: Optional cap on the number of chunks yielded (demo runs).

    Yields:
        Chunk dicts with guaranteed ``chunk_id``, ``text`` and ``content_hash``.
    """
    path = path or config.CHUNKS_JSONL
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks artifact not found: {path}. Run `python -m embed.chunk` first."
        )
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "content_hash" not in c:
                c["content_hash"] = content_hash(c["text"])
            yield c
            n += 1
            if limit is not None and n >= limit:
                return


def _select_misses(
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return cache-miss chunks (de-duplicated by content hash) to embed.

    Chunks whose ``content_hash`` is already cached are skipped (never re-billed,
    09 §C). Within the run, identical hashes collapse to a single embed request;
    the resulting vector is later fanned back out to all sharing ``chunk_id``s by
    :func:`collect`.

    Args:
        limit: Optional cap on chunks scanned (demo runs respect this *before*
            cache filtering, mirroring how a demo subset is selected).

    Returns:
        ``(misses, n_total, n_cached)`` where ``misses`` is one record per unique
        missing hash: ``{content_hash, text, chunk_ids: [...]}``.
    """
    cache = load_cache()
    by_hash: dict[str, dict[str, Any]] = {}
    n_total = 0
    n_cached = 0
    for c in _iter_chunks(limit=limit):
        n_total += 1
        h = c["content_hash"]
        if h in cache:
            n_cached += 1
            continue
        rec = by_hash.get(h)
        if rec is None:
            by_hash[h] = {"content_hash": h, "text": c["text"], "chunk_ids": [c["chunk_id"]]}
        else:
            rec["chunk_ids"].append(c["chunk_id"])
    return list(by_hash.values()), n_total, n_cached


# --------------------------------------------------------------------------- #
# Cost estimate (always safe — no network, no billing)                        #
# --------------------------------------------------------------------------- #
def estimate_cost(limit: int | None = None) -> dict[str, Any]:
    """Estimate token total and USD cost for embedding the cache-miss chunks.

    Pure local computation (tiktoken) — never calls OpenAI and never bills. This
    is the mandatory pre-submission cost gate's data source (09 §G).

    Args:
        limit: Optional cap on chunks considered (demo subset).

    Returns:
        ``{n_chunks, n_cached, n_unique_misses, total_tokens, est_usd,
           price_per_1m, batch_discount, model}``. ``n_chunks`` is the number of
        chunks scanned; ``est_usd`` reflects the **Batch** (discounted) price for
        only the unique cache-misses.
    """
    misses, n_total, n_cached = _select_misses(limit=limit)
    enc = _encoder()
    total_tokens = 0
    for rec in misses:
        total_tokens += len(enc.encode(rec["text"]))
    price = _price_per_1m()
    est_usd = (total_tokens / 1_000_000) * price * _BATCH_DISCOUNT
    return {
        "model": config.EMBED_MODEL,
        "n_chunks": n_total,
        "n_cached": n_cached,
        "n_unique_misses": len(misses),
        "total_tokens": total_tokens,
        "price_per_1m": price,
        "batch_discount": _BATCH_DISCOUNT,
        "est_usd": round(est_usd, 4),
    }


def _print_estimate(est: dict[str, Any]) -> None:
    """Print the cost estimate as a human-readable gate banner."""
    print("=" * 64)
    print("OpenAI Batch embedding — COST ESTIMATE (no billing yet)")
    print("-" * 64)
    print(f"  model              : {est['model']}")
    print(f"  chunks scanned     : {est['n_chunks']:,}")
    print(f"  already cached     : {est['n_cached']:,}  (re-embed = $0)")
    print(f"  unique to embed    : {est['n_unique_misses']:,}")
    print(f"  total input tokens : {est['total_tokens']:,}")
    print(
        f"  price              : ${est['price_per_1m']}/1M tokens "
        f"x {est['batch_discount']:.0%} batch discount"
    )
    print(f"  ESTIMATED COST     : ${est['est_usd']:.4f}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Build sharded batch input files (cache-misses only)                         #
# --------------------------------------------------------------------------- #
def build_batch_input(limit: int | None = None) -> list[Path]:
    """Write sharded OpenAI Batch input JSONL files for cache-miss chunks.

    Each line is an embeddings request keyed by the chunk's **content hash** as
    ``custom_id`` (so identical text is embedded once and fanned out on
    collection). Shards respect the 50,000-line / ~200MB Batch limits.

    Args:
        limit: Optional cap on chunks considered (demo subset).

    Returns:
        Paths of the shard files written (empty if nothing to embed). A manifest
        mapping ``content_hash -> chunk_ids`` is also written so :func:`collect`
        can fan vectors back out to every chunk.

    Raises:
        FileNotFoundError: If the chunks artifact is missing.
    """
    misses, _, _ = _select_misses(limit=limit)
    # Clear stale shards.
    for old in config.ARTIFACTS_DIR.glob(f"{_BATCH_IN_PREFIX}*.jsonl"):
        old.unlink()

    shard_paths: list[Path] = []
    manifest: dict[str, list[str]] = {}
    if not misses:
        _BATCH_MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
        return shard_paths

    shard_idx = 0
    lines_in_shard = 0
    bytes_in_shard = 0
    fh = None

    def _open_shard(idx: int):
        path = config.ARTIFACTS_DIR / f"{_BATCH_IN_PREFIX}{idx:04d}.jsonl"
        shard_paths.append(path)
        return path.open("w", encoding="utf-8")

    try:
        fh = _open_shard(shard_idx)
        for rec in misses:
            h = rec["content_hash"]
            manifest[h] = rec["chunk_ids"]
            body = {
                "custom_id": h,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": config.EMBED_MODEL, "input": rec["text"]},
            }
            line = json.dumps(body, ensure_ascii=False) + "\n"
            encoded = line.encode("utf-8")
            # Roll to a new shard if this line would breach a limit.
            if lines_in_shard >= _MAX_LINES_PER_SHARD or (
                bytes_in_shard + len(encoded) > _MAX_BYTES_PER_SHARD
                and lines_in_shard > 0
            ):
                fh.close()
                shard_idx += 1
                fh = _open_shard(shard_idx)
                lines_in_shard = 0
                bytes_in_shard = 0
            fh.write(line)
            lines_in_shard += 1
            bytes_in_shard += len(encoded)
    finally:
        if fh is not None:
            fh.close()

    _BATCH_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return shard_paths


# --------------------------------------------------------------------------- #
# Submit (cost-gated; never auto-runs)                                        #
# --------------------------------------------------------------------------- #
def submit(confirm: bool = False, limit: int | None = None) -> list[str]:
    """Submit the sharded batch input to OpenAI — **only when ``confirm``**.

    The cost gate (09 §G) is mandatory: this function prints the estimate and
    **refuses to submit unless ``confirm=True``**. Full-corpus embedding is never
    auto-run. A demo run must pass ``limit <= config.DEMO_MAX_CHUNKS``.

    Args:
        confirm: Must be ``True`` to actually create the (billed) batches.
        limit: Optional cap on chunks (demo subset). When set, it must not exceed
            ``config.DEMO_MAX_CHUNKS`` unless ``confirm`` is given for the full
            corpus explicitly (``limit=None``).

    Returns:
        The created batch ids (empty list if there was nothing to embed).

    Raises:
        PermissionError: If ``confirm`` is not ``True`` (cost gate).
        ValueError: If ``limit`` exceeds the demo cap without full confirmation.
    """
    if limit is not None and limit > config.DEMO_MAX_CHUNKS:
        raise ValueError(
            f"limit={limit} exceeds DEMO_MAX_CHUNKS={config.DEMO_MAX_CHUNKS}. "
            f"Use limit=None for the full corpus (still requires confirm=True)."
        )

    est = estimate_cost(limit=limit)
    _print_estimate(est)

    if est["n_unique_misses"] == 0:
        print("Nothing to embed (all chunks cached). No batch submitted.")
        return []

    if not confirm:
        raise PermissionError(
            "COST GATE: refusing to submit a paid embedding batch without "
            "confirm=True. Review the estimate above, then re-run with "
            "confirm=True (CLI: --confirm)."
        )

    shards = build_batch_input(limit=limit)
    if not shards:
        print("No shard files produced; nothing submitted.")
        return []

    cli = _client()
    batch_ids: list[str] = []
    for shard in shards:
        with shard.open("rb") as f:
            up = cli.files.create(file=f, purpose="batch")
        batch = cli.batches.create(
            input_file_id=up.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"project": "lawbot", "shard": shard.name},
        )
        batch_ids.append(batch.id)
        print(f"submitted shard {shard.name} -> batch {batch.id}")

    # Persist batch ids alongside the manifest for `collect`.
    ids_path = config.ARTIFACTS_DIR / "batch_ids.json"
    ids_path.write_text(json.dumps(batch_ids), encoding="utf-8")
    print(f"{len(batch_ids)} batch(es) submitted. ids -> {ids_path}")
    return batch_ids


# --------------------------------------------------------------------------- #
# Collect (poll -> embeddings.jsonl + cache update)                           #
# --------------------------------------------------------------------------- #
_TERMINAL_OK: Final[frozenset[str]] = frozenset({"completed"})
_TERMINAL_BAD: Final[frozenset[str]] = frozenset(
    {"failed", "expired", "cancelled", "cancelling"}
)


def _load_manifest() -> dict[str, list[str]]:
    """Load the ``content_hash -> chunk_ids`` fan-out manifest."""
    if not _BATCH_MANIFEST.exists():
        return {}
    return json.loads(_BATCH_MANIFEST.read_text(encoding="utf-8"))


def _parse_output_line(line: str) -> tuple[str, list[float]] | None:
    """Parse one OpenAI Batch output line into ``(custom_id, vector)``.

    Args:
        line: A raw JSONL line from the batch output file.

    Returns:
        ``(custom_id, vector)`` on success, or ``None`` for blank/error lines
        (errors are surfaced by the caller's failure accounting).
    """
    line = line.strip()
    if not line:
        return None
    rec = json.loads(line)
    if rec.get("error"):
        return None
    custom_id = rec.get("custom_id")
    resp = (rec.get("response") or {}).get("body") or {}
    data = resp.get("data") or []
    if not custom_id or not data:
        return None
    vector = data[0].get("embedding")
    if not isinstance(vector, list):
        return None
    return custom_id, vector


def collect(batch_ids: list[str], poll: bool = True) -> None:
    """Poll the given batches, download outputs, and write embeddings + cache.

    Produces ``config.EMBEDDINGS_JSONL`` with one ``{chunk_id, vector}`` line per
    chunk (fanning each unique content-hash vector out to every chunk that shared
    it via the manifest), and appends the new vectors to the content-hash cache
    so future runs skip them.

    Args:
        batch_ids: Batch ids returned by :func:`submit` (or read from
            ``artifacts/batch_ids.json`` when empty).
        poll: When ``True``, block (with ``_POLL_INTERVAL_SEC`` cadence) until
            every batch reaches a terminal state. When ``False``, only collect
            already-completed batches and skip the rest.

    Raises:
        RuntimeError: If a batch ends in a failed/expired/cancelled state, or if
            a returned vector has the wrong dimensionality.
    """
    if not batch_ids:
        ids_path = config.ARTIFACTS_DIR / "batch_ids.json"
        if ids_path.exists():
            batch_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not batch_ids:
        print("No batch ids provided or found; nothing to collect.")
        return

    cli = _client()
    manifest = _load_manifest()

    # hash -> vector for all collected outputs.
    hash_vectors: dict[str, list[float]] = {}
    n_failed_lines = 0

    for bid in batch_ids:
        # Poll to terminal state.
        while True:
            b = cli.batches.retrieve(bid)
            status = b.status
            if status in _TERMINAL_OK:
                break
            if status in _TERMINAL_BAD:
                raise RuntimeError(f"Batch {bid} ended in status={status!r}")
            if not poll:
                print(f"batch {bid}: status={status} (not complete; skipping)")
                break
            counts = getattr(b, "request_counts", None)
            done = getattr(counts, "completed", "?") if counts else "?"
            total = getattr(counts, "total", "?") if counts else "?"
            print(f"batch {bid}: status={status} ({done}/{total}); waiting...")
            time.sleep(_POLL_INTERVAL_SEC)

        b = cli.batches.retrieve(bid)
        if b.status not in _TERMINAL_OK:
            continue
        out_file_id = b.output_file_id
        if not out_file_id:
            print(f"batch {bid}: completed but no output_file_id; skipping")
            continue
        content = cli.files.content(out_file_id).text
        for line in content.splitlines():
            parsed = _parse_output_line(line)
            if parsed is None:
                if line.strip():
                    n_failed_lines += 1
                continue
            h, vector = parsed
            _validate_dim(vector)
            hash_vectors[h] = vector

    if not hash_vectors:
        print("No vectors collected.")
        return

    # Update the content-hash cache (idempotent: skip hashes already cached).
    cache = load_cache()
    fresh = [(h, v) for h, v in hash_vectors.items() if h not in cache]
    if fresh:
        append_cache(fresh)

    # Fan out hash -> vector to every chunk_id and write embeddings.jsonl.
    n_written = 0
    n_missing_manifest = 0
    with config.EMBEDDINGS_JSONL.open("w", encoding="utf-8") as out:
        for h, vector in hash_vectors.items():
            chunk_ids = manifest.get(h)
            if not chunk_ids:
                n_missing_manifest += 1
                continue
            for cid in chunk_ids:
                out.write(json.dumps({"chunk_id": cid, "vector": vector}))
                out.write("\n")
                n_written += 1

    print(
        f"collect done: {len(hash_vectors)} unique vectors -> {n_written} chunk "
        f"embeddings -> {config.EMBEDDINGS_JSONL}"
    )
    if n_failed_lines:
        print(f"  WARNING: {n_failed_lines} failed/error output line(s).")
    if n_missing_manifest:
        print(
            f"  WARNING: {n_missing_manifest} vector(s) had no manifest entry "
            f"(stale manifest?)."
        )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _main(argv: list[str]) -> int:
    """CLI entry: estimate / build-input / submit / collect.

    Args:
        argv: Process arguments (excluding the program name).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="embed.embed_batch", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_est = sub.add_parser("estimate", help="print token/cost estimate (safe)")
    p_est.add_argument("--max-chunks", type=int, default=None)

    p_in = sub.add_parser("build-input", help="write sharded batch_in*.jsonl (safe)")
    p_in.add_argument("--max-chunks", type=int, default=None)

    p_sub = sub.add_parser("submit", help="submit batches (REQUIRES --confirm)")
    p_sub.add_argument("--confirm", action="store_true")
    p_sub.add_argument("--max-chunks", type=int, default=None)

    p_col = sub.add_parser("collect", help="poll + write embeddings.jsonl")
    p_col.add_argument("batch_ids", nargs="*", default=[])
    p_col.add_argument("--no-poll", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "estimate":
        _print_estimate(estimate_cost(limit=args.max_chunks))
        return 0
    if args.cmd == "build-input":
        shards = build_batch_input(limit=args.max_chunks)
        print(f"wrote {len(shards)} shard(s): {[p.name for p in shards]}")
        return 0
    if args.cmd == "submit":
        try:
            ids = submit(confirm=args.confirm, limit=args.max_chunks)
        except PermissionError as exc:
            print(str(exc))
            return 2
        print(f"batch ids: {ids}")
        return 0
    if args.cmd == "collect":
        collect(list(args.batch_ids), poll=not args.no_poll)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
