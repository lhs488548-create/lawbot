"""Sync embedding runner for the REMAINING chunks (complement to embed_waves).

Drop-in continuation of ``embed.embed_waves`` but using the *synchronous*
OpenAI embeddings API (no 24h Batch queue) for the final stretch.

Compatibility guarantees (must match embed_waves byte-for-byte at the consumer):
* Appends to the SAME file ``config.EMBEDDINGS_JSONL`` with the SAME record shape
  ``{"chunk_id": <id>, "vector": <512 floats>}`` (json.dumps default), so the
  downstream join/FAISS build treats sync- and batch-produced rows identically.
* Same model + dims: text-embedding-3-small @ dimensions=512 (via embed_client),
  RAW vectors (normalization happens at FAISS build time, exactly like embed_waves).
* Same de-dup: one embed per unique content_hash, fanned out to every chunk_id
  sharing that hash (mirrors embed_waves._run_wave).
* Resumable + no double-spend: chunk_ids already present in EMBEDDINGS_JSONL are
  skipped (same _load_done as embed_waves), so it picks up exactly where batch
  left off and never re-embeds a completed chunk.
* Uses the OpenAI SDK (embed_client.embed_texts) — NOT raw urllib — so it is not
  affected by the Cloudflare 403 that blocks urllib.

Safe test (writes to a temp file, ~5 real chunks, ~$0.0001)::
    .venv/bin/python -m embed.sync_embed --limit 5 --out /tmp/synctest.jsonl

Full run (appends to the real embeddings file, resumable)::
    .venv/bin/python -m embed.sync_embed
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import config
from embed.embed_client import embed_texts, content_hash, append_cache, _validate_dim

TOTAL = 1_357_018  # core corpus chunk count (for progress %), matches embed_waves
FLUSH_ITEMS = 6000  # buffered chunks per flush (-> a few 2048-input requests)


def _load_done(emb: pathlib.Path) -> set[str]:
    """Return chunk_ids already embedded (same semantics as embed_waves)."""
    done: set[str] = set()
    if emb.exists():
        with emb.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    done.add(json.loads(ln)["chunk_id"])
                except Exception:
                    pass
    return done


def _stream_chunks():
    with config.CHUNKS_JSONL.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _flush(items: list[dict], emb: pathlib.Path, total_done: int,
           write_cache: bool) -> int:
    """Embed a buffer (de-duped by content_hash) and append rows. Returns new total."""
    by_hash: dict[str, str] = {}
    ids_by_hash: dict[str, list[str]] = {}
    for c in items:
        h = c.get("content_hash") or content_hash(c["text"])
        by_hash.setdefault(h, c["text"])
        ids_by_hash.setdefault(h, []).append(c["chunk_id"])

    hashes = list(by_hash)
    texts = [by_hash[h] for h in hashes]
    vectors = embed_texts(texts)  # 512-d, retry-wrapped, batched <=2048 / <=280k tok
    hv = dict(zip(hashes, vectors))

    with emb.open("a", encoding="utf-8") as out:
        for h, ids in ids_by_hash.items():
            v = _validate_dim(hv[h])  # asserts == 512
            for cid in ids:
                out.write(json.dumps({"chunk_id": cid, "vector": v}) + "\n")
                total_done += 1
    if write_cache:
        try:
            append_cache(zip(hashes, vectors))  # reuse later, cheap append
        except Exception:
            pass
    return total_done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sync embedding runner (remaining chunks).")
    ap.add_argument("--out", default=None,
                    help="output JSONL (default: config.EMBEDDINGS_JSONL). Use a temp path to test.")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N not-yet-done chunks (test).")
    ap.add_argument("--no-cache-write", action="store_true",
                    help="do not append to the content-hash cache.")
    args = ap.parse_args(argv)

    real_emb = config.EMBEDDINGS_JSONL
    out = pathlib.Path(args.out) if args.out else real_emb
    testing = out != real_emb

    # done-set is ALWAYS read from the real embeddings file, so a test run to a
    # temp --out still skips already-embedded chunks (and never re-embeds them).
    done = _load_done(real_emb)
    total_done = len(done)
    print(f"sync start: {total_done:,}/{TOTAL:,} already embedded "
          f"({100*total_done/TOTAL:.1f}%); out={out} testing={testing} limit={args.limit}",
          flush=True)

    buf: list[dict] = []
    processed = 0
    for c in _stream_chunks():
        if c["chunk_id"] in done:
            continue
        buf.append(c)
        processed += 1
        if args.limit and processed >= args.limit:
            break
        if len(buf) >= FLUSH_ITEMS:
            total_done = _flush(buf, out, total_done, not args.no_cache_write)
            buf = []
            print(f"[sync] flush -> total {total_done:,}/{TOTAL:,} "
                  f"({100*total_done/TOTAL:.1f}%)", flush=True)
    if buf:
        total_done = _flush(buf, out, total_done, not args.no_cache_write)
        print(f"[sync] final -> total {total_done:,}/{TOTAL:,} "
              f"({100*total_done/TOTAL:.1f}%)", flush=True)

    print(f"SYNC DONE: total {total_done:,}/{TOTAL:,} "
          f"({100*total_done/TOTAL:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
