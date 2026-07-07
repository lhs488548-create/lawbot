"""Memory-safe full-corpus FAISS index builder (low-RAM streaming join).

The stock ``embed.faiss_index.build_index`` accumulates every vector + meta in
RAM (fine for the 14k medical subset) and would OOM on the 1,357,018-chunk full
corpus under this box's ~7 GB RAM. This builder streams instead:

  Pass A: index chunks.jsonl by chunk_id -> byte offset (~200 MB dict).
  Pass B: stream embeddings.jsonl; for each {chunk_id, vector}, seek the chunk
          record for text/payload, drop trust_grade=="B", L2-normalize, add to a
          batched IndexFlatIP, and append the row-aligned meta line to disk.

Peak RAM ~= offset dict (~0.2 GB) + the FAISS flat index itself (1.36M*512*4 =
~2.8 GB) + one 50k batch (~0.1 GB) -> well under 7 GB. No OpenAI calls.

Outputs (NEW path; does NOT touch the live medical index):
  artifacts/full_index/index.faiss
  artifacts/full_index/meta.jsonl   (row-aligned: FAISS row i <-> meta line i)

Meta schema matches what search.retriever expects: {chunk_id, doc_id, parent_id,
text, payload}. Build is verified by embed.faiss_index.load_index-compatible
invariant ntotal == len(meta).

Test (tiny, temp out):  .venv/bin/python -m embed.build_full_index --limit 2000 --out /tmp/fulltest
Full run:               .venv/bin/python -m embed.build_full_index
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import faiss

import config

EMB = config.EMBEDDINGS_JSONL
CH = config.CHUNKS_JSONL
DIM = config.EMBED_DIM  # 512
BATCH = 50_000


def _chunkid_fast(raw: bytes) -> str | None:
    """Extract chunk_id without full JSON parse. Lines start
    {"chunk_id": "VALUE", ...} and chunk_id never contains a double quote."""
    try:
        parts = raw.split(b'"', 4)
        # parts = [b'{', b'chunk_id', b': ', b'VALUE', b', "doc_id"...]
        if len(parts) >= 4 and parts[1] == b"chunk_id":
            return parts[3].decode("utf-8")
    except Exception:
        return None
    return None


def build(limit: int | None, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_dir / "index.faiss"
    out_meta = out_dir / "meta.jsonl"
    t0 = time.time()

    # ---- Pass A: chunk_id -> offset in chunks.jsonl ----
    offsets: dict[str, int] = {}
    with open(CH, "rb") as f:
        pos = f.tell()
        line = f.readline()
        while line:
            cid = _chunkid_fast(line)
            if cid is not None:
                offsets[cid] = pos
            pos = f.tell()
            line = f.readline()
    print(f"[A] indexed {len(offsets):,} chunk offsets in {time.time()-t0:.0f}s", flush=True)

    # ---- Pass B: stream embeddings, seek-join, build ----
    index = faiss.IndexFlatIP(DIM)
    batch: list[np.ndarray] = []
    n = 0
    skipped_b = 0
    missing = 0
    bad_dim = 0

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        m = np.vstack(batch).astype(np.float32, copy=False)
        faiss.normalize_L2(m)
        # L2-norm post-check (audit P1): after normalize_L2 every nonzero row must
        # be unit-length (±1e-3). Drift means a malformed/zero vector slipped in —
        # fail loudly rather than poison the cosine (inner-product) index.
        norms = np.linalg.norm(m, axis=1)
        nz = norms > 1e-6
        if nz.any():
            drift = float(np.max(np.abs(norms[nz] - 1.0)))
            assert drift < 1e-3, f"L2 norm drift {drift:.2e} exceeds 1e-3 (bad vector?)"
        index.add(m)
        batch = []

    chf = open(CH, "rb")
    metf = open(str(out_meta) + ".tmp", "w", encoding="utf-8")
    with open(EMB, encoding="utf-8") as ef:
        for line in ef:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["chunk_id"]
            vec = rec["vector"]
            if len(vec) != DIM:
                bad_dim += 1
                continue
            off = offsets.get(cid)
            if off is None:
                missing += 1
                continue
            chf.seek(off)
            crec = json.loads(chf.readline())
            payload = crec.get("payload", {}) or {}
            if payload.get("trust_grade") == "B":
                skipped_b += 1
                continue
            batch.append(np.asarray(vec, dtype=np.float32))
            metf.write(json.dumps({
                "chunk_id": cid,
                "doc_id": crec.get("doc_id"),
                "parent_id": crec.get("parent_id"),
                "text": crec.get("text"),
                "payload": payload,
            }, ensure_ascii=False) + "\n")
            n += 1
            if len(batch) >= BATCH:
                flush()
                print(f"[B] added {n:,} (skipB={skipped_b} miss={missing}) "
                      f"{time.time()-t0:.0f}s", flush=True)
            if limit and n >= limit:
                break
    flush()
    metf.close()
    chf.close()

    assert index.ntotal == n, f"ntotal {index.ntotal} != meta {n}"
    faiss.write_index(index, str(out_index))
    pathlib.Path(str(out_meta) + ".tmp").replace(out_meta)
    print(f"[DONE] ntotal={index.ntotal:,} skippedB={skipped_b:,} "
          f"missing={missing:,} badDim={bad_dim:,} "
          f"out={out_dir} {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="artifacts/full_index")
    args = ap.parse_args()
    build(args.limit, pathlib.Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
