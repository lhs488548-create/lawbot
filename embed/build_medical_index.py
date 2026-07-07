"""Glue: medical docs → chunks → 512d embeddings → CHUNKS_VEC_JSONL → FAISS.

This is the thin end-to-end driver the FAISS build report (§5.1) flagged as the
single missing piece. It wires the already-built, individually-verified
components into one idempotent, re-run-safe command::

    embed/medical_corpus  (already run → MED_DIR/docs/*.jsonl)
        └─ chunk.build_chunks(sources=의료 docs, out=MED_DIR/chunks.jsonl,
                              parents=MED_DIR/parents.jsonl)        (헤더·content_hash)
            └─ embed_client.cached_embed(512d, content-hash 캐시)   (미스만 과금)
                └─ {chunk_id,doc_id,parent_id,text,payload,vector} → config.CHUNKS_VEC_JSONL
                    └─ faiss_index.build_index()  → config.FAISS_INDEX + config.FAISS_META

Design notes / contract alignment:

* **Source paths are pinned to the medical sub-corpus** (report §6-1 risk): the
  default ``chunk.build_chunks`` sources are the *full* corpus, so this driver
  always passes ``sources=`` / ``out_path=`` / ``parents_path=`` explicitly,
  resolved from ``config`` constants (never hardcoded Korean paths — the shell
  only ever sees ``python -m embed.build_medical_index``).
* **Only ``trust_grade == "A"`` chunks are embedded** (contract §5, report §6-2).
  This excludes B-grade metadata/별표 chunks (image/label-only) from the index.
* Vectors are written **raw** (un-normalized) into ``config.CHUNKS_VEC_JSONL``;
  ``faiss_index.build_index`` L2-normalizes at load time (contract §1, the FAISS
  builder owns normalization). sqlite-vec export normalizes the same way.
* **Idempotent**: re-running re-chunks deterministically and re-uses the
  content-hash embedding cache (``config.EMBED_CACHE_JSONL``) so only genuinely
  new text is re-billed; outputs are replaced wholesale each run.

Run (real OpenAI billing on cache-miss — sanctioned for the medical build)::

    cd /home/user1/lawbot && .venv/bin/python -m embed.build_medical_index
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
import embed.chunk as chunk
import embed.faiss_index as faiss_index
from embed.embed_client import cached_embed, load_cache


def _medical_sources() -> list[Path]:
    """Return the three medical ``docs`` JSONLs, resolved from ``config.MED_DIR``.

    Paths come only from config constants so the (Korean) directory name is never
    passed through the shell — avoiding the Git Bash → WSL encoding trap.
    """
    docs = config.MED_DIR / "docs"
    return [docs / "국가법령.jsonl", docs / "행정규칙.jsonl", docs / "판례.jsonl"]


def _chunk_medical() -> dict[str, int]:
    """Chunk the medical sub-corpus into ``MED_DIR/chunks.jsonl`` (+ parents).

    Pins ``sources``/``out_path``/``parents_path`` to the medical sub-corpus so
    the full-corpus defaults of :func:`chunk.build_chunks` are never used
    (report §6-1). Parents land in ``MED_DIR/parents.jsonl``; the medical
    ``chunks.jsonl`` is an intermediate (the canonical artifact is the vectors
    file built below).

    Returns:
        The stats dict from :func:`chunk.build_chunks`.
    """
    out_path = config.MED_DIR / "chunks.jsonl"
    parents_path = config.MED_DIR / "parents.jsonl"
    print("[1/4] chunking medical docs ->", out_path.name, "+", parents_path.name)
    stats = chunk.build_chunks(
        sources=_medical_sources(),
        out_path=out_path,
        parents_path=parents_path,
    )
    return stats


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load all chunk records from a chunks JSONL into memory.

    The medical sub-corpus is small (a few thousand chunks), so holding it in
    memory is fine and keeps the embed/merge step simple.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _embed_and_write(chunks_path: Path) -> dict[str, int]:
    """Embed A-grade chunks (512d, cached) and write ``config.CHUNKS_VEC_JSONL``.

    Filters to ``payload['trust_grade'] == 'A'`` (contract §5 / report §6-2),
    runs :func:`embed.embed_client.cached_embed` (only cache-misses hit OpenAI),
    then writes one canonical record
    ``{chunk_id, doc_id, parent_id, text, payload, vector}`` per A-grade chunk to
    ``config.CHUNKS_VEC_JSONL`` (raw vectors; FAISS normalizes at load).

    Args:
        chunks_path: The medical ``chunks.jsonl`` produced by :func:`_chunk_medical`.

    Returns:
        Stats: ``{total, embedded, skipped_b, cache_hits, cache_misses}``.
    """
    all_chunks = _load_chunks(chunks_path)
    a_chunks = [c for c in all_chunks if (c.get("payload") or {}).get("trust_grade") == "A"]
    skipped_b = len(all_chunks) - len(a_chunks)
    print(
        f"[2/4] embedding: {len(all_chunks)} chunks total, "
        f"{len(a_chunks)} A-grade to embed, {skipped_b} B-grade skipped"
    )

    # Measure cache hits/misses for honest reporting (cached_embed itself only
    # bills the misses). Distinct content_hashes drive the real OpenAI cost.
    cache_before = load_cache()
    items = [
        {"chunk_id": c["chunk_id"], "text": c["text"], "content_hash": c.get("content_hash")}
        for c in a_chunks
    ]
    distinct_hashes = {(it["content_hash"] or "") for it in items}
    distinct_hashes.discard("")
    # Hash may be absent on a record; recompute set from content_hash field which
    # the chunker always provides, so distinct_hashes is reliable here.
    cache_misses = sum(1 for h in distinct_hashes if h not in cache_before)
    cache_hits = len(distinct_hashes) - cache_misses

    vectors = cached_embed(items)  # {chunk_id -> 512d vector}; misses billed.

    config.CHUNKS_VEC_JSONL.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with config.CHUNKS_VEC_JSONL.open("w", encoding="utf-8") as out:
        for c in a_chunks:
            vec = vectors[c["chunk_id"]]
            if len(vec) != config.EMBED_DIM:  # defensive — embed_client also checks
                raise ValueError(
                    f"{c['chunk_id']}: vector dim {len(vec)} != {config.EMBED_DIM}"
                )
            rec = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "parent_id": c["parent_id"],
                "text": c["text"],
                "payload": c["payload"],
                "vector": vec,
            }
            out.write(json.dumps(rec, ensure_ascii=False))
            out.write("\n")
            written += 1

    print(
        f"      wrote {written} vector records -> {config.CHUNKS_VEC_JSONL.name} "
        f"(cache hits={cache_hits}, misses(billed)={cache_misses} of "
        f"{len(distinct_hashes)} distinct texts)"
    )
    return {
        "total": len(all_chunks),
        "embedded": written,
        "skipped_b": skipped_b,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }


def _build_faiss() -> int:
    """Build the FAISS index + meta from ``config.CHUNKS_VEC_JSONL``; return ntotal."""
    print("[3/4] building FAISS index ->", config.FAISS_INDEX.name, "+", config.FAISS_META.name)
    faiss_index.build_index()
    index, metas = faiss_index.load_index()
    print(f"      FAISS ntotal={index.ntotal}, meta rows={len(metas)}")
    return index.ntotal


def main() -> int:
    """Run the full medical index build end to end and print an aggregate report."""
    chunk_stats = _chunk_medical()
    chunks_path = config.MED_DIR / "chunks.jsonl"
    embed_stats = _embed_and_write(chunks_path)
    ntotal = _build_faiss()

    print("[4/4] DONE — medical FAISS index built.")
    print("  docs chunked   :", chunk_stats["docs"])
    print("  chunks (all)   :", chunk_stats["chunks"])
    print("  parents        :", chunk_stats["parents"])
    print("  A-grade embedded:", embed_stats["embedded"])
    print("  B-grade skipped :", embed_stats["skipped_b"])
    print("  cache hits      :", embed_stats["cache_hits"])
    print("  cache misses    :", embed_stats["cache_misses"], "(billed)")
    print("  FAISS ntotal    :", ntotal)
    print("  artifacts:")
    print("    -", config.MED_DIR / "chunks.jsonl")
    print("    -", config.MED_DIR / "parents.jsonl")
    print("    -", config.CHUNKS_VEC_JSONL)
    print("    -", config.FAISS_INDEX)
    print("    -", config.FAISS_META)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
