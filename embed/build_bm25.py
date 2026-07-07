"""Build a SQLite FTS5 (trigram) BM25 index over the full-corpus meta.jsonl.

Complements the dense FAISS index for hybrid retrieval (BM25 + RRF). FTS5 is
disk-based (low RAM — safe on the 7GB box) and the trigram tokenizer handles
Korean partial matching (e.g. query "사기" matches "사기죄", "제4조" matches
"민법 제4조"), which the default unicode61 tokenizer cannot.

Row alignment: each FTS5 ``rowid`` == the FAISS row id == the line index (0-based)
in ``meta.jsonl``, so a BM25 hit's rowid maps directly back to ``metas[rowid]``
(the same row the dense path uses). This is what lets dense and BM25 ranks be
fused by RRF without any extra join.

Two indexed columns (BM25 can weight them): ``ttl`` = law title + article_no
(strong signal for exact citation queries like "민법 제4조"), ``txt`` = the chunk
text capped to 2000 chars (content keywords like 사기/임금/보증금). The meta
boilerplate line stays in ``txt`` but BM25 IDF down-weights it.

Output: artifacts/full_index/bm25.sqlite  (env-agnostic; built from FULL_FAISS_META)

Run:  .venv/bin/python -m embed.build_bm25
"""
from __future__ import annotations

import json
import sqlite3
import time

import config

SRC = config.FULL_FAISS_META
OUT = config.FULL_FAISS_DIR / "bm25.sqlite"
TXT_CAP = 2000
BATCH = 10_000


def main() -> int:
    t0 = time.time()
    OUT.unlink(missing_ok=True)
    con = sqlite3.connect(str(OUT))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    # unicode61 tokenizer + PREFIX queries (term*) at search time: trigram fails
    # on 2-char Korean words (민법/사기/연차 -> no match); unicode61 token match
    # misses inflected/compound forms (처벌한다, 사기죄로), but a prefix query on
    # the stem (처벌*, 사기*) catches them since Korean inflection is suffixal.
    # Verified: 처벌*→처벌한다, 사기*→사기죄로, 민법*/연차* all match.
    con.execute(
        "CREATE VIRTUAL TABLE bm25 USING fts5(ttl, txt, tokenize=unicode61)"
    )
    rows: list[tuple[int, str, str]] = []
    n = 0
    with open(SRC, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            payload = rec.get("payload") or {}
            # Include article_title so topical queries match the governing article
            # (e.g. "사기" -> "형법 제347조 사기"); title+article_no alone missed it.
            ttl = (
                f"{payload.get('title') or ''} {payload.get('article_no') or ''} "
                f"{payload.get('article_title') or ''}"
            ).strip()
            txt = (rec.get("text") or "")[:TXT_CAP]
            rows.append((i, ttl, txt))
            n += 1
            if len(rows) >= BATCH:
                con.executemany("INSERT INTO bm25(rowid, ttl, txt) VALUES (?,?,?)", rows)
                rows = []
                if n % 100_000 == 0:
                    print(f"[bm25] {n:,} rows {time.time()-t0:.0f}s", flush=True)
    if rows:
        con.executemany("INSERT INTO bm25(rowid, ttl, txt) VALUES (?,?,?)", rows)
    con.commit()
    cnt = con.execute("SELECT count(*) FROM bm25").fetchone()[0]
    con.close()
    print(f"[bm25] DONE rows={cnt:,} out={OUT} {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
