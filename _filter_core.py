"""Filter CHUNKS_JSONL to the core corpus (drop 자치법규/ORD chunks).

Backs the full chunk set up to ``chunks_full.jsonl`` (once), then rewrites
``config.CHUNKS_JSONL`` to contain only non-ORD chunks (국가법령+행정규칙+판례).
The full backup lets us add 자치법규 incrementally later.
"""
import json
import os
import shutil
from pathlib import Path

import config

full = config.CHUNKS_JSONL
backup = config.ARTIFACTS_DIR / "chunks_full.jsonl"

if not backup.exists():
    shutil.copy(full, backup)
    print(f"backed up full chunks -> {backup.name}")
else:
    print(f"backup already exists -> {backup.name}")

tmp = config.ARTIFACTS_DIR / "chunks_core.tmp.jsonl"
kept = dropped = 0
with open(backup, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if str(rec.get("doc_id", "")).startswith("ORD:"):
            dropped += 1
            continue
        fout.write(line + "\n")
        kept += 1

os.replace(tmp, full)
print(f"DONE: kept(core)={kept:,}  dropped(ORD/자치법규)={dropped:,}  -> {full.name}")
