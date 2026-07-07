"""Pre-flight check before the full embedding run.

1) data integrity (core chunks count, content_hash, doc mix, resume state)
2) queue/quota state — definitive 1-request batch probe (is the Tier-1 enqueue
   limit cleared after yesterday's stuck bulk submit?)
"""
import json
import time
from pathlib import Path

import config
from embed.embed_client import _client


def count(p):
    p = Path(p)
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else None


print("===== [1] 데이터 무결성 =====")
core = count(config.CHUNKS_JSONL)
full = count(config.ARTIFACTS_DIR / "chunks_full.jsonl")
emb = count(config.EMBEDDINGS_JSONL)
print(f"chunks.jsonl (core, 임베딩 대상): {core:,}" if core else "chunks.jsonl: MISSING")
print(f"chunks_full.jsonl (자치법규 포함 백업): {full:,}" if full else "chunks_full.jsonl: MISSING")
print(f"embeddings.jsonl (이미 된 것): {emb if emb else 0} (신규 시작이면 0/MISSING)")

# 샘플 + 도메인 구성(접두 prefix로 코퍼스 비율 추정, 앞 20만줄 샘플)
from collections import Counter
pref = Counter()
has_hash = True
with open(config.CHUNKS_JSONL, encoding="utf-8") as f:
    first = json.loads(f.readline())
    has_hash = "content_hash" in first
    pref[str(first.get("doc_id", "")).split(":")[0]] += 1
    for i, line in enumerate(f):
        if i >= 200000:
            break
        try:
            pref[str(json.loads(line).get("doc_id", "")).split(":")[0]] += 1
        except Exception:
            pass
print("sample chunk keys:", list(first.keys()))
print("content_hash 존재:", has_hash, "| (없으면 드라이버가 자동계산)")
print("코퍼스 구성(앞 20만 샘플 접두):", dict(pref))
print("ORD(자치법규) 포함?:", "ORD" in pref, "(False여야 정상 — 제외했으므로)")

print("\n===== [2] 큐/쿼터 상태 — 1-요청 배치 정밀 테스트 =====")
cli = _client()
one = Path("/tmp/one_preflight.jsonl")
one.write_text(json.dumps({
    "custom_id": "pf", "method": "POST", "url": "/v1/embeddings",
    "body": {"model": config.EMBED_MODEL, "input": "사전점검", "dimensions": config.EMBED_DIMENSIONS},
}) + "\n", encoding="utf-8")
with one.open("rb") as fh:
    up = cli.files.create(file=fh, purpose="batch")
b = cli.batches.create(input_file_id=up.id, endpoint="/v1/embeddings", completion_window="24h")
print("probe batch:", b.id)
verdict = "UNKNOWN"
for _ in range(9):
    time.sleep(10)
    bb = cli.batches.retrieve(b.id)
    code = None
    errs = getattr(bb, "errors", None)
    if errs and getattr(errs, "data", None):
        code = errs.data[0].code
    print(f"  status={bb.status} code={code}")
    if bb.status == "completed":
        verdict = "CLEAR"
        break
    if bb.status in ("failed", "expired", "cancelled"):
        verdict = "BLOCKED(quota)" if code in ("request_limit_exceeded", "token_limit_exceeded") else f"FAILED({code})"
        break
    if bb.status in ("validating", "in_progress", "finalizing"):
        verdict = "CLEAR(accepted)"  # 큐가 받아줌 = 막히지 않음
        break

print("\n>>> 큐 판정:", verdict)
print(">>> 결론:", "임베딩 시작 가능 ✅" if verdict.startswith("CLEAR") else "아직 막힘/문제 — 시작 보류 ⚠️")
