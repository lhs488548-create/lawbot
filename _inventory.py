"""One-off inventory: current index size, source corpora counts, artifacts."""
from pathlib import Path
import config


def count_lines(p):
    p = Path(p)
    if not p.exists():
        return "MISSING"
    n = 0
    with open(p, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


print("===== CURRENT MEDICAL INDEX =====")
print("chunks_with_vectors.jsonl:", count_lines(config.CHUNKS_VEC_JSONL))
print("faiss index exists:", config.FAISS_INDEX.exists())
print("faiss meta lines:", count_lines(config.FAISS_META))

print("\n===== SOURCE CORPORA (files on disk) =====")
for name, d in [
    ("01_국가법령", config.LAW_DIR),
    ("02_자치법규", config.ORDINANCE_DIR),
    ("03_행정규칙", config.ADMRULE_DIR),
    ("04_판례", config.PRECEDENT_DIR),
]:
    d = Path(d)
    if d.exists():
        files = [p for p in d.rglob("*") if p.is_file()]
        print(f"{name}: {len(files)} files  @ {d}")
    else:
        print(f"{name}: MISSING @ {d}")

print("\n===== ARTIFACTS (parsed/chunked so far) =====")
for f in [
    config.DOCS_LAW_JSONL, config.DOCS_PREC_JSONL,
    config.DOCS_ADMRULE_JSONL, config.DOCS_ORD_JSONL,
    config.CHUNKS_JSONL, config.EMBEDDINGS_JSONL, config.EMBED_CACHE_JSONL,
]:
    print(f"{Path(f).name}: {count_lines(f)}")

print("\n===== DATA_ROOT =====")
print(config.DATA_ROOT, "exists:", Path(config.DATA_ROOT).exists())
