import json, itertools
from collections import Counter, OrderedDict

EMB = "artifacts/embeddings.jsonl"
CH  = "artifacts/chunks.jsonl"

# 1) embeddings.jsonl schema + last record
last = None
n = 0
with open(EMB, encoding="utf-8") as f:
    first = json.loads(f.readline()); n = 1
    for line in f:
        if line.strip():
            last = line; n += 1
print("EMB total lines:", n)
print("EMB first keys:", list(first.keys()))
def cid(rec):
    return rec.get("chunk_id") or rec.get("custom_id") or rec.get("id")
print("EMB first chunk_id:", cid(first))
if last:
    lr = json.loads(last)
    print("EMB last  chunk_id:", cid(lr))

# 2) doc_type prefix of embedded chunk_ids (sample first/last 50k via streaming counter on prefix)
emb_pref = Counter()
emb_ids = set()
with open(EMB, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        c = cid(r)
        if c:
            emb_ids.add(c)
            emb_pref[str(c).split(":")[0]] += 1
print("EMB doc_type prefix counts:", dict(emb_pref))
print("EMB distinct chunk_ids:", len(emb_ids), "(dup =", n - len(emb_ids), ")")

# 3) chunks.jsonl ordering: is it strictly LAW->ADMRULE->PREC blocks? detect transitions
order = []
seen = OrderedDict()
ch_pref = Counter()
transitions = 0
prev = None
with open(CH, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        p = str(r.get("doc_id","")).split(":")[0]
        ch_pref[p] += 1
        if p != prev:
            order.append(p)
            transitions += 1
            prev = p
print("CHUNKS total doc_type prefix counts:", dict(ch_pref))
print("CHUNKS block transition count:", transitions)
print("CHUNKS block order (first 12):", order[:12])

# 4) how many embedded ids fall in each chunks.jsonl block position (is PREC really 0?)
# Build chunk_id -> prefix is same as split; just check if any PREC embedded
prec_embedded = emb_pref.get("PREC", 0)
print("PREC embedded count:", prec_embedded)
print("ADMRULE total in core:", ch_pref.get("ADMRULE"), "embedded:", emb_pref.get("ADMRULE"))
print("LAW total in core:", ch_pref.get("LAW"), "embedded:", emb_pref.get("LAW"))
