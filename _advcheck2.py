import json
# Check: are 별표/별지/empty-text chunks present in chunks.jsonl (the bulk embed target)?
ch = "artifacts/chunks.jsonl"
n=0; empty_text=0; byeol=0; bgrade=0; sample_byeol=[]
from collections import Counter
kindc = Counter()
with open(ch, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        n += 1
        txt = r.get("text","")
        pl = r.get("payload") or {}
        kind = pl.get("kind") or r.get("kind")
        tg = pl.get("trust_grade") or r.get("trust_grade")
        kindc[kind] += 1
        if not txt or len(txt.strip())==0:
            empty_text += 1
        cid = r.get("chunk_id","")
        if "별표" in cid or "별지" in cid or kind=="별표":
            byeol += 1
            if len(sample_byeol)<3: sample_byeol.append({"chunk_id":cid,"tg":tg,"textlen":len(txt)})
        if tg == "B":
            bgrade += 1
        if n >= 400000:  # sample first 400k (covers LAW+ADMRULE = embedded region)
            break
print(f"sampled {n} chunks of chunks.jsonl")
print("empty/blank text chunks:", empty_text)
print("별표/별지 chunks:", byeol)
print("trust_grade==B chunks:", bgrade)
print("kind distribution (top):", kindc.most_common(10))
print("sample 별표:", sample_byeol)
