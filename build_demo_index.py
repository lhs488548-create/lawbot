"""데모 인덱스 빌더 (비용안전·429안전·재개가능).

핵심:
- cached_embed: content-hash 캐시 → 이미 임베딩한 청크는 재호출/재과금 0 (중단·재실행 안전).
- 토큰 페이싱: 분당 TPM_BUDGET 이하로 sleep 페이싱 → 429 원천 차단.
- 증분 저장: 슬라이스마다 캐시에 즉시 기록 → 중간에 끊겨도 이어서.

전량(319만)은 이 스크립트가 아니라 embed/embed_batch.py(Batch API, 50%↓)로.
실행: python -u build_demo_index.py
"""
from __future__ import annotations
import itertools, json, time
from pathlib import Path

import tiktoken
from ingest import parse_statute, parse_precedent, parse_admrule, parse_ordinance
from embed.chunk import chunks_of
from embed.embed_client import cached_embed
from embed import upsert_qdrant

SAMPLE = {
    "precedent": (parse_precedent, 400),
    "law":       (parse_statute,    80),
    "ordinance": (parse_ordinance, 150),
    "admrule":   (parse_admrule,   100),
}
TPM_BUDGET = 850_000           # 분당 토큰 안전예산 (Tier1 한도 100만 미만)
SLICE_TOKENS = 200_000         # 한 번에 임베딩할 토큰 묶음
ART = Path("artifacts"); ART.mkdir(exist_ok=True)
_enc = tiktoken.get_encoding("cl100k_base")


def collect_chunks() -> list[dict]:
    chunks: list[dict] = []
    for name, (mod, n) in SAMPLE.items():
        t = time.time(); docs = 0
        for doc in itertools.islice(mod.parse_all(), n):
            d = doc.model_dump() if hasattr(doc, "model_dump") else doc
            chunks.extend(chunks_of(d))
            docs += 1
        print(f"  {name:<10} 문서 {docs:>5} → 누적 청크 {len(chunks):>6,}  ({time.time()-t:.1f}s)")
    return chunks


def embed_paced(chunks: list[dict]) -> dict[str, list[float]]:
    """cached_embed를 토큰 묶음 단위로 호출 + TPM 페이싱. 반환: chunk_id -> vector."""
    vectors: dict[str, list[float]] = {}
    window_start = time.time(); window_tokens = 0
    slice_items: list[dict] = []; slice_tokens = 0
    done = 0

    def flush(items: list[dict]) -> None:
        nonlocal window_tokens
        if not items:
            return
        vectors.update(cached_embed(items))   # 캐시 미스만 실제 과금, 결과는 디스크 저장

    for c in chunks:
        n = len(_enc.encode(c["text"]))
        if slice_items and slice_tokens + n > SLICE_TOKENS:
            # TPM 페이싱: 이번 분 예산 초과하면 다음 창까지 대기
            if window_tokens + slice_tokens > TPM_BUDGET:
                wait = 60 - (time.time() - window_start)
                if wait > 0:
                    print(f"  …TPM 페이싱 {wait:.0f}s 대기")
                    time.sleep(wait)
                window_start = time.time(); window_tokens = 0
            flush(slice_items)
            window_tokens += slice_tokens; done += len(slice_items)
            print(f"  임베딩 {done:,}/{len(chunks):,}")
            slice_items, slice_tokens = [], 0
        slice_items.append(c); slice_tokens += n
    flush(slice_items); done += len(slice_items)
    print(f"  임베딩 {done:,}/{len(chunks):,}")
    return vectors


def main() -> int:
    t0 = time.time()
    print("[1/4] 샘플 파싱 + 청킹 ...")
    chunks = collect_chunks()
    print(f"  총 청크 {len(chunks):,}")

    print("[2/4] OpenAI 임베딩 (캐시+TPM페이싱) ...")
    vmap = embed_paced(chunks)

    vec_path = ART / "demo_vectors.jsonl"
    with vec_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["chunk_id"], "vector": vmap[c["chunk_id"]]}) + "\n")
    chunk_path = ART / "demo_chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("[3/4] Qdrant 컬렉션 생성 + 적재 ...")
    client, is_server = upsert_qdrant.get_client()
    print(f"  Qdrant 모드: {'server' if is_server else 'local-embedded'}")
    upsert_qdrant.ensure_collection(client)
    n_up = upsert_qdrant.upsert_all(client, chunks_path=chunk_path, embeddings_path=vec_path, ensure=False)
    print(f"  적재 포인트 수: {n_up:,}")

    print("[4/4] 검색 스모크 테스트 ...")
    from search import retriever
    retriever.set_clients(qdrant_client=client)
    for q in ["임차인 보증금 반환", "음주운전 처벌"]:
        hits = retriever.search(q, k=3)
        print(f"  Q: {q}")
        for h in hits:
            print("    -", getattr(h, "title", "?"), getattr(h, "article_no", ""))
    print(f"\n완료. {len(chunks):,} 청크, {time.time()-t0:.0f}s 소요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
