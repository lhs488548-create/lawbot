# _FAISS_BUILD_REPORT.md — 의료관련 FAISS 빌드 최종 리포트

> 작성: 2026-06-16. 빌드 통합 책임자 종합.
> 범위: **코드 + 오프라인 selftest까지** (실제 대량 임베딩/OpenAI 과금 0 — 비용규칙 준수).
> 근거 계약: `docs/_FAISS_BUILD_CONTRACT.md` · 설계: `docs/의료관련_코퍼스_범위_추출전략.md`.
> 환경: WSL `/home/user1/lawbot` = UNC `\\wsl.localhost\Ubuntu\home\user1\lawbot`.
> 파이썬 실행은 WSL venv 전용: `wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python …'`.

---

## 0. 한 줄 결론

FAISS 전환(Qdrant·redis 제거) + 512d 일관성 + 의료 서브코퍼스 추출 파이프라인의 **코드 4개 컴포넌트 + manifest 4종이 전부 완성·통과**했다. 정규 산출물(`docs/*.jsonl` 1,001건)까지 실제 생성 검증했고, **남은 일은 비용이 드는 단 한 구간(chunk→임베딩→`chunks_with_vectors.jsonl`→FAISS 빌드)뿐**이며, 그 구간을 돌리기 위한 명령·비용추정·통합 주의점을 §4~6에 정리했다.

---

## 1. 완성된 것 (파일별)

| 파일 | 소유 | 상태 | 핵심 |
|---|---|---|---|
| `embed/medical_corpus.py` | medical | ✅ 완성·실빌드 검증 | manifest(YAML) SoT → 원천 6GB **무복사 스트리밍 필터-파서** → `MED_DIR/docs/{국가법령,행정규칙,판례}.jsonl` + `_audit/*`. 멱등(원자적 교체). 임베딩 호출 없음. |
| `embed/faiss_index.py` | faiss | ✅ 완성·selftest | `build_index`/`load_index`/`export_sqlite_vec`. `IndexFlatIP(512)` + L2정규화, `FAISS_META` 행정렬, sqlite-vec(`vec0`) + float32 LE BLOB fallback. 멱등. |
| `search/retriever.py` | retriever | ✅ FAISS 이식·selftest | `search`/`Hit`/`embed_query`/`get_parent` **시그니처 100% 보존**. 내부만 FAISS. Qdrant 접근자 제거, `get_index`/`set_index`/`reset_index_cache` 추가. |
| `embed/embed_client.py` | deploy | ✅ 완성·오프라인 검증 | `_embed_request`에 `dimensions=config.EMBED_DIMENSIONS`(=512) 추가. content-hash 캐시·공개 시그니처 불변. |
| `requirements.txt` | deploy | ✅ | `faiss-cpu==1.14.3` 추가, `qdrant-client` 제거(주석화). `slowapi` 유지(in-memory rate limit, redis 불요). |
| `Dockerfile` | deploy | ✅ | Qdrant ENV 삭제. `artifacts/의료관련`은 이미지 비포함 → 런타임 read-only 볼륨 마운트. |
| `docker-compose.yml` | deploy | ✅ | `qdrant`·`redis` 서비스/볼륨 제거. `caddy`+`api`만. FAISS는 `./artifacts/의료관련:/app/artifacts/의료관련:ro` 마운트. |
| `artifacts/의료관련/manifest/statutes_whitelist.yaml` | medical | ✅ | Tier1(22)+Tier2(19) 폴더명, NFC exact(전각 가운뎃점 `ㆍ` U+318D 포함). |
| `…/manifest/admrule_targets.yaml` | medical | ✅ | 부처 스코프 2트리 + 규칙명 키워드 22종 + B등급 메타링크 정책. |
| `…/manifest/admrule_denylist.yaml` | medical | ✅ | 직제·계약·장학·국립병원예규 deny(targets보다 우선). |
| `…/manifest/precedent_match_rules.yaml` | medical | ✅ | Phase-1(형사+일반행정 대법원, 참조조문 exact) 화이트리스트 32 + alias 6. Phase-2는 정의만(`enabled:false`). |

### 실제 추출 산출물 (이번에 생성됨 — 임베딩 전 단계, 과금 0)

```
artifacts/의료관련/docs/국가법령.jsonl   107 Document  (전량 A등급, 42 법령폴더)
artifacts/의료관련/docs/행정규칙.jsonl   272 Document  (A270 / B2)
artifacts/의료관련/docs/판례.jsonl       622 Document  (형사402 + 일반행정220, 전량 대법원·A등급)
──────────────────────────────────────  1,001 Document
artifacts/의료관련/_audit/{match_log.csv, dedup_log.csv, absent_null.csv}
```

각 줄 = `Document.model_dump_json()` (= `embed/chunk.py`가 그대로 소비). 1,001건 전부 `Document.model_validate_json` 라운드트립 통과, chunk.py 소비 키(`doc_id, doc_type, title, effective_from, source_url, trust_grade, jurisdiction, law_kind, articles, meta`) 보유 확인. 재실행 시 byte-identical(멱등). raw `.md` 복사 0건.

---

## 2. selftest 통과 현황 (전부 오프라인·OpenAI 호출 0)

이번 통합 검증에서 WSL venv로 **3개 selftest 전부 재실행 → 통과(EXIT=0)** 확인:

| 명령 | 결과 |
|---|---|
| `python -m search.retriever --selftest` | **SELFTEST PASSED: 12 checks** (FAISS ranking, metadata/parent_id post-filters, match-any, as_of_date point-in-time, over-fetch, parent-promotion, validation) |
| `python -m embed.faiss_index --selftest` | **SELFTEST OK: all checks passed (offline, no OpenAI calls)** — IndexFlatIP ntotal=32, meta 행정렬, top-1 코사인~1.0, 멱등 재빌드, sqlite export(BLOB 512 float32 단위정규화·입력벡터 일치). `[INFO] vec_chunks absent → BLOB fallback`. |
| `python -m embed.medical_corpus --selftest` | **selftest PASS** (11체크: 법령명 경계/구법/alias 매칭, 화이트리스트 실재, admrule 키워드·denylist 게이트, 형사 대법원 판례 참조조문→의료법 매칭) |

`retriever` selftest는 `embed_query`를 stub + 순수 numpy `_DummyFlatIP`로 FAISS 대체 → 과금 0. `faiss_index`·`medical_corpus`는 더미 벡터/디스크 메타데이터만 사용. **세 selftest 모두 실 OpenAI·네트워크 미접속.**

추가 통합 검증(이번 빌드):
- 5개 모듈(`config`, `embed.faiss_index`, `search.retriever`, `embed.medical_corpus`, `embed.embed_client`) import 성공.
- `embed_client._embed_request`에 `dimensions=config.EMBED_DIMENSIONS` 존재 확인(소스 312행).

---

## 3. 불변 규칙 보존 확인

### 3.1 `retriever.search` 시그니처·`Hit` 보존 — ✅ 100% 유지

런타임 introspection 결과(이번 검증):
```
search(query: str, k: int = 8, flt: Mapping[str, Any] | None = None, as_of_date: str | None = None) -> list[Hit]
Hit  dataclass fields: ['id', 'score', 'payload']   (.text/.chunk_id/.parent_id/.doc_id/.doc_type/.title/.article_no/.effective_from/.source_url/.trust_grade/.location() 접근자 유지)
embed_query(query: str) -> list[float]
get_parent(parent_id: str) -> dict[str, Any] | None
```
계약 §3과 정확히 일치. 상위 모듈(`search/{rag,verify,ad_review,source_pack,statutes}`, API)은 무수정. 신규 노출은 인덱스 캐시 헬퍼(`get_index`/`set_index`/`reset_index_cache`)뿐 — 기존 표면 변경 없음.

### 3.2 512d 일관성 — ✅

```
config.EMBED_MODEL      = text-embedding-3-small
config.EMBED_DIM        = 512   (코드 Final 고정, env 무시)
config.EMBED_DIMENSIONS = 512
```
- 임베딩 요청: `embeddings.create(model=…, input=…, dimensions=config.EMBED_DIMENSIONS=512)`.
- 반환 검증: `embed_client._validate_dim`이 `len != 512` 거부(1536 레거시 캐시 줄 자동 skip).
- FAISS: `IndexFlatIP(config.EMBED_DIM)` + 적재 직전 L2정규화(numpy + `faiss.normalize_L2` 이중) → 내적=코사인.
- retriever `embed_query`도 검색측 L2정규화 → 인덱싱/질의 차원·정규화 동일.

### 3.3 sqlite-vec 호환성 — ✅ (BLOB fallback 검증, vec0 가속은 운영 환경 의존)

`export_sqlite_vec`는 정규 JSONL 하나에서:
- `chunks(chunk_id PK, doc_id, parent_id, text, payload JSON, embedding BLOB)` — `embedding`은 **항상** 512 float32 LE로 채워 vec0 확장 없이도 벡터 읽기 가능.
- `vec_chunks`(sqlite-vec `vec0`, `embedding float[512]`, rowid↔chunks.rowid) — sqlite-vec 확장 탑재 시 생성.

selftest 환경엔 vec0 확장이 없어 `vec_chunks`는 생략되고 BLOB fallback이 채워짐(`[INFO]`로 명시). **호환 요구("FAISS·sqlite-vec 둘 다 적재")는 BLOB fallback으로 충족.** 실제 sqlite-vec 가속 질의가 필요하면 운영 환경에 vec0 확장만 로드하면 동일 DB에서 `vec_chunks`가 생성된다.

---

## 4. 사용법 — 코퍼스 빌드 → 임베딩 → FAISS 빌드 → 서버

모든 파이썬 실행은 WSL venv. 한글 경로·전각 가운뎃점 때문에 **Git Bash→WSL heredoc은 인코딩이 깨지므로 한글 인자는 임시 `.py` 경유 권장**.

```bash
# ── 0) 사전 정리(권장, 1회): .env의 stale EMBED_DIM 줄 삭제 → import RuntimeWarning 소멸
#     (기능 영향 없음 — config가 항상 512 강제. §7-1 참조)

# ── 1) 의료 서브코퍼스 추출 (무복사 필터-파서, 임베딩 없음, 멱등) ──────────────
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -m embed.medical_corpus'
#   → artifacts/의료관련/docs/{국가법령,행정규칙,판례}.jsonl  (+ _audit/*)

# ── 2) 청킹 (헤더·content_hash, 임베딩 없음) ───────────────────────────────────
#   ★주의: embed/chunk.py build_chunks()의 기본 sources는 풀코퍼스
#         artifacts/docs_*.jsonl 이다(의료 서브셋이 아님). 의료 서브코퍼스를 청킹하려면
#         반드시 의료 docs 경로를 명시해야 한다(§5-6 통합 주의):
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -c "
import config, embed.chunk as c
med = config.MED_DIR/\"docs\"
c.build_chunks(
    sources=[med/\"국가법령.jsonl\", med/\"행정규칙.jsonl\", med/\"판례.jsonl\"],
    out_path=config.MED_DIR/\"chunks.jsonl\",
    parents_path=config.PARENTS_JSONL,
)"'
#   → artifacts/의료관련/chunks.jsonl  +  artifacts/parents.jsonl
#   ※ trust_grade=='A'만 임베딩 채택(행정규칙 B 2건 제외)은 이 chunk/임베딩 결합 단계의 책임.

# ── 3) 임베딩(512d) + 정규화 → 정규 산출물 chunks_with_vectors.jsonl ───────────
#   ★실제 OpenAI 과금 발생 구간 — 이번 빌드에서는 미실행. 운영 시 OPENAI_API_KEY 설정 후 실행.
#   embed/embed_client.cached_embed(512d, content-hash 캐시: 미스만 호출)로 임베딩 →
#   각 벡터 L2정규화 → {chunk_id,doc_id,parent_id,text,payload,vector[512]} 라인으로
#   config.CHUNKS_VEC_JSONL(= artifacts/의료관련/chunks_with_vectors.jsonl)에 기록.
#   (이 결합 스크립트는 아직 미구현 — §5-1 잔여작업. 우선 1콜 차원 확인:)
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -m embed.embed_client --selfcheck'  # 실 1콜, 512 최종확인

# ── 4) FAISS 인덱스 빌드 (임베딩 호출 없음 — 벡터는 src에 이미 존재) ───────────
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -c "import embed.faiss_index as f; f.build_index()"'
#   → artifacts/의료관련/faiss/{index.faiss, meta.jsonl}
#   (선택) sqlite-vec export:
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -c "import embed.faiss_index as f, config; f.export_sqlite_vec(config.MED_DIR/\"chunks.sqlite\")"'

# ── 5) 서버 (FAISS 인덱스를 read-only 볼륨 마운트) ─────────────────────────────
#   호스트에서 1~4로 인덱스를 빌드한 뒤:
docker compose up -d        # caddy + api (qdrant/redis 없음), ./artifacts/의료관련:…:ro 마운트
```

각 selftest(코드 변경 시 회귀 확인):
```bash
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -m search.retriever  --selftest'
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -m embed.faiss_index  --selftest'
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -m embed.medical_corpus --selftest'
```

---

## 5. 남은 일

### 5.1 [핵심 잔여] chunk→임베딩→정규 산출물 결합 스크립트 (미구현)
`docs/*.jsonl → chunk.build_chunks → cached_embed(512d) → L2정규화 → config.CHUNKS_VEC_JSONL`을 한 번에 잇는 결합기(또는 §4-3 인라인 스크립트의 정식화)는 아직 없다. 각 컴포넌트(`build_chunks`, `cached_embed`, `faiss_index.build_index`)는 완성·검증됐으므로 **얇은 글루 코드 1개**만 추가하면 된다. `build_index`는 이 결합기가 만든 `chunks_with_vectors.jsonl`을 그대로 소비하도록 이미 호환 검증됨.

### 5.2 [실제 의료 임베딩 실행 절차·비용추정] — 이번 빌드 미실행(과금 0)
- 절차: §4의 1→2→3→4. 3단계가 유일한 과금 구간. `OPENAI_API_KEY` 설정, content-hash 캐시(`artifacts/embed_cache.jsonl`)로 재실행 시 미스만 호출.
- **비용추정**(설계문서 §5, `text-embedding-3-small` @ $0.02/1M tokens, dimensions=512):

  | 구분 | 본 빌드 건수 | 토큰(추정) | 비고 |
  |---|---|---|---|
  | 국가법령 | 107 doc (42 법령) | ~2M | |
  | 행정규칙(A) | 270 doc | ~0.4–0.7M | B 2건 임베딩 제외 |
  | 판례 Phase-1 | 622 doc | ~6–8M | 대법원 전문 |
  | **합계** | **999 임베딩 doc** | **~8–11M tokens** | |

  → **임베딩 1회 약 $0.16–0.22** (Batch API 적용 시 50%↓ → ~$0.08–0.11). 법령·고시만이면 $0.05 미만. **비용은 무시 가능**하나 비용규칙상 이번 빌드에선 estimate만 제시하고 미실행.
  - 주: 설계문서는 판례 1,500–4,000건을 추정했으나 실제 Phase-1 채택은 **622건**(고precision 정책, §7-2). 따라서 실측 토큰·비용은 위 추정치보다 **낮을 가능성**이 크다.

### 5.3 [sqlite-vec export 검증] — 부분 완료
- BLOB fallback 경로는 selftest로 검증(512 float32 LE, 단위정규화·입력벡터 일치). **vec0 가상테이블(`vec_chunks`) 경로는 미검증** — 현재 venv에 sqlite-vec 확장 부재. 운영 환경에서 vec0 확장 로드 후 `export_sqlite_vec` 1회 실행하여 `vec_chunks` 생성·rowid 매핑·질의를 확인할 것. 호환 산출물(BLOB)은 확장 없이도 정상 생성되므로 차단 이슈 아님.

### 5.4 [판례 리콜 확장(향후)]
Phase-2(사건명/판시사항 키워드 union·하급심)는 `precedent_match_rules.yaml`에 정의만(`enabled:false`). 품질측정 후 `phase2.enabled=true` + 매처에 phase2 경로 추가로 리콜 확대 가능.

---

## 6. 통합 주의점 / 리스크

1. **[빌드 순서 — chunk.py 기본 sources 불일치]** `embed/chunk.py build_chunks()`의 기본 `sources`는 풀코퍼스 `artifacts/docs_*.jsonl`이고, `out_path` 기본값은 `artifacts/chunks.jsonl`이다. medical_corpus가 만드는 `artifacts/의료관련/docs/{국가법령,행정규칙,판례}.jsonl`과 **경로가 다르다.** 의료 서브코퍼스를 청킹하려면 §4-2처럼 `sources=`/`out_path=`를 **반드시 명시**해야 한다(인자 없이 호출하면 의료 서브셋이 아닌 풀코퍼스/빈 경로를 집어 잘못된 인덱스가 만들어질 수 있음). 5.1 결합 스크립트에서 이 경로를 고정할 것.

2. **[trust_grade B 처리 책임 경계]** 행정규칙 B 2건은 커버리지 정직성을 위해 `docs/행정규칙.jsonl`에는 기록되지만, **임베딩 제외(A만 채택)는 chunk/임베딩 결합 단계의 책임**(계약 §5). medical_corpus는 docs까지만 책임진다.

3. **[.env stale `EMBED_DIM=1536`]** 루트 `.env` 6행에 잔존. `config.py`가 무시하고 512로 강제하되 import마다 `RuntimeWarning` 출력(기능 무해). `.env`는 어느 빌더 소유도 아니므로 미수정 — **운영자가 `.env`의 `EMBED_DIM` 줄을 삭제**하면 경고 소멸. compose/Dockerfile에서는 `EMBED_DIM` 주입을 이미 제거함.

4. **[faiss-cpu 의존성]** `requirements.txt`에 `faiss-cpu==1.14.3` 추가 완료. venv에는 설치 확인. `docker compose up` 실빌드/헬스체크는 배포 환경에서 1회 확인 필요(WSL에 docker 미설치라 compose는 YAML 구조 검증만 수행됨).

5. **[retriever ↔ requirements 순서]** `qdrant-client`를 requirements에서 제거했고 retriever는 이미 FAISS로 이식 완료 → import 정합. (과거 리스크였던 retriever-qdrant 잔존 import는 해소됨.)

6. **[경로/인코딩 운영주의]** 한글 경로·전각 가운뎃점(`ㆍ` U+318D) NFC exact 매칭에 의존. Windows 도구(Read/Write/Edit)는 UNC `\\wsl.localhost\…`로 동일 파일 접근. 파이썬은 WSL venv로만, 한글 인자는 임시 `.py` 파일 경유.

---

## 7. 부록 — 본 빌드에서 건드리지 않은 것 (불변)

- `ingest/`(파서 4종), `header/`, `embed/chunk.py`, `search/{rag,verify,ad_review,source_pack,statutes}`, API: **무수정**(retriever만 호출). medical_corpus/faiss_index는 이들의 공개 함수만 소비.
- `config` 상수는 전부 import만, 재정의 없음.
