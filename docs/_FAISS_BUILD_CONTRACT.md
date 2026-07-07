# _FAISS_BUILD_CONTRACT.md — 의료관련 빌드 인터페이스 계약

> Owner: **Contracts**. 빌더들은 이 계약의 시그니처·레코드 스키마를 **그대로** 구현한다.
> 상위 모듈(`search/{rag,verify,ad_review,source_pack,statutes}`, API)은 무수정 — 오직
> `search/retriever.py`의 `search()`/`Hit`/`get_parent()`만 호출한다.
> 근거: `docs/의료관련_코퍼스_범위_추출전략.md` (Tier1/2 화이트리스트·행정규칙·판례 2-phase·512d·FAISS).

---

## 0. 불변 규칙 (절대 변경 금지)

| 항목 | 값 | 강제 위치 |
|---|---|---|
| 임베딩 모델 | `text-embedding-3-small` | `config.EMBED_MODEL` |
| 차원 | **512** (`dimensions=512`) | `config.EMBED_DIM == config.EMBED_DIMENSIONS == 512` (코드 고정) |
| 벡터 정규화 | **L2 정규화** (‖v‖₂=1) | 빌더가 적재 직전 정규화 |
| 인덱스 | `faiss.IndexFlatIP` (내적 = 코사인 동치) | `embed/faiss_index.py` |
| 벡터스토어 | FAISS + sqlite-vec **둘 다 적재 가능**. Qdrant·redis **제거** | — |
| 비용 | 이번 빌드는 **코드 + 오프라인 selftest까지**. 실제 대량 임베딩(OpenAI 과금) **금지**, estimate만 | — |

- `config.EMBED_DIM`은 코드에 512로 고정되어 env override를 무시한다. 모든 빌더는
  적재/검색 진입점에서 `assert len(vector) == config.EMBED_DIM`를 둔다(잘못된 차원 조기 차단).

### ⚠️ 통합 리스크 — 빌드 전 1회 정리 필요 (Contracts 권한 밖)
- 루트 `.env`에 `EMBED_DIM=1536`이 남아 있음. config.py가 이를 **무시하고 512로 강제**하되
  import 시 `RuntimeWarning`을 띄운다. `.env`는 본 계약의 수정 허용 범위(config.py + 본 문서)
  밖이므로 **운영자가 `.env`에서 `EMBED_DIM` 줄을 삭제**하면 경고가 사라진다. 기능엔 영향 없음.

---

## 1. 정규 레코드 — `chunks_with_vectors.jsonl` (`config.CHUNKS_VEC_JSONL`)

스토어 비종속 **정규 산출물**. FAISS와 sqlite-vec **둘 다** 이 파일 하나에서 적재된다.
JSONL 한 줄 = 청크 1개:

```json
{
  "chunk_id":  "LAW:000325:법률#제27조#0",
  "doc_id":    "LAW:000325:법률",
  "parent_id": "LAW:000325:법률",
  "text":      "<L1 인용헤더>\n<L2 맥락헤더>\n<정규화 본문>",
  "payload": {
    "doc_type":       "law|ordinance|admrule|precedent",
    "title":          "의료법",
    "article_no":     "제27조",
    "effective_from": "2024-08-07",
    "source_url":     "https://www.law.go.kr/...",
    "trust_grade":    "A|B",
    "...":            "그 외 09 §D-2 메타(jurisdiction, law_kind, part_idx, license, match_rule, matched_term, attachment_link …)"
  },
  "vector": [ /* 512 float, L2 정규화 (‖v‖₂ == 1.0 ± 1e-3) */ ]
}
```

규칙:
- `chunk_id`/`doc_id`/`parent_id`/`text`/`payload`는 `embed/chunk.py`의 청크 레코드와 동일 의미.
  `parent_id == doc_id`(parent = 법령·판례 전체). `chunk_id`는 전역 유일.
- `payload`는 `search/retriever.py`의 `Hit` 접근자가 읽는 키들을 **반드시** 포함:
  `doc_type, title, article_no, effective_from, source_url, trust_grade`
  (+ 필터키 `jurisdiction, law_kind, parent_id`). 기존 페이로드 레이아웃과 100% 호환.
- `vector`는 길이 512, **L2 정규화 후 저장**(IndexFlatIP 내적=코사인 전제). 정규화 책임은
  벡터 생성 측(`embed/faiss_index.build_index`)에 있으며, 입력이 미정규화면 적재 직전 정규화한다.

생성 경로: `embed/chunk.py`(chunk+content_hash) → `embed/embed_client.cached_embed`(512d, 캐시) →
정규화 → `chunks_with_vectors.jsonl`. (대량 임베딩은 이번 빌드에서 **실행 금지**; 빌더는 오프라인
selftest용 더미 벡터 또는 estimate만.)

---

## 2. `embed/faiss_index.py` — 공개 API (멱등)

```python
def build_index(src: pathlib.Path = config.CHUNKS_VEC_JSONL) -> None: ...
def load_index() -> tuple["faiss.Index", list[dict]]: ...
def export_sqlite_vec(out: pathlib.Path) -> None: ...
```

### `build_index(src=config.CHUNKS_VEC_JSONL) -> None`
- `src`(정규 JSONL)를 스트리밍하여 다음 2개 산출물을 생성:
  - `config.FAISS_INDEX` = `faiss.IndexFlatIP(512)`. 각 `vector`를 **L2 정규화 후** `index.add`.
    입력 순서 = 행 번호(row i). `index.ntotal == 줄 수`.
  - `config.FAISS_META` = JSONL. **행 i ↔ 입력 행 i 정렬**, 각 줄
    `{chunk_id, doc_id, parent_id, text, payload}` (벡터 제외 = 메타).
- 멱등: 동일 `src`면 결과 동일. 출력 디렉터리(`config.FAISS_DIR`) 없으면 생성, 기존 파일은 원자적 교체.
- 검증: `assert len(vector)==config.EMBED_DIM`; 정규화 후 `‖v‖≈1`; 빈/중복 `chunk_id` 거부.
- 비용: 임베딩 호출 없음(벡터는 `src`에 이미 존재). OpenAI 미접속.

### `load_index() -> (index, list[meta])`
- `config.FAISS_INDEX` 로드 + `config.FAISS_META`를 `list[dict]`로 로드(행 순서 보존).
- 반환 `(index, metas)`에서 `index.ntotal == len(metas)` 불변. `retriever`가 이걸 호출해
  검색 결과 row id → `metas[row]` → `Hit` 구성.
- 파일 부재 시 명확한 에러(빌드 선행 필요 안내).

### `export_sqlite_vec(out: pathlib.Path) -> None`
- 정규 JSONL을 sqlite-vec/HMS가 읽기 쉬운 형태로 내보냄. 권장 스키마(둘 다 채움):
  - `chunks(chunk_id TEXT PRIMARY KEY, doc_id, parent_id, text, payload JSON)`
  - `vec_chunks` (sqlite-vec `vec0`, `embedding float[512]`) — rowid ↔ `chunks` 매핑.
    sqlite-vec 미설치 환경 호환을 위해 `vector`를 `chunks.embedding BLOB`(512 float32 LE)로도 보관.
- 멱등: 동일 `out`·동일 `src` → 동일 DB(기존 테이블 DROP/재생성 또는 upsert). 벡터는 정규화본 저장.

---

## 3. `search/retriever.py` — 내부만 FAISS로 교체 (시그니처·`Hit` 100% 유지)

상위 모듈 무수정을 위해 **공개 표면은 절대 변경 금지**:

```python
@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]
    # .text .chunk_id .parent_id .doc_id .doc_type .title .article_no
    # .effective_from .source_url .trust_grade .location() 접근자 그대로 유지

def embed_query(query: str) -> list[float]: ...          # 512d, L2 정규화된 벡터
def search(query, k=config.DEFAULT_TOP_K, flt=None, as_of_date=None) -> list[Hit]: ...
def get_parent(parent_id: str) -> dict[str, Any] | None: ...
```

내부 동작(교체 대상):
1. `embed_query(query)` → 512d 벡터를 **L2 정규화**(검색측도 정규화해야 내적=코사인). 기존
   쿼리 LRU 캐시·`tenacity` 재시도 유지. `dimensions=512` 강제(§4).
2. `load_index()`로 `(index, metas)` 확보(프로세스 1회 로드·캐시; 스레드세이프).
3. FAISS top-`(k*over)` 검색(over-fetch). `over`는 post-filter로 잘려나갈 분을 흡수하는 배수
   (권장 기본 `over≈5`, `flt`/`as_of_date` 없으면 `over=1`).
4. 결과 row id들 → `metas[row]` → `Hit(id=chunk_id, score=내적(=코사인), payload=meta.payload + {text, chunk_id, doc_id, parent_id})`.
   `Hit.payload`는 기존 키 레이아웃 유지(접근자가 그대로 동작).
5. `flt`(`doc_type/jurisdiction/law_kind/effective_from/parent_id`)와 `as_of_date`
   (`effective_from <= as_of_date`, 누락/비ISO는 제외)는 **파이썬 post-filter**로 적용 후 **상위 k개**로 잘라 반환.
   - `as_of_date`는 ISO `YYYY-MM-DD`만 허용(기존 `_validate_iso_date` 재사용). 문자열 `<=` = 시간순 `<=`.
   - 허용 필터키는 기존 `ALLOWED_FILTER_KEYS` 유지(미허용 키 → `ValueError`).
6. `get_parent(parent_id)`는 **기존 `config.PARENTS_JSONL` 그대로** 사용(변경 없음).
7. `--selftest`: OpenAI·네트워크 없이 더미 512d 벡터 + 임시 FAISS 인덱스로 랭킹/필터/as_of/parent-promotion 검증
   (기존 in-memory Qdrant 셀프테스트를 FAISS in-memory로 대체, 동일 12체크 의미 보존).

제거: `qdrant_client`/`models` import, `build_filter`의 Qdrant `Filter` 생성, `_qdrant_query`,
`get_qdrant_client`. 필터는 파이썬 dict 매칭으로 재구현(같은 시맨틱).

---

## 4. `embed/embed_client.py` — `dimensions=512`

- `embed_texts`/`embed_query`(및 내부 `_embed_request`)는 OpenAI 호출 시
  `dimensions=config.EMBED_DIMENSIONS`(=512)를 **반드시** 전달:
  ```python
  _client().embeddings.create(
      model=config.EMBED_MODEL,
      input=inputs,
      dimensions=config.EMBED_DIMENSIONS,   # 512
  )
  ```
- 반환 벡터 `len == config.EMBED_DIM`(=512) 검증(`_validate_dim` 그대로, 512 기준).
- content-hash 캐시(`config.EMBED_CACHE_JSONL`)는 512d 벡터만 적재(차원 불일치 줄 skip — 기존 로직이
  `config.EMBED_DIM`로 비교하므로 1536d 레거시 캐시 줄은 자동 무시됨). **모델/차원 변경 시 캐시 키 공간이
  달라지므로 레거시 1536 캐시는 재사용 불가** — 새 캐시로 빌드.
- `cached_embed`는 미스만 OpenAI 호출. 이번 빌드는 대량 임베딩 **실행 금지**(estimate/selftest만).

---

## 5. `embed/medical_corpus.py` — 의료 서브코퍼스 추출

```python
def build_medical_corpus() -> None: ...   # manifest(YAML) 기반 → docs JSONL
```

- `config.MED_DIR/manifest/`의 YAML(`statutes_whitelist.yaml`, `admrule_targets.yaml`,
  `admrule_denylist.yaml`, `precedent_match_rules.yaml`)을 SoT로, **원천 6GB 무복사 스트리밍 필터-파서**.
- 기존 파서 재사용(무수정): `ingest/parse_statute.py`·`parse_admrule.py`·`parse_precedent.py`.
- 산출: `config.MED_DIR/docs/{국가법령.jsonl, 행정규칙.jsonl, 판례.jsonl}` (필터링된 `Document`)
  + `config.MED_DIR/_audit/{match_log.csv, absent_null.csv, dedup_log.csv}`.
- 규칙: 화이트리스트 폴더 실재 사전 assert(누락=빌드실패) · `trust_grade=B`(이미지/라벨only) 임베딩 제외·메타링크만
  · 재발령 dedup(`parse_admrule._revision_rank`) · 판례 Phase-1(형사+일반행정 **대법원만**, `## 참조조문` 정확매칭)
  + `match_rule`/`matched_term` 메타 기록 · `absent_null.csv`로 부재 4종 NULL 명시(커버리지 착시 차단).
- 이후 단계: `docs/*.jsonl` → `embed/chunk.py` → `embed/embed_client.cached_embed`(512d) → 정규화 →
  `config.CHUNKS_VEC_JSONL` → `embed/faiss_index.build_index()`.

---

## 6. 빌드 파이프라인 (요약)

```
manifest(YAML)
   └─ embed/medical_corpus.build_medical_corpus()      → MED_DIR/docs/*.jsonl  (무복사 필터-파서)
        └─ embed/chunk.build_chunks()                  → chunks(+parents)      (헤더·content_hash)
             └─ embed/embed_client.cached_embed(512d)  → vectors              (캐시·estimate만)
                  └─ (L2 정규화 결합)                    → CHUNKS_VEC_JSONL      ★정규 산출물
                       ├─ embed/faiss_index.build_index() → FAISS_INDEX + FAISS_META
                       └─ embed/faiss_index.export_sqlite_vec(out) → sqlite-vec/HMS
                            └─ search/retriever.search() (시그니처 불변, 내부 FAISS)
```

---

## 7. config 상수 (빌더가 import할 단일 출처)

| 상수 | 값 |
|---|---|
| `EMBED_MODEL` | `text-embedding-3-small` |
| `EMBED_DIM` / `EMBED_DIMENSIONS` | `512` (코드 고정, env 무시) |
| `EMBED_MAX_TOKENS` | `8191` |
| `MED_DIR` | `ARTIFACTS_DIR/'의료관련'` |
| `FAISS_DIR` | `MED_DIR/'faiss'` |
| `CHUNKS_VEC_JSONL` | `MED_DIR/'chunks_with_vectors.jsonl'` |
| `FAISS_INDEX` | `FAISS_DIR/'index.faiss'` |
| `FAISS_META` | `FAISS_DIR/'meta.jsonl'` |
| `PARENTS_JSONL` | `ARTIFACTS_DIR/'parents.jsonl'` (기존, parent-promotion) |
| `DEFAULT_TOP_K` | `8` |
| `EMBED_CACHE_JSONL` | `ARTIFACTS_DIR/'embed_cache.jsonl'` (content-hash 캐시) |

빌더는 위 상수를 `config`에서만 import하고 **재정의 금지**. 경로는 모두 POSIX(WSL) 기준이며
Windows 도구는 UNC `\\wsl.localhost\Ubuntu\home\user1\lawbot\...`로 동일 파일에 접근한다.
