# 나만의 lawbot API 구축 자료 — 한국 법률 RAG 스택 선택·파이프라인·체크리스트

> 대상 코퍼스: `D:\법파일\*.zip` (총 약 1,394만 항목, 9.3GB, 전량 공공저작물 — 저작권법 §7)
> 원천 프로젝트: [github.com/legalize-kr](https://github.com/legalize-kr) (precedent-kr / ordinance-kr / admrule-kr)
> 출처: 판례·행정규칙 = law.go.kr(국가법령정보센터), 자치법규 = ELIS(elis.go.kr)
> 본 문서는 멀티에이전트 리서치(phase3: 임베딩·벡터DB·청킹/검색·생성/인용·평가·API 프레임워크) 결과를 실무 구축 자료로 정리한 것이다.

---

## 0. 한눈에 보는 권장 스택 (TL;DR)

| 계층 | 1순위 권장 | 핵심 이유 | 대안 |
|---|---|---|---|
| **임베딩** | `nlpai-lab/KURE-v1` (BGE-m3 파인튜닝, 1024d, MIT, 8192토큰) | MTEB-ko 검색 1위(NDCG@10 0.6947), 법률 도메인 포함, 로컬 비용 0 | `dragonkue/BGE-m3-ko`, `BAAI/bge-m3` |
| **벡터DB** | **Qdrant**(셀프호스팅) 또는 **pgvector + pgvectorscale** | ACORN 필터(지역·종류·시행일) 최강 + dense·BM25 하이브리드 RRF / SQL 메타 제어 | Milvus(억대 확장), Weaviate, Pinecone(PoC) |
| **청킹** | 조문 단위(`current.articles/*.txt`) 그대로 + 부모-자식 | 이미 법적 인용 단위(조)로 사전 분할됨 | 판례는 `##` 섹션 + SAC 요약 주입 |
| **검색** | 하이브리드(BM25+dense) → RRF → 크로스인코더 리랭크 | 조문번호·법령명 정확매칭 + 의미검색 동시 확보 | — |
| **리랭커** | `dragonkue/bge-reranker-v2-m3-ko` (자체호스팅) | 한국어 AutoRAG F1≈0.91(0.9123), Apache-2.0 (GPU 50~100ms는 추정치) | Cohere Rerank 3.5(API) |
| **생성 LLM** | 하이브리드 라우팅(로컬 검색 + 상용 생성) | 한국어 법률 추론은 상용 우위, 데이터 주권 필요 시 로컬 | Claude/Gemini / HyperCLOVA X·Qwen2.5 |
| **API 프레임워크** | **FastAPI + Haystack 2.x** (또는 LlamaIndex) | 규제·법률 도메인 적합, 한국어 프로덕션 선례(urstory-rag) | LangChain/LangGraph(에이전트 확장) |
| **인용/출처** | Anthropic Citations / Cohere RAG citations 스키마 차용 | 구조화 인용 객체 + span 검증 표준 | — |
| **평가** | LegalBench-RAG(검색) + RAGAS(생성) + KBL/LRAGE(한국어 벤치) | 검색·생성·환각 3축 + 한국어 법률 정합 | 골든셋 + RAG Triad 회귀 |

> **핵심 전제 3가지**
> 1. 인덱싱 대상은 **`current.articles`의 현행 조문(약 329만) + 판례(123,743) ≈ 약 341만 청크**다. ZIP '총 항목수' 약 1,394만은 **과거 버전 스냅샷·서식 등을 포함한 raw 파일 수**이므로 인덱싱에서 반드시 제외한다.
> 2. 모든 데이터가 공공저작물(§7)이라 임베딩·파인튜닝·재배포·상용화에 라이선스 제약이 없다(구조·메타데이터는 legalize-kr 기준 MIT).
> 3. 법률 도메인은 **환각·오인용이 치명적**이다 — 인용 강제·사후 검증·면책 고지를 파이프라인에 명시적으로 둔다.

---

## 1. 코퍼스 정확 통계 (corpus_stats.txt 실측)

> 출처: `D:/법파일/_samples/corpus_stats.txt`

| ZIP | 크기 | 총 항목수(raw) | 법령(meta.json) | **현행 조문(.txt)** | 판례(.md) |
|---|---:|---:|---:|---:|---:|
| cases-판례.zip | 485.3 MB | 123,769 | 0 | 0 | **123,743** |
| statutes-광주광역시.zip | 1,267.7 MB | 1,979,293 | 5,270 | **272,687** | 0 |
| statutes-세종특별자치시.zip | 275 MB | 407,233 | 1,357 | **72,841** | 0 |
| statutes-전라남도.zip | 3,554 MB | 5,737,068 | 17,543 | **953,907** | 0 |
| statutes-전북특별자치도.zip | 2,802.9 MB | 4,296,575 | 12,128 | **678,513** | 0 |
| statutes-행정규칙.zip | 927.4 MB | 1,395,058 | 21,052 | **1,309,883** | 0 |
| **합계** | **~9.3 GB** | **≈13,938,996** | **36,298(자치)+21,052(행정)** | **3,287,831** | **123,743** |

- **인덱싱 대상 청크 ≈ 3,287,831(현행 조문) + 123,743(판례) ≈ 3,411,574개**
- 판례 법원등급: 대법원 68,175 / 하급심 55,566 / 미분류 1
- 판례 사건종류: 일반행정 45,104 · 민사 42,132 · 형사 21,667 · 세무 10,057 · 특허 3,373 · 가사 1,390
- 자치법규 종류(예, 광주): 조례 3,729 · 규칙 737 · 훈령 555 · 예규 230 · 의회규칙 19
- 행정규칙 종류: 고시 10,262 · 훈령 6,387 · 예규 3,866 · 공고 208 · 지침 167 · 국무총리훈령 100 · 대통령훈령 52

> ⚠️ **데이터 커버리지 한계(반드시 인지)**
> - 이 코퍼스는 **자치법규 4개 광역(광주·세종·전남·전북) + 국가 행정규칙 + 판례**뿐이다.
> - **국가 1차 법령(법률·시행령·시행규칙: 민법·형법·도로교통법 등)은 0건**이다. 행정규칙·자치법규는 모법(상위 법률)의 위임에 근거하므로, 모법 없이 답하면 법적 근거가 끊긴다.
> - 서울·경기·부산 등 13개 시도 자치법규, 헌재 결정례, 법령해석례, 행정심판 재결례도 미포함이다.
> - 따라서 서비스명/마케팅은 **"한국 자치법규·행정규칙·판례 검색 RAG"**로 정직하게 한정하고, 모법 부재 시 "상위 법률 미수록" 가드레일을 노출해야 한다.

---

## 2. 임베딩 모델 비교

법률 RAG의 검색 품질을 좌우하는 1차 변수다. 한국어 검색 특화 모델이 범용 다국어/상용 모델을 능가한다.

| 모델 | 차원/최대토큰 | 라이선스 | 한국어 검색 성능 | 비용 | 법률 적합도 |
|---|---|---|---|---|---|
| **KURE-v1** (nlpai-lab, BGE-m3 파인튜닝) | 1024d / 8192 | MIT | **MTEB-ko 1위** NDCG@10 0.6947, Recall@10 0.7968 | 로컬 0 (GPU 필요) | **매우 높음** (판례·조문 장문) |
| **BGE-m3-ko** / `dragonkue/BGE-m3-ko` | 1024d / 8192 | Apache-2.0 | AutoRAG F1 0.7456, dense+sparse+multivector 하이브리드 | 로컬 0 | 높음 (한·영 혼용, 하이브리드 강점) |
| **KoE5 / e5-large** | 1024d / 512 | MIT | 베이스라인, 512토큰 잘림, query/passage prefix 필수 | 로컬 0 | 중간 (짧은 조문만) |
| **ko-sroberta / ko-sbert** | 768d / 128 | — | KorSTS 85.6이나 검색 최적화 아님, 128토큰 | 로컬 0, CPU 가능 | **낮음** (법조문 부적합) |
| **OpenAI text-embedding-3 / Cohere v4 / v3** | 1536d~ / 128K | 상용 | 한국어 상대적 낮음, 운영 단순·무한확장 | $0.01~0.13/1M토큰 | 중간~낮음 (PoC·외부전송 한계) |

**권장**: `KURE-v1` 1순위. 한국어 법률 검색에서 BGE-m3·multilingual-e5·OpenAI-3-large를 모두 상회하고, 8192토큰으로 긴 조문·판례 섹션을 손실 없이 수용하며, MIT라 자체호스팅·파인튜닝·상용화 모두 자유롭다. 하이브리드(dense+sparse)를 단일 모델로 처리하려면 `BGE-m3` 계열을 택한다.

> ⚠️ **주의**: KURE 저장소의 "법률 도메인 포함 1위"는 KURE 자체 평가 근거이며, **본 코퍼스(조례·행정규칙) 전용 벤치마크는 별도로 수행**해야 한다(자체 골든셋 권장). 또한 임베딩 환경(GPU)과 서빙 환경(CPU TEI)의 **모델/토크나이저 버전이 한 글자라도 다르면 벡터 공간이 어긋나** 검색 품질이 조용히 망가지므로 버전을 고정한다.

---

## 3. 벡터 데이터베이스 비교

평가 4기준: ① 메타데이터 필터(지역·종류·시행일) ② 하이브리드 검색 ③ 셀프호스팅 여부 ④ 비용.

| DB | 메타필터 | 하이브리드 | 셀프호스팅 | 본 규모(~341만) 적합 | 비고 |
|---|---|---|---|---|---|
| **Qdrant** ⭐ | **ACORN 알고리즘, 강한 필터에서도 속도 유지** | dense+BM25 sparse 병렬→RRF 네이티브 | ✅ Apache-2.0 + Cloud | 단일~소수 노드로 충분, p50 ~3-4ms | Rust, 셀프호스팅 비용 최저권 |
| **pgvector + pgvectorscale** ⭐ | **SQL WHERE = 가장 정밀** | ParadeDB/pg_search로 결합 가능 | ✅ Postgres | ~329만 무난, 50M 초과 시 한계 | 단일 DB 운영, 시행일 range·조인 강점 |
| **Milvus / Zilliz** | 파티션 기반 | 네이티브 dense+sparse | ✅ + Cloud | **과잉**(억대 확장 대비용) | 운영 복잡도 최고 |
| **Weaviate** | 보통 | **가장 매끄러움**(단일 쿼리) | ✅ + Cloud | 적합하나 필터는 Qdrant 우세 | 복합 필터 지연 |
| **Pinecone** | 보통 | sparse-dense | ❌ 매니지드만 | PoC 한정 | 1B 벡터 storage 기준 대략 $1,350~3,600/월(쿼리·쓰기 포함 시 변동, 단일 공시가 없음), 락인·데이터주권 |
| **Chroma** | 약함 | 약함 | ✅ 로컬 | **부적합**(프로토타입) | 파이썬 친화 |
| **FAISS** | ❌ 직접구현 | ❌ | 라이브러리 | **부적합**(단독) | ANN 속도만, DB 아님 |

**권장**
- **1순위(균형)**: **Qdrant 셀프호스팅**. 지역·법령종류·시행일 복합 필터를 ACORN으로 가장 빠르게 처리하고, dense+BM25 sparse를 RRF로 네이티브 융합하며, 비용이 최저권이다.
- **공동 1순위(운영 단순)**: **pgvector + pgvectorscale**. 메타데이터를 SQL로 정밀 제어(시행일 range·기관 조인·trust_grade 필터)하고 단일 DB로 운영. 단 현행 조문 위주(~329만)로 인덱싱하고 50M 초과 시 한계.
- **확장 로드맵**: 전국 17개 시도 + 전체 과거버전(수천만~억대)이 확실하면 처음부터 **Milvus**.

> **공통 권고**: ① 1차 인덱싱은 `current.articles`만, 과거버전은 선택적·증분. ② **어느 DB든 한국어 형태소(kiwi/mecab/nori) 기반 BM25 sparse + dense 하이브리드 필수**(한국어 법률·금융 문서에서 BM25가 dense 단독보다 우수한 사례 다수). ③ `effective_from`/`promulgated_on`을 인덱싱해 시점검색을 메타 필터로 처리.

> ⚠️ **RAM 용량 함정**: dense 1024d FP32 raw = 341만×1024×4B ≈ **14GB**(양자화 전). 스칼라 양자화(int8) 적용 시 ~3.5GB이나 HNSW 그래프·sparse 인덱스·payload(한국어 텍스트)를 더하면 **실사용 RAM은 8~12GB**다. "8GB VPS로 충분"은 비현실적이며 **최소 16GB**를 권장한다.

---

## 4. 청킹 전략

본 코퍼스의 최대 강점: **자치법규·행정규칙이 이미 조문 단위(`current.articles/{seq}_{조문번호}_{조문제목}.txt`)로 사전 분할**되어 있어 청킹 로직이 거의 불필요하다.

### 4-1. 자치법규·행정규칙 (조문 단위)

```
# {법령명}                          ← 1행: 청크 헤더(메타로 승격)
# {조문번호} — {조문제목}            ← 2행: 청크 헤더(메타로 승격)
                                    ← 빈 줄
① 본문 ... <개정 2023.8.10.> ...    ← 임베딩 대상 본문(주석 분리 후)
② 본문 ...
```

- **1청크 = 1조문**(또는 항)을 기본 단위로. 첫 2행 헤더는 분리해 메타데이터(`ord_name`, 조문번호, 조문제목)로 승격하고 **본문만 임베딩**한다.
- **부모-자식(small-to-big)**: 항·호를 자식으로 색인해 정밀 검색하고, 부모(조문 전체)를 LLM에 전달해 컨텍스트 보존.
- **인라인 `<개정 …>` 주석**은 정규식으로 추출해 `amendment_dates` 메타로 빼고 본문 임베딩에서 제거(노이즈 감소). `① ②` 원문자도 정규화.

> ⚠️ **행정규칙 과편화 처리(중요)**: 행정규칙은 95.8%의 법령에서 **동일 조문번호가 2개 이상 txt로 쪼개져** 있다(예: 제4조가 5개 파일). 평균 조문 62.2는 분할 파일을 조문으로 오집계한 결과다. **`(법령ID + 조문번호)` 기준으로 흩어진 파일을 재병합(merge)**해 '조문 단위' 청크로 복원하는 것이 행정규칙 인덱싱의 핵심 전처리다. 또한 행정규칙 **27%(고시 다수)는 `current.articles`가 아예 없으므로**(본문이 attachments에만 존재) 최소한 제목+소관부처+종류+시행일을 메타 임베딩으로 색인해 검색 가능성을 확보한다.

### 4-2. 판례 (섹션 인지 + SAC)

```
## 판시사항    → 짧은 핵심 요약, 섹션=1청크 (고가중치)
## 판결요지    → 섹션=1청크
## 참조조문    → 인용 그래프 추출용
## 참조판례    → 인용 그래프 추출용
## 판례내용    → 길다 → 500~1000자 재귀 분할(200 오버랩)
```

- 판시사항·판결요지는 **사람이 쓴 고품질 요약**이라 섹션 단위 단일 청크로 고가중치 색인.
- 판례내용은 **boilerplate 유사도로 DRM(문서 오검색) 위험이 극단적으로 높으므로(계약문서 사례 95%+)**, SAC(Summary-Augmented Chunking)대로 **사건번호·사건명 포함 ~150자 문서요약을 각 청크 앞에 프리펜딩**(DRM 약 50% 감소).
- **섹션 결측 주의**: 표본 기준 판시사항 38%·판결요지 44%·참조판례 65%가 누락(특히 하급심). 결측 문서는 판례내용 첫 문단/주문·이유에서 의사-요지를 추출해 검색 리콜을 균질화한다.
- **비식별화(`○○○`) 정규화**: 표본 35%에 가명 처리가 있다. 임베딩 전 `[PERSON_n]` 토큰 치환으로 노이즈/동명 충돌을 완화하고, 인물 기반 질의 불가를 가드레일에 명시.

---

## 5. 검색 아키텍처

```
질의
 └─[메타 사전필터: region/ord_kind/effective_from/court_level/사건종류]
     ├─ BM25 sparse (형태소: nori/mecab-ko-dic/kiwi)  ─┐
     └─ dense 임베딩 (KURE-v1)                        ─┴→ RRF 융합(top-50)
                                                          └→ 크로스인코더 리랭크(bge-reranker-v2-m3-ko, top-8)
                                                              └→ LLM 컨텍스트
```

| 단계 | 권장 | 근거 |
|---|---|---|
| **1차 검색** | 하이브리드 BM25 + dense, **RRF 융합** | 정확 매칭(조문번호·법령명)+의미검색 동시. RRF가 점수 정규화 문제 회피, 하이브리드 후보 리랭킹이 단일 방식 일관 능가 |
| **형태소 분석** | nori(Elasticsearch)/mecab-ko-dic, kiwi | 조사·어미 분리로 한국어 BM25 매칭 품질 확보 (AutoRAG: okt 우수) |
| **융합 가중** | 가중 RRF, BM25:vector ≈ 1.0:0.7, 낮은 k(예 10) | 질의별 최적 alpha가 달라 랭크 기반 RRF가 점수결합보다 안정 |
| **리랭커** | `dragonkue/bge-reranker-v2-m3-ko`(자체) / Cohere Rerank 3.5(API) | 한국어(금융·일반) 리랭커 최고급, Apache-2.0 (GPU 50~100ms는 추정치). SLA 중요 시 Cohere |
| **메타필터** | Qdrant 사전필터(payload-aware) | 저선택도 필터 시 HNSW 끄고 payload 인덱스만 사용 → 더 빠름 |

> ⚠️ **상충 데이터 인지**: 일부 법률 연구(arXiv 2510.06999, gte-large)에서는 **dense-only가 하이브리드보다 나았다**는 상반 결과가 있다. 따라서 BM25:dense 가중·청크크기·리랭커 효용은 **자체 한국어 법률 쿼리셋으로 A/B 검증** 후 확정한다.

---

## 6. 생성 LLM·인용 강제·grounded generation

### 6-1. LLM 선택지

| 옵션 | 장점 | 단점 | 적합도 |
|---|---|---|---|
| **상용 프런티어**(Claude/GPT-4o·4.1/Gemini) | KBL 최상위(GPT-4 48.1%, Claude-3.5 42.5%), 네이티브 citations·structured output, MCP 통합 용이 | 데이터 외부전송, 토큰 비용 누적, RAG에도 17~33% 환각 | 높음(데이터 정책 해결 시 1순위) |
| **로컬/온프레**(HyperCLOVA X, Qwen2.5, EEVE, Solar) | 데이터 주권, 추론비 고정, 공공·B2G 유리 | 한국어 법률 정확도 열세, constrained decoding 직접 구축 | 중상(데이터 주권 필수 시) |
| **하이브리드 라우팅** ⭐ | 민감도·난이도별 로컬/상용 분리, 데이터 반출 최소화 | 아키텍처 복잡, 라우팅 오판 가드레일 필요 | 높음(실무 균형) |

> ⚠️ **현실 인지**: KBL 벤치마크에서 **상용 모델조차 변호사시험 정답률 42~48%**이며 RAG(open-book) 효과도 분야별 +7.4(형사)~-5.8(공법)로 불균일하다. **모델 단독 추론을 신뢰하지 말고 retrieval 결과에만 근거하도록 강제**해야 한다. 또한 공개 법률 데이터에는 최대 21% 오류가 섞일 수 있으므로(KBL 경고) `trust_grade` 필터·인용검증이 필수다.

### 6-2. 인용 강제 5단계 (2025 최신)

1. **구조화 출력**: JSON Schema + constrained decoding(상용=네이티브 structured output, 로컬=XGrammar+vLLM/SGLang)으로 모든 주장에 `citation` 필드(법령ID·조문번호·사건번호·source_url) **의무화**.
2. **attribute-first / quote-span**: 답변 문장이 실제 retrieved span과 겹치는지 사후 검증(CiteGuard/LLM-Cite/QUOTE-TUNING류).
3. **존재 검증(reference resolver)**: 인용된 조문번호/사건번호가 **코퍼스에 실재하는지** 정규식 추출 → `alr_bdt_id`/판례일련번호 대조(가짜 인용, 예: "형법 제9999조" 차단). `korean-law-mcp`의 `verify_citations`식 패턴 참고.
4. **correctness ≠ faithfulness**: 인용이 문장을 지지하는지(correctness)뿐 아니라 모델이 실제 그 문서에 의존했는지(faithfulness)도 평가(최대 57%가 사후합리화된 가짜 근거라는 보고).
5. **면책 자동화**: 답변마다 시스템 레벨에서 면책 고지 삽입.

### 6-3. 인용 객체 스키마 (Anthropic Citations / Cohere RAG citations 차용)

```json
{
  "answer": "...",
  "citations": [
    {
      "cited_text": "① 시장은 ... 한다.",
      "source_id": "ADMRULE:54908",        // alr_bdt_id 또는 판례일련번호(precSeq)
      "document_title": "부정청탁 신고사무 처리지침",
      "location": "제4조 제1항",            // 조문번호 또는 char 범위
      "source_url": "https://www.law.go.kr/...",   // meta.json source_url / 프론트매터 출처URL
      "effective_from": "2023-08-10",
      "trust_grade": "B"
    }
  ],
  "disclaimer": "본 답변은 법률 자문이 아니라 공공저작물(저작권법 §7) 원문 안내 목적입니다. 시행일·개정 여부를 확인하고 구체 사안은 변호사와 상담하십시오."
}
```

- 스트리밍은 SSE로 토큰을 흘리되, 인용은 Cohere처럼 **별도 `citation-start` 이벤트**로 전송하고 `fast`/`accurate` 두 모드 옵션화(법률은 정밀도 중요 → `accurate` 기본값).
- 응답에 **버전·이력 정보 포함**: 자치법규는 과거 스냅샷이 있으므로 `effective_from`/`promulgated_on`과 "시행일 기준" 답변(as-of-date 패턴)을 지원.

---

## 7. 법적·윤리적 가드레일 (한국 배포)

| 항목 | 의무/리스크 | 설계 대응 |
|---|---|---|
| **변호사법 §109** | 비변호사의 유상 법률사무 취급 금지(7년↓/5천만원↓) | '법률정보 제공·검색'으로 한정, 구체 사건 결론·문서작성·대리 금지. 유료화 시 변호사 검수 결합 |
| **로폼 판례(대법 2025두35483)** | 표준서식 자동작성은 적법, **개별 사실관계 기반 판단·생성형 문서작성은 위반 소지** | 결론적 법률판단 회피, '원문 인용+출처 안내' 성격 유지 |
| **AI기본법(2026.1.22 시행)** | 생성형 AI 사전 고지 + 결과물 AI 생성 표시(미이행 과태료 3천만원↓) | UI·약관에 사전 고지, 답변에 'AI 생성' 가시 표시. 고영향 AI 해당 시 위험관리·문서 5년 보관 |
| **개인정보보호법** | 챗봇 입력 민감정보(§23) 별도 명시 동의, 목적외 학습 금지(이루다 선례) | 민감정보 동의 화면, 학습 비활용 기본값, 입력 최소화·마스킹, 국외이전 고지 |
| **데이터 라이선스** | 본문=공공저작물(§7) 자유이용 / 구조·메타=MIT | 출처표시(국가법령정보센터/ELIS) + '비공식본·정확성 미보증' 병기, 커밋 해시 대신 안정 키(법령ID·사건번호) 사용 |

> legalize-kr 생태계는 force-push로 Git 이력을 재작성하므로 **커밋 해시를 영구 식별자로 쓰지 말고** `alr_bdt_id`·판례일련번호·`source_url`을 인용 키로 고정한다.

---

## 8. RAG 평가 방법론

검색·생성·환각 3축의 계층형 하이브리드 평가를 권장한다.

| 층 | 도구/지표 | 역할 |
|---|---|---|
| **검색** | LegalBench-RAG 방식(Recall@k·Precision@k·MRR·nDCG) | judge 편향 없는 1차 게이트. 조문 청크·판례 `##참조조문`을 정답 span으로 |
| **생성** | RAGAS(faithfulness·answer relevancy·context precision/recall) | 충실성·정답성. 한국어 강한 judge + 다중 judge 합의 + 법률용 custom threshold |
| **환각** | span-level citation verification, 인용 존재 검증 | 환각률을 **명시 KPI로 추적**(상용 도구도 17~43% 환각) |
| **한국어 벤치** | KBL(판례 15만·법령 22만 조문 RAG) / LRAGE(ablation) / KMMLU-Pro | 외부 비교 기준·통합 ablation·회귀 |
| **회귀** | 골든셋 + RAG Triad + CI 연동(judge 버전 고정) | 모델·프롬프트·인덱스 변경 시 드리프트 자동 감지 |

> ⚠️ legalize-kr는 평가 벤치마크를 제공하지 않으므로 **자치법규·행정규칙용 한국어 골든셋(법률 전문가 검수, 수백~수천 QA)을 직접 구축**해야 한다. LegalBench-RAG는 영미 계약서 기반이라 데이터 재사용 불가 — 방법론만 차용한다. **"RAG=환각 해결"로 전제하지 말 것.**

---

## 9. API 프레임워크 비교

| 옵션 | 강점 | 약점 | 한국 법률 적합도 |
|---|---|---|---|
| **FastAPI + Haystack 2.x** ⭐ | 타입드 컴포넌트, 단계별 계측, 규제산업 권장, **한국어 선례 urstory-rag** | 초기 설계 학습(약 1주) | **매우 높음** |
| FastAPI + LlamaIndex | 빠른 구현(2~3일), 150+ 커넥터, 하이브리드 내장 | 복잡 에이전트는 조합 필요 | 높음(PoC) |
| FastAPI 직접 구현 | 최소 의존성, 조문 사전분할로 청킹 불필요 | 하이브리드·리랭크 직접 유지보수 | 높음(튜닝 책임) |
| FastAPI + LangChain/LangGraph | 최다 통합, 에이전트 워크플로 | 오버헤드 최대, 잦은 API 변경 | 중간(인용검증 에이전트 확장 시) |

**권장 구성**
- **API 계층** = FastAPI: SSE `StreamingResponse` 스트리밍, 전 구간 async, Pydantic 검증 + OpenAPI 자동 문서.
- **RAG 오케스트레이션** = Haystack 2.x(규제·법률 적합). 빠른 PoC면 LlamaIndex로 시작 후 이관.
- **횡단 요소**: 인증=JWT+API key, 레이트리밋=slowapi+Redis(IP가 아닌 키 기준), 캐싱=Redis(임베딩·응답 SHA-256 키), LLM=async 클라이언트 + 지수백오프(RateLimit만 재시도), Nginx `proxy_buffering off`(SSE 보장), `/health` + 수평 확장.
- **품질·관측성** = RAGAS 자동평가 + Langfuse 트레이싱 + 가드레일(PII·인젝션·환각).
- **레퍼런스 아키텍처**: `urstory/urstory-rag`(FastAPI+Haystack 2.9+PGVector+ES Nori+bge-reranker-v2-m3-ko+Redis+RAGAS+Langfuse)를 그대로 차용하고 **임베딩만 KURE-v1로 교체**.

### 권장 엔드포인트 설계

```
POST /v1/answers              # 질의 → 인용 포함 답변(SSE 지원)
POST /v1/search               # 원시 하이브리드 검색(필터 파라미터)
GET  /v1/statutes/{lawId}/articles/{articleNo}   # 조문 직접 조회
GET  /v1/precedents/{precSeq}                     # 판례 조회
GET  /healthz
# + legalize-mcp 방식 MCP 서버로 동일 기능을 AI 에이전트에 도구로 노출(차별화)
```

---

## 10. 검색→생성→인용 파이프라인 설계

```
[1] 인제스트(ETL)
    zip 스트리밍 추출 → current.articles + 판례 .md만 선별(과거 스냅샷·README 제외)
    → 유형별 파서(판례 YAML+섹션 / 법령 헤더2줄+meta.json 조인)
    → 정규화(① ②→(1), <개정> 주석 분리·메타화, 유니코드 정규화)
    → 행정규칙 (법령ID+조문번호) 재병합
    → 통합 doc_id·chunk_id 부여({alr_bdt_id}#{조문번호})
    → 중복 해시 제거
    → 임베딩(KURE-v1 dense + 형태소 BM25 sparse)
    → 벡터DB 적재(payload: region/ord_kind/effective_from/trust_grade/source_url/court_level/사건종류)
        ※ 별표·histories는 메타로만 보존
[2] 검색
    질의 → 메타 사전필터 → BM25+dense 하이브리드 → RRF(top-50) → 크로스인코더 리랭크(top-8)
[3] 생성
    검색 청크를 출처 인용과 함께 컨텍스트 조립
    → "제공된 컨텍스트에만 근거, 없으면 모른다고 답하라(refusal)" 시스템 프롬프트
    → 구조화 출력(JSON Schema, constrained decoding)으로 citation 필드 의무화
[4] 인용 검증
    인용 조문/판례가 retrieved 집합·코퍼스에 실재하는지 기계 검증(reference resolver)
    → span 겹침(faithfulness) 확인 → 미지원 문장 삭제/플래그
[5] 후처리
    effective_from으로 현행성 표시, trust_grade 노출, 면책 고지 자동 삽입
    → SSE 스트리밍 + citation-start 이벤트
```

### 통합 데이터 모델 (3유형 단일 스키마)

| 공통 필드 | 판례 전용 | 법령 전용 |
|---|---|---|
| `doc_id`, `doc_type`, `title`, `source_url`, `license`, `trust_grade` | `case_no`, `court_name`, `court_level`, `case_category`, `선고일자`, `ref_articles`, `ref_cases` | `alr_bdt_id`, `ord_kind`, `region.{sido,sigungu,branch}`, `effective_from`, `promulgated_on`, `estrev_label`, `조문번호`, `조문제목`, `histories[]`, `attachments[]` |

> ⚠️ **region 의미 충돌**: 행정규칙은 `region.sido`=소관부처, `region.sigungu`=기관으로 자치법규(지역명)와 의미가 다르다. 통합 시 `jurisdiction_type`(중앙/지방)·`competent_ministry`·`agency`로 정규화하고 `doc_type`별 분기한다. ID도 자치법규=숫자, 행정규칙=`ADMRULE:n`, 판례=precSeq로 이질적이므로 `doc_id` 네임스페이스를 통일한다.

---

## 11. 임베딩 비용·규모 추정

> 토큰 추정은 **실측 기반**으로 보정해야 한다. 행정규칙 current.articles 샘플 평균 ~115자(≈130~170토큰)로, 흔히 가정하는 "조문 ~400토큰"은 과대다. 판례는 .md 1파일 평균 ~4,250토큰(p90 8,277, 최대 99,679) → **8,192토큰 한도 초과분이 잘리므로 섹션 분할 필수**.

| 항목 | 추정 |
|---|---|
| 인덱싱 대상 청크 | 현행 조문 ~329만 + 판례 섹션 재청킹(문서당 6~10청크) ≈ **약 400만 청크** |
| 총 토큰(실측 보정) | 조문 ~362M + 판례 ~526M ≈ **약 0.89B** (가정에 따라 0.9~2.0B) |
| 1회 임베딩 비용(상용 저가형 $0.01~0.02/1M) | **약 $9~32** (1회성) |
| 로컬 임베딩(KURE-v1 GPU) | 모델·전력 비용만, API 비용 0 (CPU는 수십 시간 소요 → 초기 배치는 GPU 임대 권장) |

> **결론**: 1회성 전량 임베딩은 수십~수백 달러 규모로 데이터량(9.3GB) 대비 관리 가능. **청크 텍스트+해시를 저장**해 모델 교체 시 증분 재임베딩(모델 변경=전량 재임베딩 강제이므로 비용 대비). 과거버전까지 잘못 포함하면 1,394만으로 3배 팽창하므로 **반드시 제외**.

---

## 12. 구축 체크리스트

### 데이터 준비
- [ ] `corpus_stats.txt`로 인덱싱 대상 확정: 현행 조문 ~329만 + 판례 123,743 ≈ 341만 (과거 스냅샷 제외)
- [ ] **current.articles 대 과거버전 실제 비율을 인제스트 전 측정**(RAM 예산 검증)
- [ ] 라이선스 확인: 본문=공공저작물(§7), 구조/메타=MIT, 출처 약관(law.go.kr/ELIS) 검토
- [ ] 데이터 커버리지 한계 명시: 국가 법률 본문·13개 시도·헌재/해석례 부재 → 서비스명 정직하게 한정
- [ ] (선택) 확장: `legalize-pipeline`(LAW_OC 키) + `compiler`로 전국 17개 시도·국가법령 보강

### 전처리/인제스트
- [ ] 유형별 파서: 판례 YAML 프론트매터+`##`섹션 / 법령 헤더2줄+meta.json 조인 (legalize-kr `cli-tools` 재사용, MIT)
- [ ] 텍스트 정규화: `① ②`→`(1)`, `<개정 …>` 주석 분리·메타화, 유니코드 정규화, `○○○`→`[PERSON_n]`
- [ ] **행정규칙 (법령ID+조문번호) 재병합**(95.8% 과편화), 무조문 27% 메타 색인
- [ ] 판례 SAC: 사건번호·사건명 포함 ~150자 문서요약 프리펜딩
- [ ] 통합 `doc_id`/`chunk_id`({alr_bdt_id}#{조문번호}) + 중복 해시 제거
- [ ] payload 구성: region/ord_kind/effective_from/promulgated_on/trust_grade/source_url/court_level/사건종류

### 모델/인프라
- [ ] 임베딩: KURE-v1(또는 BGE-m3-ko) — **임베딩·서빙 모델/토크나이저 버전 고정**
- [ ] 임베딩 서빙: HuggingFace TEI (동적 배칭, ONNX)
- [ ] 벡터DB: Qdrant(named vectors dense+sparse, HNSW, 스칼라 양자화) 또는 pgvector+pgvectorscale — **최소 16GB 노드**
- [ ] 형태소 BM25: nori/mecab-ko-dic/kiwi sparse 인덱스
- [ ] 리랭커: bge-reranker-v2-m3-ko(자체) 또는 Cohere Rerank 3.5(API)
- [ ] 메타DB(선택): Postgres — meta.json/histories 조인·시점 쿼리

### 검색·생성
- [ ] 하이브리드(BM25+dense) → 가중 RRF(BM25:vector≈1.0:0.7) → 크로스인코더 리랭크(top-8)
- [ ] 메타 사전필터(지역·종류·시행일·법원등급), 기본 현행본만 검색, trust_grade A 가중
- [ ] 시스템 프롬프트: retrieved 컨텍스트에만 근거, 없으면 refusal
- [ ] 구조화 출력(JSON Schema, constrained decoding)으로 citation 필드 의무화
- [ ] 인용 존재 검증(reference resolver) + span faithfulness 검증 + 미지원 문장 삭제
- [ ] 면책 고지 자동 삽입 + effective_from/trust_grade 노출 + as-of-date 지원

### API/운영
- [ ] FastAPI + Haystack 2.x, SSE 스트리밍(citation-start 이벤트, accurate 기본), async
- [ ] 인증 JWT+API key, 레이트리밋 slowapi+Redis, 응답/임베딩 캐시(SHA-256)
- [ ] (선택) MCP 서버 노출(Claude Desktop/Cursor 호환)
- [ ] 비밀키 환경변수/Secrets 관리, Caddy/Nginx 자동 HTTPS

### 법무/평가/거버넌스
- [ ] 변호사법 §109 가드레일(법률정보 제공 한정, 결론적 판단 회피)
- [ ] AI기본법 사전 고지 + AI 생성 표시, 개인정보 민감정보 동의·학습 비활용
- [ ] 골든셋(전문가 검수) 구축 → LegalBench-RAG(검색)+RAGAS(생성)+환각 KPI 추적
- [ ] KBL/LRAGE 외부 벤치 정합, CI 회귀 테스트(judge 버전 고정)
- [ ] 인용 키는 커밋 해시 아닌 alr_bdt_id·precSeq·source_url 사용
- [ ] 데이터 갱신 파이프라인(증분 재임베딩, force-push 동기화 전략)

---

## 13. 단계별 도입 로드맵 (저비용 → 프로덕션)

| 단계 | 스택 | 월 비용(추정) | 비고 |
|---|---|---|---|
| **PoC/MVP** | 단일 VPS(16GB) Docker Compose: Caddy+FastAPI+Qdrant+TEI(embed/rerank) + 외부 LLM 종량제 | ~₩2만 + LLM 종량 | 거의 무료 시작, 단일 노드 SPOF |
| **매니지드** | Qdrant Cloud + 상용 임베딩/리랭크/LLM + Render/Modal + Upstash | ~$80~140 | 운영 부담 최소, 벤더 락인 |
| **프로덕션** | K8s(CPU+GPU 노드풀) + 자체 GPU 임베딩/리랭커 + Qdrant + 상용 LLM + 관측성/CI-CD | ~$2,600~3,200 | 확장성 최상, DevOps 역량 필요 |

> ⚠️ **단계 선택 원칙**: 일 1만 쿼리 미만이면 **GPU 상시 가동(월 ~$580)은 매몰비용**이다. 쿼리 임베딩은 API/CPU TEI로 충분하고 GPU는 초기/재인덱싱 배치에만 온디맨드로 빌린다. 손익분기(일 1만+ 쿼리, 월 $500+) 도달 후 프로덕션 K8s로 이관한다. 고트래픽 시 비용 지배항은 검색이 아니라 **LLM 생성**이므로 시맨틱 캐시·소형모델 라우팅이 핵심 절감 레버다.

---

## 14. 출처

**오픈소스 lawbot 선례**
- hunsii/LawBot — Sentence-BERT 임베딩 + 코사인 유사도 판례검색 + KORANI 13B: https://github.com/hunsii/LawBot
- boostcampaitech5 nlp-08 LawBot — 질문필터+유사판례+legal-llama-2-ko: https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-08
- urstory/urstory-rag — 한국어 프로덕션 RAG(FastAPI+Haystack+PGVector+bge-reranker-ko): https://github.com/urstory/urstory-rag

**임베딩·리랭커**
- nlpai-lab/KURE-v1: https://huggingface.co/nlpai-lab/KURE-v1 · https://github.com/nlpai-lab/KURE
- dragonkue/BGE-m3-ko: https://huggingface.co/dragonkue/BGE-m3-ko
- dragonkue/bge-reranker-v2-m3-ko: https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko
- AWS 한국어 Reranker로 RAG 성능 올리기: https://aws.amazon.com/ko/blogs/tech/korean-reranker-rag/

**벡터DB·검색**
- Qdrant 하이브리드/필터: https://qdrant.tech/documentation/search/hybrid-queries · https://qdrant.tech/articles/vector-search-filtering/
- Qdrant 1.15 BM25/멀티링궐 토크나이저: https://qdrant.tech/blog/qdrant-1.15.x/
- pgvector vs Qdrant 벤치: https://www.tigerdata.com/blog/pgvector-vs-qdrant
- AutoRAG BM25 한국어 토크나이저: https://marker-inc-korea.github.io/AutoRAG/nodes/retrieval/bm25.html
- Nori 한국어 형태소 분석: https://www.elastic.co/docs/reference/elasticsearch/plugins/analysis-nori

**청킹·환각·인용**
- Reliable Retrieval / SAC·DRM (arXiv 2510.06999): https://arxiv.org/html/2510.06999v1
- LegalBench-RAG (arXiv 2408.10343): https://arxiv.org/html/2408.10343v1
- Correctness ≠ Faithfulness (arXiv 2412.18004): https://arxiv.org/abs/2412.18004
- Stanford 법률 AI 환각 실증(JELS 2025): https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf
- Anthropic Citations API: https://platform.claude.com/docs/en/build-with-claude/citations
- Cohere RAG Citations: https://docs.cohere.com/docs/rag-citations
- chrisryugj/korean-law-mcp(인용검증): https://github.com/chrisryugj/korean-law-mcp

**한국어 법률 벤치마크·LLM**
- KBL (arXiv 2410.08731): https://arxiv.org/html/2410.08731v1 · https://github.com/lbox-kr/kbl
- LRAGE (arXiv 2504.01840): https://github.com/hoorangyee/LRAGE
- KMMLU-Pro (arXiv 2507.08924): https://arxiv.org/html/2507.08924
- RAGAS: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

**평가 프레임워크/규제**
- 변호사법 §109: https://www.law.go.kr/법령/변호사법/제109조
- 저작권법 §7(보호받지 못하는 저작물): https://casenote.kr/법령/저작권법/제7조
- AI기본법: https://www.law.go.kr/lsInfoP.do?lsiSeq=268543
- 로폼 판례(대법 2025두35483): https://www.lawtimes.co.kr/news/articleView.html?idxno=218044

**코퍼스 원천**
- legalize-kr 조직: https://github.com/legalize-kr
- precedent-kr / ordinance-kr / admrule-kr / cli-tools / legalize-pipeline (각 저장소)
- 국가법령정보 공동활용 OpenAPI: https://open.law.go.kr/LSO/openApi/guideList.do
- 로컬 통계: `D:/법파일/_samples/corpus_stats.txt`

