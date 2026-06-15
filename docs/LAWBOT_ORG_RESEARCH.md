# lawbot.org 조사 — 정식 버전 레퍼런스

> 2026-06-15 조사. lawbot.org(한국 법률 데이터 인프라 서비스)의 공개 기능을 전수 조사해
> **우리 정식 버전 빌드에 참고**하기 위한 문서. 각 항목에 "우리 현황 / 정식 TODO" 매핑 포함.
> 출처: lawbot.org 공개 페이지(home, about, docs, docs/mcp, verify, search, source-pack), npm 레지스트리.

## 0. 한눈에 — lawbot.org 정체성

- 포지셔닝: **"챗봇이 아니라 법률 데이터 인프라."** 답을 *생성*하지 않고, 다른 AI가 만든 답의 **인용을 검증**한다.
- 슬로건: *"법률 AI는 거짓말을 자주 합니다. 우리는 그걸 잡아냅니다."*
- 현재 **베타 v0.1.0** — 일부 페이지는 404(미완), `/verify`는 Phase 1 목업 응답. 라이브 DB 연동은 Phase 4 예정.
- **납품 형태 3가지**: REST API · 호스팅 인덱스 · **MCP 서버**.

## 1. 핵심 기능 4종

| lawbot.org 기능 | 설명 | **우리 현황** | **정식 TODO** |
|---|---|---|---|
| **RAG API** | 조/항/호/목 단위 청킹 + `as_of_date` 시점 검색 | ✅ `/v1/ask`, `/v1/statutes/search` 있음 | 전량 임베딩 후 품질 확보 |
| **Source Pack** | 적용조문·판례·행정해석을 LLM 인용용 markdown으로 번들 | ✅ `/v1/source-pack` 있음 | 출력 포맷 정렬 |
| **Citation Firewall** | AI 답변의 인용 존재·정확·현행성 검증 (§3) | ✅ `/v1/verify` + law.go.kr 검증 있음 | 신뢰점수 0~100 노출 |
| **MCP Server** | Claude/Cursor/ChatGPT에서 도구로 사용 (§2) | ❌ 없음 | **신규 추가** |

## 2. MCP 서버 (우리가 추가할 것)

**중요 발견: `@lawbot/mcp` npm 패키지는 공개 레지스트리에 없음.** 실제로는 **로컬 git clone + stdio 방식**으로 동작.

- 설치(lawbot.org 방식): repo 클론 → `pnpm install` → `pnpm build && pnpm start`. (Node/pnpm 스택)
- 클라이언트 등록: `claude_desktop_config.json` 의 `mcpServers`에 `command/args/cwd/env(LAWBOT_API_URL, LAWBOT_API_KEY)` 추가.
- **노출 MCP 도구 4종**:
  - `search_authorities` — 법령·판례·해석 통합검색(시점검색 지원)
  - `get_article` — 법령명+조문번호로 단일 조문 조회
  - `make_source_pack` — LLM 인용용 근거 번들 생성
  - `verify_citations` — Citation Firewall(인용 검증)
- 아키텍처: MCP는 **HTTP API를 감싸는 래퍼**(DB 직접 연결 X) → 인증·레이트리밋·감사로그 중앙화.

> **우리 적용**: 우리는 Python/FastAPI라 Node 대신 Python MCP SDK(`mcp` 패키지)로 서버를 만들고,
> 이미 있는 `/v1/ask·/v1/statutes/search·/v1/verify·/v1/source-pack`을 MCP 도구 4종으로 감싸면 됨.
> **RAG·검색 로직 재사용 → "포장지"만 추가**라 가벼움(대략 1일 안쪽).

## 3. Citation Firewall / `/v1/verify` (신뢰점수 반영)

사용자가 본 "검증결과 + 신뢰점수 + 존재여부 확인"이 바로 이것.

**입력**: `text`(검증할 LLM 답변) + 기준일.

**4단계 검증 파이프라인:**
1. **인용 추출** — regex + LLM로 법령명·조문번호·사건번호·인용문 식별
2. **존재 검증** — DB 조회로 실제 존재 확인 (**허위 사건번호 탐지**)
3. **문구 정확성** — 인용문 vs 원문 정규화 비교, **95%+ 일치** 요구 (Levenshtein, **변조 인용 탐지**)
4. **시점 유효성** — 기준일에 해당 조문이 유효했는지 (**폐지·개정 탐지**)

**출력 / 점수**: 빨강/노랑/초록 **플래그** + about 페이지 파이프라인의 **04 Trust 단계 = 0~100 신뢰점수** → block/warn/proceed 결정 + A/B/C/D 등급.

> **우리 적용**: `search/verify.py`에 이미 존재·문구·시점 검증 골격 있음(law.go.kr OpenAPI 대조).
> **정식 TODO**: ① 검증 결과를 **0~100 신뢰점수**로 환산해 응답에 노출 ② 각 인용별 빨강/노랑/초록 플래그
> ③ `/chat`·`/verify` UI에 점수·존재여부 시각화. (사용자가 본 화면이 목표 UX)

## 4. API 엔드포인트 (lawbot.org)

| Method | Path | 용도 |
|---|---|---|
| GET | `/v1/statutes/search` | 법령 검색(`q`, `as_of`, `limit`) — 출처·라이선스·신뢰등급 반환 |
| POST | `/v1/retrieve` | 통합 RAG(법령·판례·해석) |
| POST | `/v1/source-pack` | LLM 인용용 번들 생성 |
| POST | `/v1/verify` | Citation Firewall |

- **인증**: 헤더 `x-api-key`. (우리는 `Authorization: Bearer lk_...` — 차이 있음, 통일 검토)
- **레이트리밋**: 무료 데모 30 req/min(IP별), 발급키는 tier별 호출·동시성 제한.
- 응답 공통 메타: `output · as_of · source · trust_grade`.

## 5. 가격 / 티어

- **공개된 가격 없음**(미공개). 무료 데모(30 req/min) + tier별 API 키 + 엔터프라이즈 VPC/온프렘(문의).

## 6. 데이터 / 임베딩 인프라 (우리와 가장 큰 차이)

| 항목 | lawbot.org | **우리** |
|---|---|---|
| 임베딩 모델 | **Qwen3-Embedding-8B, 4096차원**(자체 호스팅, OpenAI 호환 endpoint) | OpenAI `text-embedding-3-small`, **1536차원** |
| 벡터스토어 | **Postgres + pgvector**(HNSW 2000차원 한계로 <1만건은 flat scan) | **Qdrant** |
| 검색 | 한국어 BM25(pg_trgm) + 벡터 **하이브리드(RRF k=60)** | 벡터 검색 (BM25 하이브리드는 우리 TODO) |
| 코퍼스 | 국가법령 5,670 / 자치 14,366+ / 8개 판례분야 (MVP는 <1만건) | 311,005 문서 / 약 319만 청크 (전량 임베딩 시) |
| 데이터 출처 | 국가법령정보 공동활용·사법정보공유포털·공공데이터포털 (KOGL 1·2·3) | 동일 출처 |

> **시사점**: ① lawbot.org는 자체 임베딩(Qwen3)으로 비용 0 + 4096차원. 우리는 OpenAI(비용 발생, 1536).
> 정식에서 비용이 크면 **오픈 임베딩 자체호스팅** 검토 가치 있음. ② **BM25+벡터 하이브리드(RRF)**는
> 한국어 법령 검색 정확도에 중요 — 우리도 도입 검토. ③ lawbot.org MVP도 실제 색인은 <1만건 → 우리 데모 규모와 비슷.

## 7. 차별점 / 로드맵

- 차별점 4: **생성보다 검증** · **SaaS 아닌 인프라** · **출처 동반 출력**(타임스탬프+출처+등급) · **Citation Firewall**.
- 매니페스토: 근거 없으면 침묵(임계 미만 신뢰도 → 생성 안 함) · 출처+시점 결합 · "사람이 자문, 기계는 정보제공" · 감사가능성 기본(한국 AI기본법 2026-01-22 대응).
- 로드맵: Phase 4 = verify 라이브 DB 연동, Phase 6 = VPC/온프렘.
- 고객군: AI 앱 개발자 · 리걸테크 스타트업 · 기업 법무/컴플라이언스 · 규제기관.
- SDK 없음 — REST(`x-api-key`) + MCP 서버가 유일 통합 수단.

## 8. 우리 정식 버전 액션 요약

1. **전량 임베딩 + Qdrant 프로덕션** (최우선, 데이터 품질).
2. **MCP 서버 추가** — Python MCP SDK로 기존 `/v1/*` 4종 래핑(§2).
3. **신뢰점수(0~100) + 인용 플래그 UI** — verify 결과 시각화(§3, 사용자 요청 UX).
4. **BM25+벡터 하이브리드(RRF)** 검색 도입 검토(§6).
5. (선택) **임베딩 비용** 커지면 오픈 임베딩 자체호스팅 검토(§6).
6. UX: 비법률 입력 안내 · `/chat` raw JSON 제거 · 인증 통일.
