# lawbot 빌드 진행상황 (계속 갱신)

최종 갱신: 2026-06-29. 범용 한국법 검색·자문 제품으로 만드는 작업의 단계별 완료 기록.
계획 전문은 `지시서/README.md`, 배경은 `docs/lawbot_최종_정리본.md`.

서빙 환경변수(전체 인덱스+하이브리드+데모): `LAWBOT_INDEX=full LAWBOT_HYBRID=1 LAWBOT_DEMO=1`

---

## 단계 요약 (현재 위치)

| Phase | 내용 | 상태 |
|---|---|---|
| 0 | 재료(임베딩+전체 인덱스) | ✅ 완료·검증 |
| 1 | 전체 인덱스 검색 가능화 | ✅ 완료·검증 |
| 2 | 검색 품질(하이브리드·버그) | ✅ 사실상 완료 |
| 3 | 범용화 + 제품 표면 | ✅ 완료(P3·P5·P6) |
| 4 | 운영 + 배포 | ✅ 사실상 완료(MCP·캐시·/metrics·토큰미터·일일쿼터·컨테이너검증·런북) — 실제 NCP 서버 임대(돈)만 남음 |

---

## Phase 0 — 재료 ✅
- 임베딩 100%: `artifacts/embeddings.jsonl` = **1,357,018**개(512d), 무결성 PASS(중복·누락·차원오류 0).
- 전체 FAISS 인덱스: `embed/build_full_index.py`(메모리안전 스트리밍, B등급 118,896 제외) → `artifacts/full_index/` (index.faiss 2.54GB + meta.jsonl 3.17GB, **ntotal 1,238,122**). 검증 PASS(ntotal==meta·dim512·구조검색).
- BM25 인덱스: `embed/build_bm25.py`(FTS5 unicode61) → `artifacts/full_index/bm25.sqlite`(124만행, FAISS 행 정렬).

## Phase 1 — 전체 인덱스 검색 ✅
- 저RAM retriever: `search/retriever.py`에 `_DiskMetas`(meta 디스크 오프셋 조회). 전체 인덱스 검색 시 **RSS 2.62GB**(7GB 안전).
- config: `FULL_FAISS_*`, `ACTIVE_INDEX`(env `LAWBOT_INDEX=full`). 의료 인덱스 기본값(롤백 안전).
- 검증: 비의료 질의(근로기준법)가 전체 코퍼스에서 검색됨, 의료 selftest 12체크 무회귀.

## Phase 2 — 검색 품질 ✅
- **하이브리드(BM25+RRF, 타겟형)**: `_bm25_search`(불용어·구두점 처리·prefix `term*`로 한국어 어미 매칭, ttl 가중 5.0), `_is_citation_like`(제N조·○○법 감지 시에만 BM25 융합), 가중 RRF(dense1.0/BM25 `RRF_W_BM25=2.0`), hit.score=실코사인(reconstruct)→게이트 호환, 법명 질의 시 doc_type=law 조문 우선 재정렬.
- **골든셋 측정(31문항)**: dense **61.3%** → 하이브리드 **64.5%** Hit@5, MRR 0.467→0.516, Article-hit 33→67%, 격식질의 무회귀.
- **버그 수정**: #1 as_of_date(미래조문 인용 차단·법적리스크) / #3 Citation Firewall(죽은 get_qdrant_client 제거→parents+full_text 조문존재확인, 가짜조문 차단) / #6 ad_review에 admrule 포함. #4·#5 기구현, #2 정상.
- 질의정규화(사전 기반): 구현했으나 골든셋 효과 0 → 기본 off(코드 보존). 구어격차 진짜해법=LLM 재작성(유료).
- **한계(정직)**: "법명+주제 구어"(예: 연차휴가 vs 연차 유급휴가) 일부 미스 — 한국어 복합어/어휘격차, free 방법 한계. LLM 재작성 또는 mecab 필요.

## Phase 3 — 범용화 + 제품 표면 🔄
- ✅ **P3 범용화**: rag 프롬프트 의료고정→범용 한국법(비의료 질의 거부 안 함, 기능테스트 확인), 비법률 잡담 안내 유지. api 설명·커버리지 범용화. `ANSWER_DISCLAIMER` 거짓 "자치법규 커버리지" 제거.
- ✅ **P6 무로그인 데모 API**: `config.DEMO_MODE`(env `LAWBOT_DEMO=1`) + `demo_or_require_key`(키 있으면 키, 없고 데모면 익명+IP리밋, off면 401). ask·ad-review만 연결(프로덕션 키 유지·무회귀).
- ✅ **P6 페이지(chat.html)**: 데모 시 키 없이 호출, **광고검토 카드 필드버그 수정**(verdict/claim/rationale/law_basis/suggested_fix), 교정본·요약 렌더, 예시칩 범용화, 키입력 선택화.
- ✅ **P6 데모 서버 실테스트**: `LAWBOT_DEMO=1 LAWBOT_INDEX=full LAWBOT_HYBRID=1`로 uvicorn 기동(포트 **8010** — 8000은 MediLaw HMS 점유). 키 없이 `/v1/ask` 200·답변, `/chat` 200, healthz 200 확인. **Windows 브라우저 `http://localhost:8010/chat`로 테스트 가능.**
- ✅ **P5 인용 신호등**: 인용마다 `trust_flag`(green=본문확보·현행 / yellow=시행예정 또는 본문미확보) + `trust_note`를 **즉시 산출**(지연0, 메타 trust_grade·status 기반). chat.html 좌측 컬러바+점+툴팁. 본격 0~100 점수는 `/v1/verify`(law.go.kr 대조)로 분리 — ask 경로 지연 방지. 단위검증·test_rag 19 passed 무회귀.

### #1 as_of 하드필터 — 테스트 후 되돌림(중요)
오늘 하드필터가 **핵심 현행법을 숨기는 부작용** 발견: 코퍼스 `effective_from`이 "조문 시행일"이 아니라 "법의 최종 개정일"이라, 2026-09-13 개정 예정인 **형법 전 조문이 미래로 찍혀 today 필터에 통째 배제**됨(형법 검색 불가). → as_of 하드필터 **기본 off로 환원**, 미래시행은 인용의 effective_from + 프롬프트의 현행/시행예정 구분으로 **투명 처리**. as_of는 명시 opt-in 유지.

### 울트라코드 세션(2026-06-29 후반) — 데이터·감사·LLM재작성
- **검증된 골든 데이터 +40**(8개 분야 구어 질의, chunks.jsonl grep으로 법령 실재 확인) → 골든셋 **71문항**.
- **현실 구어 베이스라인(정직)**: 71문항에서 하이브리드만 = **Hit@5 31%**(격식 31문항일 땐 64.5%였음 → 실제 사용자 말투 성능은 낮음).
- **LLM 질의재작성 도입**(구어→법률용어, `config.QUERY_REWRITE`/env `LAWBOT_REWRITE`, retriever `_llm_rewrite` LRU캐시): 71문항 **31%→60.6%**. 콜로퀴얼 격차의 진짜 해법. ⚠️운영비: 쿼리당 ~$0.0003(캐시로 반복 0). **서빙 on 권장**.
- **재작성 강화(조문번호 포함)**: 라이브에서 "성인 몇 살부터야?"가 민법 제4조 대신 851조를 물어오는 실패 발견 → 재작성 프롬프트에 "확실한 경우 제N조 포함" 추가(LLM의 법지식을 검색에 활용; 답변은 여전히 retrieved 본문 grounding이라 환각無, 틀린 추측은 BM25 미매칭으로 폴백) → **60.6%→62.0%**, "성인 몇 살" 라이브 민법 제4조 #1 회복.
- **적대적 감사 워크플로(28에이전트)**: 20발견→**12 confirmed**. 고위험 3건 **수정 완료**: ① verify 가지번호(제N조의M) 조문검증 오탐(CRITICAL) ② retriever 하이브리드 ACTIVE_INDEX!=full 시 rowid 오정렬→환각인용(HIGH) ③ chat.html source_url href XSS(HIGH). 
  - **추가 수정 완료(2차)**: 죽은 `_qdrant_payload_match` 스텁·docstring 전면 정정(MED), ad-review **업로드 10MB 상한**(`config.MAX_UPLOAD_BYTES`, 413, DoS 차단), 광고카드 CSS를 실제 verdict(위반/위반소지/주의/적정/확인필요)에 맞춤, AD_LAW_TITLES 죽은상수 주석화. 판례 case_no는 **데이터층 TODO**(parents에 사건번호 필드 없음 → 판례 재파싱 필요)로 코드는 정직 처리(죽은 호출 제거). 전 모듈 import·retriever 셀프테스트 12체크 무회귀.
  - 잔여(경미·문서): ad_review docstring의 422/502·B등급 문구 불일치(문서만).

### 법제처/law.go.kr 딥리서치 + 차용(2026-06-29) — `docs/법제처_lawbot_조사.md`
- 법제처 Lawbot(지능형 법령검색) 실재 확인. law.go.kr OPEN API(OC) 191개 — 현행 조항호목(`lawjosub`)·판례(사건번호·판시·요지·참조조문·전문)·해석례·영문법령 등.
- **차용 ① 구현 완료 — [현행]/[시행예정] 시행일 라벨링**: 인용에 `effective_from`+`status`(현행/시행예정/미상) 부여. 미래시행 조문(예: 형법 2026-09-13)을 "시행예정"으로 투명 표기 → #1 법적리스크의 올바른 해법(하드필터 대신). rag(Citation·build_context·verify_citations)+chat.html 배지. 단위검증 통과(형법→시행예정, 민법→현행).
- **차용 로드맵(미구현)**: ② 판례 case_no를 law.go.kr `target=prec` API로 검증(우리 parents 사건번호 갭 해결) ③ 인용 본문 대조(LCS·bigram Jaccard) 강화 ④ 인용환각 5범주 분리검증 ⑤ Citation Grounding 메트릭(존재·관련·시간유효)을 골든 평가에 추가.

### 회귀 테스트 + suite green(2026-06-30)
- 수정 6개 모듈 pytest: 처음 **107 passed / 4 failed / 11 errors** → 실패 전부 **pre-existing(Qdrant→FAISS 이관 때 안 고쳐진 stale 테스트)**, 내 세션 변경과 무관함을 확인(내가 크게 고친 test_verify·test_rag·test_ad_review·test_statutes는 전부 PASS).
- **stale 테스트 FAISS로 갱신**: `test_retriever.py` 픽스처를 `QdrantClient(:memory:)`→`_DummyFlatIP`+`set_index`로 포팅(28 PASS), build_filter 3개를 `isinstance(models.Filter)`→predicate 동작검증으로, obsolete `test_as_of_date_client_side_fallback`(Qdrant server-side range 거부 폴백, FAISS는 항상 client-side) 제거. `test_main` 계약테스트를 openapi-schema→**schema∪app.routes 합집합**으로 수정(`include_in_schema=False`인 /v1/embeddings·/v1/precedents·articles 라우트 포함).
- 결과: **121 passed / 3 skipped / 0 fail / 0 error**. 내 변경 회귀 0, suite green.

### 울트라코드 파트감사(2026-06-30) + 답변품질 보완
- **실측 테스트**: 검색 웜 ~10ms(콜드 첫1회 ~150s 인덱스적재), 전체 pytest **292 passed**(stale 4건 수정: chunk dup-skip·embed dimensions·golden count·qdrant). 6분야 답변 스윕 법령인용 5/6, 미니 Hit@5 **11/12=92%**.
- **답변품질 실버그 수정(라이브 검증)**: ① "시행예정" 오탐(현행법 오표기) → 신뢰색에서 분리·중립 날짜표기 ② 답변이 판례만 인용 → `_ensure_statutes`(법조문 보장검색) + 재작성 조문번호 강화 → 가족(민법§839의2)·세무(소득세법) 등 정답 법령 인용.
- **지시서 9파트 울트라코드 감사(10에이전트)** → PM 종합(평균 ~74%, P7 운영 최약 45%). 수정 완료:
  - ✅ **P5** /v1/ask·ask_stream 응답에 top-level **trust_score(0~100)** 집계(green=100/yellow=60 평균) + chat.html 인용헤더 "신뢰도 N/100".
  - ✅ **P7** **`/metrics` 엔드포인트**(요청수·상태분포·평균지연·index ntotal·캐시적중, 미들웨어 집계) — 라이브 확인(trust_score=100, metrics 카운팅 작동).
  - ✅ **P3** api DESCRIPTION AI Q&A "의료법령"→"한국법령" 범용화 / **P9** run_eval docstring stale(gpt-4o·Qdrant) 정정 / **P6** 데모 키입력 숨김·광고이슈 심각도순 정렬 / **P8** DEPLOY.md FAISS 아키텍처 배너+런북 안내.
  - **프런트 설명 페이지**: chat.html에 작동 3단계(검색·근거답변·인용검증)·이용법·면책·일상어 예시칩 추가.
  - ✅ **P7 토큰 미터**(ask + ad_review): `_generate`/`_structured_call`가 OpenAI usage 포착 → `AskResult.usage`/`ReviewResult.usage` → api `_record_cost`가 /metrics llm_tokens_total + 비데모 키별 record_tokens 기록. 라이브(ask usage 11869 == metrics). 이전엔 항상 0이던 미터 작동.
  - ✅ **P7 일일 토큰쿼터**(인메모리, 단일서버): `config.DAILY_TOKEN_CAP_BY_TIER`(free 500k·pro 5M·enterprise/admin 무제한), api `_DailyUsage`+`_enforce_daily_cap`가 비용 엔드포인트(ask·ad-review)에서 LLM 호출 전 초과 시 429. demo·cap0 스킵, 날짜롤오버·재시작 리셋. 단위테스트+회귀 통과. (멀티레플리카는 Redis 필요-문서화.)
  - ✅ **P1 L2노름 사후 assert**: build_full_index.flush()가 normalize_L2 후 비영벡터 단위노름(±1e-3) 검증(차기 빌드 안전장치, 서빙 무영향, 컴파일 OK).
- **남은 minor(non-blocking)**: law.go.kr 서킷브레이커(retry·timeout·캐시·graceful-degrade는 기존 → 재시도 지연만 절감 효과), P2 BM25 헤더제외(재빌드 필요), P5 Levenshtein 문구유사도. → **P7 운영 45%→대부분 완료, 감사 HIGH 전부 해소.**

### 검색 품질 — free 방법 천장(정직)
하이브리드 64.5%가 무료 방법 한계. 남은 미스(cite-001 연차, 형법 사기 등)는 **질의어 자체가 굴절형**("사기죄의","연차휴가")이라 prefix 매칭이 어근과 안 맞는 **한국어 형태소 문제**. BM25 ttl에 article_title 추가했으나 이 이유로 골든셋 무변화(무회귀, 유지). 추가 향상엔 **mecab 형태소분석기(C확장 빌드) 또는 LLM 질의재작성(쿼리당 과금) = 비용 결정** 필요.

## Phase 4 — 운영 + 배포 🔄
- ✅ **MCP 서버 검증·정비**(`mcp_server/server.py`, stdio, HTTP 래퍼 4도구 ask/search/verify/review): 라이브(8010) end-to-end 확인 — search(콜로퀴얼→민법 제4조), verify(형법 제347조 100/100🟢, 민법 제4조의2 0/100🔴 firewall 차단). 정비: `lawbot_review` 이슈 필드 stale 수정(severity/note→verdict/claim/rationale/suggested_fix+교정본), `lawbot_verify`에 trust_score/신호등 표시. **/v1/verify를 데모 keyless로 전환**(`demo_or_require_key`, 프로덕션·테스트는 키 유지) → MCP 완전 keyless 데모 가능.
- ✅ **law.go.kr 응답 TTL 캐시**(`verify.py` `_law_get`): 성공 응답을 `config.LAW_CACHE_TTL`(기본 3600s, env `LAWBOT_LAW_CACHE_TTL`) 동안 (path,params) 키로 캐시(thread-safe, 4096 상한, 오류 미캐시). 반복 인용검증(형법 제347조 등) 시 law.go.kr 재호출 제거 → 지연·rate-limit 절감. 단위검증(3호출→HTTP 2회)·test_verify 14 passed. (구현 중 데코레이터/캐시블록 충돌 SyntaxError를 py_compile로 출하 전 포착·수정.)
- ✅ **배포 설정 full 인덱스로 갱신**: docker-compose 마운트 `./artifacts:/app/artifacts:ro`(full_index+parents.jsonl), env `LAWBOT_INDEX=full HYBRID=1 REWRITE=1 DEMO=1 LAW_OC GEN_MODEL=gpt-5-mini` 추가. Dockerfile 아티팩트 경로 full_index로. **healthz를 죽은 Qdrant 프로브→FAISS ntotal(mmap, 비차단)** 재배선(faiss 백엔드 키, points=ntotal=1,238,122). 검증: `docker compose config` 유효, probe ntotal 정확, test_main 16 passed.
- ✅ **로컬 컨테이너 빌드·기동 실증 완료**(NCP 결정 전 무료 검증). `docker compose build api` → run(artifacts ro 마운트) → **healthz 200·전 백엔드 ready·points=1,238,122·faiss:true·/chat 200·degraded 없음**. **실증이 배포 블로커 3건 포착·수정**: ① requirements.txt에 `sqlmodel`·`SQLAlchemy` 누락(.venv엔 있어 테스트는 통과·컨테이너 크래시) ② `numpy` 명시 핀 추가 ③ **config.py가 `API_KEYS_DB` env 무시·경로 하드코딩** → 컨테이너 키DB 쓰기 실패(`os.getenv` 반영으로 수정). 회귀 27 passed. (qdrant_client는 lazy/빌드전용이라 런타임 무관 확인.)
- ✅ **배포 런북·스크립트 준비**(무료, 사용자 선택): `docs/배포_런북.md`(NCP 단일서버·도메인 있음HTTPS/없음IP-HTTP 분기·아티팩트 rsync·.env·docker compose·검증·트러블슈팅), `scripts/deploy_ncp.sh`(코드+full_index+parents.jsonl 전송+원격 기동, 문법·동작 검증), `.env.example` 현재 아키텍처로 갱신(Qdrant 제거·full-index env·gpt-5-mini·EMBED_DIM 핀 주석).
- ◻️ 남음: 키 쿼터·일일상한, law.go.kr CB, 메트릭, **실제 NCP 서버 띄우기**(서버 디스크≥30GB·RAM≥4GB·공인IP·도메인 — 돈·계정 결정. 런북대로 하면 됨).
- ⚙️ 운영 메모: 데모 서버 종료는 `fuser -k 8010/tcp`(bash -c 안 pgrep은 자기매칭으로 불안정).

---

## 정정된 오인 (검증으로 바로잡음)
- ~~parents.jsonl 3.3GB가 답변경로 OOM 블로커~~ → **비호출**(get_parent는 retriever 정의+셀프테스트만). 답변경로 7GB 정상. parents 저RAM 수정 불필요.

## 변경된 주요 파일
- `embed/build_full_index.py`(신규), `embed/build_bm25.py`(신규)
- `search/retriever.py`(_DiskMetas·하이브리드·정규화·조문우선), `search/rag.py`(as_of 기본·프롬프트 범용), `search/ad_review.py`(as_of·admrule), `search/verify.py`(firewall #3)
- `config.py`(FULL_FAISS·ACTIVE_INDEX·HYBRID·QUERY_NORM·DEMO_MODE·disclaimer)
- `api/main.py`(demo_or_require_key·설명 범용), `web/chat.html`(데모·카드버그·교정본)

## 다음 작업 순서
1. 데모 서버 기동 + 페이지 end-to-end 실테스트(질문/광고검토).
2. P5 신뢰점수/신호등.
3. Phase 4 운영·배포(NCP, MCP).
