# 00. lawbot 빌드 RESUME / 핸드오프 (토큰 소진 시 재개용)

> 갱신: 2026-06-15. **이 문서만 읽고 "▶ 재개 방법"부터 그대로 이어가면 된다.**
> 핵심: 빌드는 **하네스 워크플로우**로 진행 중이며, **resume 가능**(완료 에이전트는 캐시 복구).

---

## ★ 현재 상태 (2026-06-15 갱신)
- 빌드 1차 완료: **19개 모듈 + 테스트 281 passed/4 skip**, 21개 모듈 import OK. 파서 4종 실데이터 검증 통과(중복제거 동작 확인).
- 자치법규 **전국 18개 시도 완비**(17시도+충청광역연합) 확인.
- OpenAI 크레딧 **$35로 증액**됨.
- 진행 중: **마무리 워크플로우 Run `wf_d3e2db94-24d`** (Task wzx90lrvw) = 통합점검 → 청킹/임베딩 품질강화(판례 집중) → 채팅+PDF 페이지 → 실인덱스+/ask 스트리밍 실데모.
  - resume: `Workflow({scriptPath:"...lawbot-finish-wf_d3e2db94-24d.js", resumeFromRunId:"wf_d3e2db94-24d"})`
- 이전 빌드 Run: 최종 `wf_05039cd9-b31`(Fix/Integrate/GoLive는 세션한도로 중단됐으나 코드는 정상).

## ✅ 데모 실동작 검증 완료 (2026-06-15, 529 복귀 후)
- 데모 색인 **정상 완료**: 청크 8,296 임베딩 → Qdrant 로컬 **8,165 적재**(중복 chunk_id 131건은 UUID5로 dedup, 정상). demo_err.log의 Traceback은 종료 시 qdrant `__del__` 경고로 **무해**.
- 서버 기동 OK(`bash ~/lawbot/run_demo_server.sh`): `/docs /chat /console /openapi` 모두 200.
- **버그 발견·수정**: API/retriever가 Qdrant를 **서버모드(localhost:6333)로만** 붙어 로컬 임베디드 데모데이터를 못 읽었음 → `search/retriever.py:get_qdrant_client()`를 `upsert_qdrant.get_client()`(서버우선·로컬폴백)로 교체, `api/main.py:_probe_qdrant()`도 retriever 공유클라이언트 재사용(로컬 path는 프로세스당 1핸들). **이 두 수정 유지 필수**.
- `/v1/ask` **엔드투엔드 정상**(200, 7.4s): 근거기반 답변 + citations(제6조, source_url, trust_grade) + disclaimer. 요청 필드는 `query`(※`question` 아님). 키 발급: `auth.issue_key(tenant,tier="admin")`, 헤더 `Authorization: Bearer lk_...`.
- **남은 한계**: ① 데모코퍼스가 각 파서 앞쪽 N건(판례400/법령80/자치150/행정100) 샘플이라 도로교통법 등 핵심법령 누락 → "음주운전" 질의에 5·18법이 나옴(관련성 낮음). 정식은 전량 임베딩 필요. ② 인증 불일치: `/v1/statutes/search`는 무인증 200, `/v1/ask`는 키 필수 → 정식 전 통일 필요. ③ `/v1/ad-review`는 multipart(PDF) 엔드포인트.

## ☑ 로컬 데모 진행 상황 (2026-06-15 최신)
- 마무리 워크플로우(`wf_d3e2db94-24d`): 통합✅ 품질강화(판례·법령·자치 3종)✅ 채팅페이지(web/chat.html, /chat 라우트)✅ / **GoLive는 Anthropic 529 과부하로 중단** → 코드 무관, 내가 직접 데모 색인 진행 중.
- **테스트 281 passed**, 21모듈 import OK. 파서 산출물 `artifacts/docs_*.jsonl` 존재(단 판례는 50건만 → 데모 색인에서 2,000건 재파싱).
- Docker가 WSL에 없음 → **Qdrant 로컬 임베디드(local-path) 모드**로 구동(코드 `upsert_qdrant.get_client()` 폴백 지원).
- 데모 색인 빌더: `~/lawbot/build_demo_index.py` (판례2000+법령800+자치1200+행정500 → 청킹 → OpenAI sync 임베딩 → 로컬 Qdrant). 로그: `artifacts/demo_build.log`.
- **다음**: 색인 완료 → `uvicorn api.main:app` 기동 → `http://localhost:8000/chat`(채팅+PDF), `/console`(키발급), `/docs` 로 사용자 직접 테스트.
- 로컬구동 명령: `cd ~/lawbot && .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000` (Qdrant 임베디드라 별도 docker 불필요).

## ▶ 재개 방법 (제일 먼저)

1. **빌드 워크플로우 resume** (완료분 캐시, 나머지만 재실행):
   ```
   Workflow({
     scriptPath: "C:\\Users\\SMHRD-\\.claude\\projects\\--wsl-localhost-Ubuntu-home-user1----NEW2\\ac821cc9-a0c6-4a8e-8910-a5e55b86e3e3\\workflows\\scripts\\lawbot-build-final-wf_05039cd9-b31.js",
     resumeFromRunId: "wf_05039cd9-b31"
   })
   ```
   - 최종 빌드 Run ID: **wf_05039cd9-b31** (Task `w7p122okj`).
   - (이전 중단된 run: v1 `wf_82897953-220`, v2 `wf_fadd4de3-433` — 사용 안 함.)
2. 진행상황 확인: `/workflows`, 또는 `~/lawbot/BUILD_REPORT.md`·`DEMO.md` 존재여부.
3. 빌드 끝났으면 → 아래 "■ 빌드 후 남은 일" 순서대로.

---

## 환경 (확정·구축완료)

| 항목 | 값 |
|---|---|
| 코드 위치 | `~/lawbot` (WSL: `/home/user1/lawbot`, UNC: `\\wsl.localhost\Ubuntu\home\user1\lawbot`) |
| 파이썬 | **3.12.13** (uv로 설치). venv: `~/lawbot/.venv` (의존성 설치 완료) |
| 실행 | `wsl -d Ubuntu -- bash -lc 'cd ~/lawbot && .venv/bin/python ...'` |
| Docker | 29.4.0 (Qdrant 로컬용) |
| OS | Windows 11 호스트 + WSL Ubuntu 22.04 |

## 데이터 (정리완료, 단일 정본)

`/home/user1/체크/NEW2/원천데이터/` (README.md 참조):
- `01_국가법령/kr` — 약 5,673 법령문서
- `02_자치법규` — 전국 18개 시도, **159,890건** (약 233만 조문)
- `03_행정규칙` — 약 21,700건
- `04_판례` — 123,742건
- **합계: 311,005 문서 / 약 319만 청크(조문·섹션) / 헤더포함 약 14.6억 임베딩 토큰** (표본 추정, 정본은 빌드 parser가 산출).
- 형식: 전부 YAML 프론트매터 + 조문/섹션 markdown. 중복 zip(`법률데이터/`, 9.1GB)은 부분집합 → 미사용(삭제 가능).

## 키 / 비용 (`~/lawbot/.env`, 커밋 금지)

- `OPENAI_API_KEY` — 유효(사용자가 추후 폐기·재발급 예정). **크레딧 $20**.
- `LAW_OC=<LAW_OC_REDACTED>` — law.go.kr 국가법령 OpenAPI, **검증됨**(도로교통법 조회 성공).
- 비용수칙: 임베딩=**text-embedding-3-small(1536)만**, 헤더=결정적($0). **전량 임베딩 = Batch 약 $14.6**(sync는 $29.2이므로 금지), 반드시 승인 게이트. content-hash 캐시로 재임베딩 0. $20 들어가나 여유 ~$5 → 대량사용 시 $30~40 충전 권장. **재작업 방지: 표본 ~5만청크 검증 후 전량 1회.**

## 설계 문서 (절대 기준)

- `분석/08_lawbot_빌드_하네스_플레이북.md` — Phase별 빌드 Task + 코드골격.
- `분석/09_청킹_임베딩_헤더_설계.md` — 청킹(parent-child)·임베딩(3-small)·**헤더 거버넌스(2층 결정적 헤더+검증기)**·lawbot.org 정렬·테스트 전략. **최우선.**
- `분석/01~07` — 리서치/데이터분석/검증.

## 확정 결정 (재논의 불필요)

- 스택: OpenAI(3-small + gpt-4o-mini) · Qdrant · FastAPI · sqlmodel/SQLite(키).
- 청킹: 조문/섹션=child, 법령/판례=parent. 8191초과만 2차분할.
- 헤더: L1 인용헤더 + L2 결정적 맥락헤더, 스키마+검증기+테스트로 지속 감독.
- 환각방지: grounded-only + 인용 강제 + 인용 사후검증 + Citation Firewall(law.go.kr).
- **lawbot.org 형식 정렬**: `/v1/statutes/search · /v1/verify · /v1/source-pack · /v1/ask · /v1/ad-review · /v1/embeddings · /v1/keys · /console`, 응답메타 `trust_grade·source_url·license·as_of_date`.
- 멀티테넌트: 외부 이용자 셀프서비스 API 키 발급.
- 변호사 대상 → 변호사법 §109 소비자 가드 없음(전문가 모드).
- **페이지: 단순 "채팅방 + PDF 업로드" LLM 챗 페이지 하나만**(거창한 대시보드 X — 사용자 최종 지시).
- 별도 프로젝트 `MediLaw_AI`(바탕화면 산출문서)가 이 lawbot을 apikey로 가져다 씀. lawbot은 독립 서비스로만 집중.

## 빌드 워크플로우 구조 (Run wf_05039cd9-b31)

`Contracts → Build(19모듈 병렬) → Verify(8) → Fix → Integrate → GoLive(실데모)`

**19 모듈**: parse_statute, parse_precedent, parse_admrule, parse_ordinance, header, chunk, embed_openai, upsert_qdrant, retriever, rag, verify, source_pack, statutes_search, embeddings_api, multitenant, api_server, console_ui, ad_review, deploy_eval.

각 모듈 산출: `~/lawbot/`의 해당 디렉터리 파일 + 단위테스트. 공유계약: `~/lawbot/_BUILD_CONTRACT.md`.

## ■ 빌드 후 남은 일 (순서)

1. `~/lawbot/BUILD_REPORT.md`·`DEMO.md` 확인 — 실제 질의응답·인용·광고판정 동작 검증.
2. **페이지를 채팅+PDF 단순형으로 교체**(console_ui가 대시보드면 단일 chat.html로 정리).
3. 로컬 구동 확인: Qdrant(docker) + `uvicorn api.main:app` → `/console`·`/v1/*` 동작.
4. (사용자 승인 후) **전량 임베딩 실행**: 전체 파싱→청킹→헤더→Batch 임베딩(비용추정~$9 출력→승인)→Qdrant 전량 적재.
5. **git init → 커밋 → 원격 push**(.env 제외 확인).
6. **클라우드 배포**: Qdrant Cloud + Render/Fly + Caddy(HTTPS) + 시크릿 주입. 외부 이용자 키발급 온보딩.

## ★ 정식 버전 반영 TODO (사용자 지시 2026-06-15 — 데모 테스트 중 발견, 지금은 보류)

- **[UX] 비법률/인사·잡담 입력 처리**: "안녕" 같은 일반 채팅에 지금은 "근거가 불충분합니다"로 딱딱하게 거부함. 정식에선 **"저는 한국 법령·판례 기반 법률 질문에 답하는 lawbot입니다. 법률 관련 질문을 해주세요." 식 안내**로 응답하게. 단 **법률 질문의 grounded-only(근거불충분) 거부는 유지**(환각방지). 구현: `search/rag.py` `SYSTEM_PROMPT`에 "비법률/인사 입력이면 answer에 안내문구, citations 빈배열" 절 추가하거나, ask 핸들러 앞단에 비법률 입력 분기. (위치: rag.py:105 SYSTEM_PROMPT, :131 "[검색결과가 빈약할 때]" 절 근처)
- **[UI] 원본 JSON 노출 제거**: `/chat`에서 답변과 함께 ```json 블록(raw response)이 그대로 보임 → answer/citations만 정제 렌더링하도록 `web/chat.html` 정리.

## 미해결/주의

- 전량 임베딩은 아직 미실행(데모는 ≤2만 청크). 실행 전 비용 재확인.
- 행정규칙 본문결측(trust=B)·별표 본문은 메타만 — 표기 유지.
- 페이지 형태(채팅+PDF)는 빌드 후 정리 예정.
- OpenAI 키는 작업 종료 후 사용자 폐기·재발급 예정.

---
**한 줄 상태**: 환경·데이터·설계·키 준비 완료 → 최종 빌드 워크플로우(wf_05039cd9-b31) 실행 중 → 끝나면 데모 검증·페이지 정리·전량임베딩(승인)·git/클라우드 배포.
