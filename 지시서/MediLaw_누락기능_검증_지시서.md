# MediLaw 누락 기능 검증·보완 지시서

작성일 2026-06-30 · 대상 레포 `2025-SMHRD-KDT-HealthCare-3/MediLaw` (`develop` 기준, origin과 동기화 확인)

산출문서(기획서·요구사항정의서·화면설계서·WBS·테이블명세서·DB요구사항분석서)와 실제 `develop` 코드를
한 줄씩 대조해, "문서가 약속했는데 화면까지 내려오지 않은" 기능만 추렸다. 각 항목은 근거 파일·라인까지
달았고, front만 고치면 되는지 back도 손봐야 하는지 구분했다.

## 검증 결론 요약

| 항목 | 문서 근거 | 코드 실태 | 필요 작업 |
|---|---|---|---|
| PDF/텍스트 수정 전·후 비교 | FR_14·FR_21, UC_04·06, 기획서 메뉴구조도 "수정안(before/after)", WBS "좌우분할" | 데이터는 이미 응답에 다 옴. 화면만 안 그림 | **front only** (이번 작업) |
| 법령 개정 전·후 조문 비교 | FR_25, UC_09, 기획서, 화면설계 슬라이드12 | rag `/v1/laws/diff` 완성, 프론트가 호출만 안 함 | front only (백엔드 준비됨) |
| 연관 판례·법령 그래프 | FR_23, UC_08 | rag `/v1/related-graph` 완성, 프론트 화면 없음 | front only |
| 체크리스트 PDF 리포트 | 화면설계 슬라이드11 #4, 기획서 | PDF 생성 코드 자체가 레포에 없음 | back(생성) + front(다운로드) |
| 관리자 화면 / API 호출 로그 | FR_29, UC_12, 화면설계 메뉴 | admin 엔드포인트 일부만(검증결과 O, 호출로그 X), UI 없음 | back(로그 모델) + front(화면) |

> MCP·멀티테넌트 API 키(FR_27·28)는 lawbot/rag 엔진 쪽 담당이라 MediLaw 레포 기준으로는 누락 아님.

---

## 1. 수정 전·후 비교 (이번 작업 — front only)

### 검증 (왜 front만으로 되는가)

- 엔진 `backend/fastapi/app/schemas.py:200-208` `ReviewFinding`에 **`suggestion`(대안 문구 = after)**이 정식 필드로 있고,
  어댑터 `backend/fastapi/app/pdf/review_adapter.py:94`가 `suggestion=risk.after`로 채운다. 즉 비스트림
  `/documents/review` 응답의 finding마다 원문(`segment_text`)과 수정문(`suggestion`)이 같이 온다.
- product `backend/app/services/ai_ad_copy_service.py:147-149`가 그 `findings`를 `legal_basis.findings`에 그대로
  저장하고, 라우터 `ai_ad_copy_router.py:48-71`이 응답으로 다시 풀어 준다. 문서 전체 원문은 `input_text`,
  전체 수정본은 `revision_recomm`로 함께 내려온다.
- 프론트 `frontend/src/pages/AdReview.tsx:81-104`는 이미 `legal.findings`를 받지만 `f.suggestion`을 **안 읽고**,
  `result.inputText`(원문)도 285번 줄 비교용으로만 쓰고 **화면에 안 그린다.**

→ 백엔드는 데이터를 이미 다 주고 있다. **프론트가 안 그릴 뿐**이라 backend 수정 불필요.

### 변경 내용 (`frontend/src/pages/AdReview.tsx`, `frontend/src/i18n/strings.ts`)

1. `ChecklistItem`에 `suggestion?: string` 추가, findings 매핑에 `suggestion: f.suggestion` 추가.
2. 위험 카드마다 **수정 전(원문 `segment_text`) / 수정 후(대안 `suggestion`)**를 2단으로 나란히 표시.
3. 결과 하단의 단일 "수정 추천" 박스를 **문서 전체 수정 전(`inputText`) / 수정 후(`revision`) 2단 비교**로 교체.
4. i18n 키 추가: `ad.compareTitle`(수정 전후 비교), `ad.beforeLabel`(수정 전), `ad.afterLabel`(수정 후) — ko/en.

### 검증 방법

- `cd frontend && npm run build` (= `tsc -b && vite build`) 타입체크·빌드 통과 확인.
- 광고검토 화면에서 위반 소지 문구 입력 시, 카드마다 수정 전/후가 보이고 하단에 전체 전·후 비교가 뜨는지 확인.

---

## 2. 법령 개정 전·후 조문 비교 (다음 후보 — front only)

- 백엔드 준비 완료: `backend/fastapi/app/routers/laws.py`의 `GET /v1/laws/diff` → `LawDiffResponse`
  (`schemas.py:315-332`, added/removed/changed + before/after 조문). node 브릿지가 `/api/rag/*`를 통과시키므로
  프론트에서 `GET /api/rag/v1/laws/diff` 호출 가능.
- 할 일: `frontend/src/api/lawApi.ts`에 `fetchLawDiff` 추가, `frontend/src/pages/LawUpdates.tsx`에 법령별
  "개정 전후 조문 비교" 패널(현행 ↔ 시행예정 조문 대조) 추가. 화면설계 슬라이드12 개요와 일치시킨다.

## 3. 연관 판례·법령 그래프 (front only, 신규 화면)

- 백엔드 준비 완료: `POST /v1/related-graph` → `RelatedGraphResponse`(root→issues→cases, `schemas.py:335-383`).
- 할 일: 신규 라우트/화면 + 마인드맵 렌더. 요구사항 UC_08엔 있으나 화면설계서엔 통째로 빠져 있어, 화면설계서에도
  화면을 추가해야 정합이 맞는다.

## 4. 체크리스트 PDF 리포트 (back + front)

- 현재 레포에 PDF 생성 라이브러리가 전혀 없다(reportlab/weasyprint 등 부재). `tb_summary.summary_file`은
  파일명 문자열만 저장.
- 할 일: product에 체크리스트→PDF 생성 엔드포인트(back), 프론트 체크리스트 화면에 다운로드 버튼(front).
  화면설계 슬라이드11 #4("리포트(PDF) 저장")와 맞추려면 필요.

## 5. 관리자 화면 / API 호출 로그 (back + front)

- 백엔드: `backend/app/routers/admin_router.py`에 `/api/admin/users`·`/verifications`·`/summaries`는 있으나
  **API 호출 로그 모델·엔드포인트가 없다**(현재 로깅은 stdout만).
- 할 일: 호출 로그 저장 모델·조회 API(back), 관리자 라우트/화면(front). 화면설계 메뉴의 "관리자(조문검증
  결과·API 호출 로그)"와 맞추려면 필요.

---

## 부수 정합성 메모

- MediLaw `README.md`가 스택을 "Node + Express"로만 적어 실제 3계층(product FastAPI 8001 / rag FastAPI 8000 /
  Node 브릿지 4000)을 반영 못 한다. 발표 전 갱신 권장.
- 화면설계서 슬라이드13(영어 입력)은 별도 화면처럼 그렸지만 구현은 전역 토글이다(큰 문제 아님).
- 팀표지 PDF 첫 장 제목("선박용 엣지 AI 응급진단…")·팀명("MDTS")이 타 프로젝트 값. → "MediLaw AI" / "H-LAB"로.
  편집 원본(HWP/PPT)이 산출문서 폴더에 없어 통합발표회 한글 템플릿에서 수정해 재출력 필요.
