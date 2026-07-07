# 플랜B — lawbot(API) + MediLaw_AI(Node 제품) 2-프로젝트 분리 구조

> 작성: 2026-06-16. 사용자 지시 "lawbot은 LLM 챗 UI 없이 순수 API, MediLaw_AI가
> 그 API를 받아서 챗봇 + PDF 교정 재생성, 서버는 Node" 를 분석한 대안 설계.
> 관련 메모리: 프로젝트 방향(MediLaw_AI 제품), NCP 배포 계획.

---

## 0. 한 줄 결론

**가능합니다. 오히려 더 깔끔합니다.** lawbot을 "법률 grounding API"로만 두고,
MediLaw_AI를 별도 **Node 제품 서버**로 만들어 lawbot을 API 키로 호출하는 구조는
교과서적인 마이크로서비스 분리입니다. 언어가 달라도(Python ↔ Node) 통신은
HTTP/JSON + Bearer 키 한 줄이라 문제 없습니다.

---

## 1. 두 프로젝트의 역할 (책임 분리)

```
[브라우저 — 병원/의료기관 마케터]
        │  업로드 · 채팅 (HTTPS)
        ▼
┌──────────────────────────────────────────────────────────┐
│ MediLaw_AI  —  Node 서버 (Express 또는 NestJS)             │  ← 새로 만들 것
│  · 사용자 인증/세션, 파일 업로드 처리                       │
│  · 챗 UI (프런트) + 챗 응답 중계                            │
│  · 의료광고 소비자 가드·면책·AI생성표시                     │
│  · PDF 재생성 (원본 하이라이트 + 수정안 + 근거리포트)        │
│  · lawbot API 키 보관(서버측, 브라우저에 절대 노출 X)        │
└──────────────────────────────────────────────────────────┘
        │  Bearer 키 + HTTP/JSON
        ▼
┌──────────────────────────────────────────────────────────┐
│ lawbot  —  Python / FastAPI  (LLM 챗 UI 없음, 순수 API)    │  ← 이미 빌드됨
│  · POST /v1/ask         grounded 법률 Q&A (챗봇 백엔드)     │
│  · POST /v1/ad-review   PDF 검토 + corrected_copy(수정문안) │
│  · POST /v1/source-pack 인용가능 원문 번들                  │
│  · POST /v1/verify      law.go.kr 현행 대조                 │
│  · POST /v1/statutes/search                                │
│  · POST /v1/keys, /healthz                                 │
└──────────────────────────────────────────────────────────┘
        ▼
 Qdrant(벡터)  ·  OpenAI(임베딩·생성)  ·  law.go.kr(검증)
```

### 책임 경계 (누가 무엇을)

| 기능 | lawbot (Python) | MediLaw_AI (Node) |
|---|:---:|:---:|
| 법령·판례 검색(RAG) | ✅ | — |
| 환각방지·인용검증(grounding) | ✅ | — |
| 광고 검토 + 수정문안 생성 | ✅ (`corrected_copy`) | — |
| 법률 Q&A 답변 생성 | ✅ (`/v1/ask`) | — |
| 챗봇 **화면**·세션·스트리밍 중계 | — | ✅ |
| 파일 업로드 UI/처리 | — | ✅ |
| **수정 PDF 재생성·하이라이트** | — | ✅ |
| 의료광고 소비자 가드·고지 | — | ✅ |
| 사용자 계정·결제 | — | ✅ |

> 원칙: **"법률 두뇌"는 전부 lawbot, "제품·화면·문서출력"은 전부 MediLaw.**
> 이러면 lawbot은 독립 데이터인프라로 재사용 가능하고, MediLaw는 법률 로직을
> 다시 구현하지 않아 환각 위험을 떠안지 않는다.

---

## 2. "lawbot에 LLM 챗 없이" 의 정확한 의미

- lawbot의 **웹 채팅 페이지(`/chat`, web/chat.html)는 Plan B에서 안 씀** —
  화면은 MediLaw가 제공. 앞서 논의한 `/chat` UX 개선(잡담 처리·raw JSON 제거)
  **TODO는 폐기**해도 됨. (그만큼 일이 줆)
- 단 `/v1/ask`·`/v1/ad-review`는 **LLM을 쓰는 API 엔드포인트**로 그대로 유지.
  "LLM 챗이 없다"는 건 *소비자 챗 UI가 없다*는 뜻이지, LLM 호출 자체가 없다는
  뜻이 아니다. (챗봇의 답 생성은 결국 lawbot `/v1/ask`가 함)

---

## 3. ★ 핵심 결정 — MediLaw의 LLM/PDF를 어디서 처리하나

지시에 "fastapi 사용"과 "서버는 node"가 섞여 있어, 두 갈래가 가능하다.

### 안 A (권장) — MediLaw = 순수 Node, LLM은 lawbot 재사용
- MediLaw는 **Python/FastAPI를 두지 않음.** 챗봇 답·광고 수정문안은 전부
  lawbot의 `/v1/ask`·`/v1/ad-review`를 호출해 받는다.
- MediLaw Node가 하는 LLM 관련 일 = **없음** (lawbot이 다 함). Node는 중계·가드·
  PDF·UI만.
- 장점: 언어 하나로 단순, 환각방지 로직 재구현 0, 인용검증 그대로 상속.
- **이 경우 "fastapi"는 불필요** — lawbot이 유일한 FastAPI.

### 안 B — MediLaw에 별도 Python/FastAPI 서비스 추가
- MediLaw가 자체 프롬프트로 LLM을 직접 돌리고(예: 의료광고 특화 톤), lawbot은
  **검색·근거(`/v1/source-pack`)만** 제공.
- 구조: 브라우저 → Node(게이트웨이/UI) → FastAPI(LLM·PDF) → lawbot(근거).
- 장점: 프롬프트·출력 완전 자체 통제. 단점: 서비스 3개, 환각방지·인용검증을
  MediLaw가 다시 책임져야 함(품질·법적 리스크 ↑).

> **권장 = 안 A.** 의료광고 특화가 필요하면 lawbot 호출 시 `question` 파라미터로
> `"의료법 제56조 의료광고 금지유형 위주로"`를 넘기면 FastAPI 추가 없이 특화된다.
> 안 B는 "lawbot 답으로 부족하다"가 실측으로 확인됐을 때만.

---

## 4. MediLaw_AI (Node) 기술 스택 제안 (안 A 기준)

| 레이어 | 권장 | 비고 |
|---|---|---|
| 런타임/서버 | **Node 20 + Express**(단순) 또는 **NestJS**(구조화) | 둘 다 가능 |
| lawbot 호출 | `fetch`/`axios` + Bearer 키(`process.env`) | 키는 서버에만 |
| 파일 업로드 | `multer` | PDF → lawbot `/v1/ad-review` multipart 전달 |
| 챗 스트리밍 | SSE(`text/event-stream`) | §6 참고(lawbot 스트리밍 여부에 의존) |
| **PDF 재생성** | `pdf-lib`(주석·하이라이트) + `puppeteer`(HTML→PDF) | §5 |
| 프런트 | 가벼운 React/Vite 또는 서버사이드 템플릿 | 업로드·before/after·챗 |
| 배포 | Docker 컨테이너 1개, lawbot과 같은 compose | §7 |

---

## 5. PDF 교정 "재생성" — Node에서의 현실적 구현

원본 PDF 레이아웃을 그대로 둔 채 텍스트만 바꾸는 건 PDF 구조상 비현실적.
표준은 **3종 산출물 묶음**:

1. **원본 + 위반 하이라이트** — `pdf-lib`로 위반 문구 위치에 형광/주석 박스.
   (lawbot이 준 `issues[].claim` 문자열을 원본에서 찾아 표시)
2. **수정안 PDF** — lawbot `corrected_copy`를 깔끔한 HTML로 만들어
   `puppeteer`로 PDF 렌더.
3. **검토 리포트 PDF** — `issues`(verdict·severity·law_basis·citations·
   law.go.kr 링크)를 표로 렌더.

→ 세 파일을 ZIP 또는 한 PDF로 병합해 다운로드. **Node 생태계가 HTML→PDF
(puppeteer)·PDF 편집(pdf-lib)에 강해서 오히려 Python보다 쉬움.**

---

## 6. 데이터 흐름 예시 (안 A)

### (가) 광고 PDF 교정
```
브라우저 ──PDF업로드──► Node(/review)
  Node ──multipart(file, question="의료법 §56 위주")──► lawbot POST /v1/ad-review
  lawbot ──► { summary, issues[], corrected_copy, citations[] }
  Node:
    · 의료광고 가드·면책·AI표시 부착
    · PDF 3종 재생성(원본하이라이트 + 수정안 + 리포트)
  Node ──► 브라우저(다운로드 + before/after 화면)
```

### (나) 챗봇 (법률 Q&A)
```
브라우저 ──질문──► Node(/chat)
  Node ──{query}──► lawbot POST /v1/ask
  lawbot ──► { answer, citations[], disclaimer, ai_generated }
  Node ──► 브라우저(답변 + 인용칩 + 의료 면책)
```

> **스트리밍 주의**: 현재 lawbot `/v1/ask`는 완성형 JSON 반환(토큰 스트리밍
> 아님, `_BUILD_CONTRACT` 기준). 토큰 단위 타이핑 효과를 원하면 (1) lawbot에
> SSE 추가하거나 (2) MediLaw가 OpenAI를 직접 호출하되 근거는 lawbot
> `/v1/source-pack`으로 받는 안 B 하이브리드. MVP는 비스트리밍으로 충분.

---

## 7. 배포 (NCP 단일 서버, Docker — 기존 계획과 호환)

같은 NCP 서버의 docker compose에 **Node 컨테이너 하나만 추가**:

```
caddy ──► medilaw (Node :3000)  ──► api (lawbot FastAPI :8000) ──► qdrant + redis
            (공개 도메인/IP)         (내부 네트워크, 외부 비공개 가능)
```

- Caddy는 외부엔 MediLaw(Node)만 노출, lawbot은 내부 네트워크로만 접근(보안↑).
- lawbot 키 발급(`/v1/keys`)으로 MediLaw용 tenant 키 1개 만들어 Node 환경변수에.
- 두 컨테이너가 한 박스 → 호출 지연 거의 0, 운영 단순.
- (확장 시) MediLaw와 lawbot을 서버 2대로 분리해도 코드 변경 없음(URL만).

---

## 8. 가능성 체크리스트 (결론: 전부 ✅)

- [x] Python lawbot ↔ Node MediLaw 통신 — HTTP/JSON, 문제 없음
- [x] lawbot에서 챗 UI 제거하고 순수 API로 — `/chat` 안 쓰면 끝(엔드포인트 그대로)
- [x] Node가 PDF 교정·재생성 — pdf-lib + puppeteer로 가능(오히려 강점)
- [x] Node 챗봇 — `/v1/ask` 중계로 가능(스트리밍만 별도 고려)
- [x] 의료광고 소비자 가드 — MediLaw 레이어에서 부착(lawbot은 전문가 모드 유지)
- [x] NCP 단일 서버 배포 — compose에 Node 서비스 1개 추가

### 선행 의존성 (잊지 말 것)
1. **lawbot 전량 임베딩** — 의료법·약사법·화장품법·식약처/복지부 고시 색인.
   안 하면 MediLaw가 호출해도 "근거 불충분"만 나옴(데모 80개 법령엔 의료법 없음).
2. lawbot NCP 배포 + Qdrant 양자화(메모리) — 별도 계획대로.
3. 실제 의료광고 PDF 1건으로 `/v1/ad-review` 사전 검증.

---

## 9. 미해결/결정 필요

- **안 A vs 안 B** (§3): 기본은 A(순수 Node). 품질 부족 확인 시 B로.
- **Node 프레임워크**: Express(빠른 MVP) vs NestJS(장기 구조). 취향.
- **챗 스트리밍 필요 여부**(§6): MVP 비스트리밍 → 추후 SSE.
- **MediLaw 코드 위치**: lawbot repo와 **별도 폴더/프로젝트**(예:
  `~/medilaw_ai`)로 분리 권장 — 두 프로젝트 독립 유지.

---

## 10. 권장 진행 순서

1. (진행 중) lawbot 전량 임베딩 + NCP Docker 배포 + 양자화.
2. lawbot에서 MediLaw용 tenant API 키 발급.
3. 실제 의료광고 PDF로 `/v1/ad-review` 품질 검증(안 A로 충분한지 판단).
4. MediLaw_AI(Node) 신규 프로젝트 스캐폴딩: 업로드→`/v1/ad-review`→PDF 3종.
5. 챗봇(`/v1/ask` 중계) + 의료광고 가드·면책 부착.
6. compose에 Node 컨테이너 추가 → 같은 NCP 서버에 배포.
