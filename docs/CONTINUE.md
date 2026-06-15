# CONTINUE — 다른 환경에서 작업 이어가기

> 집/다른 컴퓨터에서 정식 버전 작업을 이어갈 때 **이 문서부터** 읽으세요.
> 세부 상태·결정의 정본은 [`분석/00_빌드_RESUME_핸드오프.md`](분석/00_빌드_RESUME_핸드오프.md).

## 0. 이 프로젝트가 뭔지 (30초 요약)

한국 법령·판례 근거로 답하는 **환각방지 법률 RAG API "lawbot"**. lawbot.org 형식 참고, 독립 서비스로 빌드.
스택: OpenAI `text-embedding-3-small`(1536) + `gpt-4o-mini` · Qdrant · FastAPI · SQLite(키). Python 3.12.

## 1. 새 환경 세팅 (순서대로)

```bash
# (1) repo 클론
git clone https://github.com/lhs488548-create/lawbot.git && cd lawbot

# (2) Python 3.12 가상환경 + 의존성
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# (3) 비밀키 — .env 생성 (repo에 없음, 직접 채우기)
cp .env.example .env
#   OPENAI_API_KEY=sk-...   LAW_OC=...   (본인 키)

# (4) 원천데이터 확보 — DATA.md 참조 (6GB, git에 없음)
#     클라우드 드라이브로 옮기거나 공공 출처에서 재수집 → ~/체크/NEW2/원천데이터/

# (5) 데모 색인 + 서버
.venv/bin/python -u build_demo_index.py
bash run_demo_server.sh   # → http://localhost:8000/chat /console /docs
```

## 2. 지금까지 한 것 (2026-06-15 기준)

- ✅ 19개 모듈 빌드 + **테스트 281 passed**.
- ✅ 데모 색인·서버 동작 검증: `/v1/ask` 근거기반 답변 + 인용 정상.
- ✅ 버그 수정: API/retriever가 Qdrant 로컬 임베디드 폴백을 쓰도록 (`retriever.get_qdrant_client` → `upsert_qdrant.get_client`). **유지할 것.**

## 3. 정식 버전 남은 일 (우선순위)

1. **전량 임베딩** — 319만 청크 Batch API 임베딩(~$14.6, 승인 게이트) → 도로교통법·주택임대차보호법 등 전체 색인. (지금 데모는 가나다순 앞쪽 샘플뿐이라 대부분 질문이 "근거 불충분"으로 나옴 — 버그 아님, 데이터 부족.)
2. **Qdrant 프로덕션** — 로컬 임베디드(폴더) → Qdrant Cloud/서버. `.env`의 `QDRANT_URL`만 바꾸면 코드 자동 전환. (자세히: `분석/04_배포_및_구축계획.md`)
3. **클라우드 배포** — Render/Fly + Caddy(HTTPS) + 시크릿 주입.
4. **MCP 서버 추가** — Claude/Cursor에서 lawbot을 도구로. (자세히: [`LAWBOT_ORG_RESEARCH.md`](LAWBOT_ORG_RESEARCH.md) §MCP)
5. **UX 개선** — 비법률/인사 입력 안내문구, `/chat` 원본 JSON 노출 제거. (`분석/00` "정식 버전 반영 TODO")
6. **인증 통일** — `/v1/statutes/search` 무인증 → 키 필수로.

## 4. 확정 결정 (재논의 불필요)

- 청킹: 조문/섹션=child, 법령/판례=parent. 헤더: L1 인용 + L2 맥락(결정적).
- 환각방지: grounded-only + 인용 강제 + 사후 인용검증 + Citation Firewall(law.go.kr).
- 페이지: "채팅 + PDF 업로드" 단순형 하나.
- 대상: 변호사(전문가 모드) → 소비자 가드 없음.

## 5. 비밀키 메모

- 키는 **`.env`에만**. 문서·코드의 실제 키값은 `<...REDACTED>`로 지움(작동 무관).
- 키 다시 필요하면 `.env`에 본인 값 넣으면 끝. OpenAI 키·LAW_OC는 사용자가 보관/로테이션.
