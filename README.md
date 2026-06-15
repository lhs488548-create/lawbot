# lawbot — 한국 법률 데이터 인프라 API

한국 법령·판례·행정규칙·자치법규를 근거로 답하는 **환각방지(grounded-only) 법률 RAG API**.
lawbot.org 형식을 참고해 직접 빌드 중인 독립 서비스.

> **다른 환경(집 등)에서 작업 이어가기 → [`docs/CONTINUE.md`](docs/CONTINUE.md) 부터 읽으세요.**

## 무엇인가

- **RAG API** (`/v1/ask`): 질문 → 법령·판례 검색 → 근거 기반 LLM 답변 + 인용(조문/사건번호/출처 URL/신뢰등급).
- **인용 검증 / Citation Firewall** (`/v1/verify`): AI가 만든 인용이 실제 존재·정확·현행인지 검증.
- **법령 검색** (`/v1/statutes/search`), **Source Pack** (`/v1/source-pack`), **광고심사** (`/v1/ad-review`).
- **멀티테넌트 키 발급** (`/console`) + 채팅+PDF 웹페이지 (`/chat`).

응답 메타: `trust_grade · source_url · license · as_of_date`. 환각방지: grounded-only + 인용 강제 + 사후 인용검증.

## 스택

OpenAI `text-embedding-3-small`(1536) + `gpt-4o-mini` · Qdrant(벡터) · FastAPI · SQLite(키). Python 3.12.

## 빠른 시작

```bash
# 1) 의존성 (uv 또는 pip)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 2) 비밀키 설정 (커밋 금지 — .env는 .gitignore에 있음)
cp .env.example .env   # OPENAI_API_KEY, LAW_OC 등 채우기
# 3) 데모 색인 (원천데이터 필요 — DATA.md 참조)
.venv/bin/python -u build_demo_index.py
# 4) 서버 기동
bash run_demo_server.sh        # → http://localhost:8000/chat /console /docs
```

## 디렉터리

| 경로 | 내용 |
|---|---|
| `ingest/` | 파서 4종(국가법령·판례·행정규칙·자치법규) |
| `embed/` | 청킹·임베딩·Qdrant 적재 (`upsert_qdrant.py`, `embed_client.py`) |
| `header/` | 2층 결정적 헤더 + 검증기 |
| `search/` | retriever · rag · verify · statutes · source_pack · ad_review |
| `api/` | FastAPI 서버 · 멀티테넌트 키(auth/db/keys) · 콘솔 |
| `web/` | 채팅+PDF 페이지(`chat.html`) |
| `tests/` | 단위테스트 (281 passed) |
| `docs/` | 설계·핸드오프·리서치 문서 (`docs/분석/`, `CONTINUE.md`, `LAWBOT_ORG_RESEARCH.md`) |

## 데이터 · 비밀키

- **원천데이터(6GB)·빌드 산출물(artifacts, 3.6GB)는 이 repo에 없습니다** (GitHub 100MB/repo 용량 한계). 확보·복원 방법은 [`DATA.md`](DATA.md) 참조.
- `.env`(OpenAI·LAW_OC 키)는 **절대 커밋 안 함**(`.gitignore`). 키는 사용자가 별도 보관·로테이션.

## 현재 상태 (2026-06-15)

데모 동작 검증 완료(`/v1/ask` 인용까지 정상). 정식 버전 남은 일: 전량 임베딩 → Qdrant 프로덕션 → 클라우드 배포 + MCP 서버·UX 개선. 자세히는 `docs/분석/00_빌드_RESUME_핸드오프.md` 의 "정식 버전 반영 TODO".
