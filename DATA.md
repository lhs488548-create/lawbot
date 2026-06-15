# DATA — 원천데이터 & 빌드 산출물 확보 가이드

> **왜 repo에 데이터가 없나?** GitHub는 파일 1개 **100MB 초과를 거부**하고 repo 전체도 수 GB를 권장하지 않습니다.
> 이 프로젝트의 데이터/산출물은 합계 **약 19GB**라 git에 올릴 수 없습니다. 코드·문서만 repo에 있고,
> 데이터는 아래 방법으로 **다른 환경에서 따로 확보/복원**합니다.

## 용량 요약 (repo 제외 대상)

| 경로 | 용량 | 성격 | 복원 방법 |
|---|---|---|---|
| `원천데이터/` | **6.0 GB** | 법령·판례·행정규칙·자치법규 원본(YAML+md) | ① 클라우드 드라이브로 전송 ② 또는 공공 출처에서 재수집 (아래) |
| `법률데이터/` (zip) | 9.1 GB | **중복 부분집합 — 미사용** | 복원 불필요 (삭제 가능) |
| `lawbot/artifacts/` | 3.6 GB | 파싱·청킹·임베딩 산출물 | 코드로 **재생성**(아래) |
| `lawbot/.venv/` | 246 MB | 가상환경 | `pip install -r requirements.txt` 로 재생성 |

## 원천데이터 (6GB) — 다른 환경으로 옮기는 법

`원천데이터/` 구조 (`docs/원천데이터_README.md` 참조):
- `01_국가법령/` — 약 5,673 법령문서
- `02_자치법규/` — 전국 18개 시도, 159,890건
- `03_행정규칙/` — 약 21,700건
- `04_판례/` — 123,742건
- 형식: 전부 **YAML 프론트매터 + 조문/섹션 markdown**

**옮기는 방법 (택1):**
1. **클라우드 드라이브 / 외장드라이브** (권장) — `원천데이터/`를 통째로 압축해 Google Drive·OneDrive 등으로 전송 후 집 컴퓨터 `~/체크/NEW2/원천데이터/`에 풀기. 6GB라 git보다 이게 빠르고 안전.
2. **공공 출처에서 재수집** — 원 출처:
   - 국가법령정보 공동활용 (law.go.kr OpenAPI, `LAW_OC` 키 필요)
   - 공공데이터포털 (data.go.kr)
   - 사법정보공유포털 (판례)
   - 수집 스크립트/계획은 `docs/분석/05_데이터_갭_및_추가수집_계획.md` 참조.

> 데모만 돌려볼 거면 전체 6GB가 다 필요하진 않음 — `build_demo_index.py`가 각 분류 앞쪽 일부만 샘플링.

## artifacts (3.6GB) — 재생성

데이터만 있으면 코드로 다시 만들 수 있음(올릴 필요 없음):

```bash
# 데모 색인 (소량 샘플 → 로컬 Qdrant)
.venv/bin/python -u build_demo_index.py
#  → artifacts/demo_chunks.jsonl, demo_vectors.jsonl, qdrant_local/ 생성

# (정식) 전량 파싱·청킹·임베딩 — 비용 발생, 승인 후
#  embed/embed_batch.py (OpenAI Batch API, ~$14.6) 참조
```

## 비밀키 (.env) — repo에 없음, 직접 생성

`.env`는 `.gitignore`라 GitHub에 안 올라감. 새 환경에서:

```bash
cp .env.example .env
# .env 편집:
#   OPENAI_API_KEY=sk-...        (본인 OpenAI 키)
#   LAW_OC=...                   (law.go.kr OpenAPI OC 값)
#   QDRANT_URL=http://localhost:6333   (정식: Qdrant Cloud URL)
```

> 문서에서 키 실제값은 보안상 `<...REDACTED>`로 지웠음. 작동에는 영향 없음 — **`.env`에만 넣으면 됨.**
