# 08. lawbot 빌드 하네스 플레이북 (BUILD HARNESS)

> 작성: 2026-06-15 · 기반: 01~07 리서치/검증 문서 + `data/`·`법률데이터/` 실측
> 목적: **읽고 끝나는 리서치 문서가 아니라, 에이전트/하네스가 그대로 "실행"해서 lawbot을 빌드하는 작업 지시서.**
> 확정 스택(사용자 결정 2026-06-15): **OpenAI 임베딩(text-embedding-3) + OpenAI GPT + Qdrant + FastAPI**, **지금 6개 코퍼스로 즉시 MVP**.

---

## 0. 이 문서 사용법 (하네스 방식이란)

이 문서는 **Phase → Task** 트리로 되어 있다. 각 Task는 다음 6칸을 가진다.

| 칸 | 의미 |
|---|---|
| 🎯 목표 | 이 Task가 끝나면 무엇이 존재하는가 |
| 📥 입력 | 어떤 파일/값이 있어야 시작 가능한가 |
| 🛠 작업 | 무엇을 하는가 (사람 또는 에이전트가) |
| 📤 산출물 | 생기는 파일/아티팩트 (경로 고정) |
| ✅ DoD | "끝났다"의 검증 기준 (이게 통과해야 다음 Task로) |
| 💻 스켈레톤 | 바로 채워 쓰는 코드 골격 |

**하네스 실행 = 각 Task를 독립 단위로 보고**, Claude Code의 `Workflow`/`Agent`에 "이 Task의 🎯📥🛠📤✅를 줄 테니 💻를 완성·실행하라"고 넘기는 방식. Task끼리 입력/산출물 경로가 맞물려 있어 파이프라인으로 자동 연결된다. (§9에 실행용 Workflow 스크립트 골격 포함.)

> 비전공자 메모: 지금 당장 모든 코드를 이해할 필요는 없다. `06_학습_로드맵.md`의 ①~⑩을 병행하면서, 이 문서의 Phase 0→5를 위에서부터 하나씩 "통과(DoD)"시키면 된다.

### 확정된 핵심 결정 (다시 안 물어봄)

| 항목 | 결정 | 근거 |
|---|---|---|
| 임베딩 | **OpenAI `text-embedding-3-small`(1536d) 기본**, 품질 필요시 `-large`(3072d) | GPU 불필요, 키만 있으면 즉시. 비용 §6 |
| 답변 LLM | **OpenAI `gpt-4o-mini` 기본**, 어려운 질의 `gpt-4o` 폴백 | 저렴·자료 풍부 |
| 벡터 DB | **Qdrant** (개발=Docker 로컬, 배포=Qdrant Cloud 무료티어) | 메타 필터+하이브리드 지원 |
| API | **FastAPI** + API 키 발급/관리 내장 | "API 발급 형태로 서비스" 요구 충족 |
| 청킹 단위 | **조문(법령·행정규칙·자치법규) / 섹션(판례)** = 1차 청크 | 원천이 이미 조문 단위로 분리됨 §4 |
| MVP 데이터 | **현재 6개 코퍼스**(국가법령+판례+행정규칙+자치법규4) | 즉시 0→1 |
| 검색 | dense 우선, **하이브리드(BM25+dense)+리랭커는 Phase 3.5 옵션** | OpenAI-only로 GPU 회피 |

---

## 1. 내가(사용자) 너에게 줘야 할 것 — 자재 요청서

빌드를 막힘없이 진행하려면 아래를 준비해 주면 된다. **★는 Phase 0에서 즉시 필요**, 나머지는 해당 Phase 직전에 주면 됨.

### 1-1. API 키 / 계정 (`.env`에 넣음)

| # | 무엇 | 어디서 | 언제 | 비고 |
|---|---|---|---|---|
| ★1 | **OpenAI API Key** (`OPENAI_API_KEY`) | platform.openai.com → API keys | Phase 0 | 임베딩+답변 둘 다 사용. 결제수단 등록 필요. **Usage limit(월 상한) 꼭 설정** |
| 2 | **Qdrant** (로컬은 불필요) | Docker로 로컬 실행 → 키 없음. 배포 시 Qdrant Cloud `QDRANT_URL`+`QDRANT_API_KEY` | Phase 2(로컬)/5(클라우드) | 무료티어 1GB로 MVP 가능 |
| 3 | (선택) **law.go.kr OpenAPI OC** (`LAW_OC`) | open.law.go.kr 회원가입→OC 신청(1~2일) | Phase 2확장 | 신선도 갱신·헌재/해석례 추가용. MVP엔 불필요 |
| 4 | (배포) 호스팅 계정 | Render/Railway/Fly.io 중 1 | Phase 5 | 카드 등록 |

> **키 전달 방법**: 채팅에 평문으로 붙여넣지 말고, 로컬 `.env` 파일에 직접 넣어줘. 이 문서/코드는 `.env`에서만 읽는다(아래 §Phase0). `.gitignore`에 `.env` 포함 확인.

### 1-2. 결정/확인이 필요할 때 물어볼 것 (지금은 기본값으로 진행)

- 임베딩 모델 small↔large 전환 시점(품질 테스트 결과 보고 §Phase3 DoD에서 결정)
- 답변 LLM gpt-4o-mini↔gpt-4o 승급 기준(골든셋 점수 §Phase5)
- 서비스 도메인/브랜드명(배포 Phase5 전)
- API 요금제(무료/유료 키 등급) 정책(Phase4)

### 1-3. 내가 안 줘도 되는 것 (이미 디스크에 있음)

- 원천 데이터 전부: `원천데이터/`(01_국가법령·02_자치법규·03_행정규칙·04_판례) — 중복정리 완료(`원천데이터/README.md`)
- 실측 통계: `_samples/corpus_stats.txt`
- 리서치/검증 근거: `분석/01~07`

---

## 2. 최종 아키텍처

```
                         ┌─────────────────────── 빌드 타임(1회/주기) ───────────────────────┐
 data/legalize-kr  ─┐    │  [P1] 파서 → 통합 Document  ─→ [P2] 청킹 → OpenAI 임베딩(Batch)   │
 data/precedent-kr ─┼──▶ │                                        ↓                          │
 data/admrule-kr   ─┤    │                              Qdrant 업서트(dense + payload)        │
 법률데이터/*.zip   ─┘    └────────────────────────────────────────┬──────────────────────────┘
 (자치법규 4개)                                                     │
                         ┌──────────────────── 런타임(질의마다) ────┴──────────────────────────┐
   사용자 질문 ─▶ FastAPI /v1/ask ─▶ (메타 사전필터) ─▶ Qdrant dense 검색 top-K              │
                         │                              [+P3.5 옵션: BM25 하이브리드 + 리랭커]   │
                         │                                        ↓                            │
                         │        OpenAI GPT (검색결과를 컨텍스트로, 인용 강제 JSON 스키마)     │
                         │                                        ↓                            │
                         │        인용 사후검증(인용 ID가 실제 DB에 존재하는지) + 면책문구       │
                         └─▶ {answer, citations[], disclaimer}  ◀───────────────────────────────┘
   API 키 인증 + Rate limit ─ 모든 엔드포인트 앞단
```

핵심 원칙(01·03 검증 반영):
1. **답변은 검색된 원문만 근거로**(RAG). 모델 내부지식 단독 답변 금지.
2. **모든 답변에 인용**(법령명·조문번호·판례 사건번호·출처 URL). 인용은 LLM이 지어내지 못하게 **JSON 스키마 강제 + 사후 존재검증**.
3. **변호사법 §109 자세**: "정보제공/검색"만, "구체 사건 법률상담·결론·서면작성" 금지 → 시스템 프롬프트+면책문구로 강제.

---

## 3. 데이터 실측 인벤토리 & 형식별 구조 (코드가 의존하는 진실)

> 정본 통계: `_samples/corpus_stats.txt`. 인덱싱 대상은 **현행 조문/판례만**(과거버전 제외).

| 코퍼스 | 위치 | 형식 | 건수 | 인덱싱 단위 |
|---|---|---|---|---|
| 국가법령(법률·시행령·시행규칙·대통령령 등) | `원천데이터/01_국가법령/kr/{법령명}/{구분}.md` | YAML+MD, `##### 제N조` 구분 | 약 5,673 법령문서 | 조문 |
| 판례 | `원천데이터/04_판례/{사건종류}/{등급}/{법원}_{선고일}_{사건번호}.md` | YAML+`## 섹션` | 123,742 | 섹션(긴 건 재분할) |
| 행정규칙 | `원천데이터/03_행정규칙/{부처}/.../{규칙명}/본문.md` | YAML+본문(`제N조(...)` 인라인) | 약 21,700 | 조문 |
| 자치법규(**전국 18개 시도**) | `원천데이터/02_자치법규/{시도}/.../{법령명}/본문.md` | YAML+본문(`##### 제N조` 헤더) | **159,890** (약 233만 조문) | 조문 |

> ✅ 데이터는 `원천데이터/`로 정리·중복제거 완료(`원천데이터/README.md` 참조). 자치법규는 zip(4개 시도)이 아니라 **클론(markdown, 18개 시도)** 을 정본으로 사용 → 파서가 행정규칙과 동일.

**형식별 파싱 키 포인트** (Phase 1 파서가 이걸 처리):

- **국가법령 `.md`**: 프론트매터 키 = `제목, 법령ID, 법령구분, 소관부처, 공포일자, 시행일자, 상태, 출처`. 본문은 `##### 제N조 (제목)` 헤더로 조문 분리. `①②③` 항, `<개정 ...>` 개정주석 포함. 한 법령 폴더에 `법률.md`·`시행령.md`·`시행규칙.md`가 따로 있으면 **각각 별도 문서**로 취급(법령구분 메타로 구분).
- **판례 `.md`**: 프론트매터 = `판례일련번호, 사건번호, 사건명, 법원명, 법원등급, 사건종류, 출처, 선고일자`. 본문 섹션 = `## 판결요지`, `## 판례내용`(내부 `【심급】【주문】【이유】` 블록). 섹션 결측 흔함 → 있는 섹션만.
- **행정규칙 `본문.md`**: 프론트매터 = `행정규칙ID, 행정규칙명, 행정규칙종류, 소관부처명, 발령일자, 시행일자, 첨부파일(별표)`. 본문은 `#####` 헤더가 **아니라** `제N조(제목) 내용...` 인라인 → 정규식 `^제\d+조(의\d+)?\s*\(` 로 분리. 일부는 본문 결측(라벨만).
- **자치법규 `본문.md`**: 프론트매터 = `자치법규ID, 자치법규명, 자치법규종류(조례·규칙), 지자체구분(광역·기초), 공포일자, 시행일자, 첨부파일(별표)`. 본문은 **행정규칙과 동일하게** `제N조(제목)` 인라인 → 같은 정규식으로 분리. (zip의 `current.articles` 형식은 더 이상 사용 안 함.)

---

## 4. 청킹 / 임베딩 단위 결정 (왜 이렇게 쪼개나)

### 4-1. 청킹 규칙

| 문서종류 | 1차 청크 | 2차 분할 | 메타로 뺄 것 |
|---|---|---|---|
| 법령/자치법규 조문 | **조문 1개 = 청크 1개** | 조문이 8,191토큰 초과(드묾)면 항 단위 분할 | 법령명·조문번호·시행일·소관부처(헤더는 임베딩 텍스트에 prefix로 1줄만 포함) |
| 행정규칙 조문 | 조문 1개 = 청크 1개(재병합 후) | 동일 | 규칙명·종류·부처 |
| 판례 | **섹션 1개 = 청크 1개**(판결요지/판례내용 등) | 섹션이 길면 800~1,200토큰, 오버랩 200 | 사건번호·법원·선고일·사건종류 |

원칙:
- 원천이 **이미 조문/섹션 단위**라 의미 단위가 깔끔함 → 일반 텍스트 스플리터 불필요.
- **헤더(법령명·조문번호)는 임베딩 텍스트 맨 앞에 1줄 prefix**로 넣어 "어느 법 몇 조"인지 벡터에 묻게 함. 나머지 메타는 payload로만(검색 필터·인용용).
- 판례 긴 섹션엔 **SAC(요약 prefix) 옵션**(03 §): 섹션 앞에 한두 문장 요약을 붙이면 검색 회수율↑. MVP는 생략 가능, Phase3.5에서 A/B.

### 4-2. 임베딩 모델 (확정: OpenAI)

- 기본 **`text-embedding-3-small`** — 1536차원, 입력 최대 8,191토큰/건, `dimensions` 파라미터로 축소 가능(MVP는 1536 그대로).
- 품질 부족 판단 시 **`text-embedding-3-large`**(3072d)로 교체 — 코드에서 모델명·차원만 바꾸면 됨(Qdrant 컬렉션 차원도 같이). 그래서 §Phase2 스켈레톤은 모델/차원을 **상수 1곳**에서만 설정.
- **Batch API 사용**(50% 할인, 24h 윈도우) — 약 400만 청크 1회 임베딩이라 배치가 합리적.

> 단위·모델 변경은 전부 `config.py`의 상수만 고치면 전파되도록 설계(아래).

---

## 5. 빌드 하네스 — Phase별 Task

권장 디렉터리(신규 생성):
```
lawbot/
  .env                 # 키 (gitignore)
  config.py            # 모든 상수 1곳
  ingest/              # P1 파서
    parse_statute.py   # 국가법령
    parse_precedent.py # 판례
    parse_admrule.py   # 행정규칙
    parse_ordinance.py # 자치법규 zip
    schema.py          # 통합 Document/Chunk 모델
  embed/               # P2
    chunk.py
    embed_batch.py     # OpenAI Batch 제출/수거
    upsert_qdrant.py
  search/              # P3
    retriever.py
    rag.py             # 검색→GPT→인용검증
  api/                 # P4
    main.py            # FastAPI
    auth.py            # API 키 발급/검증 + rate limit
  eval/                # P5
    golden_set.jsonl
    run_eval.py
  artifacts/           # 중간 산출물(jsonl 등, gitignore)
```

---

### Phase 0 — 환경·계정·데이터 검증

#### Task 0.1 — 프로젝트 골격 + 비밀키
- 🎯 `lawbot/` 디렉터리, `.env`, `config.py` 생성. OpenAI 키 동작 확인.
- 📥 사용자 자재 ★1(OpenAI 키).
- 🛠 venv 생성 → 의존성 설치 → `.env` 작성 → 키 1회 호출 테스트.
- 📤 `lawbot/config.py`, `lawbot/.env`, `requirements.txt`.
- ✅ DoD: `python -c "from openai import OpenAI; print(OpenAI().models.list().data[0].id)"` 가 모델 id 출력.
- 💻:
```bash
python -m venv .venv && source .venv/bin/activate   # win: .venv\Scripts\activate
pip install openai qdrant-client fastapi uvicorn pydantic python-dotenv pyyaml tiktoken slowapi tenacity
pip freeze > requirements.txt
```
```python
# config.py — 모든 상수는 여기서만 바꾼다
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
EMBED_MODEL      = "text-embedding-3-small"   # 품질 필요시 "text-embedding-3-large"
EMBED_DIM        = 1536                         # large면 3072
EMBED_MAX_TOKENS = 8191
GEN_MODEL        = "gpt-4o-mini"               # 승급시 "gpt-4o"
QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")  # 로컬이면 None
COLLECTION       = "lawbot"
DATA_ROOT        = r"\\wsl.localhost\Ubuntu\home\user1\체크\NEW2"   # 원천 루트
```

#### Task 0.2 — 데이터 실측 재확인
- 🎯 corpus_stats와 디스크 실제가 일치함을 확인(코드가 의존하는 경로/건수).
- 📥 `data/`, `법률데이터/`, `_samples/corpus_stats.txt`.
- 🛠 각 코퍼스 건수 카운트 → corpus_stats와 대조.
- ✅ DoD: 국가법령≈3,062폴더, 판례=123,743 .md, 행정규칙 `본문.md` 수, 자치법규 zip 4개 존재. 차이 나면 로그.
- 💻: `find data/precedent-kr -name '*.md' ! -name README.md | wc -l` 등으로 확인(이미 §3에서 1차 확인됨).

---

### Phase 1 — 인제스트 & 통합 정규화 (파서)

> 목표: 4개 형식 → **하나의 `Document` 모델**로 통일. 이후 단계는 형식을 몰라도 됨.

#### Task 1.0 — 통합 스키마
- 🎯 모든 문서를 표현하는 `Document`/`Article` 모델 + 안정적 ID 규칙.
- 📤 `ingest/schema.py`.
- ✅ DoD: 4개 파서가 전부 이 모델 인스턴스를 반환.
- 💻:
```python
# ingest/schema.py
from pydantic import BaseModel
from typing import Literal, Optional

DocType = Literal["law", "ordinance", "admrule", "precedent"]

class Article(BaseModel):       # 법령류: 조문 / 판례: 섹션
    article_no: str             # "제4조" | "판결요지" 등
    title: Optional[str] = None
    text: str

class Document(BaseModel):
    doc_id: str                 # 안정적 고유 ID (아래 규칙)
    doc_type: DocType
    title: str                  # 법령명/사건명
    jurisdiction: str           # "국가" | "광주" | "전남" ... | 법원명
    law_kind: Optional[str]=None# "법률"|"시행령"|"조례"|"대통령훈령"|사건종류
    effective_from: Optional[str]=None  # 시행일/선고일
    source_url: Optional[str]=None
    trust_grade: str = "A"      # A=원문있음, B=메타만
    articles: list[Article]
    meta: dict = {}             # 부처/별표/법령ID 등 원본 보존

# doc_id 규칙 (절대 commit hash 쓰지 말 것 — 07 검증)
#   law      : LAW:{법령ID}:{법령구분}        예) LAW:011463:법률
#   ordinance: ORD:{지자체}:{법령ID}
#   admrule  : ADMRULE:{행정규칙ID}
#   precedent: PREC:{판례일련번호}
```

#### Task 1.1 — 국가법령 파서
- 🎯 `data/legalize-kr/kr/**.md` → `Document[]`.
- 🛠 YAML 프론트매터 파싱 + 본문 `##### 제N조` 분리.
- 📤 `ingest/parse_statute.py`, 산출 `artifacts/docs_law.jsonl`.
- ✅ DoD: 임의 5건 스폿체크 — 조문 수·제목·시행일 맞음. 본문 없는 폴더는 skip 로그.
- 💻:
```python
# ingest/parse_statute.py
import re, glob, yaml, json
from ingest.schema import Document, Article

FM = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
ART = re.compile(r"^#{3,6}\s*(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?", re.M)

def parse_file(path):
    raw = open(path, encoding="utf-8").read()
    m = FM.match(raw);  fm = yaml.safe_load(m.group(1)); body = m.group(2)
    hits = list(ART.finditer(body))
    arts = []
    for i, h in enumerate(hits):
        seg = body[h.end(): hits[i+1].start() if i+1 < len(hits) else len(body)]
        arts.append(Article(article_no=h.group(1), title=h.group(2), text=seg.strip()))
    return Document(
        doc_id=f"LAW:{fm.get('법령ID')}:{fm.get('법령구분')}",
        doc_type="law", title=fm.get("제목",""), jurisdiction="국가",
        law_kind=fm.get("법령구분"), effective_from=str(fm.get("시행일자","")),
        source_url=fm.get("출처"), articles=arts,
        meta={k: fm.get(k) for k in ("법령ID","법령MST","소관부처","상태")},
    )

if __name__ == "__main__":
    with open("artifacts/docs_law.jsonl","w",encoding="utf-8") as out:
        for p in glob.glob("원천데이터/01_국가법령/kr/*/*.md"):
            try:
                d = parse_file(p)
                if d.articles: out.write(d.model_dump_json()+"\n")
            except Exception as e: print("SKIP", p, e)
```

#### Task 1.2 — 판례 파서
- 🎯 `data/precedent-kr/**.md` → `Document[]`(섹션=Article).
- 🛠 프론트매터 + `## ` 섹션 분리. `○○○`→`[당사자]` 등 비식별 정규화(03). 12만건이라 스트리밍.
- 📤 `ingest/parse_precedent.py` → `artifacts/docs_prec.jsonl`.
- ✅ DoD: 사건번호·법원·선고일 채워짐, 섹션 결측건도 깨지지 않음.
- 💻: `## (.+)` 로 섹션 split, 각 섹션 → `Article(article_no=섹션명)`. doc_id=`PREC:{판례일련번호}`. (1.1과 동일 골격, 헤더 정규식만 `^##\s+(.+)$`)

#### Task 1.3 — 행정규칙 파서
- 🎯 `data/admrule-kr/**/본문.md` → `Document[]`.
- 🛠 본문 `제N조(...)` **인라인** 정규식 `^(제\d+조(?:의\d+)?)\s*\(([^)]*)\)` 분리. 본문결측(라벨만)이면 `trust_grade="B"`, 메타만.
- 📤 `ingest/parse_admrule.py` → `artifacts/docs_admrule.jsonl`.
- ✅ DoD: 본문 있는 건은 조문 1+개, 결측건은 B등급으로 별도 카운트.

#### Task 1.4 — 자치법규 파서 (행정규칙 파서 재사용)
- 🎯 `원천데이터/02_자치법규/**/본문.md`(**전국 18개 시도**) → `Document[]`.
- 🛠 **Task 1.3(행정규칙) 파서를 그대로 재사용** — 같은 `본문.md` + `제N조()` 인라인 구조. 프론트매터 키만 자치법규용으로 매핑(`자치법규ID·자치법규종류·지자체구분.광역`). doc_id=`ORD:{광역}:{자치법규ID}`.
- 📤 `ingest/parse_ordinance.py` → `artifacts/docs_ord.jsonl`.
- ✅ DoD: 약 5.2만 법령 파싱, 시도 18개 모두 포함, 조례·규칙 구분·시행일 채워짐. (zip 사용 안 함.)
- 💻:
```python
# ingest/parse_ordinance.py (요지) — 1.3과 동일, glob/매핑만 교체
import glob
for p in glob.glob("원천데이터/02_자치법규/**/본문.md", recursive=True):
    fm, arts = parse_bonmun(p)   # 1.3의 본문.md 파서 공용화
    doc = Document(
        doc_id=f"ORD:{fm['지자체구분']['광역']}:{fm['자치법규ID']}",
        doc_type="ordinance", title=fm["자치법규명"],
        jurisdiction=fm["지자체구분"]["광역"], law_kind=fm["자치법규종류"],
        effective_from=str(fm.get("시행일자","")), source_url=fm.get("출처"),
        articles=arts, meta={"지자체기관명": fm.get("지자체기관명")})
```

> Phase 1 종료 산출물: `artifacts/docs_{law,prec,admrule,ord}.jsonl` = **통합 Document 전량**.

---

### Phase 2 — 청킹 → OpenAI 임베딩(Batch) → Qdrant 적재

#### Task 2.1 — 청킹
- 🎯 Document → `Chunk[]`(임베딩 단위). §4 규칙 적용.
- 📥 P1 jsonl 4개.
- 🛠 조문/섹션을 청크로, 헤더 prefix 부착, 8,191토큰 초과만 2차 분할(tiktoken으로 길이 측정).
- 📤 `artifacts/chunks.jsonl` — `{chunk_id, doc_id, text, payload{...}}`.
- ✅ DoD: 청크 수 약 340~400만, 모든 청크 토큰 ≤ 8,191, payload에 필터키(jurisdiction·doc_type·law_kind·effective_from) 존재.
- 💻:
```python
# embed/chunk.py (요지)
import tiktoken, json
enc = tiktoken.get_encoding("cl100k_base")
def chunks_of(doc):
    for a in doc["articles"]:
        header = f"[{doc['title']} {a['article_no']}{' '+a['title'] if a.get('title') else ''}]"
        body = f"{header}\n{a['text']}"
        toks = enc.encode(body)
        parts = [body] if len(toks) <= 8000 else split_overlap(body, 1000, 200)  # 판례 긴 섹션
        for j, p in enumerate(parts):
            yield {
              "chunk_id": f"{doc['doc_id']}#{a['article_no']}#{j}",
              "doc_id": doc["doc_id"], "text": p,
              "payload": {"doc_type":doc["doc_type"],"title":doc["title"],
                          "jurisdiction":doc["jurisdiction"],"law_kind":doc.get("law_kind"),
                          "article_no":a["article_no"],"effective_from":doc.get("effective_from"),
                          "source_url":doc.get("source_url"),"trust_grade":doc.get("trust_grade","A")},
            }
```

#### Task 2.2 — OpenAI Batch 임베딩
- 🎯 모든 청크 텍스트 → 벡터. **Batch API(50%↓)**.
- 📥 `artifacts/chunks.jsonl`.
- 🛠 ① chunks를 Batch 입력 `.jsonl`(각 줄 `{custom_id, method:POST, url:/v1/embeddings, body:{model,input}}`)로 변환 → ② `files.create(purpose="batch")` → ③ `batches.create(...)` → ④ 완료까지 폴링 → ⑤ 결과 다운로드해 `chunk_id→vector` 매핑.
- 📤 `artifacts/embeddings.jsonl` (`{chunk_id, vector[1536]}`).
- ✅ DoD: 임베딩 수 = 청크 수, 차원=EMBED_DIM. 실패 custom_id 재시도 로그.
- ⚠️ 비용 가드: 제출 전 **토큰 총량 추정 → 예상비용 출력 → 사람 확인(y/n)** 게이트. (§6)
- 💻:
```python
# embed/embed_batch.py (요지)
from openai import OpenAI; from config import EMBED_MODEL
cli = OpenAI()
# 1) build batch input
with open("artifacts/batch_in.jsonl","w",encoding="utf-8") as w:
    for c in read_jsonl("artifacts/chunks.jsonl"):
        w.write(json.dumps({"custom_id":c["chunk_id"],"method":"POST",
          "url":"/v1/embeddings","body":{"model":EMBED_MODEL,"input":c["text"]}})+"\n")
# 2~3) submit
f = cli.files.create(file=open("artifacts/batch_in.jsonl","rb"), purpose="batch")
b = cli.batches.create(input_file_id=f.id, endpoint="/v1/embeddings", completion_window="24h")
print("batch:", b.id)   # 4) 이후 batches.retrieve(b.id) 폴링 → output_file_id 다운로드
# ※ 50,000줄/200MB 배치 한도 → chunks를 샤딩해 여러 배치로 제출
```

#### Task 2.3 — Qdrant 컬렉션 생성 + 업서트
- 🎯 벡터+payload를 Qdrant에 적재(검색 가능 상태).
- 📥 `embeddings.jsonl` + `chunks.jsonl`(payload).
- 🛠 로컬 Qdrant(Docker) → 컬렉션 생성(size=EMBED_DIM, distance=Cosine) → payload 인덱스(jurisdiction·doc_type·law_kind·effective_from 필터용) → 배치 업서트.
- 📤 Qdrant `lawbot` 컬렉션(약 400만 포인트).
- ✅ DoD: `count` = 청크 수, 필터검색 `doc_type="law"` 동작, 임의 쿼리 top-5가 관련 조문 반환.
- 💻:
```bash
docker run -p 6333:6333 -p 6334:6334 -v $PWD/qdrant_storage:/qdrant/storage qdrant/qdrant
```
```python
# embed/upsert_qdrant.py (요지)
from qdrant_client import QdrantClient, models
from config import QDRANT_URL, COLLECTION, EMBED_DIM
q = QdrantClient(url=QDRANT_URL)
q.recreate_collection(COLLECTION,
    vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE))
for field in ["doc_type","jurisdiction","law_kind","effective_from"]:
    q.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
# uuid는 chunk_id 해시 → point id, vector + payload 업서트 (batch 1000개씩)
```

---

### Phase 3 — 검색 + RAG 파이프라인

#### Task 3.1 — Retriever (dense + 메타필터)
- 🎯 질문 → 관련 청크 top-K(필터 지원).
- 🛠 질문 임베딩(같은 EMBED_MODEL) → Qdrant search(필터: 예 "전남 조례만", "법률만") → top-K payload+text.
- 📤 `search/retriever.py`.
- ✅ DoD: 법령 질문은 법령이, 판례 질문은 판례가 상위. 필터 적용 정확.
- 💻:
```python
# search/retriever.py
from openai import OpenAI; from qdrant_client import QdrantClient, models
from config import EMBED_MODEL, QDRANT_URL, COLLECTION
oc, q = OpenAI(), QdrantClient(url=QDRANT_URL)
def search(query, k=8, flt: dict|None=None):
    v = oc.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    qf = models.Filter(must=[models.FieldCondition(key=kk, match=models.MatchValue(value=vv))
                             for kk,vv in (flt or {}).items()]) if flt else None
    return q.search(COLLECTION, query_vector=v, query_filter=qf, limit=k, with_payload=True)
```

#### Task 3.2 — RAG /ask (검색→GPT→인용검증)
- 🎯 질문 → {answer, citations[], disclaimer}. 인용 강제+검증.
- 🛠 ① retriever top-K → ② 컨텍스트 구성(각 청크에 `[n]` 번호+메타) → ③ GPT 호출(시스템: "검색결과만 근거, 인용 필수, 법률상담 금지"; **Structured Outputs**로 citations 스키마 강제) → ④ 인용 ID가 실제 검색결과/ DB에 있는지 사후검증, 없으면 경고/제거 → ⑤ 면책문구 부착.
- 📤 `search/rag.py`.
- ✅ DoD: 답변의 모든 citation.source_id가 컨텍스트에 실재. "이혼하려면 어떻게?" 같은 상담성 질문엔 정보제공+면책으로 응답(결론·조언 회피).
- 💻:
```python
# search/rag.py (요지)
from openai import OpenAI; from config import GEN_MODEL; from search.retriever import search
oc = OpenAI()
SYS = ("너는 한국 법령·판례 '정보 검색' 도우미다. 아래 [검색결과]에 있는 내용만 근거로 답하라. "
       "각 사실에 [번호] 인용을 달고, 결과에 없으면 '확인 불가'라고 말하라. "
       "구체적 사건의 법률상담·승소예측·법률문서 작성은 하지 말고 '변호사 상담 권유'로 안내하라.")
CITES = {  # Structured Outputs json_schema
  "type":"object","properties":{
    "answer":{"type":"string"},
    "citations":{"type":"array","items":{"type":"object","properties":{
       "source_id":{"type":"string"},"title":{"type":"string"},
       "location":{"type":"string"},"source_url":{"type":"string"}},
       "required":["source_id","title","location"]}}},
  "required":["answer","citations"]}
def ask(query, k=8, flt=None):
    hits = search(query, k, flt)
    ctx = "\n\n".join(f"[{i+1}] ({h.payload['doc_type']}) {h.payload['title']} "
                      f"{h.payload['article_no']} | id={h.id} | {h.payload.get('source_url','')}\n{h.payload['text'][:1200]}"
                      for i,h in enumerate(hits))
    r = oc.chat.completions.create(model=GEN_MODEL, response_format={"type":"json_schema",
          "json_schema":{"name":"cited_answer","schema":CITES,"strict":True}},
        messages=[{"role":"system","content":SYS},
                  {"role":"user","content":f"[검색결과]\n{ctx}\n\n[질문]\n{query}"}])
    import json; out = json.loads(r.choices[0].message.content)
    valid_ids = {h.payload['title'] for h in hits} | {str(h.id) for h in hits}
    out["citations"] = [c for c in out["citations"] if c["title"] in valid_ids or c["source_id"] in valid_ids]
    out["disclaimer"] = "본 답변은 법령·판례 정보 제공이며 법률자문이 아닙니다. 구체적 사안은 변호사 상담을 권합니다."
    return out
```

#### Task 3.5 (옵션) — 하이브리드 + 리랭커
- 🎯 회수율↑: BM25(한국어 형태소, Kiwi/Nori) sparse + dense → RRF 합산 → (선택) 크로스인코더 리랭크 top-8.
- 비고: 리랭커는 로컬 GPU 모델이라 OpenAI-only MVP에선 **후순위**. Qdrant 내장 sparse(BM25/FastEmbed)로 GPU 없이 하이브리드만 먼저 도입 가능.
- ✅ DoD: 골든셋(§5) nDCG@10가 dense 단독 대비 향상되면 채택.

---

### Phase 4 — FastAPI + API 키 발급/관리 + 가드레일

#### Task 4.1 — API 키 발급/검증 + Rate limit
- 🎯 "API 발급 형태 서비스" 충족: 키 생성/조회/폐기 + 인증 미들웨어 + 호출량 제한.
- 🛠 키 저장(MVP=SQLite, 운영=Postgres/Redis). `Authorization: Bearer <key>`. 익명=읽기검색 IP제한, 키=상향, `/ask`(LLM비용)는 키 필수.
- 📤 `api/auth.py`.
- ✅ DoD: 키 발급→그 키로 `/v1/ask` 성공, 무키 401, 한도초과 429.
- 💻:
```python
# api/auth.py (요지) — 키=난수 hash 저장, slowapi로 rate limit
import secrets, hashlib, sqlite3
def issue_key(owner, tier="free"):
    raw = "lk_"+secrets.token_urlsafe(24)
    db().execute("INSERT INTO api_keys(hash,owner,tier) VALUES(?,?,?)",
                 (hashlib.sha256(raw.encode()).hexdigest(), owner, tier))
    return raw   # 평문은 이때 한 번만 반환
def verify(key)->dict|None:
    row = db().execute("SELECT owner,tier FROM api_keys WHERE hash=? AND revoked=0",
                       (hashlib.sha256(key.encode()).hexdigest(),)).fetchone()
    return {"owner":row[0],"tier":row[1]} if row else None
```

#### Task 4.2 — FastAPI 엔드포인트
- 🎯 외부서비스 API.
- 📤 `api/main.py`.
- ✅ DoD: `uvicorn api.main:app` 기동, `/healthz` 200, `/v1/ask` 인용 응답, `/docs`(OpenAPI) 자동생성.
- 💻:
```python
# api/main.py (요지)
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from search.rag import ask; from search.retriever import search
from api.auth import verify
app = FastAPI(title="lawbot API")
def auth(authorization: str = Header(None)):
    u = verify(authorization.removeprefix("Bearer ").strip()) if authorization else None
    if not u: raise HTTPException(401, "API key required")
    return u
class AskReq(BaseModel): query: str; filter: dict|None=None; k:int=8
@app.get("/healthz")            def health(): return {"ok": True}
@app.post("/v1/ask")            def v1_ask(r: AskReq, u=Depends(auth)): return ask(r.query, r.k, r.filter)
@app.post("/v1/search")         def v1_search(r: AskReq):  # 익명 허용(읽기)
    return [{"id":h.id, **h.payload} for h in search(r.query, r.k, r.filter)]
```
- 가드레일(03/04 요약, 코드에 반영):
  - 모든 `/ask` 응답에 `disclaimer` 강제(이미 rag.py).
  - 시스템 프롬프트로 변호사법 §109 회피(상담·결론·서면 금지).
  - AI기본법: 응답 메타에 "AI 생성" 표기 필드 추가.
  - 개인정보: 판례 비식별 정규화 유지, 질의 로그에 PII 저장 금지.

---

### Phase 5 — 평가(골든셋) & 배포

#### Task 5.1 — 골든셋 평가
- 🎯 품질 수치화 → small↔large, 4o-mini↔4o, 하이브리드 도입 판단 근거.
- 🛠 30~100개 질문+정답조문/판례 `eval/golden_set.jsonl` 작성 → retrieval Hit@K·답변 인용정확도·할루시네이션율 측정(Stanford 기준 17~33% 베이스라인 인지).
- ✅ DoD: 베이스라인 점수표 산출. 회귀 비교 가능.

#### Task 5.2 — 배포
- 🎯 외부 접속 가능한 HTTPS API.
- 🛠 Qdrant Cloud(무료티어) 업로드 → Render/Railway에 FastAPI 컨테이너 배포 → Caddy/플랫폼 자동 HTTPS → 환경변수에 키 주입.
- ✅ DoD: 공개 URL `/healthz` 200, 발급키로 `/v1/ask` 동작, 사용량/비용 모니터링 켜짐.
- 💻: `Dockerfile`(python-slim + uvicorn) + 플랫폼 `render.yaml`. (배포 상세는 `04_배포_및_구축계획.md` Phase1/B 참조 — 단 임베딩/LLM은 OpenAI로 치환.)

---

## 6. 비용 추정 (OpenAI 기준 · 발주 전 공식가 재확인)

> 가격은 변동 가능. 아래는 작성 시점 기준 **추정**이며, 코드의 비용게이트(Task 2.2)가 실제 토큰으로 다시 계산해 보여준다.

| 항목 | 가정 | 모델 | 1회/월 비용(추정) |
|---|---|---|---|
| 초기 임베딩(빌드) | 약 8.9억 토큰(02 추정) | `3-small` Batch | **약 $9** (large Batch면 약 $58) |
| 질의 임베딩 | 질문당 ~수십 토큰 | `3-small` | 사실상 무시 가능 |
| 답변 생성 | 질의당 컨텍스트 ~6K입력+~0.7K출력 | `gpt-4o-mini` | 질의 1천건/월 ≈ 수 달러 |
| Qdrant | 무료티어→유료 | — | $0 ~ 소액 |
| 호스팅 | Render 등 | — | $0 ~ $25 |

→ **MVP 부트스트랩 총비용: 임베딩 1회 약 $9 + 월 운영 수~수십 달러.** OpenAI 대시보드에서 **월 사용한도(hard limit)** 설정 필수.

---

## 7. 법적/안전 가드레일 체크리스트 (출시 전 필수, 01·03·04 정본)

- [ ] 변호사법 §109: "법령·판례 **정보제공/검색**" 포지셔닝. 구체 사건 상담·결론·서면작성 금지(프롬프트+UI 문구).
- [ ] AI기본법(2026 시행): AI 생성물 표시 + 고지.
- [ ] 모든 답변에 **면책문구** + 출처(인용) 표기. 데이터 커버리지 정직 고지(국가법령·판례·행정규칙+자치법규 4개 시도, **나머지 13개 시도·헌재·해석례 미포함**).
- [ ] 개인정보: 판례 비식별 유지, 질의/응답 로그에 사용자 PII·민감정보 비저장(기본), 학습 미사용 명시.
- [ ] 라이선스: 본문=공공저작물(저작권법 §7 취지), 메타=원천 라이선스 → 출처표기 유지.
- [ ] 할루시네이션율 골든셋 측정(KPI), 인용 사후검증 항상 ON.

---

## 8. 데이터 보정 TODO (정확도용 — MVP 후 또는 병행)

- 행정규칙 **조문 과편화 재병합**((행정규칙ID+조문번호) 기준) — 검색 품질 직결(02).
- 행정규칙 **본문결측 건(B등급)** 분리 표기 — 답변에서 "메타만 있음" 고지.
- 참조 엔티티링킹(평문 "제8조 제2항" → 실제 doc_id 링크) — 후속.
- 판례 섹션 결측 보완(판례내용에서 판시사항/요지 휴리스틱 추출) — 후속.

---

## 9. 하네스 실행 방법 (이 문서로 자동 빌드)

이 문서의 Task들은 Claude Code `Workflow`로 파이프라인 실행할 수 있다. 각 Task의 🎯📥🛠📤✅를 에이전트에 주고 💻를 완성·실행시키되, **DoD를 검증 게이트로** 둔다.

```js
// 분석/lawbot_build_workflow.js (골격) — Workflow({scriptPath:"...lawbot_build_workflow.js"}) 로 실행
export const meta = {
  name: 'lawbot-build',
  description: '08 플레이북 Phase1~4를 에이전트로 빌드/검증',
  phases: [{title:'Ingest'},{title:'Embed'},{title:'Search'},{title:'API'}],
}
const TASKS = {
  ingest: ['parse_statute','parse_precedent','parse_admrule','parse_ordinance'],
}
// Phase1: 4개 파서를 병렬 생성 → 각자 DoD(스폿체크) 통과까지
phase('Ingest')
const parsers = await parallel(TASKS.ingest.map(name => () =>
  agent(`08_플레이북 Task 1.x의 ${name} 스켈레톤을 완성하고 실행해 artifacts/${name}.jsonl 생성.
         DoD(임의 5건 스폿체크)까지 검증하고 결과 요약 반환.`,
        {label:`ingest:${name}`, phase:'Ingest'})))
// Phase2: 청킹→배치임베딩(비용게이트는 사람확인)→Qdrant 업서트  ... (순차)
// Phase3: retriever/rag 작성 + 골든셋 일부로 스모크 테스트
// Phase4: FastAPI+auth 기동, /healthz·/ask 통과
```

> 주의: Phase 2의 **OpenAI Batch 임베딩(비용 발생)** 은 자동 실행하지 말고, 토큰·비용 추정 출력 후 **사람 승인** 게이트를 둔다(Task 2.2 ⚠️). 키는 `.env`에서만 읽고 로그·코드에 노출 금지.

---

## 10. 지금 바로 다음 한 걸음

1. 사용자 → `lawbot/.env`에 `OPENAI_API_KEY` 넣기(자재 ★1) + OpenAI 월 사용한도 설정.
2. **Phase 0 Task 0.1·0.2** 통과(골격+키검증+데이터카운트).
3. **Phase 1** 파서 4개 → `artifacts/docs_*.jsonl`.
4. **Phase 2** 소규모(예: 국가법령 1,000조문)로 청킹→임베딩→Qdrant→검색까지 **엔드투엔드 1회** 성공시킨 뒤 전량 확장(06 로드맵 "1K→10K→전량" 원칙).

> 막히면: 이 문서의 해당 Task 번호(예 "Task 2.2")를 대면 그 칸의 🛠💻✅ 기준으로 이어서 진행한다.
