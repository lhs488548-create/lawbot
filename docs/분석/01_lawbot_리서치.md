# lawbot 리서치 — 정의·사례·시장·데이터·연구동향·API 설계·규제

> 본 문서는 한국 법률 챗봇("lawbot") 프로젝트를 위한 1차 리서치 종합본이다. lawbot의 정의와 다의성, 국내외 오픈소스/상용 사례, 한국 법률 AI 시장 지형, 본 코퍼스의 원천인 `github.com/legalize-kr` 데이터 출처, 법률 RAG 연구 동향, 법률 챗봇 API 설계 관행, 그리고 규제·윤리(변호사법·면책·라이선스)를 다룬다.
>
> **데이터 기준일**: 2026-06-04 · **코퍼스**: `D:\법파일\*.zip` (공공저작물, 저작권법 §7) · **정확 통계 출처**: `D:/법파일/_samples/corpus_stats.txt` (본 문서 작성 시 직접 Read로 검증)

---

## 0. 한 장 요약 (TL;DR)

| 항목 | 핵심 내용 |
|---|---|
| **"lawbot"의 정의** | 단일 제품명이 아닌 **다의어**: (a) 다수의 오픈소스 GitHub 저장소명, (b) 대한민국 법제처(MOLEG)의 지능형 법령검색 시스템 별칭, (c) "법률 챗봇/법률 Q&A 봇"이라는 일반 개념 |
| **공통 아키텍처** | 한국·영어권 오픈소스 모두 **RAG(임베딩 검색 + LLM 요약)** 패턴으로 수렴. "법률 조언"이 아닌 "유사 판례·조문 정보 제공"으로 포지셔닝 |
| **한국 시장** | 로앤컴퍼니(로톡·빅케이스·슈퍼로이어)와 엘박스(LBox) 양강. 수백만 건 판례 데이터가 핵심 자산. B2B/B2G·구독제 수익모델 |
| **본 코퍼스 원천** | `github.com/legalize-kr` — 한국 법령·판례·자치법규·행정규칙을 Markdown+Git으로 버전관리. 본문=공공저작물(저작권법 §7), 구조/메타=MIT |
| **연구 핵심 발견** | RAG는 환각을 **줄이되 제거하지 못함**(상용 법률 도구도 17~33% 환각, Stanford). 인용검증·trust_grade 필터·시점 정합 필수 |
| **규제 핵심** | 변호사법 §109(비변호사 유상 법률사무 금지)가 최대 리스크. 무료·정보제공형은 위험 낮음, 유료·구체적 사건 자동상담은 위험. AI기본법(2026.1.22 시행) 고지·표시 의무 |

---

## 1. "lawbot"이란 무엇인가 — 정의와 다의성

### 1.1 세 갈래의 의미

`lawbot`은 고유한 단일 제품을 가리키지 않는다. 검색·조사 결과 다음 세 가지를 동시에 지칭하는 다의어다.

1. **다수의 오픈소스 GitHub 저장소 이름** — `hunsii/LawBot`, `boostcampaitech5 .../LawBot`, `Vasumathi2002/LawBot`, `storminstakk/Justice_Juggernaut` 등. 저장소마다 목적·구조가 다르다.
2. **대한민국 법제처(MOLEG)의 정부 공식 지능형 법령검색 시스템 별칭** — `law.go.kr/LSW/aai/main.do`. 자연어 질의(예: "이사 후 전입신고 방법")로 법령 정보를 검색해 주는 공공 서비스로, 오픈소스 봇과는 **별개**다. (출처: [law.go.kr](https://www.law.go.kr/LSW/aai/main.do))
3. **"법률 챗봇/법률 Q&A 봇"이라는 일반 개념** — 법령·판례를 근거로 사용자 질의에 답하는 대화형 시스템 일반.

> ⚠️ **혼동 주의**: GitHub `lawbot` 토픽 태그에 달린 공개 저장소는 소수(예: `storminstakk/Justice_Juggernaut`, `Vasumathi2002/LawBot`)에 불과하다. 따라서 토픽 태그보다 **저장소명 검색**이 더 유용하다. (출처: [github.com/topics/lawbot](https://github.com/topics/lawbot))

### 1.2 법률 도메인에서 RAG가 사실상 표준이 된 이유

법률 챗봇이 거의 예외 없이 **RAG(Retrieval-Augmented Generation)** 구조를 택하는 이유는 다음과 같다. (출처: [Harvard JOLT](https://jolt.law.harvard.edu/digest/retrieval-augmented-generation-rag-towards-a-promising-llm-architecture-for-legal-work), [AWS: What is RAG](https://aws.amazon.com/what-is/retrieval-augmented-generation/))

- 법령·판례·규정의 **고품질 DB가 풍부**하다.
- 비싼 재학습 없이 **벡터DB를 자주 갱신**할 수 있다(법령은 수시 개정).
- **외부 진실 근거(grounding)**로 환각률을 낮출 수 있다.
- 답변에 **출처를 인용**할 수 있어 검증 가능성이 높다.

---

## 2. 대표 오픈소스 사례

### 2.1 한국어권 — 동명 프로젝트 두 건

#### (A) `hunsii/LawBot` — "LLM을 활용한 대화형 유사 판례 검색 시스템"

가장 완성도 높은 동명 한국어 오픈소스. (출처: [github.com/hunsii/LawBot](https://github.com/hunsii/LawBot))

| 구성요소 | 내용 |
|---|---|
| 임베딩 | Sentence-BERT |
| 검색 | 코사인 유사도, 약 6만 건 판례(대법원 약 8.5만 건 수집 → 전처리) |
| 생성 | KORANI v1 (13B) LLM으로 요약 |
| 추론 | Polyglot-JAX (TPU) |
| 데모 | Flask 웹 |
| 포지셔닝 | 직접 조언 대신 **판례 근거 제시**로 환각 회피 |

#### (B) `boostcampaitech5 .../LawBot` (네이버 부스트캠프 nlp-08, 2023.08.18 서비스 종료)

3단계 모델 파이프라인. (출처: [github.com/boostcampaitech5/level3_nlp_finalproject-nlp-08](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-08))

1. **질문 필터링 모델** — 법률 질의 여부 판별
2. **유사판례 모델** — 최대 3건 검색 (판례DB 77,382건)
3. **법률 LLM** — `kfkas/legal-llama-2-ko-7b-Chat`(약 25,000 법률 QA로 파인튜닝)

> 스택: React/Tailwind + Express/FastAPI + PyTorch(V100×4) + Docker/AWS.

#### 공통 아키텍처 패턴

> 두 한국어 프로젝트는 **"판례 임베딩 검색(RAG) + 한국어 파인튜닝 LLM 요약"** 구조를 공유하며, 변호사법 리스크를 의식해 **"법률 조언"이 아닌 "유사 판례·가이드라인 제공"**으로 일관되게 포지셔닝한다. ← 본 프로젝트 설계의 핵심 선례.

### 2.2 영어권 — RAG + 벡터DB + LLM + Streamlit 패턴

| 프로젝트 | 스타/라이선스 | 스택 요약 | 대상 |
|---|---|---|---|
| [`lawglance/lawglance`](https://github.com/lawglance/lawglance) | 266★ / Apache-2.0 | ChromaDB + OpenAI + LangChain + Django/Streamlit | 인도 법률 |
| [`suryanshgupta9933/Law-GPT`](https://github.com/suryanshgupta9933/Law-GPT) | 49★ / MIT | Llama-7B-chat + LangChain + vectorstore + Streamlit, PDF/DOCX 인제스트 | 인도 법률 |
| [`itsmesneha/Legal-CHATBOT`](https://github.com/itsmesneha/Legal-CHATBOT) | (LawGPT) | RAG + 임베딩 + 법률문서 코퍼스 | 일반 |
| [`milistu/LegaBot`](https://github.com/milistu/LegaBot) | — | RAG + 법조문 지식베이스 참조 | 일반 |
| [`reethuthota/Legal_Chatbot`](https://github.com/reethuthota/Legal_Chatbot) | — | RAG + 시맨틱 검색 + 벡터DB + LLM | 일반 |

> 거의 모두 **RAG + 벡터DB + LLM + Streamlit/Web UI** 패턴으로 수렴한다. "PDF에 질문하기" 구조가 전형적.

---

## 3. 한국 법률 AI 시장 지형

### 3.1 양강 구도: 로앤컴퍼니 vs 엘박스

```
┌─────────────────────────────────────────────────────────────┐
│  로앤컴퍼니 (LAW&COMPANY)        │  엘박스 (LBox)               │
│  ─────────────────────────       │  ────────────────────────    │
│  · 로톡(LawTalk)  변호사 매칭     │  · 판례검색 (코어)            │
│  · 빅케이스(BigCase) 판례 AI검색  │  · 엘박스 AI (RAG)            │
│  · 슈퍼로이어 변호사 AI어시스턴트 │  · 엘파인드 (매칭)            │
│                                  │  · 스칼라 (법률콘텐츠)        │
└─────────────────────────────────────────────────────────────┘
       양사 공통: 수백만 건 판례 데이터가 핵심 자산
```

### 3.2 주요 서비스 상세

| 서비스 | 출시/운영 | 데이터 규모 | 핵심 기능 | 수익모델 |
|---|---|---|---|---|
| **로톡** | 로앤컴퍼니 | — | 변호사 광고·상담 매칭(전화 15분/영상 20분/방문 30분) | 변호사 광고료 |
| **빅케이스** | 2022.1 | 약 53만 본문 + 약 260만 미리보기 = **약 313만 검색가능** | '서면으로 검색', 'AI 요점보기'(국내 최초) | 회원가입 시 무료 |
| **슈퍼로이어** | 2024.7 | **약 458만 건의 판례**(법령·행정규칙·유권해석·결정례 등은 별도 자료로 함께 활용; 이후 약 490만~530만+로 증가) | 법률리서치·초안작성·문서요약·문서/판례 기반 대화 | 스탠다드 월 99,000원 / 프로 월 198,000원 (변호사 자격자) |
| **엘박스(LBox)** | 2019 창업 | 판례 **약 280만 건** (THE VC는 '500만 건' 표기) | 판례검색·엘박스 AI(RAG)·엘파인드·스칼라 | 구독·B2G |

출처: [전자신문(빅케이스)](https://www.etnews.com/20220125000057), [스타트업투데이](https://www.startuptoday.kr/news/articleView.html?idxno=44021), [AI타임스(슈퍼로이어)](https://www.aitimes.com/news/articleView.html?idxno=161158), [법률신문(엘박스)](https://www.lawtimes.co.kr/news/191116), [THE VC](https://thevc.kr/lbox)

> ⚠️ **판례 보유 건수는 단순 비교 금지**: 사업자마다 산정 기준(본문 vs 미리보기, 상·하급심 포함 여부)이 달라(예: 빅케이스 53만 본문/313만 검색가능, 엘박스 280만~500만 표기 편차) 정의를 명확히 한 뒤 인용해야 한다.

### 3.3 투자·확장

- **엘박스**: 2022.12 SV인베스트먼트·KB인베스트먼트·산업은행 등 180억 + 2023.2 삼성벤처투자 20억 = **시리즈 B 200억원**. 2022년 매출 약 9억원(8.8억)·영업손실 약 25.2억원[검증정정: 79억은 미검증]. 현재 시리즈 C·B2G 계약 35건·IPO 준비(THE VC). (출처: [법률신문](https://www.lawtimes.co.kr/news/191116), [THE VC](https://thevc.kr/lbox))
- **B2G 사례**: 전국 약 13만 경찰이 엘박스 판례 검색을 무료로 이용. 공공기관 계약이 주요 채널 중 하나. (출처: [데이터넷](https://www.datanet.co.kr/news/articleView.html?idxno=182489))

### 3.4 인접·해외 서비스

로폼([lawform.io](https://www.lawform.io/en), 문서자동작성·전자서명), 팔로([follaw.co.kr](https://follaw.co.kr)), 알법([albup.co.kr](https://albup.co.kr), 변호사 매칭), 대한변협 '나의 변호사'(klaw.or.kr), 해외 Lexis+ AI 한국 진출 등.

> ℹ️ **"유스로우"** 라는 명칭의 서비스는 한국어·영어 검색에서 독립적으로 확인되지 않았다. **[추정]** 슈퍼로이어 또는 알법/팔로/로폼 등과 혼동되었거나 소규모·비활성 서비스일 가능성. 보고서 인용 전 정확한 사명/도메인 재확인 필요.

### 3.5 공개 API 현황

> 조사 범위에서 로톡·빅케이스·슈퍼로이어·엘박스 모두 **외부 개발자용 공개 REST API를 정식 상품으로 명시하지 않았다.** **[추정]** B2B 통합은 개별 계약/SDK 형태로 보이며, 문서화된 셀프서비스 API는 확인 불가. 반면 본 코퍼스의 원천 `github.com/legalize-kr`와 cli-tools는 **CLI·MCP 도구**를 표방해 에이전트 친화적 접근을 제공한다.

---

## 4. 데이터 출처 — `github.com/legalize-kr`

### 4.1 조직 개요

`legalize-kr`는 대한민국의 법령·판례·행정규칙·자치법규를 Git 저장소로 관리하는 오픈소스 조직이다. **핵심 슬로건: "모든 법령은 Markdown 파일, 모든 개정은 Git 커밋"**(공포일/선고일 = 커밋일). 법령 포털이 '스냅샷'만 제공하는 한계를 넘어 법령의 시간적 **변천(개정 이력)**을 추적하는 것이 차별점이다. (출처: [legalize.kr](https://legalize.kr), [.github profile README](https://github.com/legalize-kr))

### 4.2 저장소 구성 (2026-06-04 기준)

| 저장소 | 역할 | 스타(약) | 라이선스 |
|---|---|---|---|
| `legalize-kr` | 법령(법률·대통령령·부령) | 1,392★ | MIT(구조)/공공저작물(본문) |
| `precedent-kr` | **판례** (본 코퍼스 원천) | 101★ | MIT/공공저작물 |
| `admrule-kr` | **행정규칙** (본 코퍼스 원천) | 8★ | MIT/공공저작물 |
| `ordinance-kr` | **자치법규** (17개 시도 표방) | 7★ | MIT/공공저작물 |
| `legalize-pipeline` | Python 수집 ETL | 23★ | Apache-2.0/MIT |
| `compiler` | Rust, `.cache → bare Git` | 54★ | Apache-2.0/MIT |
| `cli-tools` | CLI·MCP 조회 도구 | — | MIT |
| `agent-skills` | Claude/Cursor 등 에이전트 플러그인 | — | — |
| `legalize-web` / `.github` | 홈페이지 / 조직 프로필 | — | — |

출처: [legalize-kr 조직 저장소 목록](https://github.com/orgs/legalize-kr/repositories)

### 4.3 데이터 흐름

```
국가법령정보센터 law.go.kr OpenAPI (XML, LAW_OC 키)
        │
        ▼
legalize-pipeline (Python: 수집·Markdown 변환·검증)
        │  → .cache/
        ▼
compiler (Rust 4개 바이너리: legalize / precedent / admrule / ordinance)
        │  → bare Git 저장소 직접 생성
        ▼
본 코퍼스 D:\법파일\*.zip (Markdown+YAML+meta.json 산출물)
```

### 4.4 라이선스 이원화 (재배포 시 필수 준수)

- **본문 텍스트**(법령·판례·행정규칙·자치법규): **공공저작물 = 대한민국 정부저작물, 저작권법 §7** → 영리 포함 자유 이용.
- **저장소 구조·메타데이터**: **MIT**.
- **도구류**: pipeline·compiler = Apache-2.0+MIT 듀얼, cli-tools = MIT.

### 4.5 AI/RAG 연계 인프라

조직은 별도의 완성형 "lawbot"이나 RAG 챗봇 **서비스를 운영하지 않는다.** 대신 RAG/에이전트 소비를 1차 목표로 하는 **인프라형 프로젝트**다.

- **MCP 서버** (약 9~11개 도구: `laws_list`, `laws_get`, `laws_article`, `search`, `precedents_*`, `admrules_*`, `ordinances_*` 등) — Claude Desktop/Cursor 등 MCP 호스트에서 자연어 소비.
- **CLI** (`legalize-cli`) — GitHub REST API 기반, 인증 없이 조회, `--json`(schema_version 1.0) 구조화 출력.
- **agent-skills** — 모델 비종속 플러그인. '법률 자문 도구 아님' 면책 명시.
- **`legalize.kr/llms.txt`** — LLM 컨텍스트.

### 4.6 주의사항 (조직이 직접 경고)

> ⚠️ **Git 이력 불안정성**: 파서 개선 시 force-push로 커밋이 재작성되므로 **커밋 해시는 장기 식별자로 불안정**하다. 영구 참조에는 **법령ID(`alr_bdt_id`)·사건번호·판례일련번호·선고일·`law.go.kr` URL** 같은 원천 식별자를 사용하라. (← 본 코퍼스가 커밋 해시가 아닌 ID/메타 기반으로 청크를 관리하는 이유와 부합)

> ⚠️ **자치법규 출처 불일치**: 본 코퍼스 과제 설명서는 자치법규 출처를 **ELIS(elis.go.kr)**로 명시하나, 현행 `ordinance-kr` README는 "**law.go.kr OpenAPI**에서 가져온다"고 밝혀 충돌한다. **[추정]** 코퍼스 빌드 시점과 현행 README 간 출처 정책 변경이 있었거나, 코퍼스 메타(`source_url=elis.go.kr`)가 빌드 단계에서 부가된 것으로 보인다. 사용자 대면 출처 문구 확정 전 양쪽 검증 필요.

### 4.7 본 코퍼스의 정확 통계 (`corpus_stats.txt` 직접 검증)

| ZIP | 총 항목수 | 법령(meta.json) | 현행 조문(.txt) | 판례(.md) | 압축 크기 |
|---|---:|---:|---:|---:|---:|
| cases-판례.zip | 123,769 | — | — | **123,743** | 485.3 MB |
| statutes-광주광역시.zip | 1,979,293 | 5,270 | 272,687 | — | 1,267.7 MB |
| statutes-세종특별자치시.zip | 407,233 | 1,357 | 72,841 | — | 275 MB |
| statutes-전라남도.zip | 5,737,068 | 17,543 | 953,907 | — | 3,554 MB |
| statutes-전북특별자치도.zip | 4,296,575 | 12,128 | 678,513 | — | 2,802.9 MB |
| statutes-행정규칙.zip | 1,395,058 | 21,052 | 1,309,883 | — | 927.4 MB |
| **합계** | **약 13,938,996 (약 1,394만)** | **36,298 (자치)+21,052 (행정)** | **3,287,831 (현행 조문)** | **123,743** | 약 9.3 GB |

- **판례 법원등급**: 대법원 68,175 / 하급심 55,566 / 미분류 1
- **판례 사건종류**: 일반행정 45,104 · 민사 42,132 · 형사 21,667 · 세무 10,057 · 특허 3,373 · 가사 1,390 · 기타 11 · 선거·특별 8
- **행정규칙 종류**: 고시 10,262 · 훈령 6,387 · 예규 3,866 · 공고 208 · 지침 167 · 국무총리훈령 100 · 대통령훈령 52 · 기타 10

> 📌 **결정적 포인트**: 코퍼스 본 판례 123,743건과 `precedent-kr` 커밋 수(약 123,744)가 일치 → **동일 원천 확인**. `admrule-kr`의 meta `license_source: "admrule-kr"`, `parser_version: "markdown_admrule/0.1.0"`도 저장소 기원을 직접 확인해 준다.

> ⚠️ **"총 항목수 ≫ 조문수"의 이유**: 자치법규 zip은 **과거 버전 스냅샷 디렉토리**(`versions/`)를 포함해 총 항목이 현행 조문의 5~8배로 부풀려진다(예: 세종 407,233 vs 현행 72,841 = 약 5.6배). RAG 인덱싱 대상은 **`current.articles/*.txt`**에 한정해야 한다. (행정규칙은 `version_count=1`로 과거버전이 없어 총항목 ≈ 현행 조문.)

> ⚠️ **데이터 커버리지 한계 (제품 설계 시 정직하게 고지)**: 본 코퍼스는 (1) **국가 1차 법령(법률·시행령·시행규칙: 민법·형법 등) 본문이 0건**, (2) 자치법규가 **17개 시도 중 4곳(광주·세종·전남·전북)뿐**(서울·경기·부산 등 결여), (3) 헌재 결정례·법령해석례·행정심판 재결례 0건이다. 따라서 일반 사용자가 'lawbot'에 기대하는 민법·형법 질의에 답할 수 없다. 서비스 범위를 **"자치법규·행정규칙·판례 검색"**으로 정직하게 한정하는 것이 안전하다.

---

## 5. 법률 RAG 연구 동향

### 5.1 환각은 RAG로도 사라지지 않는다 (실증)

Stanford RegLab/HAI의 *Hallucination-Free?* 연구(2024 preprint → *Journal of Empirical Legal Studies* 2025)는 **RAG 기반 상용 법률 도구조차 환각을 못 막음**을 실증했다. (출처: [Stanford HAI](https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries), [Legal_RAG_Hallucinations.pdf](https://law.stanford.edu/wp-content/uploads/2024/05/Legal_RAG_Hallucinations.pdf))

| 도구 | 정확도 | 환각률 |
|---|---:|---:|
| Lexis+ AI | 65% | ≥17% |
| Westlaw AI-Assisted Research | 42% | ~33% |
| GPT-4 (참고) | — | ~43% |
| Ask Practical Law | — | 60%+ 응답 거부 |

> **함의**: RAG는 환각을 **줄이되 제거하지 못한다.** "RAG=환각 해결"로 전제하지 말고, 인용검증·trust_grade 필터를 명시적 KPI로 추적하라.

### 5.2 검색 정밀도가 핵심 — LegalBench-RAG

[LegalBench-RAG](https://arxiv.org/abs/2408.10343)(arXiv 2408.10343)는 6,858개 전문가 주석 질의-답변 쌍/79M자 코퍼스로 **검색 단계**를 평가한다. 큰 청크는 컨텍스트 초과·비용·환각을 유발하므로, **최소 단위의 관련 텍스트 정밀 추출**이 인용 생성과 환각 억제의 전제임을 강조한다.

### 5.3 한국어 핵심 벤치마크

| 벤치마크 | 규모 | 특징 |
|---|---|---|
| **KBL** ([arXiv 2410.08731](https://arxiv.org/html/2410.08731v1), EMNLP 2024 Findings) | 지식 7태스크(510) + 추론 4태스크(288) + 변호사시험 4영역(2,510) | RAG 평가용 11k 법령·조례(52M토큰) + 15만 판례(320M토큰) 코퍼스. **본 코퍼스와 구조·규모 매우 유사** |
| **KCL** ([arXiv 2512.24572](https://arxiv.org/html/2512.24572)) | 변호사시험 기반 | 문항별 근거 판례 제공, MCQA + Essay(rubric LLM-as-Judge) |
| **KMMLU-Pro** ([arXiv 2507.08924](https://arxiv.org/html/2507.08924)) | 전문 한국어 평가(법률 포함) | — |

> ⚠️ **KBL의 두 가지 경고**:
> 1. RAG 효과가 **비일관적** — GPT-4는 법지식 +2.4~3.3%, 변호사시험은 형사 +7.4%/공법 -2.5% 혼재. Claude 계열은 미미·음(-)의 효과.
> 2. 전문가 검증 없는 공개 법률 데이터에 **최대 21% 오류**가 섞일 수 있다 → `trust_grade` 필터·인용검증 필수.

### 5.4 한국어 검색 임베딩 — KURE-v1

[`nlpai-lab/KURE-v1`](https://github.com/nlpai-lab/KURE)(고려대 NLP&AI연구실, MIT)은 `bge-m3`를 한국어 질의-문서 약 200만 쌍(하드네거티브)으로 파인튜닝. MTEB-ko-retrieval에서 `multilingual-e5-large`·`bge-m3` 대비 Recall/NDCG SOTA(NDCG@10 0.6947 vs bge-m3 0.6872 등), **법률 도메인 데이터 포함**, 8192토큰, 1024차원. → 본 코퍼스 RAG 임베딩 후보로 적합.

### 5.5 청킹 신뢰성 — SAC(Summary-Augmented Chunking)

[SAC](https://arxiv.org/html/2510.06999v1)(arXiv 2510.06999)는 법률 문서의 **보일러플레이트 유사성**이 Document-Level Retrieval Mismatch(DRM, NDA에서 95%+)를 유발함을 보이고, 문서 단위 약 150자 요약을 각 청크에 프리펜딩하는 SAC로 DRM을 약 절반 감소시킨다. 청크 500자/요약 150자가 균형 최적. **본 코퍼스에서는 `meta.json`의 `ord_name`/`ord_kind`/조문제목을 청크 앞에 프리펜딩**하면 구현이 쉽다.

### 5.6 인용 보장 — correctness ≠ faithfulness

[*Correctness is not Faithfulness in RAG Attributions*](https://arxiv.org/abs/2412.18004)(arXiv 2412.18004, SIGIR ICTIR 2025)는 인용이 문장을 **지지하는지**(correctness)와 모델이 실제로 그 문서에 **인과적으로 의존했는지**(faithfulness)를 구분 — 최대 **57%의 인용이 사후합리화**된 가짜 근거임을 보고. 법률 인용 보장 설계 시 핵심 함의.

### 5.7 한국 법률 RAG 인용검증 실무 레퍼런스

[`chrisryugj/korean-law-mcp`](https://github.com/chrisryugj/korean-law-mcp)은 법제처 42개 API를 17개 MCP 도구로 압축. `verify_citations`는 조문 인용 정규식 추출 + 30자 lookback 법령명 역추적 + 법제처 DB 병렬 교차검증으로 **가짜 인용(예: 형법 제9999조)을 탐지**한다. `impact_map`(역방향 인용 그래프), `time_travel`(시점 diff) 제공. → 환각 억제·citation 보장 실무 레퍼런스.

### 5.8 그 외 한국어 법률 NLP 생태계

`yeontaek/Korea-Law-LLM`(Polyglot-ko-12.8B + 36,650 법률 instruction), `maj34/Legal_Specific_KoLLM`, 영어권 SaulLM 패밀리(Equall, NeurIPS 2024) 등.

---

## 6. 법률 챗봇 API 설계 관행

### 6.1 두 부류로 나뉜 시장

1. **엔터프라이즈 법률 AI** (Harvey, LexisNexis Protégé, Thomson Reuters CoCounsel) — 공개 셀프서비스 API가 아닌 **영업 계약 기반**. Bearer 토큰/OAuth 2.0, RBAC, 감사 로그, SSO/SCIM, 지역별 데이터 레지던시 표준.
2. **인용/출처 반환 표준** — Anthropic Citations API와 Cohere RAG citations가 사실상 표준.

### 6.2 인용 반환 스키마 — 사실상 표준 (차용 권장)

**Anthropic Citations API** (출처: [platform.claude.com/docs](https://platform.claude.com/docs/en/build-with-claude/citations))

```jsonc
{
  "citations": [
    {
      "cited_text": "인용 원문",
      "document_index": 0,          // 0-인덱스
      "document_title": "법령명/사건명",
      "type": "char_location",       // 평문: char_location, PDF: page_location, 커스텀: content_block_location
      "start_char_index": 120,       // 0-인덱스, 끝 배타적
      "end_char_index": 340
    }
  ]
}
```

**Cohere RAG citations** (출처: [docs.cohere.com](https://docs.cohere.com/docs/rag-citations))도 거의 동일: `message.citations[]`에 `start`/`end`/`text`/`sources[]{type, id, title, snippet}`. 문서 id는 `doc:0, doc:1…` 자동 생성(커스텀 가능).

### 6.3 스트리밍 설계

- SSE 스트리밍에서 인용은 **별도 `citation-start` 이벤트**로 전송.
- Cohere는 **`fast`**(생성 중 인라인 인용, 지연 낮음·정밀도 약간 손해) vs **`accurate`**(응답 완성 후 인용, 인덱스 정렬 정확, 기본값) 두 모드 제공.
- → **법률 답변은 정밀도 우선이므로 `accurate`를 기본값 권장.**

### 6.4 데이터 원천 API — 국가법령정보 공동활용 OpenAPI

| 구분 | 내용 |
|---|---|
| 검색 | `http://www.law.go.kr/DRF/lawSearch.do` |
| 본문 | `http://www.law.go.kr/DRF/lawService.do` |
| 판례 본문 예 | `lawService.do?target=prec&OC={인증키}&ID={판례일련번호}&type={HTML\|XML\|JSON}` |
| 인증 | `OC`(신청 시 발급 인증키) 쿼리 파라미터 |
| 타입 | 기본 XML, JSON/HTML 선택 (단 **국세청 판례는 HTML만**) |
| 트래픽 | 개발계정 일 10,000건, 운영계정은 활용사례 등록 시 증액 |

출처: [open.law.go.kr precInfoGuide](https://open.law.go.kr/LSO/openApi/guideResult.do?htmlName=precInfoGuide)

> 📌 판례 API 반환 필드(사건명·사건번호·선고일자·법원명·판시사항·판결요지·참조조문·참조판례·판례내용)는 **본 코퍼스 YAML 프론트매터 구조와 정확히 일치**한다.

### 6.5 본 코퍼스 기반 API 설계 권장안

```
POST /v1/answers                 # 질의 → 인용 포함 답변 (SSE 지원)
GET  /v1/precedents/{precSeq}    # 판례 단건
GET  /v1/statutes/{lawId}/articles/{articleNo}
GET  /v1/search                  # 통합 검색 (region/ord_kind/effective_from 필터)
GET  /healthz
```

인용 객체 매핑 권장:

```jsonc
{
  "citations": [{
    "cited_text": "...",
    "source_id": "ADMRULE:54908",          // alr_bdt_id 또는 판례일련번호(precSeq)
    "document_title": "○○ 규정",            // ord_name / 사건명
    "location": "제4조",                    // 조문번호 또는 char 범위
    "effective_from": "2023-08-10",         // 시점 정합
    "trust_grade": "B",                     // 사용자에게 신뢰도 노출
    "source_url": "https://www.law.go.kr/..." // meta.json source_url / 프론트매터 출처URL
  }]
}
```

### 6.6 인증/요금제 단계화

| 단계 | 인증 | 레이트리밋 |
|---|---|---|
| 공개 read-only (공공저작물) | 비인증 | 낮은 한도 (참고: 국가법령 OpenAPI 일 10,000건) |
| API 키 발급 | API key | 상향 |
| AI 답변 (LLM 비용 발생) | Bearer 토큰 | 토큰 종량제 또는 좌석제 |
| 엔터프라이즈 | OAuth 2.0 + RBAC + 감사 로그 + SSO | 협의 |

### 6.7 한국 사례 — 엘박스(LBox)

RAG 기반으로 판례와 자체 콘텐츠 플랫폼 **'스칼라'를 인용**해 답변하는 대표 사례. (출처: [lbox.kr/ai](https://lbox.kr/ai), [ai-landing.lbox.kr](https://ai-landing.lbox.kr/blog/law-inquery))

---

## 7. 규제·윤리

### 7.1 변호사법 — 최대 리스크

**변호사법 제109조 제1호**: 변호사가 아니면서 금품·이익을 받거나 약속하고 법률사건/법률사무(감정·대리·중재·화해·청탁·**법률상담·법률관계 문서 작성** 등)를 취급·알선하면 **7년 이하 징역 또는 5천만원 이하 벌금**. 법률 AI 챗봇도 적용 대상이라는 데 법조계가 대체로 동의한다. (출처: [국가법령정보센터 §109](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EB%B3%80%ED%98%B8%EC%82%AC%EB%B2%95/%EC%A0%9C109%EC%A1%B0))

**적법/위법 분기점** (출처: [법률신문](https://www.lawtimes.co.kr/news/articleView.html?idxno=197014)):

| 구분 | 위법 소지 |
|---|---|
| 무료 + 일반 법리 설명·유사 판례 검색·정보 제공 | 낮음 |
| 유료 + 구체적 사건 답변(법률상담·대리·문서작성) | **높음** |
| 표준 서식 공란 채우기형 자동화(로폼) | **적법** (대법원 2026.2.12, 2025두35483 심리불속행 기각) |
| 제휴 변호사 검토·직인 '사건 검토서비스' / 생성형 AI의 개별 사실관계 문서작성 | **위반 소지** |

> **로폼 판결의 경계선**: "표준 서식 자동화"와 "개별 사실관계 기반 법률판단" 사이에 적법 경계가 그어진다. (출처: [법률신문 2025두35483](https://www.lawtimes.co.kr/news/articleView.html?idxno=218044))

### 7.2 로톡 사태 — 광고형 플랫폼의 적법성 확립

| 일자 | 결정 |
|---|---|
| 2015·2017 | 검찰 변호사법 위반 고발 모두 **무혐의** |
| 2021.8 | 법무부 "로톡 서비스 변호사법 위반 아님" 공식 확인 |
| 2021.11.29 | 공정위, 변협의 가입변호사 징계를 사업자단체 금지행위로 판단 |
| 2022.5.26 | **헌재(2021헌마619)**: 변협 광고규정 일부(유권해석 반하는 광고·소개행위 금지) **위헌** |
| 2023.9.26 | **법무부**, 변협이 징계한 로톡 변호사 123명 **징계취소**(120명 혐의없음, 3명 불문경고) |

출처: [KISO 저널](https://journal.kiso.or.kr/?p=12455), [한국일보](https://www.hankookilbo.com/News/Read/A2022051116500002666)

> ⚠️ **[추정]** 로톡 판단은 '변호사 광고·연결 플랫폼'에 대한 것이라, **챗봇이 직접 답변을 생성하는 모델에 그대로 적용된다고 단정하기 어렵다.** 챗봇이 스스로 법률사무를 '취급'할수록 §109 리스크가 별도 검토되어야 한다.

### 7.3 데이터 라이선스 — 저작권법 §7

저작권법 제7조는 다음을 **'보호받지 못하는 저작물'**(공공저작물)로 규정 → 자유 이용 가능. (출처: [CaseNote §7](https://casenote.kr/%EB%B2%95%EB%A0%B9/%EC%A0%80%EC%9E%91%EA%B6%8C%EB%B2%95/%EC%A0%9C7%EC%A1%B0))

1. 헌법·법률·조약·명령·조례·규칙
2. 국가·지자체 고시·공고·훈령 등
3. 법원 판결·결정·명령, 행정심판 의결·결정 등
4. 위 편집물·번역물(국가·지자체)
5. 사실 전달에 불과한 시사보도

> 다만 법령정보센터는 "**법적 효력 있는 공식본이 아니며 정확성을 보증하지 않음**"을 안내 → 서비스 측에 **정확성·최신성 책임**이 남는다. 출처표시(공공누리 KOGL 제1유형 참고)는 신뢰성·투명성 차원에서 권장된다.

### 7.4 AI 기본법 (2026.1.22 시행)

생성형 AI 기반 서비스의 3대 의무. (출처: [help-me.kr](https://www.help-me.kr/blog/article/korea-ai-act-2026-compliance-guide/), [국가법령정보센터 lsiSeq=268543](https://www.law.go.kr/lsInfoP.do?lsiSeq=268543))

1. **AI 기반 운용 사실 사전 고지** (미이행 시 과태료 3천만원 이하)
2. **결과물에 AI 생성 표시**(가시적/비가시적)
3. **딥페이크 표시**

> **고영향(고위험) AI** 해당 시: 위험관리방안 수립, 설명요구권 보장, 사람의 관리·감독, 이용자 보호조치, 관련 문서 **5년 보관**. 법률 분야는 기본권 영향이 커 고영향 해당 여부 사전 검토 필요. 국외 사업자도 적용.

### 7.5 개인정보보호법

챗봇 입력에는 사건 정황·건강·사상 등 **민감정보(법 §23)**가 포함될 수 있어 **별도 명시 동의** 필요. 이루다 사건 및 후속 판결에서 "수집 목적을 벗어난 대화데이터의 AI 학습 이용은 실질적 동의 없는 위법 처리"로 판단. (출처: [대륜](https://www.daeryunlaw.com/trend/10305), [개인정보위 생성형AI 안내서 2025.8](https://www.privacy.go.kr/front/bbs/bbsView.do?bbsNo=BBSMSTR_000000000049&bbscttNo=20731))

---

## 8. 본 코퍼스 → lawbot 적용 권장사항

### 8.1 아키텍처 정합성

> 본 코퍼스(판례 12.4만 건 + 자치법규/행정규칙 조문 단위 사전분할)는 `hunsii/LawBot`·boostcamp nlp-08과 **동일한 "RAG 기반 유사 판례/조문 검색 + 한국어 LLM 요약" 아키텍처에 그대로 투입 가능**하다. `current.articles/*.txt`가 이미 조문 단위로 분할돼 있어 별도 청킹 없이 임베딩 인덱싱에 이상적이다.

### 8.2 권장 RAG 스택

| 계층 | 권장 | 근거 |
|---|---|---|
| 임베딩 | **KURE-v1**(한국어 법률 포함 MTEB-ko 1위, MIT) 또는 BGE-M3 | §5.4 |
| 검색 | **하이브리드**(dense + 형태소 BM25, nori/kiwi) → RRF → 크로스인코더 리랭킹 | §5.5, 한국어 법률에서 BM25 보완 필수 |
| 리랭커 | `dragonkue/bge-reranker-v2-m3-ko`(자체호스팅) 또는 Cohere Rerank 3.5 | — |
| 벡터DB | **Qdrant**(필터·하이브리드 강점) 또는 **pgvector**(SQL 메타필터·단일DB) | 약 340만 현행 청크 규모 적합 |
| 청킹 | 자치법규/행정규칙 = `current.articles` 1조문=1청크(과거버전 제외), 판례 = `##` 섹션 분할 + SAC | §5.5 |
| 인용 | Anthropic/Cohere 스키마 차용, `alr_bdt_id`/`precSeq`/`source_url` 매핑 | §6.2 |

### 8.3 필수 가드레일 체크리스트

- [ ] 인덱싱은 **`current.articles/*.txt`에 한정**, 과거버전 스냅샷 제외(중복·구버전 오염 방지)
- [ ] 행정규칙 **조문 과편화**(95.8% 법령에서 동일 조문번호 분할) → `(alr_bdt_id + 조문번호)` 기준 **재병합**
- [ ] 메타데이터 필터(`region`/`ord_kind`/`effective_from`/`court_level`/`사건종류`/`trust_grade`)를 벡터검색에 결합
- [ ] 시점 정합: `effective_from` + `histories[]`로 **현행 조문만** 인덱싱, 시점 질의는 별도 보조 인덱스
- [ ] 생성 후 **인용검증**(`verify_citations`식 조문번호 정규식 추출 → ID 대조)으로 존재하지 않는 조문/판례 차단
- [ ] **모든 답변에 면책고지** 시스템 레벨 자동 삽입: *"본 서비스는 법률정보 제공 도구이며 변호사의 법률자문이 아닙니다. 답변은 부정확하거나 최신 법령·판례를 반영하지 못할 수 있으니 실제 사안은 반드시 변호사와 상담하십시오."*
- [ ] **AI 생성 표시** + **AI 기반 사전 고지**(AI기본법 2026.1.22)
- [ ] **민감정보 별도 동의** + 입력 데이터 모델 학습 비활용 기본값(개인정보보호법)
- [ ] 출처표시: 공공저작물(저작권법 §7) + 원천(`law.go.kr`/ELIS, `github.com/legalize-kr`) + MIT(구조/메타) 고지
- [ ] 식별자는 **커밋 해시 금지**, `alr_bdt_id`·`precSeq`·`사건번호`·선고일 사용(force-push 대응)
- [ ] **데이터 커버리지 한계 고지**: 국가 1차 법령 부재, 4개 시도 자치법규만 → 답변 범위 라우팅("상위 법률 미수록" 경고)
- [ ] `trust_grade` A/B 노출, B는 별도 표기. 폐지·개정 조문 인용 시 경고

### 8.4 규제 안전 포지셔닝 요약

```
✅ "법령·판례 정보 제공 / 유사 판례 안내 / 조문 검색"   ← 안전
✅ 무료 또는 변호사 검수 결합형 유료
✅ 출처 인용 + 인용검증 + 면책고지 상시
─────────────────────────────────────────────
⚠️  "특정 사건 결론·전략·소송 대리·문서 자동작성"      ← 변호사법 §109 위험
⚠️  유료 + 비변호사 자동응답으로 구체적 사건 상담
⚠️  무료 챗봇을 미끼로 한 변호사 영업 유도(광고규제)
```

---

## 9. 참고문헌 (주요 출처)

**오픈소스 사례**
- [hunsii/LawBot](https://github.com/hunsii/LawBot) — 한국어 RAG 유사판례 검색
- [boostcampaitech5 nlp-08 LawBot](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-08)
- [lawglance/lawglance](https://github.com/lawglance/lawglance), [suryanshgupta9933/Law-GPT](https://github.com/suryanshgupta9933/Law-GPT)
- [GitHub Topics: lawbot](https://github.com/topics/lawbot)

**데이터 원천**
- [legalize-kr 조직](https://github.com/legalize-kr) · [precedent-kr](https://github.com/legalize-kr/precedent-kr) · [admrule-kr](https://github.com/legalize-kr/admrule-kr) · [ordinance-kr](https://github.com/legalize-kr/ordinance-kr) · [cli-tools](https://github.com/legalize-kr/cli-tools) · [legalize-pipeline](https://github.com/legalize-kr/legalize-pipeline)
- [국가법령정보 공동활용 OpenAPI](https://open.law.go.kr/LSO/openApi/guideList.do) · [법제처 Lawbot(정부)](https://www.law.go.kr/LSW/aai/main.do)

**한국 시장**
- [로톡](https://www.lawtalk.co.kr/) · [빅케이스](https://bigcase.ai/) · [슈퍼로이어](https://superlawyer.co.kr/) · [엘박스](https://lbox.kr/v2)
- [전자신문(빅케이스)](https://www.etnews.com/20220125000057) · [AI타임스(슈퍼로이어)](https://www.aitimes.com/news/articleView.html?idxno=161158) · [법률신문(엘박스)](https://www.lawtimes.co.kr/news/191116)

**연구**
- [Stanford HAI: AI on Trial](https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries) · [Legal_RAG_Hallucinations PDF](https://law.stanford.edu/wp-content/uploads/2024/05/Legal_RAG_Hallucinations.pdf)
- [LegalBench-RAG (2408.10343)](https://arxiv.org/abs/2408.10343) · [KBL (2410.08731)](https://arxiv.org/html/2410.08731v1) · [SAC (2510.06999)](https://arxiv.org/html/2510.06999v1) · [Correctness≠Faithfulness (2412.18004)](https://arxiv.org/abs/2412.18004)
- [KURE (한국어 임베딩)](https://github.com/nlpai-lab/KURE) · [korean-law-mcp](https://github.com/chrisryugj/korean-law-mcp)

**API 설계**
- [Anthropic Citations API](https://platform.claude.com/docs/en/build-with-claude/citations) · [Cohere RAG Citations](https://docs.cohere.com/docs/rag-citations) · [Harvey API](https://developers.harvey.ai/api-reference/authentication) · [LexisNexis Protégé API](https://www.lexisnexis.com/en-us/products/lexis-api.page)

**규제**
- [변호사법 §109](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EB%B3%80%ED%98%B8%EC%82%AC%EB%B2%95/%EC%A0%9C109%EC%A1%B0) · [저작권법 §7](https://casenote.kr/%EB%B2%95%EB%A0%B9/%EC%A0%80%EC%9E%91%EA%B6%8C%EB%B2%95/%EC%A0%9C7%EC%A1%B0)
- [KISO 저널(로톡 징계취소)](https://journal.kiso.or.kr/?p=12455) · [법률신문(로폼 2025두35483)](https://www.lawtimes.co.kr/news/articleView.html?idxno=218044) · [AI기본법](https://www.law.go.kr/lsInfoP.do?lsiSeq=268543) · [개인정보위 생성형AI 안내서](https://www.privacy.go.kr/front/bbs/bbsView.do?bbsNo=BBSMSTR_000000000049&bbscttNo=20731)

---

*본 문서는 멀티에이전트 리서치 결과(phase1 중심)와 로컬 코퍼스 통계(`D:/법파일/_samples/corpus_stats.txt` 직접 검증)를 종합해 작성되었다. `[추정]`으로 표기된 항목과 "유스로우" 등 미확인 사항은 인용 전 재검증이 필요하다.*

