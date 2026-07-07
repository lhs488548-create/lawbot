# 법제처 Lawbot · law.go.kr OPEN API 조사 + 우리 lawbot 차용안

작성 2026-06-29 (딥리서치 102에이전트, 적대적 검증). 출처는 각 항목에 표기.

## 1. 법제처 "Lawbot" (실재)
- URL: `law.go.kr/LSW/aai/main.do` — **"지능형 법령검색 시스템"**, 일상 자연어 질의로 44개 법률분야 검색. "법제처 Lawbot"으로 브랜딩.
- OPEN API 노출: **검색 API** + **연관법령 API** 2종(open.law.go.kr 가이드).
- ⚠️ 내부 검색방식(키워드/임베딩/하이브리드/LLM)은 **공개 문서로 확인 불가**. "AI/지능형"은 부분적 자기 브랜딩.

## 2. law.go.kr 국가법령정보 공동활용 OPEN API (가장 실용적 자산)
- **OC 인증** 필수(이메일 등록으로 발급, open.law.go.kr 신청). data.go.kr 게이트웨이는 별도 `serviceKey` 인증.
- **총 191개 엔드포인트**, 대부분 `목록조회(lawSearch.do)` + `본문조회(lawService.do)` 이원 구조 → "목록 검색 후 일련번호로 본문 인출".
- 핵심 엔드포인트(우리에게 바로 유용):
  - **현행법령 조항호목 본문**: `lawService.do?target=lawjosub` (OC, type=HTML/XML/JSON) — **조/항/호/목 단위**. 우리 청크 단위와 직접 정렬, 인용 본문 대조에 이상적.
  - **판례 목록**: `lawSearch.do?target=prec&search=1|2` (1=사건명, 2=본문검색).
  - **판례 본문**: `lawService.do?target=prec&ID=<판례일련번호>` → **사건명·사건번호·선고일자·법원명·판시사항·판결요지·참조조문·참조판례·전문**. (HuggingFace `joonhok-exo-ai/korean_law_open_data_precedents` ~85,830건이 동일 구조 미러링)
  - 그 외: 법령해석례, 헌재결정례, 행정심판례, 영문법령, 자치법규(`target=ordin`), 별표·서식, 법령용어, 조약, 위원회 결정문.
- (우리 verify.py는 이미 `config.LAW_OC`로 law.go.kr를 호출 중 — 이 엔드포인트들로 확장 가능.)

## 3. 우리 lawbot 차용안 (우선순위)
1. **[현행]/[시행예정] 시행일 라벨링** (LexDiff식 앵커링) — 미래시행 조문(코퍼스 12.3%, effective_from이 개정일)이 현행처럼 보이지 않게 인용에 시행일 상태 표기. 우리 #1(하드필터 되돌림) 보완책으로 **정확히 부합**. → **즉시 구현 대상.**
2. **판례 case_no를 law.go.kr API로 검증** — 우리 parents.jsonl엔 사건번호 필드가 없음(감사 HIGH). `target=prec` 본문 API로 사건번호 존재·일치를 확인 → Citation Firewall 판례 게이트 복구. (LAW_OC 필요)
3. **인용 본문 대조(content matching)** — 존재여부만이 아니라 생성답변의 인용문과 원문 본문을 LCS(≥30자)·문자 bigram Jaccard(≥0.25)·임베딩으로 대조(LexDiff `citation-content-matcher`). 우리 verify에 wording 강화.
4. **인용 환각 5범주 분리검증**(Princeton 'Who Checks the Citations?'): 비존재·사건명불일치·핀사이트오류·축자오인용·내용왜곡 — 범주별 다른 검증. 단일 이진판정 X.
5. **Citation Grounding 메트릭**: precision(존재)·relevance(관련)·temporality(시간유효) 3분해 — 우리 골든셋 평가에 추가.

## 4. 현실 경계(정직)
- 어떤 RAG도 100% 인용정확도 불가. **상용 법률AI 인용 환각 17~33%**. 출처가 실재해도 틀린 답을 뒷받침할 수 있음 → **답변 정확성의 독립 검증 단계 필수**(우리 Citation Firewall + as_of + 골든셋이 그 역할).

## 출처(대표)
- open.law.go.kr/LSO/openApi/guideList.do, guideResult(lsNwJoListGuide·precListGuide·precInfoGuide)
- data.go.kr 15000115(공유서비스)·15058878(조항호목)·15059269/15057123(판례)
- github.com/chrisryugj/lexdiff (Verbatim RAG·content matcher·4신호)
- arxiv 'Who Checks the Citations?'(Princeton, 5범주), arxiv Citation Grounding 메트릭
