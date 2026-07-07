# P5 — Citation Firewall: 0~100 신뢰점수 + 신호등 표시

## 목적
lawbot.org 대비 부족한 "신뢰점수·신호등" UX를 채운다. 답변 인용의 존재·문구·현행 검증 결과를 0~100 점수와 빨강/노랑/초록으로 시각화한다.

## 선행조건
- P4-#3(verify.py 1차 DB 게이트) 수정 완료 — 그래야 점수가 의미 있다.

## 대상 파일
- `search/verify.py` — 검증 파이프라인(존재·문구·현행).
- `api/main.py` — `/v1/verify` 및 답변 응답 envelope에 점수/플래그 포함.
- `web/chat.html` — 시각화(P6와 연동).

## 현재 상태 (검증 결과)
- **이미 있음**: `verify.py`의 `verify_citation`이 boolean `verified` + `trust_grade`(A/B) + `current`/`db_match`/`api_match`/`note` 반환. 존재·현행·시점 체크 구현됨.
- **없음(신규)**: 숫자 0~100 점수, 빨강/노랑/초록 flag, 응답 envelope의 `verify_score`/`trust_score`/`flag` 필드.
- ⚠️ **문구 비교는 현재 "조문번호 정확 일치"** 방식 — 아래의 "Levenshtein ≥95% 유사도"는 **신규 작업**(현재 없음).
- ⚠️ 점수의 `db_match` 입력은 **P4-#3 수정 전까지 불안정**(Qdrant 경로 죽어 제목조회로 degrade) → 선행조건 지킬 것.

## 작업 단계
1. **점수 산출** (`search/verify.py`):
   - 인용별 결과(인용 추출 → 존재 → 문구 일치 → 시점 유효)를 종합해 **0~100 신뢰점수** 계산. 문구 유사도(Levenshtein 등)는 신규 구현.
   - 인용별 플래그: 초록(존재+문구+현행 OK) / 노랑(경미 불일치·시점주의) / 빨강(미존재·문구변조·폐지).
   - 답변 전체 등급(A/B/C/D)도 선택적으로.
2. **응답 envelope**: `/v1/ask`·`/v1/ad-review`·`/v1/verify` 응답에 `citations[].flag`, `citations[].verify_score`, 전체 `trust_score` 필드 추가. 기존 인용 사후검증 흐름 위에 얹는다.
3. **시각화** (P6에서 구현): 인용 배지에 색상 + 점수, 클릭 시 검증 상세(원문 대조).

## 완료 기준 (DoD)
- `/v1/verify`에 텍스트를 넣으면 인용별 플래그 + 0~100 점수 반환.
- 일부러 변조한 인용(조문번호 바꿈)이 빨강으로 표시됨.
- 존재하지 않는 가짜 조문이 빨강 + 낮은 점수.
- `/v1/ask` 응답에 trust_score가 포함됨.

## 환각 방지 체크
- 점수는 **실제 검증 결과**에서만 계산(모델이 점수를 지어내지 않게). 문구 비교는 코퍼스/law.go.kr 원문 대조.
- law.go.kr 호출은 P7의 타임아웃/캐시/CB 정책을 따른다.
