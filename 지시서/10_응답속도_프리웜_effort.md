# 10 — 응답 속도: 인덱스 프리웜 + reasoning effort 분리 + rewrite 경량화

> 실측(2026-06-30, 도커 풀스택 caddy+api, full 인덱스 124만, effort=low):
> 콜드 첫 쿼리 **46초**, 워밍 실질답변 **10~28초(중앙값 21초)**.
> 지연을 지배하는 건 로컬(WSL 12GB·4코어·ext4)이 아니라 **gpt-5-mini 생성(추론+긴 근거형 출력 토큰) + `REWRITE=1`의 추가 LLM 호출**. FAISS 검색은 medical 2.5ms / full 216ms로 무시 수준. NCP로 옮겨도 이 시간은 안 변함(OpenAI측 비용).

## 목표
1. **첫 쿼리 콜드 46초 제거** — full 인덱스는 첫 질문 때 lazy 로드된다. 서버 기동 시 백그라운드로 미리 올린다.
2. **추론 시간 축소** — 메인 생성은 `reasoning_effort=low`(이미 적용), 질의 재작성(rewrite)은 키워드 추출 수준이라 **minimal**로 더 가볍게.
3. 모델·effort·프리웜은 **env로 조절**(코드 수정 없이 튜닝).

## 대상 파일
- `config.py` — `GEN_REASONING_EFFORT`(기본 `low`, 적용됨), `reasoning_effort_kwargs(model, effort=None)`(override 인자 추가), `GEN_REWRITE_EFFORT`(신규, 기본 `minimal`), `PREWARM_INDEX`(신규 env `LAWBOT_PREWARM`, 기본 on).
- `search/retriever.py` — `_llm_rewrite`가 `GEN_REWRITE_EFFORT` 사용.
- `api/main.py` — `lifespan`에서 데몬 스레드로 `retriever.get_index()` 프리웜(healthz 즉시 OK 유지, 블로킹 금지).

## 작업 단계
1. `reasoning_effort_kwargs(model, effort=None)`: effort 미지정이면 `GEN_REASONING_EFFORT` 사용. **gpt-5/o-계열 모델일 때만** 파라미터 부착(그 외 모델은 400 → 빈 dict). 가드 유지.
2. `GEN_REWRITE_EFFORT = os.getenv("GEN_REWRITE_EFFORT","minimal")`.
3. `retriever._llm_rewrite`: `**config.reasoning_effort_kwargs(config.GEN_MODEL, config.GEN_REWRITE_EFFORT)`.
4. `PREWARM_INDEX = os.getenv("LAWBOT_PREWARM","1")=="1"`.
5. `lifespan`: probe 뒤, PREWARM이면 `threading.Thread(daemon=True)`로 `backends.retriever.get_index()` 호출. 예외는 로깅만(무해). `get_index()`는 내부 락을 쓰므로 동시 요청과 충돌·중복로드 없음.

## 완료 기준 (DoD — OpenAI 호출 없이 검증)
- import OK: `config`, `search.rag`, `search.retriever`, `api.main`.
- `config.GEN_REWRITE_EFFORT == "minimal"`; `reasoning_effort_kwargs("gpt-5-mini","minimal") == {"reasoning_effort":"minimal"}`; `reasoning_effort_kwargs("gpt-4o-mini") == {}`.
- 도커 재빌드 후 컨테이너 `healthy`, **로그에 "index pre-warm" 완료 메시지**(첫 쿼리 없이 인덱스 적재됨).
- `python -m search.retriever --selftest` 회귀 없음.

## 환각 방지 / 주의
- effort 파라미터는 reasoning 모델(gpt-5/o-계열) 전용. 모델을 gpt-4o-mini 등으로 바꾸면 자동으로 빠짐(가드).
- 프리웜은 첫 쿼리 지연을 startup으로 옮길 뿐 — 메모리(12GB)에 full 인덱스 2.4GB + 오프셋표만 상주(parents 미적재).
