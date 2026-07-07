# P8 — 배포: NCP 단일서버 · 공인 IP HTTPS · MCP 연결

## 목적
전체 인덱스를 얹은 lawbot을 네이버클라우드(NCP) 단일 서버에 Docker로 올려 실제로 접속·테스트 가능하게 한다.

## 선행조건
- P1(전체 인덱스), P3(범용화), P6(테스트 페이지) 완료 권장.

## 대상 파일
- `Dockerfile`, `docker-compose.yml` — 이미 caddy+api 구성(qdrant/redis 제거됨), FAISS 볼륨 마운트.
- `config.py` — 경로/도메인 env.
- `mcp_server/server.py` — MCP는 HTTP 래퍼(배포 후 엔드포인트 연결).
- `DEPLOY.md` — 런북(**이미 존재** → 신규 생성 아니라 업데이트).

## 작업 단계
1. **서버 스펙 확정 (먼저 결정)**:
   - **실측(2026-06-29 빌드 완료)**: index.faiss **2.54GB** + meta.jsonl **3.17GB**(ntotal 1,238,122). 현재 `load_index`가 meta를 RAM에 통째로 올려 **검색만으로 ~8.5GB 상주** → **7GB 박스 OOM 확인.** LLM·OS 여유까지 하면 **16GB 이상 필요.**
   - **권장 4vCPU/16GB**(~₩140k/월). 8GB는 meta 디스크 조회 전환(저RAM retriever) + 양자화 없이는 위험.
   - 이 결정이 P1의 인덱스 타입(Flat/IVF/HNSW)·양자화·meta 적재방식을 좌우 → P1과 합의.
2. **데이터 전달**:
   - GitHub 미사용 방침. **rsync로 원천/산출물 전달.** 단 임베딩 산출물(수 GB)·인덱스는 크므로 **서버에서 빌드**하거나 인덱스 파일만 전송(50GB+ 원천 전송 회피).
   - 비밀키(.env: OPENAI_API_KEY, LAW_OC)는 서버에서 직접 주입, 레포/이미지에 안 굽기.
   - ⚠️ **`docker-compose.yml`의 api `environment:`에 `LAW_OC` 슬롯이 없음**(OPENAI_API_KEY만 있음). `LAW_OC: ${LAW_OC:-}` 추가할 것 — 없으면 law.go.kr 현행 검증이 조용히 DB-only로 degrade.
3. **HTTPS**:
   - 도메인 있으면 Caddy 자동 인증서. **도메인 없으면 공인 IP + `<공인IP>.nip.io`**로 Let's Encrypt(베어 IP엔 인증서 발급 불가).
4. **기동·점검**:
   - `docker compose up -d` → `/healthz` 200, 테스트 페이지 접근(경로는 **`/chat`** — 루트 `/`는 라우트 없음/404). 무인증 접근은 **P6의 데모키 메커니즘이 먼저 배포돼야** 가능(그 전엔 키 없이 질의·검토 불가).
   - ⚠️ **healthz는 "잔재 정리"가 아니라 실제 코드 변경**: 현재 `api/main.py`의 `_probe_qdrant`가 Qdrant를 호출하고 `/healthz`의 `points`가 **Qdrant 카운트**라 FAISS 배포에선 항상 null. FAISS `ntotal`을 healthz에 새로 배선해야 DoD의 "ntotal 표시"가 충족됨. `console.html`의 Qdrant pill, `search/verify.py`의 Qdrant scroll(P4-#3)도 같이 정리.
5. **MCP 연결**: `mcp_server/server.py`의 `LAWBOT_API_BASE`를 배포 URL로, 4도구(ask/search/verify/review) 동작 확인.

## 완료 기준 (DoD)
- 외부에서 HTTPS로 테스트 페이지 접속·질의·광고검토 성공.
- `/healthz` 정상(인덱스 로드·ntotal 표시).
- MCP 클라이언트에서 4도구 호출 성공.
- DEPLOY.md에 실제 실행 기록 남김.

## 환각 방지 체크
- compose/Dockerfile 현재 내용 확인 후 수정(이미 FAISS 구성됨, Qdrant 되살리지 말 것).
- 비용: 배포 자체는 OpenAI 무관. 서버에서 인덱스 빌드 시 벡터 재임베딩 금지(이미 있음).
