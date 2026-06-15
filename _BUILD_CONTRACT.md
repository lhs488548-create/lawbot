# lawbot — BUILD CONTRACT

> Owner: **Contracts**. This file plus `config.py`, `ingest/schema.py`, and
> `requirements.txt` are the shared foundation. **Builders do not modify these
> shared files** — they add their own modules implementing the interfaces below.
> Source of truth for tasks: `분석/08_lawbot_빌드_하네스_플레이북.md`.

Product: a **production** Korean-legal RAG service for **lawyers** (expert
audience), delivered as a **multi-tenant cloud API**. Not a demo.

---

## (a) How to run (WSL venv — do NOT recreate it)

Python lives in an **already-created** WSL venv (Python 3.12.13). Never run
`python -m venv` again.

```bash
# Run any script / module:
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python <script-or -m module>'

# Add a dependency (then add the pin to requirements.txt — Contracts only):
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && ~/.local/bin/uv pip install --python .venv/bin/python <pkg>'

# Start the API:
wsl -d Ubuntu -- bash -lc 'cd /home/user1/lawbot && .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000'
```

Run modules as packages (`-m ingest.parse_statute`) so that
`from ingest.schema import ...` and `from config import ...` resolve. The
project root `/home/user1/lawbot` must be on `sys.path` (it is when you `cd`
there and run `.venv/bin/python -m ...`).

## (b) Path bridge (UNC ⇄ WSL — the same files)

| Purpose | Path |
|---|---|
| Read/Write/Edit tools (Windows) | `\\wsl.localhost\Ubuntu\home\user1\lawbot\...` |
| Python / shell (WSL) | `/home/user1/lawbot/...` |
| Source data (WSL) | `/home/user1/체크/NEW2/원천데이터` (= `config.DATA_ROOT`) |
| Source data (Windows) | `\\wsl.localhost\Ubuntu\home\user1\체크\NEW2\원천데이터` |

Code uses **POSIX paths only** (via `config.DATA_ROOT` etc.). The UNC form is
for the Windows-side file tools.

## Secrets policy (hard rules)

- Secrets are read **only** from `.env` via `config.py`. Never hard-code,
  never `print`/log a key, never write a key into an artifact.
- `.env`, `.venv/`, `artifacts/`, `*.db`/`*.sqlite3`, `batch_*.jsonl`,
  `__pycache__/` are git-ignored. Do not commit them.

---

## (c) Parser interface (Phase 1)

Each corpus parser exposes a module-level **public generator**:

```python
def parse_all() -> Iterator[Document]: ...
```

returning `ingest.schema.Document` instances (never raw dicts). Running a
parser module as a script writes its corpus JSONL (one `Document.model_dump_json()`
per line) to the fixed artifact path:

| Module | `parse_all()` source glob | Output (see `config`) |
|---|---|---|
| `ingest/parse_statute.py`   | `LAW_DIR/*/*.md`               | `DOCS_LAW_JSONL`     (`artifacts/docs_law.jsonl`) |
| `ingest/parse_precedent.py` | `PRECEDENT_DIR/**/*.md` (skip README) | `DOCS_PREC_JSONL` (`artifacts/docs_prec.jsonl`) |
| `ingest/parse_admrule.py`   | `ADMRULE_DIR/**/본문.md`        | `DOCS_ADMRULE_JSONL` (`artifacts/docs_admrule.jsonl`) |
| `ingest/parse_ordinance.py` | `ORDINANCE_DIR/**/본문.md`      | `DOCS_ORD_JSONL`     (`artifacts/docs_ord.jsonl`) |

`build_doc_id(doc_type, *parts)` from `ingest.schema` builds ids:
`LAW:{법령ID}:{법령구분}`, `ORD:{광역}:{자치법규ID}`, `ADMRULE:{행정규칙ID}`,
`PREC:{판례일련번호}`.

### Verified on-disk facts (the parsers depend on these)

- All four corpora are **YAML front-matter + Markdown body**, delimited by a
  leading `---\n...\n---\n` block.
- **National law** (`01_국가법령/kr/{법령명}/{구분}.md`): one file per kind
  (`법률.md`, `시행령.md`, `시행규칙.md`, `대통령령.md`, ...) ⇒ **each file is a
  separate Document**, distinguished by `법령구분`. FM keys: `제목, 법령ID,
  법령MST, 법령구분, 소관부처, 공포일자, 시행일자, 상태, 출처, 첨부파일`.
  Count ≈ **5,673 files**.
- **Ordinance** (`02_자치법규/{광역}/.../{법령명}/본문.md`): FM keys
  `자치법규ID, 자치법규명, 자치법규종류(조례·규칙), 지자체구분{광역,기초},
  공포일자, 시행일자, 출처, 첨부파일`. Count ≈ **159,890 files** (all 18 시도).
- **Admin rule** (`03_행정규칙/{부처}/.../{규칙명}/본문.md`): FM keys
  `행정규칙ID, 행정규칙명, 행정규칙종류, 소관부처명, 발령일자, 시행일자,
  본문출처, 출처, 첨부파일`. Count ≈ **21,700 files**.
- **Precedent** (`04_판례/{사건종류}/{등급}/{법원}_{선고일}_{사건번호}.md`):
  FM keys `판례일련번호, 사건번호, 사건명, 법원명, 법원등급, 사건종류, 출처,
  선고일자`. Count ≈ **123,743 files** (excluding README.md).

### IMPORTANT correction vs. the 08 playbook prose

The playbook (§3) says ordinances/admin-rules use **inline** `제N조(제목)`
bodies while national law uses `##### 제N조` headers. **On this dataset all
three legal-text corpora use `#####`-style article headers.** Therefore a
single article splitter works for law, ordinance, and admin-rule:

```python
ART = re.compile(r"^#{3,6}\s*(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?", re.M)
```

Keep an inline fallback (`^(제\d+조(?:의\d+)?)\s*\(([^)]*)\)`) for any stray
files, but `#####` is primary. Precedents split on `^##\s+(.+)$` sections.
Admin-rules/ordinances whose body is empty/label-only ⇒ `trust_grade="B"`,
`articles=[]` (still emit the Document with metadata).

Parsers must: validate front matter exists, skip+log malformed files (never
crash the run), and **stream** (12万 precedents) rather than load all in memory.

---

## (d) Header / Chunk / Embed / Qdrant / Retriever / RAG interfaces (Phases 2–3)

> Source of truth for chunking/headers/embeddings is **`분석/09_청킹_임베딩_헤더_설계.md`**.
> Where 09 and the 08 playbook differ, **09 wins**.

### Header governance — `header/` (09 §D) ★

The **deterministic, $0, rule-based** two-layer header is the heart of retrieval
quality (Anthropic Contextual Retrieval: −35% top-20 failures). It is built in
**one place only** so it never drifts. **No LLM is used to build headers** (cost
rule). Three modules:

```python
# header/schema.py  — single definition of required header fields per doc_type
HEADER_REQUIRED: dict[str, list[str]]      # e.g. law -> ["title","article_no","effective_from"]

# header/build.py   — the ONLY place chunk headers are produced
def build_headers(doc: dict, article: dict) -> tuple[str, dict]: ...
#   returns (embed_text, payload):
#     embed_text = "<L1 인용헤더>\n<L2 맥락헤더>\n<정규화 본문>"   (09 §D-1)
#     payload    = the structured meta dict (09 §D-2, keys below)

# header/validate.py — governance check, run in ingest pipeline AND pytest
def validate_chunk(chunk: dict) -> list[str]: ...   # [] if valid, else error strings
def validate_file(path) -> dict: ...                # {n, n_bad, errors[...]} report
```

L1/L2 header format (09 §D-1), built by rule (parent title + position +
key entities — 소관부처/지자체/사건종류), never by LLM:

```
[법령] 도로교통법 > 제2장 > 제17조(자동차등의 속도) · 시행 2026-04-02 · 소관 경찰청   <- L1 인용헤더
이 조문은 '도로교통법'(법률)의 일부로, …를 규정한다.                                  <- L2 맥락헤더
(정규화된 조문 본문)                                                                  <- 본문
판례 L1: [판례] 대법원 2022-05-25 2022누50008 · 선거·특별 · 판결요지
```

`validate_chunk` asserts: (a) L1 has the doc_type's required fields
(`HEADER_REQUIRED`), (b) a non-empty L2 line exists, (c) payload carries all
filter keys. **The ingest pipeline and `pytest` both run the validator; a header
regression fails the build (= continuous supervision, 09 §D-3).**

### Normalization (09 §B-4), applied to **embedding text only** (original preserved)

`①②③`→`(1)(2)(3)` 병기, `<개정 …>` 주석은 본문에서 분리해 payload(`amendments`)로,
판례 `○○○`→`[당사자]`, Unicode **NFC**, 중복공백 정리.

### Chunking — `embed/chunk.py` (09 §B-1 parent/child)

```python
def chunks_of(doc: dict) -> Iterator[dict]: ...   # one parsed Document (dict)
def build_chunks() -> None: ...                    # docs_*.jsonl -> CHUNKS_JSONL (+ PARENTS_JSONL)
```

**child** chunk record (`artifacts/chunks.jsonl`, one per line) — the search /
embedding unit. `chunk_id`/`parent_id` via `ingest.schema.build_chunk_id` /
`parent_id_of`. `text` is the **two-layer-header + body** from
`header.build_headers` (do not re-implement the header here):

```jsonc
{
  "chunk_id": "LAW:014565:법률#제4조#0",     // build_chunk_id(doc_id, article_no, part_idx)
  "doc_id":   "LAW:014565:법률",
  "parent_id":"LAW:014565:법률",             // == doc_id (parent = whole law/precedent)
  "text":     "<L1 인용헤더>\n<L2 맥락헤더>\n<정규화 본문>",
  "content_hash": "<sha256 of text>",        // 09 §C cache key (idempotent re-embed)
  "payload": {
    "doc_type": "law", "title": "...", "jurisdiction": "국가",
    "law_kind": "법률", "article_no": "제4조", "article_title": "성년",
    "chapter_path": "제1장", "part_idx": 0,
    "effective_from": "2024-07-03", "as_of": null,
    "source_url": "https://...", "license": "<config.DEFAULT_LICENSE>",
    "kind": "본문",                            // "본문" | "별표"
    "trust_grade": "A",
    "parent_id": "LAW:014565:법률"
  }
}
```

**parent** record (`artifacts/parents.jsonl`) — the generation / source-pack
unit (child hit → promote to parent full text, 09 §E-1):

```jsonc
{ "parent_id": "LAW:014565:법률", "doc_type": "law", "title": "...",
  "law_kind": "법률", "jurisdiction": "국가", "effective_from": "...",
  "source_url": "...", "license": "...", "trust_grade": "A",
  "full_text": "<모든 조문 원문 연결>" }
```

Rules: the two-layer header goes **inside** the embedded `text` (so "which law,
which article, what context" lands in the vector); structured meta is
payload-only. Measure length with `tiktoken.get_encoding(config.EMBED_ENCODING)`;
only split when > `config.EMBED_MAX_TOKENS` (rare; long precedent sections),
using `config.CHUNK_WINDOW_TOKENS`/`config.CHUNK_OVERLAP_TOKENS` windows on
sentence boundaries where possible, same `parent_id`, increment `part_idx`.
별표/별지 with body → separate child `kind="별표"`; label-only → payload link,
`trust_grade="B"`, not embedded (09 §B-3). Payload **must** contain the filter
keys `doc_type, jurisdiction, law_kind, effective_from` (+ `trust_grade`,
`license`, `parent_id`).

### Embedding wrapper + cache — `embed/embed_client.py` (09 §C)

A thin OpenAI wrapper used by retriever, the small/sync path, and `/v1/embeddings`:

```python
def embed_texts(texts: list[str]) -> list[list[float]]: ...   # sync, batched <=2048, tenacity retry
def embed_batch(texts: list[str]) -> list[list[float]]: ...   # alias / batched form
def content_hash(text: str) -> str: ...                       # sha256 of normalized text (cache key)
def cached_embed(items: list[dict]) -> dict[str, list[float]]:# items: [{chunk_id,text,content_hash}]
```

- **Content-hash cache (09 §C, mandatory):** before embedding, look up
  `content_hash` in `config.EMBED_CACHE_JSONL`; only embed cache-misses, then
  append new `{content_hash, vector}` lines. **Same text is never re-embedded /
  re-billed.** Cache key = sha256 of the *normalized embedding text*.
- Always `config.EMBED_MODEL`; every returned vector has `len == config.EMBED_DIM`.
- **large model and LLM-generated chunk context are forbidden** (budget rule).

### Embedding (Batch) — `embed/embed_batch.py`

```python
def build_batch_input() -> list[Path]: ...  # CHUNKS_JSONL -> batch_in*.jsonl (sharded <=50k lines / <=200MB), cache-misses only
def estimate_cost() -> dict: ...            # {total_tokens, est_usd, n_chunks, n_cached}
def submit(confirm: bool = False) -> list[str]: ...  # returns batch ids; refuses unless confirm
def collect(batch_ids: list[str]) -> None:  # poll -> EMBEDDINGS_JSONL {"chunk_id","vector":[EMBED_DIM]} + update cache
```

- **Cost gate (mandatory, Task 2.2 ⚠️ / 09 §G):** print token total + estimated
  USD and require explicit human `confirm=True` before any paid submission.
  **Full-corpus embedding is never auto-run** — estimate + approval only. Demo
  embeds ≤ `config.DEMO_MAX_CHUNKS` (20k) chunks. Use `config.EMBED_MODEL`;
  output dim must equal `config.EMBED_DIM`.
- Batch lines: `{custom_id: chunk_id, method: "POST", url: "/v1/embeddings",
  body: {model: EMBED_MODEL, input: text}}`. **Skip chunks whose `content_hash`
  is already cached.** Shard to respect 50k-line / 200MB limits. Log + retry
  failed `custom_id`s.

### Qdrant upsert — `embed/upsert_qdrant.py`

```python
def ensure_collection() -> None: ...   # size=EMBED_DIM, distance=Cosine; payload KEYWORD indexes
def upsert_all() -> None: ...          # join chunks+embeddings, batch upsert (~1000/req)
```

- Collection name = `config.COLLECTION`; vector size = `config.EMBED_DIM`,
  distance Cosine. Create KEYWORD payload indexes on
  `doc_type, jurisdiction, law_kind, effective_from`.
- **Point id = deterministic UUID5 of `chunk_id`** (so re-ingest is idempotent).
  Keep `chunk_id` in payload for joins. Connect with `QDRANT_URL` +
  `QDRANT_API_KEY` (None locally).

### Retriever — `search/retriever.py` (09 §E-1)

```python
def embed_query(query: str) -> list[float]: ...
def search(query: str, k: int = config.DEFAULT_TOP_K,
           flt: dict[str, str] | None = None,
           as_of_date: str | None = None) -> list[Hit]: ...
def get_parent(parent_id: str) -> dict | None: ...   # child hit -> parent full text (PARENTS_JSONL)
```

`flt` is `{payload_key: value}` AND-ed into a Qdrant filter (e.g.
`{"doc_type": "law"}`, `{"jurisdiction": "전라남도"}`). `as_of_date` (ISO
`YYYY-MM-DD`) restricts to rows whose `effective_from <= as_of_date` (point-in-
time current-law lookup, 09 §A/E). Each `Hit` exposes `.id`, `.score`,
`.payload` (with `text` + the payload keys above incl. `parent_id`). Embed the
query with the **same** `config.EMBED_MODEL`. Wrap OpenAI/Qdrant calls with
`tenacity` retry. `get_parent` powers parent-promotion for `ask`/`source-pack`.

### RAG — `search/rag.py`

```python
def ask(query: str, k: int = config.DEFAULT_TOP_K,
        flt: dict | None = None, model: str | None = None) -> AskResult: ...
```

Returns:

```jsonc
{
  "answer": "...",
  "citations": [
    {"source_id": "<chunk-id-or-doc-id>", "title": "...",
     "location": "제4조 | 판결요지", "source_url": "https://...",
     "doc_type": "law", "trust_grade": "A"}
  ],
  "used_context": [ /* the hits shown to the model, for audit */ ],
  "model": "gpt-4o-mini",
  "ai_generated": true,
  "disclaimer": "<config.ANSWER_DISCLAIMER>"
}
```

Pipeline: retrieve top-K → build numbered `[n]` context (each block tagged with
its `source_id` = the hit id, doc_type, title, article/section, url) →
GPT call with **Structured Outputs** (`response_format` json_schema, strict) →
**citation post-verification**: drop any citation whose `source_id` is not in
the retrieved hit ids (anti-hallucination, always ON) → attach
`config.ANSWER_DISCLAIMER` and `ai_generated=true`. Use `config.GEN_MODEL`
(escalate to `GEN_MODEL_FALLBACK` only when explicitly requested). When the top
score < `config.MIN_RETRIEVAL_SCORE`, return a "근거 불충분" answer with empty
`citations` (never fabricate). `ask` accepts `as_of_date` and forwards it to
`search`.

### Statutes search — `search/statutes.py` (09 §A, lawbot.org `/v1/statutes/search`)

```python
def statutes_search(query: str, k: int = config.DEFAULT_TOP_K,
                    filter: dict | None = None,
                    as_of_date: str | None = None) -> list[dict]: ...
```

Unified law+precedent search with article/section precision. Each result row
carries the **common meta** `{trust_grade, source_url, license, as_of_date,
effective_from}` plus `{doc_id, doc_type, title, article_no, score, text}`.
No LLM — pure retrieval (cheap, key-gated but no generation cost).

### Citation Firewall — `search/verify.py` (09 §E-2, lawbot.org `/v1/verify`)

```python
def verify_citation(citation: dict, as_of_date: str | None = None) -> dict: ...
#   citation: {law_name?/title?, article_no?} or {case_no?/사건번호?}
#   returns: {verified: bool, trust_grade, current: bool, source_url,
#             effective_from, as_of_date, note, db_match: bool, api_match: bool|None}
```

Three checks: ① DB existence (Qdrant/parents), ② **law.go.kr OpenAPI
(`config.LAW_API_BASE`, OC=`config.LAW_OC`) 현행·문구 대조**, ③ `as_of_date`
point-in-time validity. Detects 폐지/오인용/허위사건. **OC token never logged or
returned.** When the law API is unavailable, fall back to DB-only with
`api_match=None` and a `note`.

### Source Pack — `search/source_pack.py` (09 §E-3, lawbot.org `/v1/source-pack`)

```python
def build(query: str, k: int = config.DEFAULT_TOP_K,
          filter: dict | None = None,
          as_of_date: str | None = None) -> dict: ...
#   returns: {markdown: "<인용가능 번들>", sources: [{...common meta...}], as_of_date}
```

Retrieve child hits → promote to ≤`config.SOURCE_PACK_MAX_PARENTS` parents →
emit an **LLM-citable markdown bundle** (법령>조문 원문 + 시행일 + source_url +
trust_grade + license). Deterministic assembly, **no generation cost**.

---

## (e) FastAPI endpoint contract (Phase 4) — `api/main.py`

All under prefix `/v1` except `/healthz`, `/console`. Auth via
`Authorization: Bearer <key>` (see (f)). OpenAPI auto-served at `/docs`. The
endpoint set is **aligned to lawbot.org** (09 §A) plus the lawyer-mode add-ons
`/v1/ask` and `/v1/ad-review`. **All search/verify/source-pack/ask responses
carry the common meta `{trust_grade, source_url, license, as_of_date,
effective_from}`** and accept an optional `as_of_date` (ISO `YYYY-MM-DD`).

| Method & path | Auth | Body / params | Response |
|---|---|---|---|
| `GET /healthz` | none | — | `{"ok": true, "collection": "...", "points": <int>}` |
| `POST /v1/statutes/search` | key (anon read may be IP-limited) | `{query:str, filter?:dict, k?:int=8, as_of_date?:str}` | `{results:[{doc_id,doc_type,title,article_no,score,text, ...common meta}]}` — `search.statutes.statutes_search` |
| `POST /v1/verify` | key | `{citation:{...} \| citations:[...], as_of_date?:str}` | `{results:[{verified,trust_grade,current,source_url,note, db_match,api_match}]}` — Citation Firewall `search.verify.verify_citation` |
| `POST /v1/source-pack` | key | `{query:str, filter?:dict, k?:int=8, as_of_date?:str}` | `{markdown, sources:[...common meta], as_of_date}` — `search.source_pack.build` |
| `POST /v1/ask` | **key required** (LLM cost) | `{query:str, filter?:dict, k?:int=8, as_of_date?:str}` | `AskResult` (see (d)) |
| `POST /v1/embeddings` | key | OpenAI-compatible `{input: str\|[str], model?:str}` | OpenAI-shape `{object:"list", data:[{embedding,index}], model, usage}` — internal `embed.embed_client` wrapper (forces `config.EMBED_MODEL`/`EMBED_DIM`) |
| `POST /v1/ad-review` | key | **multipart/form-data**: `file` (PDF) **or** `text`; optional `question` | `{summary, issues:[...], citations:[...], ai_generated:true, disclaimer}` |
| `GET /v1/statutes/{law_id}/articles/{article_no}` | key | path: 법령ID + e.g. `제4조` (URL-encoded) | `{doc_id, title, law_kind, article_no, text, source_url, effective_from}` or 404 |
| `GET /v1/precedents/{seq}` | key | path: 판례일련번호 | `{doc_id, 사건번호, 사건명, 법원명, 선고일자, sections:[...], source_url}` or 404 |
| `POST /v1/keys` | **admin key** | `{tenant:str, tier?:str="free", rate?:str}` | `{key:"lk_...", tenant, tier, rate}` — **plaintext key returned once only** |
| `GET /v1/keys` | admin key | — | `[{tenant, tier, rate, usage, revoked, created_at}]` (no plaintext keys) |
| `DELETE /v1/keys/{key_id}` | admin key | path: key id/hash prefix | `{revoked: true}` |
| `GET /console` | none (page); actions use key | — | minimal multi-tenant self-service console (API keys · usage · sources · sync logs), served from `web/` |

Errors: missing/invalid key → **401**; rate limit exceeded → **429** (with
`Retry-After`); not found → **404**; bad input → **422** (Pydantic). Every
`/ask` and `/ad-review` response carries `disclaimer` + `ai_generated:true`
(AI Basic Act notice). Do not log user PII; do not persist query bodies with
personal data.

`/v1/statutes/search`, `/v1/verify`, `/v1/source-pack`, `/v1/embeddings` are the
**core lawbot.org-aligned data-infra surface** (no generation cost except none).
`/v1/ask`/`/v1/ad-review` are the lawyer add-ons (LLM cost). `/v1/ad-review`
extracts text (pypdf for PDF, raw for text), retrieves relevant
statutes/precedents, and returns an **expert-mode** issue-spotting review with
verified citations (see (g)). `/console` + `/v1/keys` provide the multi-tenant
self-service surface; the console UI lives in `web/`.

---

## (f) Multi-tenant API key schema (Phase 4) — `api/auth.py`

Store (MVP SQLite at `config.API_KEYS_DB`; prod Postgres/Redis behind the same
functions). **Never store plaintext keys** — store a SHA-256 hash.

| Column | Type | Meaning |
|---|---|---|
| `key_hash` | TEXT PK | `sha256(raw_key)`; raw is `lk_<token_urlsafe(24)>`, shown once |
| `tenant` | TEXT | tenant / customer id (multi-tenant isolation key) |
| `tier` | TEXT | `free` \| `pro` \| `enterprise` \| `admin` |
| `rate` | TEXT | rate-limit spec, e.g. `"60/minute"` (slowapi format) |
| `usage` | INTEGER | cumulative request count (incremented per call) |
| `revoked` | INTEGER | 0 \| 1 |
| `created_at` | TEXT | ISO timestamp |

Public functions:

```python
def init_db() -> None: ...
def issue_key(tenant: str, tier: str = "free", rate: str | None = None) -> str: ...  # returns raw key once
def verify(raw_key: str) -> dict | None: ...   # -> {tenant,tier,rate,usage} or None; increments usage
def list_keys() -> list[dict]: ...             # never returns plaintext
def revoke(key_id: str) -> bool: ...
```

Rate limiting via `slowapi`, keyed by `tenant`+`tier` (per-key limits), with an
IP fallback for anonymous `/v1/search`. `/v1/ask` requires a key. `admin`-tier
keys gate the `/v1/keys` management endpoints.

## (g) Expert mode policy (lawyers, NOT consumers)

Audience is **practicing lawyers**. We therefore do **NOT** add the
변호사법 §109 consumer guard (no "I can't give legal advice / consult a lawyer"
refusals, no forced disclaimers that block substance). Instead:

- **Full professional analysis & drafting assistance** is allowed (issue
  spotting, statute/precedent synthesis, draft-document review on
  `/v1/ad-review`).
- **Grounding is mandatory anyway, for quality:** answer **only** from
  retrieved originals; never answer from model parametric knowledge alone.
- **Citations are mandatory** (law name + article no / case number + URL) and
  every citation is **post-verified against retrieved context** (drop
  unverifiable ones). This is a quality control, not a consumer safety gate.
- When retrieval is insufficient, the model must say **"확인 필요"** /
  "근거 불충분" rather than fabricate. B-grade (metadata-only) sources must be
  flagged as such.
- Keep the lightweight provenance/AI notice (`config.ANSWER_DISCLAIMER`,
  `ai_generated:true`) — that satisfies the AI Basic Act, not consumer refusal.

## (h) Service-grade prompt authoring guide

Prompts are production assets. Every generation prompt must:

1. **Role**: senior Korean legal research assistant for lawyers (expert mode,
   precise legal Korean register / 법률 문어체).
2. **Grounding clause**: "아래 [검색결과]에 있는 내용만 근거로 답하라. 검색결과에
   없는 사실은 추측하지 말고 '확인 필요'라고 명시하라." (no outside knowledge).
3. **Citation enforcement**: every assertion carries a `[n]` marker mapping to a
   context block; final structured `citations[]` via **Structured Outputs**
   (strict json_schema). `source_id` must equal a provided context id.
4. **Hallucination suppression**: forbid inventing article numbers, case
   numbers, dates, or URLs; if unsure, downgrade to "확인 필요".
5. **Uncertainty surfacing**: explicitly distinguish 현행/개정, flag B-grade
   (metadata-only) sources, note coverage gaps honestly.
6. **Format discipline**: answer in Korean legal prose; structured fields exactly
   per the json_schema; no extra prose outside the schema when structured output
   is requested.
7. **Robustness**: define behavior for empty/low-score retrieval (return a
   "근거 불충분" answer with empty `citations`), and for multi-jurisdiction
   conflicts (present each with its citation rather than merging silently).

Builders place reusable prompt constants near their consumer (`search/rag.py`,
the ad-review handler) and keep the json_schema in one place per endpoint.

## (i) Test convention (09 §F) — `tests/`

- Unit tests live in `tests/test_<module>.py` (pytest). Run them with the WSL
  venv from the project root so imports resolve:
  `cd /home/user1/lawbot && .venv/bin/python -m pytest -q tests`.
- **Per-module green:** parsers×4, chunk, **header validator**, embed wrapper,
  retriever, verify, source_pack, statutes, rag, auth/keys.
- **Header governance gate:** a test runs `header.validate.validate_file` over a
  sample of produced chunks and asserts **0 missing-required-field rows** (or
  only the allow-list). Header regressions fail CI.
- **OpenAI calls in tests:** at most **1–2** real calls total (cost rule);
  prefer mocking the client. Never embed > `config.DEMO_MAX_CHUNKS` in a test.
- **Integration (E2E):** ingest→chunk→header→(demo)embed→Qdrant→
  search/ask/verify/source-pack returns a cited answer.
- Tests must not print secrets and must not require live network beyond the
  sanctioned 1–2 OpenAI calls (skip-if-unset for Qdrant/law.go.kr).

## Directory layout (Contracts-owned skeleton)

```
ingest/  header/  embed/  search/  api/  web/  eval/  tests/  artifacts/
```

`header/` (09 §D) and `web/` (`/console` UI) are added by this revision. Builders
own the modules **inside** these dirs; Contracts owns `config.py`,
`ingest/schema.py`, `requirements.txt`, and this file.

---

## DoD ledger (Contracts, Phase 0)

- venv is Python **3.12.13** (no recreation).
- `.env` load works and `OPENAI_API_KEY` **and** `LAW_OC` are present (boolean
  check only — keys never printed).
- UNC⇄WSL path bridge proven (a Windows-written file is read by WSL Python).
- `config.py`, `ingest/schema.py`, `requirements.txt`, this contract, and the
  package directories (`ingest embed search api web header eval tests artifacts`)
  exist.

### Revision (09 alignment)

This revision aligns the foundation to **`분석/09_청킹_임베딩_헤더_설계.md`** (SoT):
parent/child two-layer chunking (`build_chunk_id`/`parent_id_of` + `parent_id`
payload + `PARENTS_JSONL`), the **deterministic two-layer header** module
(`header/` build+schema+validate, §D), content-hash embed cache
(`EMBED_CACHE_JSONL`, §C), `as_of_date` + common response meta
(`{trust_grade, source_url, license, as_of_date, effective_from}`), and the full
**lawbot.org endpoint set** (`/v1/statutes/search`, `/v1/verify`,
`/v1/source-pack`, `/v1/embeddings`, `/console`) with the
`statutes_search`/`verify_citation`/`source_pack.build` interfaces. Builders'
existing `embed/chunk.py`, `search/retriever.py`, `search/rag.py`, `api/*` should
be brought up to these signatures (additive: new kwargs default to old behavior).
