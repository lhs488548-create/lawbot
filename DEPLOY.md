# lawbot — Deployment & Operations Guide

Production runbook for the **lawbot** Korean-legal data-infra API (multi-tenant,
lawyer audience). It takes you from a local bring-up to a public HTTPS cloud
deployment, including external-key onboarding and the golden-set evaluation.

> Source of truth for the build: `_BUILD_CONTRACT.md`,
> `분석/08_lawbot_빌드_하네스_플레이북.md`, `분석/09_청킹_임베딩_헤더_설계.md`.
>
> **Secrets policy (hard rule):** secrets live only in a git-ignored `.env`
> (local) or the platform's secret store (cloud). Never commit `.env`, never
> print/log a key. `.env`, `.venv/`, `artifacts/`, `*.db` are git-ignored.

---

## 0. What you deploy

```
                    Caddy (TLS)  ──►  api (FastAPI, uvicorn)  ──►  Qdrant (vectors)
   client ──HTTPS──►   :443                  :8000            └─►  Redis (rate-limit state)
```

- **api** — `api.main:app`, the FastAPI surface (`/v1/statutes/search`,
  `/v1/verify`, `/v1/source-pack`, `/v1/embeddings`, `/v1/ask`, `/v1/ad-review`,
  `/v1/keys`, `/console`, `/healthz`). Stateless except the SQLite key store.
- **qdrant** — vector store (local Docker or **Qdrant Cloud** free tier).
- **redis** — shared rate-limit backend so per-key limits hold across replicas.
- **caddy** — automatic Let's Encrypt HTTPS in the self-hosted compose stack.
  On Render/Fly the platform is the TLS edge, so Caddy is dropped there.

---

## 1. Local bring-up (Docker Compose — the whole stack)

### 1.1 Prerequisites
- Docker + Docker Compose v2 (`docker compose version`).
- An **OpenAI API key** with a monthly usage hard-limit set in the OpenAI
  dashboard (cost guard).

### 1.2 Configure
```bash
cp .env.example .env
# edit .env:  set OPENAI_API_KEY=sk-...   (DOMAIN=localhost is fine for dev)
```

Validate the stack renders before starting anything (this is the contract's
deploy DoD — it must succeed):
```bash
docker compose config        # prints the fully-resolved compose model, or errors
```

### 1.3 Run
```bash
docker compose up -d --build
docker compose ps            # all services healthy?
curl -fsS http://localhost:8000/healthz | jq    # via api directly (host-mapped only if you add a port)
# Through Caddy (dev TLS, self-signed CA):
curl -k https://localhost/healthz
```

`GET /healthz` returns `{"ok": true, "collection": "lawbot", "points": <n>,
"backends": {...}}`. `points` is 0 until you ingest+embed+upsert (Phases 1–2,
owned by the ingest/embed builders — see the playbook). The API still boots and
serves `/healthz` and `/docs` with an empty collection.

### 1.4 Bootstrap the first admin key (once)
The key store starts empty. Mint the first **admin** key from a trusted shell:
```bash
docker compose exec api python -m api.auth     # prints the raw admin key ONCE
```
Store it in a password manager. Use it to issue tenant keys (§4).

### 1.5 Stop / logs
```bash
docker compose logs -f api
docker compose down           # keep volumes (certs, key DB, vectors)
docker compose down -v        # ALSO wipe volumes (destroys Qdrant data + certs)
```

---

## 2. Local bring-up (no Docker — WSL venv, for development)

The repo lives in WSL with a pre-built venv (Python 3.12.13 — **do not
recreate**). Run the API directly:

```bash
# 1) Qdrant only, via Docker:
docker run -d -p 6333:6333 -p 6334:6334 \
    -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.12.4

# 2) API (reads .env; QDRANT_URL defaults to http://localhost:6333):
cd /home/user1/lawbot && .venv/bin/python -m uvicorn api.main:app \
    --host 0.0.0.0 --port 8000

# 3) admin key (once):
cd /home/user1/lawbot && .venv/bin/python -m api.auth
```

For a single instance, set `RATE_LIMIT_STORAGE_URI=memory://` in `.env` to skip
Redis.

---

## 3. Initialize git and push (first time)

The project is not yet a git repo. From the project root:

```bash
cd /home/user1/lawbot
git init
git add -A                       # .gitignore already excludes .env/.venv/artifacts/*.db
git status                       # CONFIRM no .env, *.db, or artifacts/ are staged
git commit -m "lawbot: initial production build"

# Create an EMPTY private repo on GitHub first (web UI or `gh repo create`), then:
git branch -M main
git remote add origin git@github.com:<you>/lawbot.git
git push -u origin main
```

> **Before pushing, double-check** `git status` shows no `.env`, `lawbot_keys.db`,
> or `artifacts/`. If any secret was ever staged, rotate the key immediately.

Pushing to `main` triggers `.github/workflows/deploy.yml` (CI: unit tests +
header gate, `docker compose config`, image build; deploy hook if configured).

---

## 4. External-key onboarding (issue keys to tenants)

The service is multi-tenant. Tenant keys are minted by an **admin** key against
the live API (or with the admin key directly for your own testing):

```bash
ADMIN=lk_...   # the admin key from §1.4 / §2

# Issue a tenant key (plaintext returned ONCE — give it to the tenant securely):
curl -fsS -X POST https://api.lawbot.example/v1/keys \
  -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"tenant":"acme-law","tier":"pro","rate":"120/minute"}' | jq
# -> {"key":"lk_...","tenant":"acme-law","tier":"pro","rate":"120/minute"}

# List keys (no plaintext is ever returned):
curl -fsS https://api.lawbot.example/v1/keys -H "Authorization: Bearer $ADMIN" | jq

# Revoke:
curl -fsS -X DELETE https://api.lawbot.example/v1/keys/<key_id> \
  -H "Authorization: Bearer $ADMIN"
```

A tenant then calls the API with `Authorization: Bearer <their key>`:
```bash
curl -fsS -X POST https://api.lawbot.example/v1/statutes/search \
  -H "Authorization: Bearer $TENANT_KEY" -H "Content-Type: application/json" \
  -d '{"query":"주택임대차 계약갱신요구권 행사 기간","k":8}' | jq
```
Tenants can also self-serve from `GET /console`. Missing/invalid key → **401**;
rate-limit exceeded → **429** (with `Retry-After`).

---

## 5. Cloud deploy

You need **three external accounts**; this is the onboarding order:

| # | Service | Why | Free tier |
|---|---------|-----|-----------|
| 1 | **OpenAI** (platform.openai.com) | embeddings + answers | pay-as-you-go; set a hard monthly limit |
| 2 | **Qdrant Cloud** (cloud.qdrant.io) | managed vector store | 1 GB free cluster (enough for the MVP) |
| 3 | **Render** *or* **Fly.io** | host the API container | both have small free/starter tiers |

### 5.1 Qdrant Cloud (the vector store for any cloud host)
1. Create a free cluster at <https://cloud.qdrant.io>.
2. Copy the **cluster URL** (e.g. `https://xxxx.aws.cloud.qdrant.io:6333`) and an
   **API key**.
3. Upload your vectors to it (run the embed/upsert pipeline with
   `QDRANT_URL`/`QDRANT_API_KEY` pointed at the cluster — owned by the
   embed builder; see the playbook Phase 2).
4. These become the `QDRANT_URL` + `QDRANT_API_KEY` secrets below.

### 5.2 Option A — Render (blueprint: `render.yaml`)
1. Push the repo to GitHub (§3).
2. Render → **New → Blueprint** → point at the repo. It reads `render.yaml`
   (Docker web service + managed Key Value/Redis + a 1 GB disk for the SQLite
   key store; HTTPS is automatic on `*.onrender.com`).
3. In the Render dashboard, set the **secret** env vars (marked `sync: false`):
   `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`.
4. Deploy. Then bootstrap the admin key once via the Render **Shell**:
   `python -m api.auth`.
5. Verify: `curl -fsS https://<app>.onrender.com/healthz`.

### 5.3 Option B — Fly.io (config: `fly.toml`)
```bash
flyctl launch --no-deploy --copy-config       # creates the app from fly.toml
flyctl volumes create lawbot_data --size 1 --region nrt
flyctl secrets set \
  OPENAI_API_KEY=sk-... \
  QDRANT_URL=https://xxxx.aws.cloud.qdrant.io:6333 \
  QDRANT_API_KEY=<qdrant-cloud-key>
# optional shared rate-limit backend (else single-machine in-memory):
#   flyctl secrets set RATE_LIMIT_STORAGE_URI=redis://default:...@<upstash-host>:6379
flyctl deploy
flyctl ssh console -C "python -m api.auth"     # admin key once
curl -fsS https://<app>.fly.dev/healthz
```

### 5.4 Option C — Self-hosted (compose + Caddy, your own VM)
Use §1 on a VM with a public IP and a real domain:
```bash
# .env:  DOMAIN=api.yourdomain.com   ACME_EMAIL=you@example.com
#        QDRANT_URL=...  QDRANT_API_KEY=...  (or keep the bundled qdrant service)
# Point an A/AAAA DNS record at the VM, open ports 80/443, then:
docker compose up -d --build
```
Caddy obtains and renews Let's Encrypt certs automatically. **Back up the
`caddy_data` and `api_data` volumes** (certs + key DB).

---

## 6. Configuration reference (env vars)

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| `OPENAI_API_KEY` | **yes** | — | embeddings + chat. Set an OpenAI hard monthly limit. |
| `QDRANT_URL` | yes (cloud) | `http://localhost:6333` | Qdrant Cloud cluster URL in prod. |
| `QDRANT_API_KEY` | cloud only | — | omit for local Docker Qdrant. |
| `COLLECTION` | no | `lawbot` | must match the embedded collection. |
| `EMBED_MODEL` / `EMBED_DIM` | no | `text-embedding-3-small` / `1536` | **must agree with the collection**; changing them = full re-embed. |
| `GEN_MODEL` | no | `gpt-4o-mini` | answer model. |
| `RATE_LIMIT_STORAGE_URI` | no | `memory://` | use `redis://…` across replicas. |
| `API_KEYS_DB` | no | `./lawbot_keys.db` | put on a persisted volume in prod. |
| `DOMAIN` / `ACME_EMAIL` | compose only | `localhost` / — | Caddy TLS. |
| `LAW_OC` | no | — | law.go.kr OC for the `/v1/verify` Citation Firewall. |

All knobs are centralized in `config.py`; nothing else needs editing to switch
models or stores.

---

## 7. Evaluation (golden set)

Quantify retrieval/answer quality and watch for regressions. **Default run is
retrieval-only — no LLM cost.** The LLM answer pass is opt-in and capped.

```bash
# Retrieval baseline against a running API (cheap):
cd /home/user1/lawbot && .venv/bin/python -m eval.run_eval \
    --mode http --base-url http://localhost:8000 --api-key "$LAWBOT_API_KEY"

# Add a small LLM answer + citation-firewall pass (≤3 questions, a few cents):
... --ask --ask-limit 3

# In-process (no HTTP server), retrieval only — for CI smoke / debugging:
cd /home/user1/lawbot && .venv/bin/python -m eval.run_eval --mode direct
```

Reports **Hit@K, MRR@K, Article-hit@K** and (with `--ask`) **citation accuracy**
+ **grounding rate** (≥1 verified citation or an explicit "근거 불충분" — the
anti-hallucination signal; Stanford legal-RAG baselines hallucinate 17–33%, so
grounding should be far higher). Use the scorecard to decide `small↔large`,
`gpt-4o-mini↔gpt-4o`, and whether to enable hybrid retrieval. `eval/golden_set.jsonl`
holds 20 lawyer-style questions across all four corpora; extend it as coverage grows.

---

## 8. Operations & safety

- **Cost guard:** OpenAI hard monthly limit on; full-corpus embedding is never
  auto-run (estimate + human approval gate in the embed pipeline); demo embeds
  ≤ 20k chunks; content-hash cache prevents re-embedding/re-billing.
- **Health/scaling:** `/healthz` drives platform + container health checks.
  Scale by adding replicas (keep `RATE_LIMIT_STORAGE_URI=redis://…` so limits are
  shared). The API process is single-uvicorn; the platform/Caddy load-balances.
- **Backups:** the SQLite key store (`API_KEYS_DB` on a volume) and Caddy certs.
  Qdrant Cloud is managed; for self-hosted Qdrant, back up `qdrant_storage`.
- **Key rotation:** revoke via `DELETE /v1/keys/{id}`; re-issue. If `.env` or a
  key leaks, rotate the OpenAI/Qdrant keys in the provider dashboards.
- **Compliance:** every `/ask` + `/ad-review` response carries `disclaimer` +
  `ai_generated:true` (AI Basic Act). Do not log user PII or persist query
  bodies with personal data. Coverage gaps are disclosed in the answer
  disclaimer.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `docker compose config` errors on `OPENAI_API_KEY is required` | `.env` missing/empty — `cp .env.example .env` and fill it. |
| `/healthz` shows `"qdrant": false` | Qdrant not reachable — check `QDRANT_URL`/cluster, or `docker compose ps qdrant`. |
| `503` from `/v1/ask` etc. | that backend module isn't wired yet (e.g. empty collection) — finish the embed/upsert phase. |
| `401` on every call | missing/invalid `Authorization: Bearer <key>` — issue a key (§4). |
| `429` | rate limit hit — back off per `Retry-After`, or raise the key's `rate`. |
| eval `--mode http` says "needs an API key" | export `LAWBOT_API_KEY` or pass `--api-key`. |
| Caddy can't get a cert | ports 80/443 not reachable, or DNS not pointing at the host; check `docker compose logs caddy`. |
