#!/usr/bin/env bash
# ---------------------------------------------------------------------------- #
# lawbot — deploy helper: sync code + serving artifacts to an NCP (or any SSH)
# server, then bring up the Docker Compose stack there.
#
# Usage:
#   scripts/deploy_ncp.sh user@<server-ip> [--sync-only] [--remote-dir DIR]
#
#   --sync-only     transfer files only; do NOT run docker compose on the server
#   --remote-dir    target dir on the server (default: /opt/lawbot)
#
# What it transfers:
#   * source code  (excludes artifacts/ .venv/ .git/ *.db — small)
#   * artifacts/full_index/  (index.faiss + meta.jsonl + bm25.sqlite, ~8.8GB)
#   * artifacts/parents.jsonl (Citation Firewall DB-existence, ~3.3GB)
# Raw embeddings/chunks are NOT shipped (not needed to serve).
#
# Prereqs: ssh access to the server, rsync on both ends, Docker on the server.
# Secrets: never transferred. Create /opt/lawbot/.env on the SERVER (see DEPLOY).
# ---------------------------------------------------------------------------- #
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_DIR="/opt/lawbot"
SYNC_ONLY=0
SERVER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --sync-only)  SYNC_ONLY=1; shift ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      if [ -z "$SERVER" ]; then SERVER="$1"; shift
      else echo "unknown arg: $1" >&2; exit 2; fi ;;
  esac
done

if [ -z "$SERVER" ]; then
  echo "usage: scripts/deploy_ncp.sh user@<server-ip> [--sync-only] [--remote-dir DIR]" >&2
  exit 2
fi

ART="$LOCAL_ROOT/artifacts"
for f in "full_index/index.faiss" "full_index/meta.jsonl" "full_index/bm25.sqlite" "parents.jsonl"; do
  if [ ! -e "$ART/$f" ]; then
    echo "ERROR: missing serving artifact: artifacts/$f" >&2
    echo "Build it on this machine first (embed/build_full_index.py, build_bm25.py)." >&2
    exit 1
  fi
done

echo "==> target: $SERVER:$REMOTE_DIR"
ssh "$SERVER" "mkdir -p '$REMOTE_DIR/artifacts'"

echo "==> [1/3] sync code (small; excludes artifacts/.venv/.git/*.db)"
rsync -az --delete \
  --exclude 'artifacts' --exclude '.venv' --exclude '.git' \
  --exclude '*.db' --exclude '__pycache__' --exclude '.pytest_cache' \
  "$LOCAL_ROOT/" "$SERVER:$REMOTE_DIR/"

echo "==> [2/3] sync FAISS + BM25 index (~8.8GB; resumable)"
rsync -az --info=progress2 \
  "$ART/full_index" "$SERVER:$REMOTE_DIR/artifacts/"

echo "==> [3/3] sync parents.jsonl (~3.3GB; Citation Firewall)"
rsync -az --info=progress2 \
  "$ART/parents.jsonl" "$SERVER:$REMOTE_DIR/artifacts/"

echo "==> transfer complete."

if [ "$SYNC_ONLY" -eq 1 ]; then
  cat <<EOF

Files synced. Next, on the server:
  ssh $SERVER
  cd $REMOTE_DIR
  cp .env.example .env && nano .env     # set OPENAI_API_KEY, DOMAIN, etc. (see DEPLOY)
  docker compose up -d --build
  curl -fsS http://127.0.0.1:80/healthz   # or https://<domain>/healthz
EOF
  exit 0
fi

if ssh "$SERVER" "test -f '$REMOTE_DIR/.env'"; then
  echo "==> .env present on server — bringing up the stack"
  ssh "$SERVER" "cd '$REMOTE_DIR' && docker compose up -d --build && docker compose ps"
  echo "==> done. Verify: curl -fsS http://<server>/healthz  (or https://<domain>/healthz)"
else
  cat <<EOF

==> NOTE: $REMOTE_DIR/.env does NOT exist on the server (secrets are never
    transferred). Create it, then bring the stack up:
  ssh $SERVER
  cd $REMOTE_DIR
  cp .env.example .env && nano .env       # OPENAI_API_KEY, DOMAIN, LAW_OC, ...
  docker compose up -d --build
EOF
fi
