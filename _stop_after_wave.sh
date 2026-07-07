#!/usr/bin/env bash
# 현재 진행 중인 웨이브가 끝나면(embeddings.jsonl 줄 수가 늘면) embed_waves를 멈춘다.
cd /home/user1/lawbot || exit 1
base=$(wc -l < artifacts/embeddings.jsonl)
echo "watch baseline=$base"
for i in $(seq 1 360); do
  sleep 20
  cur=$(wc -l < artifacts/embeddings.jsonl 2>/dev/null || echo "$base")
  if [ "$cur" -gt "$base" ]; then
    pkill -f embed.embed_waves
    sleep 2
    echo "STOPPED after current wave: $cur lines (was $base)"
    exit 0
  fi
  if ! pgrep -f embed.embed_waves >/dev/null; then
    echo "process already gone at $cur lines"
    exit 0
  fi
done
echo "watcher timed out (120min)"
exit 0
