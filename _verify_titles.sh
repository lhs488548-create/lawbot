#!/bin/bash
F=/home/user1/lawbot/artifacts/chunks.jsonl
echo "=== sample 민법 line ==="
grep -m1 "\"title\": \"민법\"" "$F" | head -c 600
echo
echo "=== check article fields present ==="
grep -m1 "유류분" "$F" | head -c 400
echo
