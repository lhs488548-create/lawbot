#!/usr/bin/env bash
set -u
cd ~/lawbot
# 이미 떠 있으면 종료
pkill -f "uvicorn api.main:app" 2>/dev/null || true
sleep 1
nohup .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 > artifacts/uvicorn.log 2>&1 &
echo "  서버 기동 (PID $!)"
for i in $(seq 1 25); do
  sleep 1
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/docs 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "  준비됨 (${i}s)"; break; fi
done
echo "/docs    -> $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/docs)"
echo "/chat    -> $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/chat)"
echo "/console -> $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/console)"
echo "/openapi -> $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/openapi.json)"
