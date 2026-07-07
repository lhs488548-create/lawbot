#!/usr/bin/env bash
cd /home/user1/lawbot || exit 1
key=$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)
key="${key%\"}"; key="${key#\"}"; key="${key//[[:space:]]/}"
bid=$(grep -o 'batch_[a-z0-9]*' artifacts/waves.log | tail -1)
curl -sS -H "Authorization: Bearer $key" "https://api.openai.com/v1/batches/$bid" \
 | .venv/bin/python -c "import sys,json,time; d=json.load(sys.stdin); rc=d.get('request_counts') or {}; t=rc.get('total',0); c=rc.get('completed',0); ip=d.get('in_progress_at'); now=int(time.time()); print('batch %s | %s | %d/%d (%.1f%%) | in_progress_min=%s'%(d.get('id','')[:24], d.get('status'), c, t, (100*c/t if t else 0), round((now-ip)/60,1) if ip else 'NA'))"
