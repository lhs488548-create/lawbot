"""Wave-based OpenAI Batch embedding driver (Tier-1 friendly, resumable).

The org's Batch *enqueued-token* limit (Tier 1 = 3,000,000) rejects submitting
the whole corpus at once. This driver submits ONE bounded wave at a time, polls
it to completion, collects the vectors, then submits the next — so at most one
wave (<= WAVE_BUDGET tokens) is ever enqueued. Runs unattended for days.

Key properties:
* **512-d**: every request passes ``dimensions=512`` (the plain embed_batch.py
  omitted this — its vectors would have been 1536-d). Collected vectors are
  asserted == 512.
* **Resumable**: chunk_ids already present in ``EMBEDDINGS_JSONL`` are skipped on
  restart, so a crash/reboot loses at most the in-flight wave.
* **Memory-safe (~7GB box)**: only the done-id set (~150MB) and one wave of text
  are held; vectors are appended straight to disk, never accumulated in RAM.
* **Batch-discount preserved** (~50% vs sync).

Run (detached, survives the terminal/session)::

    cd /home/user1/lawbot && nohup .venv/bin/python -m embed.embed_waves \
        > artifacts/waves.log 2>&1 &

Test one small wave first::

    cd /home/user1/lawbot && .venv/bin/python -m embed.embed_waves --max-waves 1 --budget 40000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tiktoken

import config
from embed.embed_client import _client, _validate_dim, content_hash

# Stay safely under the Tier-1 enqueued-token ceiling (3,000,000).
WAVE_BUDGET_DEFAULT = 2_600_000
POLL_SEC = 20
_TERMINAL_BAD = {"failed", "expired", "cancelled", "cancelling"}
TOTAL = 1_357_018  # core corpus chunk count (for progress %)

_enc = tiktoken.get_encoding(config.EMBED_ENCODING)
EMB = config.EMBEDDINGS_JSONL
_WAVE_IN = config.ARTIFACTS_DIR / "wave_in.jsonl"
# Enqueue-quota errors that are transient (a just-finished wave's quota has not
# been released yet) — retry rather than abort.
_QUOTA_CODES = {"token_limit_exceeded", "request_limit_exceeded"}
_MAX_QUOTA_RETRIES = 40


def _err_code(b) -> str | None:
    """Extract the first BatchError code from a failed batch, if any."""
    errs = getattr(b, "errors", None)
    data = getattr(errs, "data", None) if errs else None
    if data:
        return getattr(data[0], "code", None)
    return None


def _load_done() -> set[str]:
    """Return the set of chunk_ids already embedded (for resume)."""
    done: set[str] = set()
    if EMB.exists():
        with EMB.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    done.add(json.loads(ln)["chunk_id"])
                except Exception:
                    pass
    return done


def _stream_chunks():
    with config.CHUNKS_JSONL.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _run_wave(cli, wave_no: int, chunks: list[dict], total_done: int) -> int:
    """Submit one wave, poll to completion, append vectors. Return new total_done."""
    # De-dup identical text within the wave (embed once, fan out on collect).
    by_hash: dict[str, str] = {}
    ids_by_hash: dict[str, list[str]] = {}
    for c in chunks:
        h = c.get("content_hash") or content_hash(c["text"])
        by_hash.setdefault(h, c["text"])
        ids_by_hash.setdefault(h, []).append(c["chunk_id"])

    with _WAVE_IN.open("w", encoding="utf-8") as out:
        for h, text in by_hash.items():
            out.write(json.dumps({
                "custom_id": h, "method": "POST", "url": "/v1/embeddings",
                "body": {"model": config.EMBED_MODEL, "input": text,
                         "dimensions": config.EMBED_DIMENSIONS},
            }, ensure_ascii=False) + "\n")

    with _WAVE_IN.open("rb") as f:
        up = cli.files.create(file=f, purpose="batch")

    # Create + poll. One wave is ever enqueued, so a token/request-limit failure
    # here means the previous wave's quota has not been released yet — wait and
    # re-create the batch (reusing the uploaded file) rather than aborting.
    b = None
    quota_retries = 0
    while b is None:
        batch = cli.batches.create(
            input_file_id=up.id, endpoint="/v1/embeddings", completion_window="24h",
            metadata={"project": "lawbot", "wave": str(wave_no)},
        )
        print(f"[wave {wave_no}] submitted {len(chunks)} chunks "
              f"({len(by_hash)} unique) -> {batch.id}", flush=True)
        while True:
            try:
                bb = cli.batches.retrieve(batch.id)
            except Exception as exc:
                print(f"[wave {wave_no}] retrieve error {type(exc).__name__}; retrying", flush=True)
                time.sleep(POLL_SEC)
                continue
            if bb.status == "completed":
                b = bb
                break
            if bb.status in _TERMINAL_BAD:
                code = _err_code(bb)
                if code in _QUOTA_CODES and quota_retries < _MAX_QUOTA_RETRIES:
                    quota_retries += 1
                    wait = min(30 * quota_retries, 300)
                    print(f"[wave {wave_no}] enqueue quota busy ({code}); "
                          f"wait {wait}s, retry {quota_retries}/{_MAX_QUOTA_RETRIES}", flush=True)
                    time.sleep(wait)
                    break  # recreate the batch (reuse uploaded file)
                raise RuntimeError(f"wave {wave_no} ended status={bb.status!r} "
                                   f"errors={getattr(bb, 'errors', None)}")
            time.sleep(POLL_SEC)

    content = cli.files.content(b.output_file_id).text
    hv: dict[str, list[float]] = {}
    for ln in content.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rec = json.loads(ln)
        if rec.get("error"):
            continue
        cid = rec.get("custom_id")
        data = ((rec.get("response") or {}).get("body") or {}).get("data") or []
        if cid and data:
            hv[cid] = data[0]["embedding"]

    with EMB.open("a", encoding="utf-8") as out:
        for h, vec in hv.items():
            _validate_dim(vec)  # asserts == 512
            for cid in ids_by_hash[h]:
                out.write(json.dumps({"chunk_id": cid, "vector": vec}) + "\n")
                total_done += 1
    print(f"[wave {wave_no}] done: {len(hv)} vectors -> total {total_done:,}/{TOTAL:,} "
          f"({100*total_done/TOTAL:.1f}%)", flush=True)
    return total_done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wave-based batch embedding driver.")
    ap.add_argument("--budget", type=int, default=WAVE_BUDGET_DEFAULT,
                    help="max tokens enqueued per wave (default ~2.6M, under the 3M ceiling)")
    ap.add_argument("--max-waves", type=int, default=None, help="stop after N waves (test)")
    args = ap.parse_args(argv)

    done = _load_done()
    total_done = len(done)
    print(f"resume: {total_done:,}/{TOTAL:,} already embedded; budget={args.budget:,} tok/wave",
          flush=True)
    cli = _client()

    wave_no = 0
    buffer: list[dict] = []
    tok = 0
    pending = (c for c in _stream_chunks() if c["chunk_id"] not in done)
    for c in pending:
        n = len(_enc.encode(c["text"]))
        if buffer and tok + n > args.budget:
            wave_no += 1
            total_done = _run_wave(cli, wave_no, buffer, total_done)
            buffer, tok = [], 0
            if args.max_waves and wave_no >= args.max_waves:
                print(f"stopped after {wave_no} wave(s) (--max-waves).", flush=True)
                return 0
        buffer.append(c)
        tok += n
    if buffer:
        wave_no += 1
        total_done = _run_wave(cli, wave_no, buffer, total_done)

    print(f"ALL DONE: {total_done:,} embeddings in {EMB}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
