"""Inspect and clear the OpenAI Batch queue (Tier-1 enqueue-limit hygiene).

The org's Batch *enqueued* limits (Tier 1 = 3,000,000 tokens / 1,000,000
requests) are a point-in-time sum over all **non-terminal** batches. Batches
that fail/expire can leave their requests counted against the quota ("stuck
queue" — see embed/embed_waves.py docstring), so a fresh wave of only a few
thousand requests gets rejected with ``request_limit_exceeded`` /
``token_limit_exceeded`` even though it is tiny.

This utility lists every batch and the sum still occupying the queue, and can
cancel all non-terminal batches so a clean wave run can proceed.

    cd /home/user1/lawbot && .venv/bin/python -m embed.batch_queue status
    cd /home/user1/lawbot && .venv/bin/python -m embed.batch_queue clear   # cancels all pending

``status`` is read-only. ``clear`` only issues cancels (never deletes data) and
prints what it cancelled. After ``clear`` the freed quota may take up to ~24h to
fully release if OpenAI's counter is stuck; ``status`` will show 0 non-terminal
batches immediately, which is the signal a wave run can be retried.
"""

from __future__ import annotations

import sys

import config
from embed.embed_client import _client

# Batch statuses that still occupy the enqueue quota.
_NON_TERMINAL = {"validating", "in_progress", "finalizing", "cancelling"}
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _iter_batches(cli):
    """Yield every batch in the org, newest first, across pagination."""
    after = None
    while True:
        page = cli.batches.list(limit=100, after=after) if after else cli.batches.list(limit=100)
        data = list(page.data)
        if not data:
            return
        for b in data:
            yield b
        if not getattr(page, "has_more", False):
            return
        after = data[-1].id


def _counts(b):
    """Return (total, completed, failed) request counts for a batch."""
    rc = getattr(b, "request_counts", None)
    if rc is None:
        return (0, 0, 0)
    return (getattr(rc, "total", 0) or 0,
            getattr(rc, "completed", 0) or 0,
            getattr(rc, "failed", 0) or 0)


def status() -> int:
    """Print every batch with status + request counts; sum the live queue."""
    cli = _client()
    by_status: dict[str, int] = {}
    pending_reqs = 0
    pending = []
    n = 0
    print(f"{'batch_id':<40} {'status':<12} {'total':>8} {'done':>8} {'fail':>6}")
    print("-" * 78)
    for b in _iter_batches(cli):
        n += 1
        total, done, fail = _counts(b)
        by_status[b.status] = by_status.get(b.status, 0) + 1
        print(f"{b.id:<40} {b.status:<12} {total:>8} {done:>8} {fail:>6}")
        if b.status in _NON_TERMINAL:
            pending_reqs += total
            pending.append(b.id)
    print("-" * 78)
    print(f"total batches: {n}")
    print(f"by status: {by_status}")
    print(f"NON-TERMINAL (occupying queue): {len(pending)} batches, "
          f"~{pending_reqs:,} requests enqueued")
    if pending:
        print("  -> run 'clear' to cancel these and free the queue:")
        for bid in pending:
            print(f"     {bid}")
    else:
        print("  -> queue is clean; a wave run can be started.")
    return 0


def clear() -> int:
    """Cancel every non-terminal batch so the enqueue quota can be reclaimed."""
    cli = _client()
    cancelled = 0
    for b in _iter_batches(cli):
        if b.status in _NON_TERMINAL:
            try:
                cli.batches.cancel(b.id)
                print(f"cancel -> {b.id} (was {b.status})", flush=True)
                cancelled += 1
            except Exception as exc:
                print(f"cancel FAILED {b.id}: {type(exc).__name__}: {exc}", flush=True)
    print(f"requested cancel on {cancelled} batch(es). "
          f"Quota may take up to ~24h to fully release if the counter is stuck.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        return status()
    if cmd == "clear":
        return clear()
    print(f"usage: python -m embed.batch_queue [status|clear]\n  got: {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
