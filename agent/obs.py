"""Minimal structured logging → stderr as JSON lines (captured by Vercel logs).

Deliberately tiny and stdlib-only. Each record carries a `run_id` so all the events
for one /api/execute request can be grepped/aggregated together. This is the ops-facing
observability channel and is kept SEPARATE from the user-facing `steps[]` trace."""
import json
import sys
import time
import uuid


def new_run_id():
    return uuid.uuid4().hex[:12]


def log(event, run_id=None, **fields):
    """Emit one structured event. Never raises; logging must not break a request."""
    rec = {"ts": round(time.time(), 3), "event": event}
    if run_id:
        rec["run_id"] = run_id
    rec.update(fields)
    try:
        sys.stderr.write(json.dumps(rec, default=str) + "\n")
        sys.stderr.flush()
    except Exception:
        pass
