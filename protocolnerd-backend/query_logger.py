"""
Server-side query logging — a durable record of what users searched and what they got.

Writes ONE self-contained JSON object per line (JSON Lines) to a per-day file:

    {QUERY_LOG_DIR}/query_results_YYYY-MM-DD.log

JSON Lines rather than a single JSON array on purpose: an array can't be appended to
safely (you'd have to rewrite the whole file before the closing `]`), whereas one object
per line is append-only, crash-safe, and streamable with standard tools:

    cat query_results_2026-07-20.log | jq .
    jq -r '.original_query' query_results_*.log

Each record captures the full search: the user's original request, the queries the system
suggested, the ones the user actually ran, and the results in the exact order shown.

Design rules:
  * Never raises. A logging failure must not break a search — every write is guarded.
  * Thread-safe. Searches run on a ThreadPoolExecutor, so appends take a lock.
  * Daily rotation by UTC date. Retention is "keep everything" by default; set
    QUERY_LOG_RETENTION_DAYS to prune files older than N days.

PERSISTENCE WARNING: on the current Fargate task, QUERY_LOG_DIR sits on the container's
EPHEMERAL storage, which is destroyed on every task restart (deploy, crash, scale). To
keep these logs for months, QUERY_LOG_DIR must point at a mounted EFS volume (or the
records must be shipped off-box). See DEPLOYMENT notes.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

ENABLED = os.getenv("QUERY_LOGGING_ENABLED", "1").strip() not in ("0", "false", "False", "")
LOG_DIR = Path(os.getenv("QUERY_LOG_DIR", "storage/query_logs"))
RETENTION_DAYS = int(os.getenv("QUERY_LOG_RETENTION_DAYS", "0") or 0)  # 0 = keep forever

# --- S3 durability -----------------------------------------------------------------
# The local file lives on the container's EPHEMERAL disk (wiped on restart). When a bucket
# is set, each daily file is mirrored to S3 so it survives -- that is what makes the logs
# durable in production. Bucket unset (e.g. local dev) => S3 is a no-op, local file only.
#
# We re-upload the WHOLE current daily file (it's tiny KB) on a debounce, overwriting the
# same object, rather than streaming each line: idempotent, crash-safe, and one PUT/minute
# regardless of traffic. Worst-case loss on a hard task kill is one flush interval of records.
S3_BUCKET = os.getenv("QUERY_LOG_S3_BUCKET", "").strip()
S3_PREFIX = os.getenv("QUERY_LOG_S3_PREFIX", "query_logs/").strip()
S3_FLUSH_SECONDS = int(os.getenv("QUERY_LOG_S3_FLUSH_SECONDS", "60") or 60)

_lock = threading.Lock()
_last_prune_day: Optional[str] = None
_s3_client: Any = None
_s3_broken = False
_last_flush_ts = 0.0
_last_flush_path: Optional[Path] = None


def _s3():
    """Lazy boto3 S3 client. None (once) if boto3/credentials are unavailable."""
    global _s3_client, _s3_broken
    if _s3_broken or not S3_BUCKET:
        return None
    if _s3_client is None:
        try:
            import boto3  # noqa: WPS433 — optional dependency; local dev needs no S3
            _s3_client = boto3.client("s3")
        except Exception as e:  # noqa: BLE001
            log.warning(f"query_logger: S3 disabled ({e}); logging locally only.")
            _s3_broken = True
            return None
    return _s3_client


def _upload(path: Path) -> None:
    c = _s3()
    if c is None or not path.exists():
        return
    try:
        # put_object (single synchronous call) rather than upload_file's threaded transfer
        # manager: the daily file is tiny, and this stays reliable inside the atexit flush,
        # where boto3's thread pool is already gone ("cannot schedule new futures").
        c.put_object(Bucket=S3_BUCKET, Key=S3_PREFIX + path.name,
                     Body=path.read_bytes(), ContentType="application/x-ndjson")
    except Exception as e:  # noqa: BLE001 — S3 must never break the request path
        log.warning(f"query_logger: S3 upload of {path.name} failed ({e}).")


def _maybe_flush_s3(path: Path) -> None:
    """Debounced mirror of the current daily file to S3; finalizes the file on day rollover."""
    global _last_flush_ts, _last_flush_path
    if not S3_BUCKET:
        return
    now = time.time()
    if _last_flush_path is not None and _last_flush_path != path:
        _upload(_last_flush_path)                      # push yesterday's completed file
        _last_flush_path, _last_flush_ts = None, 0.0
    if _last_flush_path is None or (now - _last_flush_ts) >= S3_FLUSH_SECONDS:
        _upload(path)
        _last_flush_ts, _last_flush_path = now, path


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _daily_path() -> Path:
    return LOG_DIR / f"query_results_{_today()}.log"


def _slim_results(results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Keep enough to reconstruct what the user saw, in order — not full descriptions."""
    out = []
    for rank, r in enumerate(results or [], 1):
        out.append({
            "rank": rank,
            "id": r.get("id"),
            "source": r.get("source") or "protocols.io",
            "title": r.get("title") or "",
            "url": r.get("url") or r.get("uri") or "",
        })
    return out


def _prune_old_logs() -> None:
    """Delete daily files older than RETENTION_DAYS. Runs at most once per day."""
    global _last_prune_day
    if RETENTION_DAYS <= 0:
        return
    today = _today()
    if _last_prune_day == today:
        return
    _last_prune_day = today
    cutoff = datetime.now(timezone.utc).timestamp() - RETENTION_DAYS * 86400
    for f in LOG_DIR.glob("query_results_*.log"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                log.info(f"query_logger: pruned {f.name} (older than {RETENTION_DAYS}d)")
        except OSError:
            pass


def log_event(event: str, *, session_id: Optional[str], **fields: Any) -> None:
    """Append one JSON-Lines record for a single interaction event. Never raises.

    A search is a multi-step interaction, so it produces several correlated records
    (join them by session_id), written the moment each step happens:

      event="new_query"      the user starts a fresh search — logged the instant a new
                             query is detected, before any clarification or search, so it
                             is captured even if the user then abandons.
      event="clarification"  the user answered a clarification: the question that was
                             asked, its options, and the option they selected.
      event="suggestions"    the system proposed candidate search queries.
      event="search"         the search ran: selected queries, PubMed queries, and the
                             results in the exact order shown.
    """
    if not ENABLED:
        return
    try:
        record = {"ts": datetime.now(timezone.utc).isoformat(),
                  "event": event, "session_id": session_id}
        if "results" in fields:                       # slim results, keep order
            fields["result_count"] = len(fields.get("results") or [])
            fields["results"] = _slim_results(fields.get("results"))
        record.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = _daily_path()
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            _prune_old_logs()
            _maybe_flush_s3(path)
    except Exception as e:  # noqa: BLE001 — logging must never break the request path
        log.warning(f"query_logger: failed to record '{event}' ({e})")


@atexit.register
def _flush_on_exit() -> None:
    """Final S3 push on graceful shutdown so the last debounce window isn't lost."""
    if S3_BUCKET:
        with _lock:
            _upload(_daily_path())


# --- Read side (for the debug log viewer) ------------------------------------------
_NAME_RE = re.compile(r"query_results_(\d{4}-\d{2}-\d{2})\.log$")


def _date_from_name(name: str) -> Optional[str]:
    m = _NAME_RE.search(name)
    return m.group(1) if m else None


def list_dates() -> List[str]:
    """Available log dates (YYYY-MM-DD), newest first. Unions S3 (durable history) with
    any local files (the current day, freshest)."""
    dates = set()
    c = _s3()
    if c is not None:
        try:
            paginator = c.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
                for obj in page.get("Contents", []):
                    d = _date_from_name(obj["Key"].rsplit("/", 1)[-1])
                    if d:
                        dates.add(d)
        except Exception as e:  # noqa: BLE001
            log.warning(f"query_logger: S3 list failed ({e}).")
    if LOG_DIR.exists():
        for f in LOG_DIR.glob("query_results_*.log"):
            d = _date_from_name(f.name)
            if d:
                dates.add(d)
    return sorted(dates, reverse=True)


def read_records(date: str) -> List[Dict[str, Any]]:
    """Parsed JSON records for one day, in write order. Reads S3 (durable, complete across
    restarts) when configured, else the local file. Malformed lines are skipped."""
    if not _NAME_RE.search(f"query_results_{date}.log"):
        return []                                    # reject anything not a plain date
    text = ""
    c = _s3()
    if c is not None:
        try:
            obj = c.get_object(Bucket=S3_BUCKET, Key=S3_PREFIX + f"query_results_{date}.log")
            text = obj["Body"].read().decode("utf-8")
        except Exception:  # noqa: BLE001 — missing object is normal
            text = ""
    if not text:
        path = LOG_DIR / f"query_results_{date}.log"
        if path.exists():
            text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out
