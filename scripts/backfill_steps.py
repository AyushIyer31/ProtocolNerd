#!/usr/bin/env python3
"""
Backfill step data for protocols already cached without steps.

Fetches each protocol individually via GET /protocols/<id> to get its full step
content, extracts readable plain-text steps, and writes them back into the
cached JSON file.

The individual-protocol response nests the protocol under the top-level
`protocol` key (NOT `payload`), and each step's instruction text lives in the
step-level `step` field (HTML). type_id=6 components are section headers, not
instructions, so they are skipped.

Runs concurrently with a global rate limiter that keeps request dispatch under
protocols.io's 100 req/min cap (sequential is latency-bound and ~3x slower).

Usage:
    python backfill_steps.py
    python backfill_steps.py --limit 100        # only first 100 missing steps
    python backfill_steps.py --workers 8        # concurrency (default 8)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "protocolnerd-backend" / "variables.env", override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

API_BASE = "https://www.protocols.io/api/v3"
MIN_DISPATCH_INTERVAL = 0.66  # seconds between request starts → ~90 req/min, under the 100/min cap
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Global rate limiter shared across worker threads.
_rate_lock = threading.Lock()
_last_dispatch = [0.0]


def _throttle() -> None:
    """Block until at least MIN_DISPATCH_INTERVAL has passed since the last dispatch."""
    with _rate_lock:
        now = time.monotonic()
        wait = MIN_DISPATCH_INTERVAL - (now - _last_dispatch[0])
        if wait > 0:
            time.sleep(wait)
        _last_dispatch[0] = time.monotonic()


def _get_token() -> str:
    token = os.getenv("PROTOCOLS_IO_TOKEN", "").strip().strip('"')
    if not token:
        raise SystemExit("PROTOCOLS_IO_TOKEN not set in protocolnerd-backend/variables.env")
    return token


def _api_get(path: str, token: str, retries: int = 4) -> Dict[str, Any]:
    """GET with retries on transient failures (timeouts/5xx/connection resets).
    Each attempt passes through the global rate limiter."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": BROWSER_UA,
        },
    )
    for attempt in range(1, retries + 1):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                log.warning(f"HTTP {e.code} for {url}")
                return {}  # genuine client error — don't retry
        except Exception:
            pass
        if attempt < retries:
            time.sleep(min(2 ** attempt, 20))
    return {}


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def _extract_steps(proto: Dict[str, Any]) -> List[str]:
    """
    Extract readable plain-text steps from a full protocol object.

    Prefers the step-level `step` field (the instruction HTML). Falls back to
    description components, skipping type_id=6 section headers.
    """
    steps: List[str] = []
    for s in (proto.get("steps") or []):
        text = _strip_html(s.get("step") or "")
        if not text:
            parts = []
            for comp in (s.get("components") or []):
                if comp.get("type_id") == 6:  # section header, not an instruction
                    continue
                source = comp.get("source") or {}
                t = _strip_html(
                    source.get("description")
                    or source.get("body")
                    or source.get("title")
                    or ""
                )
                if t and t.lower() not in {"note", "warning", "tip"}:
                    parts.append(t)
            text = " ".join(parts).strip()
        if text:
            steps.append(text)
    return steps


def _process_one(fpath: Path, cached: Dict[str, Any], token: str) -> str:
    """Fetch + extract steps for one protocol, writing back on success.
    Returns one of: 'updated', 'no_steps', 'failed'."""
    pid = cached.get("id")
    if not pid:
        return "failed"

    data = _api_get(f"/protocols/{pid}", token)
    proto = data.get("protocol") or (data if data.get("id") else None)
    if not proto:
        return "failed"

    steps = _extract_steps(proto)
    if not steps:
        return "no_steps"

    cached["steps"] = steps
    fpath.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    return "updated"


def backfill(data_dir: Path, limit: Optional[int], token: str, workers: int) -> None:
    files = sorted(data_dir.glob("*.json"))

    # Only process files that currently have no step text.
    to_update = []
    for f in files:
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
            if not any(s.strip() for s in (p.get("steps") or []) if isinstance(s, str)):
                to_update.append((f, p))
        except Exception:
            pass

    if limit:
        to_update = to_update[:limit]

    total = len(to_update)
    log.info(f"{total} protocols need step backfill (out of {len(files)} total); workers={workers}")
    est_min = total * MIN_DISPATCH_INTERVAL / 60
    log.info(f"Estimated time at ~{60/MIN_DISPATCH_INTERVAL:.0f} req/min: ~{est_min:.0f} min")

    counts = {"updated": 0, "no_steps": 0, "failed": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, fpath, cached, token): cached.get("id")
            for fpath, cached in to_update
        }
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                log.warning(f"worker error: {e}")
                result = "failed"
            counts[result] += 1
            done += 1
            if done % 200 == 0 or done == total:
                log.info(
                    f"  [{done}/{total}] updated={counts['updated']} "
                    f"no_steps={counts['no_steps']} failed={counts['failed']}"
                )

    log.info(
        f"\nDone. {counts['updated']} updated with steps, "
        f"{counts['no_steps']} had no step data, {counts['failed']} failed."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/protocols"))
    parser.add_argument("--limit", type=int, default=None, help="Only process this many protocols")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers (default 8)")
    args = parser.parse_args()

    token = _get_token()
    backfill(args.data_dir, args.limit, token, args.workers)


if __name__ == "__main__":
    main()
