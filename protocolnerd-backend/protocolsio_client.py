"""
Live protocols.io search client.

Empirical findings about the /api/v3/protocols `key` parameter (see
probe_protocolsio_syntax.py and data/protocolsio_syntax_probe.json):

  * `page_id` is 0-indexed — the first page is page_id=0, NOT 1.
  * The `key` parameter is a PHRASE / adjacency match, not a boolean engine:
      - Single keywords work well:        rice -> 209,  CRISPR -> 346
      - Multi-word strings only match when the words occur as an adjacent
        phrase in the protocol text:      "in planta" -> 10, "gene editing" -> 36
        but "drought tolerance" -> 0,     "rice drought" -> 0
      - Quotes / OR / parentheses are NOT operators — they are matched
        literally, so '"drought tolerance"' and 'drought OR rice' both -> 0.

Consequence: the right way to use this API is to fire MANY short (1-2 word)
real keyword/phrase probes and merge + re-rank the results client-side, rather
than sending one long natural-language query (which almost always returns 0).
That merge/re-rank lives in protocol_ranker.py; the term generation that feeds
the probes lives in concept_expansion.py.
"""
from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import certifi
from dotenv import load_dotenv

from protocol_rag import _draftjs_to_text

log = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent / "variables.env", override=False)

API_BASE = "https://www.protocols.io/api/v3"
RATE_LIMIT_DELAY = 0.65  # ~92 req/min, under the 100/min limit
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _token() -> str:
    return os.getenv("PROTOCOLS_IO_TOKEN", "").strip().strip('"')


def _api_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Accept": "application/json",
            "User-Agent": BROWSER_UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20, context=SSL_CONTEXT) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        log.warning(f"protocols.io HTTP {e.code} for {url}: {e.read().decode('utf-8','ignore')[:160]}")
        return {}
    except Exception as e:
        log.warning(f"protocols.io request failed for {url}: {e}")
        return {}


def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a raw API protocol object to the fields the ranker needs."""
    authors_raw = raw.get("authors") or []
    authors = [
        (a.get("name") or f"{a.get('fname','')} {a.get('lname','')}".strip())
        for a in authors_raw
        if isinstance(a, dict)
    ]
    # Flatten the first few step descriptions into plain strings for previews
    steps_preview: list = []
    for s in (raw.get("steps") or [])[:3]:
        parts = [
            (c.get("description") or c.get("title") or c.get("body") or "")
            for c in (s.get("components") or [])
        ]
        text = " ".join(p for p in parts if p).strip() or str(s.get("description") or "").strip()
        if text:
            steps_preview.append(text)

    # Build the most reliable public link. The DOI always resolves to the
    # published page; the /view/<slug> can 404 (unpublished working copy), and
    # the API's own `url` field is the same fragile slug link. Prefer DOI.
    doi = (raw.get("doi") or "").strip()
    slug = raw.get("uri") or ""
    if doi:
        link = doi if doi.startswith("http") else f"https://{doi}"
    elif raw.get("url"):
        link = raw["url"]
    elif slug:
        link = f"https://www.protocols.io/view/{slug}"
    else:
        link = ""

    return {
        "id": raw.get("id"),
        "title": raw.get("title") or "",
        "uri": slug,           # raw slug (frontends may prepend the /view/ base)
        "url": link,           # ready-to-click link, DOI-preferred
        "doi": doi,
        "created_on": raw.get("created_on"),
        "published_on": raw.get("published_on"),
        "updated_on": raw.get("updated_on"),
        "step_count": len(raw.get("steps") or []),
        "stats": raw.get("stats") or {},
        "description": _draftjs_to_text(raw.get("description"))[:600],
        # materials_text and steps_preview are consumed by explain_matches(); keep
        # them present (even if empty) so the explanation path never KeyErrors.
        "materials_text": _draftjs_to_text(raw.get("materials_text"))[:300],
        "steps_preview": steps_preview,
        "keywords": [
            (kw.get("name") if isinstance(kw, dict) else kw)
            for kw in (raw.get("keywords") or [])
        ],
        "authors": [a for a in authors if a],
    }


def search_live(key: str, max_results: int = 10) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Search the live protocols.io API for a single keyword/phrase probe.

    Returns (total_results_reported_by_api, normalized_protocols).
    Uses page_id=0 (the API's true first page).
    """
    if not _token():
        raise RuntimeError("No PROTOCOLS_IO_TOKEN in variables.env")

    data = _api_get(
        "/protocols",
        {
            "filter": "public",
            "key": key,
            "order_field": "activity",
            "order_dir": "desc",
            "page_id": 0,
            "page_size": min(max_results, 50),
        },
    )
    pagination = data.get("pagination", {}) or {}
    total = pagination.get("total_results", 0) or 0
    items = data.get("items", []) or []
    return total, [_normalize(it) for it in items[:max_results]]


def multi_probe_search(
    probes: List[str],
    per_probe: int = 10,
    cap: int = 80,
    max_workers: int = 8,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[str]], Dict[str, int]]:
    """
    Fire many short probes IN PARALLEL and merge the results by protocol id.

    Returns:
      merged       — {protocol_id: protocol_dict}
      hit_map      — {protocol_id: [probes that returned this protocol]}
      probe_totals — {probe: total_results reported by the API}

    The probes are independent GET requests, so we run them concurrently on a
    small thread pool (a handful of requests stays well under protocols.io's
    ~100 req/min limit) — this cuts a ~15-20s sequential search to a few seconds.
    Merging then iterates probes in their original order so tie-breaking and
    hit_map ordering stay deterministic regardless of completion order.
    """
    probes = [p for p in probes if p.strip()]
    merged: Dict[int, Dict[str, Any]] = {}
    hit_map: Dict[int, List[str]] = {}
    probe_totals: Dict[str, int] = {}

    # Fire all probes concurrently; collect (total, results) per probe.
    per_probe_results: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(search_live, probe, per_probe): probe for probe in probes}
        for fut in as_completed(futures):
            probe = futures[fut]
            try:
                per_probe_results[probe] = fut.result()
            except Exception as e:
                log.warning(f"probe {probe!r} failed: {e}")
                per_probe_results[probe] = (0, [])

    # Merge in original probe order for deterministic tie-breaking.
    for probe in probes:
        total, results = per_probe_results.get(probe, (0, []))
        probe_totals[probe] = total
        for p in results:
            pid = p.get("id")
            if pid is None:
                continue
            if pid not in merged:
                if len(merged) >= cap:
                    continue
                merged[pid] = p
                hit_map[pid] = []
            if probe not in hit_map[pid]:
                hit_map[pid].append(probe)

    return merged, hit_map, probe_totals
