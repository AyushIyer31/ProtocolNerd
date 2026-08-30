#!/usr/bin/env python3
"""
Empirically probe what search syntax the protocols.io /api/v3/protocols
endpoint accepts in its `key` (keyword) parameter.

We can't read protocols.io's internal query parser, so we test it directly:
for each candidate syntax we send a request and record how many results come
back and the titles of the top hits. By comparing result counts across plain
vs. quoted vs. OR vs. parenthesised queries we can infer what the API honours.

Run:  python probe_protocolsio_syntax.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "variables.env", override=False)

API_BASE = "https://www.protocols.io/api/v3"
TOKEN = os.getenv("PROTOCOLS_IO_TOKEN", "").strip().strip('"')
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def search(key: str, page_size: int = 10) -> dict:
    params = {
        "filter": "public",
        "key": key,
        "order_field": "activity",
        "order_dir": "desc",
        "page_id": 1,
        "page_size": page_size,
    }
    url = f"{API_BASE}/protocols?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
            "User-Agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode("utf-8", errors="ignore")[:200]}
    except Exception as e:
        return {"_error": str(e)}


def summarize(key: str) -> dict:
    data = search(key)
    if "_http_error" in data or "_error" in data:
        return {"key": key, "error": data}
    items = data.get("items", []) or []
    pag = data.get("pagination", {}) or {}
    return {
        "key": key,
        "total_results": pag.get("total_results", pag.get("total_pages")),
        "returned": len(items),
        "top_titles": [(it.get("title") or "")[:70] for it in items[:3]],
    }


# Each tuple: (label, query string). Compare result counts to infer syntax support.
PROBES = [
    ("plain two words", "drought tolerance rice"),
    ("quoted phrase", '"drought tolerance"'),
    ("quoted phrase + word", '"drought tolerance" rice'),
    ("OR uppercase", "drought OR tolerance"),
    ("OR lowercase", "drought or tolerance"),
    ("pipe OR", "drought | tolerance"),
    ("parens + OR", "(drought OR water deficit) rice"),
    ("AND uppercase", "drought AND rice"),
    ("plus required", "+drought +rice"),
    ("single rare word", "Oryza"),
    ("scientific name quoted", '"Oryza sativa"'),
    ("transcription factor multiplex", "multiplex transcription factor mouse"),
    ("in planta gene", "in planta gene editing"),
]


def main():
    if not TOKEN:
        raise SystemExit("No PROTOCOLS_IO_TOKEN found in variables.env")
    print(f"Token present: {TOKEN[:8]}...  Probing {len(PROBES)} queries\n")
    results = []
    for label, q in PROBES:
        s = summarize(q)
        results.append({"label": label, **s})
        print(f"[{label:32}] key={q!r}")
        if "error" in s:
            print(f"    ERROR: {s['error']}")
        else:
            print(f"    total_results={s['total_results']}  returned={s['returned']}")
            for t in s["top_titles"]:
                print(f"      - {t}")
        print()
        time.sleep(0.7)  # stay under rate limit

    out = Path(__file__).parent.parent / "data" / "protocolsio_syntax_probe.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved raw probe results to {out}")


if __name__ == "__main__":
    main()
