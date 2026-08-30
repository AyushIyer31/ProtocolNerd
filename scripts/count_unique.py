#!/usr/bin/env python3
"""
Estimate the UNIQUE reachable public protocol count on protocols.io.

Sums of per-keyword totals double-count protocols that match several terms.
This crawls the listing pages (IDs only — not full protocol bodies) for a broad
set of high-yield terms, unions the IDs, and reports the deduplicated count.

Writes the unique ID set to data/reachable_ids.txt so a later full download can
reuse it. Resumable-ish: re-running re-crawls, but it's listing-only and fast.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "protocolnerd-backend" / "variables.env")

from fetch_protocols import _get_token, _api_get  # noqa: E402

# Broad, high-yield terms spanning the major biology domains. Broad on purpose:
# coverage, not precision. Overlap is fine — we dedupe by ID.
TERMS = [
    "cell", "RNA", "DNA", "PCR", "protein", "assay", "tissue", "culture",
    "sequencing", "imaging", "extraction", "isolation", "sample", "microscopy",
    "antibody", "bacteria", "enzyme", "staining", "buffer", "plasmid",
    "cloning", "purification", "transfection", "crispr", "western blot",
    "flow cytometry", "mouse", "plant", "yeast", "zebrafish", "histology",
    "metabolomics", "proteomics", "organoid", "stem cell", "chromatography",
    "mass spectrometry", "immunoprecipitation", "electrophoresis", "blood",
    "virus", "membrane", "neuron", "drug", "fixation", "embryo", "genome",
    "expression", "library preparation", "microbiome",
]

PAGE_SIZE = 50
SLEEP = 0.8           # pace to stay under protocols.io throttling
MAX_RETRIES = 4
OUT = ROOT / "data" / "reachable_ids.txt"


def _get_page(key, page_id):
    for attempt in range(MAX_RETRIES):
        d = _api_get("/protocols", {
            "filter": "public", "key": key, "order_field": "activity",
            "order_dir": "desc", "page_id": page_id, "page_size": PAGE_SIZE,
        }, TOKEN) or {}
        if d.get("status_code") in (0, None) and "items" in d:
            return d
        time.sleep(SLEEP * (2 ** attempt))
    return {}


TOKEN = _get_token()
ids = set()

for ti, term in enumerate(TERMS, 1):
    first = _get_page(term, 1)
    total_pages = (first.get("pagination", {}) or {}).get("total_pages", 1) or 1
    total = (first.get("pagination", {}) or {}).get("total_results")
    before = len(ids)
    for it in first.get("items", []) or []:
        if it.get("id"):
            ids.add(it["id"])
    for pid in range(2, total_pages + 1):
        page = _get_page(term, pid)
        for it in page.get("items", []) or []:
            if it.get("id"):
                ids.add(it["id"])
        time.sleep(SLEEP)
    # Write incrementally so the unique count is always available, even if
    # interrupted — and so this doubles as the ID manifest for the download.
    OUT.write_text("\n".join(str(i) for i in sorted(ids)), encoding="utf-8")
    print(f"[{ti:2}/{len(TERMS)}] {term:18} pages={total_pages:4} term_total={total} "
          f"+{len(ids)-before:4} new | UNIQUE SO FAR={len(ids)}", flush=True)

OUT.write_text("\n".join(str(i) for i in sorted(ids)), encoding="utf-8")
print(f"\nUNIQUE reachable protocols across {len(TERMS)} broad terms: {len(ids)}")
print(f"Already in your corpus: 7304")
print(f"Wrote ID set to {OUT}")
