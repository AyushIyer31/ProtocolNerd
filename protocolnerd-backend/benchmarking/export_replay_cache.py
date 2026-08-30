#!/usr/bin/env python3
"""Export the minimum cache set needed to replay the paper's experiment offline.

`cache/` is a working directory: it is gitignored, it accumulates every experiment
ever run, and it is ~9 MB. A fresh clone therefore has no cache at all, which means
every evaluation script would re-call the Claude API — the web-search baselines use
Sonnet with a search tool, so that is slow and genuinely expensive.

This script writes `replay_cache/`, which IS committed: the same entries, pruned to
just the 100 benchmark papers. `_bootstrap.py` seeds `cache/` from it on first use,
so all four experiments replay for free with no API key.

    python export_replay_cache.py

Rerun it after any change to the published numbers. It never deletes from `cache/`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
CACHE_DIR = BENCH_DIR / "cache"
REPLAY_DIR = BENCH_DIR / "replay_cache"
CANONICAL = BENCH_DIR / "citation_grounded_eval" / "citation_ground_truth_Biology_100.csv"

# Exactly the caches the four live experiments read. The legacy rerank_citegt*
# caches are deliberately excluded: they hold several entries per query from
# repeated runs, so "the" cached ranking for a paper is ambiguous.
WANTED = [
    "profiles.json",
    "rerank_final100_haiku.json",
    "rerank_final100_haiku_combined.json",
    "websearch_llm_baseline.json",
    "websearch_llm_baseline_slug_resolution.json",
    "websearch_llm_baseline_shortquery.json",
    "websearch_llm_baseline_shortquery_slug_resolution.json",
    "keyword_baseline_queries.json",
]


def main() -> int:
    rows = list(csv.DictReader(open(CANONICAL)))
    queries = {r["query"].strip() for r in rows}
    pmids = {r["pmid"] for r in rows}

    REPLAY_DIR.mkdir(exist_ok=True)
    total_in = total_out = 0

    for name in WANTED:
        src = CACHE_DIR / name
        if not src.exists():
            print(f"  skip (absent): {name}")
            continue
        data = json.loads(src.read_text())

        # Keys are either a query, a "query||candidate-ids" pair, or a PMID.
        kept = {k: v for k, v in data.items()
                if k.split("||", 1)[0].strip() in queries or k in pmids}
        # Slug-resolution caches are keyed by protocol slug, not by paper; nothing
        # matches, and they are small, so carry them whole.
        if not kept:
            kept = data

        (REPLAY_DIR / name).write_text(json.dumps(kept, indent=0, sort_keys=True))
        total_in += len(data)
        total_out += len(kept)
        print(f"  {name}: {len(kept)}/{len(data)} entries")

    size = sum(p.stat().st_size for p in REPLAY_DIR.glob("*.json")) / 1048576
    print(f"\nwrote {len(list(REPLAY_DIR.glob('*.json')))} files, "
          f"{total_out}/{total_in} entries, {size:.2f} MB -> {REPLAY_DIR.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
