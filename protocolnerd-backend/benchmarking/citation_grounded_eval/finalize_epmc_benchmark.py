"""Finalize the EPMC-grounded chemistry benchmark: add queries, write the CSV.

Reads the builder's confirmed pairs and writes each pair's search query with
the house paraphrase prompt (Prompt 8), role line naming a chemist, from P's
title and abstract only. The query never sees X. Queries are cached so a
rerun is cheap.

    python finalize_epmc_benchmark.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401
from llm_providers import call_llm
import sys as _sys
_V2 = "--v2" in _sys.argv
_GT = "epmc_ground_truth_chemistry_v2.json" if _V2 else "epmc_ground_truth_chemistry.json"
_OUT = "citation_ground_truth_Chemistry_EPMC_100_v2.csv" if _V2 else "citation_ground_truth_Chemistry_EPMC_100.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
QUERY_PROMPT = (
    "You are a bench chemist. Given a lab protocol or a research paper's title and "
    "abstract, write ONE natural-language sentence that a scientist would type into a "
    "search tool when they need this exact item. Describe the experimental goal, "
    "material or sample, and technique in your own words, in a single sentence. Do NOT "
    "copy the title verbatim, paraphrase and use everyday phrasing. Sound like a real "
    "request a scientist would type. Exactly one sentence, under 30 words. No quotes, "
    "no preamble. Return only the sentence."
)


def main() -> int:
    pairs = json.load(open(SCRIPT_DIR / _GT))
    cache_path = SCRIPT_DIR / ".epmc_query_cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    out_rows = []
    for i, pr in enumerate(pairs, 1):
        key = pr["p_pmid"]
        if key not in cache:
            user = f"TITLE: {pr['p_title']}\nABSTRACT: {pr['p_abstract'][:2000]}"
            q = (call_llm(messages=[{"role": "system", "content": QUERY_PROMPT},
                                    {"role": "user", "content": user}],
                          temperature=0.0, provider="claude",
                          model="claude-sonnet-4-6") or "").strip().strip('"')
            if not q:
                print(f"  [{i:>3}] QUERY FAILED for P {key}, skipping")
                continue
            cache[key] = q
            json.dump(cache, open(cache_path, "w"))
        # Same schema as citation_ground_truth_Biology_100.csv, so tooling and
        # readers treat the two benchmarks identically. The "protocol" columns
        # carry the protocol-describing paper X; its PMID serves as the id.
        out_rows.append({
            "pmid": pr["p_pmid"],
            "protocol_id": pr["x_pmid"],
            "protocol_title": pr["x_title"],
            "protocol_doi": pr["x_doi"],
            "protocol_link": f"https://europepmc.org/article/MED/{pr['x_pmid']}",
            "citing_pmid": pr["p_pmid"],
            "citing_title": pr["p_title"],
            "citing_abstract": pr["p_abstract"][:2000],
            "relevance_score": "",
            "query": cache[key],
            "used_in_methods_protocol_ids": pr["x_pmid"],
        })
        print(f"  [{i:>3}/{len(pairs)}] {cache[key][:76]}")
    out = SCRIPT_DIR / _OUT
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
