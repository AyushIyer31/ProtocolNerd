#!/usr/bin/env python3
"""How many top-10 slots do PubMed papers take, biology vs chemistry?

The runner builds combined_pool = 10 protocols + 5 PubMed papers and the
re-ranker keeps K=10. Ground truth is ALWAYS a protocols.io protocol id, so
every slot a paper takes is a slot that cannot hold the right answer. If
chemistry's protocol candidates are weaker (thin corpus coverage) the re-ranker
will rationally promote papers, cutting protocol slots from 10 to 5 and roughly
halving the chance of surfacing the target.

This measures that directly, on each domain's own benchmark queries, using the
same run_one path the find-rate test uses.

Usage:
    python slot_occupancy_analysis.py --limit 25
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401
import systems as S  # type: ignore
from rerank_llm import build_shortlist, llm_rerank  # type: ignore
from domains import set_current_domain  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
K = 10
COMBINED_PUBMED_CANDIDATES = 5
RERANKER_MODEL = "claude-haiku-4-5"


def run_one(query: str, cache_name: str):
    """Mirrors run_citation_ground_truth_test.run_one, but returns the final ids
    so the caller can classify each slot by source."""
    profile, cqs = S._nerd_profile_cached(query)
    shortlist, _ = build_shortlist(profile, cqs, query)
    protocol_reranked = llm_rerank(query, [S._slim(x) for x in shortlist],
                                   cache_name=cache_name, model=RERANKER_MODEL)[:K]
    pubmed_combined = S.pubmed_combined_candidates(query, COMBINED_PUBMED_CANDIDATES)
    slim = {int(x["id"]): x for x in [S._slim(y) for y in shortlist] + pubmed_combined}
    combined_pool = [slim[j] for j in protocol_reranked if j in slim] + pubmed_combined
    final_ids = llm_rerank(query, combined_pool, cache_name=f"{cache_name}_combined",
                           model=RERANKER_MODEL)[:K]
    return final_ids


def measure(label, domain, csv_path, cache_name, limit):
    set_current_domain(domain)
    os.environ["BENCHMARK_DOMAIN"] = domain
    rows = list(csv.DictReader(open(csv_path)))[:limit]
    tot_pm = tot_slots = 0
    per_query = []
    for i, r in enumerate(rows, 1):
        try:
            ids = run_one(r["query"], cache_name)
        except Exception as e:
            print(f"  [{i}] failed: {str(e)[:50]}", flush=True)
            continue
        pm = sum(1 for x in ids if int(x) >= S._PUBMED_ID_BASE)
        tot_pm += pm
        tot_slots += len(ids)
        per_query.append(pm)
        print(f"  [{i:>3}/{len(rows)}] {label}: {pm} pubmed / {len(ids)} slots", flush=True)
    return label, tot_pm, tot_slots, per_query


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    results = []
    results.append(measure("biology  ", "biology",
                           SCRIPT_DIR / "citation_ground_truth_Biology_100.csv",
                           "rerank_final100_haiku", args.limit))
    results.append(measure("chemistry", "chemistry",
                           SCRIPT_DIR / "citation_ground_truth_Chemistry_100.csv",
                           "rerank_finalchem_sonnet", args.limit))

    print("\n" + "=" * 74)
    print(f"{'domain':<12}{'pubmed slots':>14}{'total slots':>13}{'% of top-10':>14}")
    print("=" * 74)
    for label, pm, tot, per in results:
        pct = pm / tot if tot else 0
        print(f"{label:<12}{pm:>14}{tot:>13}{pct:>13.1%}")
    print("=" * 74)
    print("\nEvery PubMed slot is a slot that CANNOT hold the ground-truth protocol.")
    for label, pm, tot, per in results:
        if per:
            avg = sum(per) / len(per)
            print(f"  {label}: avg {avg:.1f} paper slots/query "
                  f"-> ~{10-avg:.1f} protocol slots available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
