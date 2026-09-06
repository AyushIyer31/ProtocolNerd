"""Two-arm find-rate runner for the EPMC-grounded chemistry benchmark.

Ground truth is a protocol-describing paper X in Europe PMC that a citing
paper P used in its Methods section. The query comes from P's title and
abstract. A find means X's PMID (or DOI) appears in the top 10, from any
retrieval lane.

Four arms. The first two mirror the shipped chemistry path with only the
literature lane varying; the second two drop the Protocols.io pool entirely
and rank the literature candidates alone, isolating the sources themselves:
  epmc                 Protocols.io + Europe PMC   (shipped configuration)
  epmc_pubmed          Protocols.io + Europe PMC + PubMed
  epmc_lit             Europe PMC only, no Protocols.io
  epmc_pubmed_lit      Europe PMC + PubMed, no Protocols.io

    python run_epmc_grounded_test.py --arm epmc
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_DIR
import systems as S
from rerank_llm import build_shortlist, llm_rerank  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ID_BASE = S._PUBMED_ID_BASE  # shared id-space: same paper from any lane dedupes
K = 10
PAPER_CANDIDATES = 5
RERANKER_MODEL = "claude-haiku-4-5"


def _slim_paper(r: Dict[str, Any]) -> Dict[str, Any]:
    """Paper candidate slim, keyed by PMID so lanes dedupe and X is matchable."""
    try:
        rid = PAPER_ID_BASE + int(r.get("pmid"))
    except Exception:
        rid = PAPER_ID_BASE + (abs(hash(r.get("title") or "")) % 10 ** 8)
    return {"id": rid, "title": r.get("title") or "",
            "description": (r.get("description") or r.get("abstract") or "")[:400],
            "source": r.get("source") or "paper",
            "pmid": str(r.get("pmid") or ""), "doi": (r.get("doi") or "").lower()}


def epmc_candidates(profile, query: str, keep: int) -> List[Dict[str, Any]]:
    """Europe PMC lane, exactly as the retriever runs it (relaxation ladder)."""
    from europepmc_client import search_with_fallback
    hits = search_with_fallback(profile, query, keep) or []
    for r in hits:
        r.setdefault("source", "europepmc")
    return [_slim_paper(r) for r in hits[:keep]]


def pubmed_candidates(query: str, keep: int) -> List[Dict[str, Any]]:
    """PubMed lane via the same helper the frozen runner uses, re-slimmed to
    carry pmid/doi for ground-truth matching."""
    from pubmed_client import search_pubmed_fanout, balanced_trim
    hits = search_pubmed_fanout(query, max(keep, 5),
                                fallback_core=S._pubmed_core(query)) or []
    for r in hits:
        r.setdefault("source", "pubmed")
    return [_slim_paper(r) for r in balanced_trim(hits, keep)]


def run_one(query: str, arm: str, cache_tag: str):
    profile, cqs = S._nerd_profile_cached(query)
    with_pubmed = arm in ("epmc_pubmed", "epmc_pubmed_lit")
    lit_only = arm.endswith("_lit")

    papers = epmc_candidates(profile, query, PAPER_CANDIDATES)
    if with_pubmed:
        seen = {p["id"] for p in papers}
        papers += [p for p in pubmed_candidates(query, PAPER_CANDIDATES)
                   if p["id"] not in seen]

    if lit_only:
        slim = {int(x["id"]): x for x in papers}
        final_ids = llm_rerank(query, papers, cache_name=f"{cache_tag}_combined",
                               model=RERANKER_MODEL)[:K]
        return final_ids, slim

    shortlist, _ = build_shortlist(profile, cqs, query)
    protocol_reranked = llm_rerank(query, [S._slim(x) for x in shortlist],
                                   cache_name=cache_tag, model=RERANKER_MODEL)[:K]
    slim = {int(x["id"]): x for x in [S._slim(y) for y in shortlist] + papers}
    combined_pool = [slim[j] for j in protocol_reranked if j in slim] + papers
    final_ids = llm_rerank(query, combined_pool, cache_name=f"{cache_tag}_combined",
                           model=RERANKER_MODEL)[:K]
    return final_ids, slim


def bootstrap_ci(vals: List[int], n: int = 10000):
    random.seed(0)
    means = sorted(sum(random.choice(vals) for _ in range(len(vals))) / len(vals)
                   for _ in range(n))
    return means[int(n * 0.05) - 1] * 100, means[int(n * 0.95) - 1] * 100


def main() -> int:
    global PAPER_CANDIDATES
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["epmc", "epmc_pubmed", "epmc_lit", "epmc_pubmed_lit"], required=True)
    ap.add_argument("--queries-csv", default=str(SCRIPT_DIR / "citation_ground_truth_Chemistry_EPMC_100.csv"))
    ap.add_argument("--paper-candidates", type=int, default=PAPER_CANDIDATES,
                    help="papers fetched per literature lane (default 5, the shipped setting)")
    ap.add_argument("--v2", action="store_true",
                    help="v2 benchmark: OA X's with tagged references")
    args = ap.parse_args()
    PAPER_CANDIDATES = args.paper_candidates
    suffix = "" if args.paper_candidates == 5 else f"_pc{args.paper_candidates}"
    if args.v2:
        suffix += "_v2"
        args.queries_csv = str(SCRIPT_DIR / "citation_ground_truth_Chemistry_EPMC_100_v2.csv")
    rows = list(csv.DictReader(open(args.queries_csv)))
    cache_tag = f"rerank_epmcgt_{args.arm}{suffix}"
    print(f"Arm {args.arm}: {len(rows)} queries, cache {cache_tag}\n", flush=True)

    report, hits = [], []
    for i, r in enumerate(rows, 1):
        gt_paper_id = PAPER_ID_BASE + int(r["protocol_id"])
        gt_doi = (r["protocol_doi"] or "").lower()
        final_ids, slim = run_one(r["query"], args.arm, cache_tag)
        rank = 0
        for pos, item_id in enumerate(final_ids, 1):
            item = slim.get(int(item_id), {})
            if int(item_id) == gt_paper_id or (gt_doi and item.get("doi") == gt_doi):
                rank = pos
                break
        # self-return guard: the query came from P's abstract, so finding P
        # itself is not finding X
        hits.append(1 if rank else 0)
        report.append({"p_pmid": r["citing_pmid"], "x_pmid": r["protocol_id"],
                       "query": r["query"], "is_hit": bool(rank), "hit_rank": rank})
        print(f"  [{i:>3}/{len(rows)}] {'HIT@' + str(rank) if rank else 'miss':<7} {r['query'][:68]}",
              flush=True)

    out = RESULTS_DIR / f"epmc_grounded_{args.arm}{suffix}_100.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    lo, hi = bootstrap_ci(hits)
    print(f"\nARM {args.arm}: find rate {sum(hits)}/{len(hits)}  "
          f"90% CI [{lo:.0f}%, {hi:.0f}%]\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
