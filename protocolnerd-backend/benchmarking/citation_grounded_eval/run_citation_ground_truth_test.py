"""
Run the citation-grounded queries (gen_citation_ground_truth.py) through the
SHIPPED ProtocolNerd pipeline and score results two ways (Prof. Shasha's Q1/Q2):

  1. Exact-match position: does the REAL cited protocol X appear in the top-10,
     and at what rank (0 if absent).
  2. STS semantic similarity: cosine similarity between each result's title and
     X's title, using the SAME all-MiniLM-L6-v2 embeddings already running in
     production (dense_index.py) -- standardized, reproducible, not an ad-hoc
     LLM judgment call. Reported for the top-1 result and the best (highest
     similarity) result anywhere in the top-10.

Usage: python run_citation_ground_truth_test.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarking/ -- for _bootstrap, systems, etc.
import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_DIR
import systems as S
from protocol_rag import _draftjs_to_text  # type: ignore
from rerank_llm import build_shortlist, llm_rerank  # type: ignore
from dense_index import embed  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--queries-csv", default=str(SCRIPT_DIR / "citation_ground_truth_Biology_100.csv"))
ap.add_argument("--out", default=str(RESULTS_DIR / "citation_ground_truth_report_100.csv"))
ap.add_argument("--cache-name", default="rerank_citegt100_haiku")
_args, _ = ap.parse_known_args()

QUERIES_CSV = Path(_args.queries_csv)
OUT_CSV = Path(_args.out)
RERANK_CACHE = _args.cache_name
COMBINED_PUBMED_CANDIDATES = 5
RERANKER_MODEL = "claude-haiku-4-5"
K = 10


def _protocol_by_id(pid: int) -> Optional[Dict[str, Any]]:
    for p in S.get_index()["protocols"]:
        if p.get("id") == pid:
            return p
    return None


def run_one(query: str):
    """Mirrors the shipped nerd_combined_reranked path; returns full top-K ids + slim lookup."""
    profile, cqs = S._nerd_profile_cached(query)
    shortlist, _ = build_shortlist(profile, cqs, query)
    protocol_reranked = llm_rerank(query, [S._slim(x) for x in shortlist],
                                   cache_name=RERANK_CACHE, model=RERANKER_MODEL)[:K]
    pubmed_combined = S.pubmed_combined_candidates(query, COMBINED_PUBMED_CANDIDATES)

    slim = {int(x["id"]): x for x in [S._slim(y) for y in shortlist] + pubmed_combined}
    combined_pool = [slim[j] for j in protocol_reranked if j in slim] + pubmed_combined
    final_ids = llm_rerank(query, combined_pool, cache_name=f"{RERANK_CACHE}_combined",
                           model=RERANKER_MODEL)[:K]
    return final_ids, slim


def _sts_batch(titles_a: List[str], titles_b: List[str]) -> np.ndarray:
    """Pairwise cosine similarity for aligned lists (a[i] vs b[i]). Vectors are
    already L2-normalized by embed(), so dot product == cosine similarity."""
    va = embed(titles_a)
    vb = embed(titles_b)
    return np.sum(va * vb, axis=1)


def main():
    rows = list(csv.DictReader(open(QUERIES_CSV)))
    print(f"Running {len(rows)} citation-grounded queries through the shipped pipeline...", flush=True)

    report = []
    for i, r in enumerate(rows, 1):
        query = r["query"]
        gt_id = int(r["protocol_id"])
        gt_title = r["protocol_title"]
        citing_paper_id = S._PUBMED_ID_BASE + int(r["citing_pmid"])

        # The paper credits a find when any protocol the citing paper's Methods
        # section confirms it used appears in the top-K -- not only the single
        # protocol the citation graph pointed at. Inputs carrying that column
        # are scored that way; an input without it falls back to the single
        # cited protocol id.
        confirmed_raw = (r.get("used_in_methods_protocol_ids") or "").replace("|", ",").replace(";", ",")
        confirmed_ids = [int(x) for x in confirmed_raw.split(",") if x.strip()] or [gt_id]

        # Ground truth is always a protocol; protocols.io titles are often terse
        # (e.g. "Catalase"), which under-scores genuinely correct matches on a
        # title-only comparison -- calibrated earlier: 0.098 title-only vs. 0.621
        # once description is included, for a pair that IS the same topic. Title+
        # description is used for the primary score; title-only is kept alongside
        # since that's the literal comparison Prof. Shasha asked for.
        gt_protocol = _protocol_by_id(gt_id)
        gt_context = ((gt_title + ". " + _draftjs_to_text(gt_protocol.get("description")))[:500]
                     if gt_protocol else gt_title)

        # ALSO score against the citing paper P1 itself (title+abstract) -- distinct
        # from X on purpose. If a mismatch scores high against P1 but low against X,
        # that's direct evidence the result is generically on-topic for the query
        # (which came from P1's abstract) rather than actually resembling the specific
        # protocol P1 cited. Included for completeness, to make that distinction
        # measurable rather than just argued.
        p1_context = (r["citing_title"] + ". " + r["citing_abstract"])[:500]

        final_ids, slim = run_one(query)

        position = 0
        for pos, item_id in enumerate(final_ids, 1):
            if item_id in confirmed_ids:
                position = pos
                break

        # Self-return: the query was built from the citing paper's OWN abstract, so a
        # live PubMed search can legitimately refind that same paper. That's not an
        # answer to "did we find protocol X" -- it's the system handing back the paper
        # the researcher is already reading. Flagged and EXCLUDED from the "best match"
        # pick below so it can't masquerade as a semantically-close near-miss (a paper
        # title vs. a protocol title rarely overlap much in text, so including it would
        # just drag down the genuine near-miss average, not represent one).
        self_return_position = 0
        for pos, item_id in enumerate(final_ids, 1):
            if item_id == citing_paper_id:
                self_return_position = pos
                break

        eligible = [(iid, slim.get(iid, {}).get("title", "") or "",
                    ((slim.get(iid, {}).get("title", "") or "") + ". " +
                     (slim.get(iid, {}).get("description", "") or ""))[:500])
                    for iid in final_ids if iid != citing_paper_id]
        result_titles = [t for _, t, _ in eligible]
        result_contexts = [c for _, _, c in eligible]
        if result_titles:
            sims_title = _sts_batch(result_titles, [gt_title] * len(result_titles))
            sims_context = _sts_batch(result_contexts, [gt_context] * len(result_contexts))
            sims_vs_paper = _sts_batch(result_contexts, [p1_context] * len(result_contexts))
        else:
            sims_title = np.array([])
            sims_context = np.array([])
            sims_vs_paper = np.array([])

        top1_title = result_titles[0] if result_titles else ""
        top1_sts_title = float(sims_title[0]) if len(sims_title) else 0.0
        top1_sts_context = float(sims_context[0]) if len(sims_context) else 0.0
        top1_sts_vs_paper = float(sims_vs_paper[0]) if len(sims_vs_paper) else 0.0
        # "Best" is chosen by the more robust (title+description) score, since that's
        # the one that isn't distorted by terse protocol titles.
        best_idx = int(np.argmax(sims_context)) if len(sims_context) else -1
        best_sts_title = float(sims_title[best_idx]) if best_idx >= 0 else 0.0
        best_sts_context = float(sims_context[best_idx]) if best_idx >= 0 else 0.0
        best_sts_vs_paper = float(sims_vs_paper[best_idx]) if best_idx >= 0 else 0.0
        best_title = result_titles[best_idx] if best_idx >= 0 else ""
        # position among the ORIGINAL top-10 (not the self-return-filtered list), so
        # this stays comparable to exact_match_position.
        best_pos = final_ids.index(eligible[best_idx][0]) + 1 if best_idx >= 0 else 0

        report.append({
            "query": query,
            "citing_pmid": r["citing_pmid"],
            "citing_abstract": r["citing_abstract"],
            "ground_truth_title": gt_title,
            "confirmed_protocol_ids": ",".join(str(i) for i in confirmed_ids),
            "self_return_position": self_return_position,
            "ground_truth_link": r["protocol_link"],
            "exact_match_position": position,
            "top1_title": top1_title,
            "top1_sts_title_only": round(top1_sts_title, 4),
            "top1_sts_title_plus_desc": round(top1_sts_context, 4),
            "top1_sts_vs_source_paper": round(top1_sts_vs_paper, 4),
            "best_sts_title_only": round(best_sts_title, 4),
            "best_sts_title_plus_desc": round(best_sts_context, 4),
            "best_sts_vs_source_paper": round(best_sts_vs_paper, 4),
            "best_sts_position": best_pos,
            "best_sts_title": best_title,
        })
        sr_tag = f" SELF-RETURN@{self_return_position}" if self_return_position else ""
        print(f"  [{i:>3}/{len(rows)}] exact_pos={position:>2}  top1_sts_vs_X={top1_sts_context:.2f}  "
              f"top1_sts_vs_P1={top1_sts_vs_paper:.2f}  best_sts_vs_X={best_sts_context:.2f}@{best_pos}"
              f"{sr_tag}  {query[:35]}", flush=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "citing_pmid", "citing_abstract",
                                         "ground_truth_title", "confirmed_protocol_ids",
                                         "self_return_position", "ground_truth_link",
                                         "exact_match_position", "top1_title", "top1_sts_title_only",
                                         "top1_sts_title_plus_desc", "top1_sts_vs_source_paper",
                                         "best_sts_title_only", "best_sts_title_plus_desc",
                                         "best_sts_vs_source_paper", "best_sts_position", "best_sts_title"])
        w.writeheader()
        w.writerows(report)

    n = len(report)
    exact_hits = sum(1 for r in report if r["exact_match_position"] > 0)
    exact_top1 = sum(1 for r in report if r["exact_match_position"] == 1)
    self_returns = sum(1 for r in report if r["self_return_position"] > 0)
    avg_top1_title = sum(r["top1_sts_title_only"] for r in report) / n
    avg_top1_context = sum(r["top1_sts_title_plus_desc"] for r in report) / n
    avg_top1_vs_paper = sum(r["top1_sts_vs_source_paper"] for r in report) / n
    avg_best_title = sum(r["best_sts_title_only"] for r in report) / n
    avg_best_context = sum(r["best_sts_title_plus_desc"] for r in report) / n
    avg_best_vs_paper = sum(r["best_sts_vs_source_paper"] for r in report) / n
    print(f"\nSelf-return (found the citing paper itself, excluded from STS): {self_returns}/{n} ({self_returns/n:.1%})")
    print(f"Exact match anywhere in top-10: {exact_hits}/{n} ({exact_hits/n:.1%})")
    print(f"Exact match at position 1:      {exact_top1}/{n} ({exact_top1/n:.1%})")
    print(f"\n-- vs. X (the actual cited protocol) --")
    print(f"Mean top-1 STS (title only):        {avg_top1_title:.3f}")
    print(f"Mean top-1 STS (title+description): {avg_top1_context:.3f}")
    print(f"Mean best-in-top10 STS (title only):        {avg_best_title:.3f}")
    print(f"Mean best-in-top10 STS (title+description): {avg_best_context:.3f}")
    print(f"\n-- vs. P1 (the citing paper itself, for comparison) --")
    print(f"Mean top-1 STS (title+description):        {avg_top1_vs_paper:.3f}")
    print(f"Mean best-in-top10 STS (title+description): {avg_best_vs_paper:.3f}")
    print(f"wrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
