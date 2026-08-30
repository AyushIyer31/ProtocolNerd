"""
Lever #2 prototype: LLM re-ranker.

Instead of the crude global lexical blend (lever #1, which regressed ambiguous
queries), let an LLM re-order the shortlist by actual relevance to the query.
Because the judge showed the right protocol is usually ALREADY retrieved but
mis-ranked, a smarter re-ranker should help specific queries WITHOUT hurting
ambiguous ones.

Design (reuses cached profiles + pools -> cheap):
  shortlist = top-15 by profile ranking  UNION  top-15 by raw TF-IDF
              (so both profile-favoured and keyword-favoured candidates get a
               fair shot at the re-ranker)
  llm_rerank(query, shortlist) -> top-10
Compared against the current profile ranking (closeness_rank) as baseline.

Metrics: nDCG@10 overall + by query type (S/A/M), same blind judge.

Usage:  python rerank_llm.py
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

import _bootstrap  # noqa: F401
from _bootstrap import EVAL_DIR, RESULTS_DIR, CACHE_DIR
import systems as S
import metrics as M
from field_ranking import closeness_rank  # type: ignore
from llm_providers import call_llm  # type: ignore
from llm_judge import judge

K = 10
# Shortlist = profile-top-N UNION lexical-top-N, drawn from a ~240-candidate pool.
# Env-overridable so we can sweep recall (a wider shortlist gives the re-ranker more
# of the pool to find a buried best match). Mirrors the backend reranker env names.
SHORTLIST_PROFILE = int(os.getenv("RERANKER_SHORTLIST_PROFILE", "15") or 15)
SHORTLIST_LEXICAL = int(os.getenv("RERANKER_SHORTLIST_LEXICAL", "15") or 15)


def _rr_path(name: str = "rerank"):
    return CACHE_DIR / f"{name}.json"

RERANK_SYSTEM = (
    "You are an expert biologist helping a scientist find the most useful lab "
    "protocol or paper for their experiment. Given their request and a numbered "
    "list of candidates, rank the candidates by how directly each one helps them "
    "RUN the described experiment (right technique + compatible organism/sample). "
    "Return ONLY a JSON array of the candidate numbers, best first, e.g. [4,1,9,...]. "
    "Include every candidate number exactly once."
)


def _load_rr(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_rr(path, c):
    path.write_text(json.dumps(c, indent=1))


def llm_rerank(query, candidates, use_cache=True, cache_name="rerank", model=None):
    """Return candidate ids ordered best->worst by the LLM (falls back to input
    order for anything the LLM omits or on failure). `cache_name` keeps different
    reranker models in separate caches; `model` sets the reranker model per-call
    (e.g. claude-haiku-4-5) so other flows (profiles) stay on their own model."""
    path = _rr_path(cache_name)
    ids = [int(c["id"]) for c in candidates]
    key = f"{query.strip()}||{','.join(map(str, ids))}"
    cache = _load_rr(path) if use_cache else {}
    if use_cache and key in cache:
        return cache[key]

    lines = [f"[{i}] {c['title']}: {(c['description'] or '')[:180]}"
             for i, c in enumerate(candidates, 1)]
    user = (f"REQUEST:\n{query}\n\nCANDIDATES:\n" + "\n".join(lines) +
            f"\n\nRank all {len(candidates)} candidates. Return ONLY a JSON array of their numbers, best first.")
    try:
        raw = call_llm(
            messages=[{"role": "system", "content": RERANK_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0.0,
            response_format={"type": "json_object"},
            model=model,
        )
        nums = [int(x) for x in re.findall(r"\d+", raw)]
        order, seen = [], set()
        for n in nums:
            if 1 <= n <= len(candidates) and n not in seen:
                seen.add(n)
                order.append(ids[n - 1])
    except Exception as e:
        print(f"    (rerank failed: {e})")
        order = []
    # append anything omitted, in original (profile) order
    for pid in ids:
        if pid not in order:
            order.append(pid)
    if use_cache:
        cache[key] = order
        _save_rr(path, cache)
    return order


def build_shortlist(profile, cqs, query, use_dense=None):
    pool = S._nerd_pool(profile, cqs, query, K, use_dense=use_dense)
    profile_ranked = closeness_rank(profile, pool, max(SHORTLIST_PROFILE, 20), raw_query=query)
    lexical_ranked = sorted(pool, key=lambda r: float(r.get("score", 0) or 0), reverse=True)
    shortlist, seen = [], set()
    for r in list(profile_ranked[:SHORTLIST_PROFILE]) + lexical_ranked[:SHORTLIST_LEXICAL]:
        if r["id"] not in seen:
            seen.add(r["id"])
            shortlist.append(r)
    baseline_top = [int(r["id"]) for r in profile_ranked[:K]]
    return shortlist, baseline_top


def main():
    rows = list(csv.DictReader(open(EVAL_DIR / "queries.csv")))
    print(f"Phase 1: profiles + shortlists ({len(rows)} queries)...", flush=True)
    data = []
    for i, r in enumerate(rows, 1):
        q = r["query"]
        profile, cqs = S._nerd_profile_cached(q)
        shortlist, baseline_top = build_shortlist(profile, cqs, q)
        data.append({"id": r["id"], "tag": r["tag"], "query": q,
                     "shortlist": [S._slim(x) for x in shortlist],
                     "baseline": baseline_top})
        print(f"  [{i}/{len(rows)}] {r['id']} shortlist={len(shortlist)}", flush=True)

    print("\nPhase 2: LLM re-ranking (cached)...", flush=True)
    for i, d in enumerate(data, 1):
        order = llm_rerank(d["query"], d["shortlist"])
        d["rerank"] = order[:K]
        print(f"  [{i}/{len(data)}] {d['id']}", flush=True)

    print("\nPhase 3: judging surfaced results (reuses cache)...", flush=True)
    lookup = {(d["id"], int(x["id"])): x for d in data for x in d["shortlist"]}
    relevance = defaultdict(dict)
    for d in data:
        ids = set(d["baseline"]) | set(d["rerank"])
        for pid in ids:
            res = lookup.get((d["id"], pid), {"id": pid, "title": "", "description": ""})
            relevance[d["id"]][pid] = judge(d["query"], res, use_cache=True)

    tags = ["S", "A", "M"]
    systems = {"profile (baseline)": "baseline", "llm_rerank": "rerank"}
    report = {"k": K, "systems": {}}
    for name, field in systems.items():
        overall, by = [], defaultdict(list)
        for d in data:
            nd = M.ndcg_at_k(d[field], relevance[d["id"]], K)
            overall.append(nd)
            by[d["tag"]].append(nd)
        report["systems"][name] = {"nDCG@10": round(M.mean(overall), 4),
                                   "by_tag": {t: round(M.mean(by[t]), 4) for t in tags}}
    (RESULTS_DIR / "rerank_llm.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 58, flush=True)
    print(f"LLM RE-RANKER vs PROFILE BASELINE  (n={len(data)}, k={K})", flush=True)
    print("=" * 58, flush=True)
    print(f"{'system':<22}{'overall':>9}{'S(31)':>9}{'A(10)':>8}{'M(9)':>8}", flush=True)
    print("-" * 58, flush=True)
    for name in systems:
        c = report["systems"][name]
        print(f"{name:<22}{c['nDCG@10']:>9.3f}{c['by_tag']['S']:>9.3f}"
              f"{c['by_tag']['A']:>8.3f}{c['by_tag']['M']:>8.3f}", flush=True)
    b = report["systems"]["profile (baseline)"]["nDCG@10"]
    r = report["systems"]["llm_rerank"]["nDCG@10"]
    print("-" * 58, flush=True)
    print(f"delta (llm_rerank - baseline): {r - b:+.3f}", flush=True)
    print("=" * 58, flush=True)


if __name__ == "__main__":
    main()
