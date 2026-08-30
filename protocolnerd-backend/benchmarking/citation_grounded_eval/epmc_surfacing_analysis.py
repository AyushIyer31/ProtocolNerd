#!/usr/bin/env python3
"""Where does Europe PMC appear for the chemistry benchmark queries?

Deliberately SEPARATE from run_citation_ground_truth_test.py. That runner must
not include Europe PMC: the ground truth is always a protocols.io protocol id,
so a paper can never BE the right answer, and any slot it takes is one that
cannot hold the right answer. Including it could only depress the find rate for
a reason unrelated to retrieval quality, and would break comparability with the
biology run.

This instead replays the same queries through the live /chat endpoint, which
does include Europe PMC, and records where it lands. Reports the surfacing rate
and prints concrete samples.

Usage (backend must be running on :8001):
    python epmc_surfacing_analysis.py [--queries-csv ...] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
URL = "http://localhost:8001/chat"


def _post(payload, timeout=300):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ask(q):
    """Two-turn flow, as the UI does it: build the profile, then confirm the
    search with it. A single-shot search_confirmed call skips the planner and
    searches with an empty profile, which is not what a user experiences."""
    first = _post({"query": q, "search_mode": "local", "top_k": 3,
                   "explain": False, "no_log": True, "skip_clarification": True})
    prof = first.get("experiment_profile") or {}
    cands = first.get("candidate_search_queries") or [q]
    return _post({"query": cands[0], "search_mode": "local", "top_k": 3,
                  "explain": False, "no_log": True, "search_confirmed": True,
                  "selected_search_query": cands[0], "candidate_search_queries": cands,
                  "experiment_profile": prof, "conversation_query": q})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-csv",
                    default=str(SCRIPT_DIR / "citation_ground_truth_Chemistry_100.csv"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "../results/epmc_surfacing_chemistry.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.queries_csv)))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Replaying {len(rows)} chemistry benchmark queries through /chat "
          f"(Europe PMC enabled)...\n", flush=True)

    out, samples, n_with = [], [], 0
    for i, r in enumerate(rows, 1):
        q = r["query"]
        try:
            d = ask(q)
        except Exception as e:
            print(f"  [{i:>3}/{len(rows)}] FAILED: {str(e)[:60]}", flush=True)
            continue
        res = d.get("results", [])
        counts = collections.Counter(x.get("source") for x in res)
        ranks = [j for j, x in enumerate(res, 1) if x.get("source") == "europepmc"]

        # Shipped-configuration find rate. The offline runner deliberately excludes
        # Europe PMC (a paper can never BE the ground truth, so its slots could only
        # depress the score); that measures protocol-retrieval quality and is what
        # compares against biology. But the shipped chemistry app DOES include it,
        # so the rate a chemist actually experiences is this one. Recording both
        # here keeps the difference visible instead of implicit.
        targets = {t.strip() for t in
                   (r.get("used_in_methods_protocol_ids") or r.get("protocol_id") or "").split("|")
                   if t.strip()}
        found_rank = next((j for j, x in enumerate(res, 1)
                           if str(x.get("id")) in targets), 0)
        if ranks:
            n_with += 1
            for j in ranks:
                samples.append({"query": q, "rank": j, "result": res[j - 1]})
        out.append({"query": q, "citing_pmid": r.get("citing_pmid", ""),
                    "epmc_ranks": "|".join(map(str, ranks)),
                    "n_epmc": len(ranks),
                    "found_rank": found_rank,
                    "sources": json.dumps(dict(counts))})
        shown = ",".join(map(str, ranks)) if ranks else "-"
        hit = f"HIT@{found_rank}" if found_rank else "miss"
        print(f"  [{i:>3}/{len(rows)}] epmc={shown:<12} {hit:<8} {str(dict(counts))[:40]}", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "citing_pmid", "epmc_ranks", "n_epmc",
                                          "found_rank", "sources"])
        w.writeheader()
        w.writerows(out)

    n = len(out)
    hits = [o for o in out if o["found_rank"]]
    epmc_slots = sum(o["n_epmc"] for o in out)
    print("\n" + "=" * 76)
    print(f"queries replayed                  : {n}")
    print(f"queries with Europe PMC in top-10 : {n_with}" + (f"  ({n_with/n:.0%})" if n else ""))
    print(f"total Europe PMC results surfaced : {len(samples)}")
    if n:
        print(f"Europe PMC slots / query          : {epmc_slots/n:.1f} of 10")
        print(f"SHIPPED-config find rate          : {len(hits)}/{n} = {len(hits)/n:.1%}")
        print("  (offline runner, Europe PMC excluded, measured 32.0% -- the gap is")
        print("   the cost of the slots Europe PMC takes from protocols.io)")
    print("=" * 76)
    # Bootstrap input, same 0/1 vector format MeanConf.py consumes.
    vals = Path(args.out).with_name("chemistry_shipped.vals")
    vals.write_text(">ProtocolNerd-chemistry-shipped\n"
                    + " ".join("1" if o["found_rank"] else "0" for o in out) + "\n")
    print(f"bootstrap input -> {vals.name}")

    if samples:
        print("\nSAMPLES — Europe PMC results that survived the re-rank:\n")
        for s in samples[:15]:
            r = s["result"]
            print(f"  [rank {s['rank']}] {(r.get('title') or '')[:76]}")
            print(f"      from query   : {s['query'][:72]}")
            print(f"      methods match: {r.get('methods_match')}")
            print(f"      {r.get('url','')}")
            print()
    else:
        print("\nNo Europe PMC result reached the top 10 for any benchmark query.")
    print(f"wrote -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
