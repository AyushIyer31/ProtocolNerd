"""Route every benchmark query through the production domain classifier.

The benchmark pairs a chemistry protocol paper X with a citing paper P, and the
query is written from P alone. P is frequently a biology study that used a
chemistry method, so a sizeable share of the queries describe biology
experiments and the router sends them to the biology domain, where the chemistry
literature lanes never run. This records the routing decision per query so the
benchmark can be restricted to the queries the chemistry domain actually serves.
"""
import csv, json, sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
import logging; logging.disable(logging.INFO)
from domains.registry import route

SRC = SCRIPT_DIR / "archive" / "citation_ground_truth_Chemistry_EPMC_100_v2.csv"
OUT = SCRIPT_DIR / ".benchmark_domain_routing.json"

def main():
    rows = list(csv.DictReader(open(SRC)))
    cache = json.load(open(OUT)) if OUT.exists() else {}
    counts = {}
    for i, r in enumerate(rows, 1):
        key = r["citing_pmid"]
        if key not in cache:
            try:
                cache[key] = route(r["query"]).name
            except Exception as e:
                print(f"  [{i}] routing failed for {key}: {e}", flush=True)
                continue
            json.dump(cache, open(OUT, "w"))
        counts[cache[key]] = counts.get(cache[key], 0) + 1
        if i % 25 == 0:
            print(f"  {i}/{len(rows)} routed: {counts}", flush=True)
    print(f"\nFINAL over {len(rows)} queries: {counts}")
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    main()
