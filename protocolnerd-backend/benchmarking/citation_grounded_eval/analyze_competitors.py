"""Reviewer 1.1: turn the competitor runs into the numbers the revision needs.

For each tool it reports the find rate with a confidence interval, the paired
comparison against ProtocolNerd's published per-paper vector (the same paired
non-parametric test the paper already uses against DP), and two descriptive
counts that matter more than the rate itself:

  protocols.io reached   how often the tool returned any Protocols.io page at
                         all, which separates "searched the right place and
                         ranked badly" from "never goes there".
  publication hits       how often a hit arrived as a Springer, Nature or
                         Protocol Exchange DOI rather than a Protocols.io entry.
                         48 of the 100 targets are published that way, so this
                         is where a paper search engine can legitimately score.

Title-tier matches are counted separately and never folded into the headline,
since they are advisory until a human confirms them.

Usage:
    python analyze_competitors.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_DIR  # noqa: E402

PUBLISHED = Path(RESULTS_DIR) / "accuracy_final_100.csv"   # ProtocolNerd, 46/100
DECISIONS = Path(RESULTS_DIR) / "competitor_title_decisions.csv"


def load_decisions() -> Dict[Tuple[str, str], bool]:
    """Human rulings on tier-3 title matches, which are advisory until someone
    confirms them. Kept in a file so the adjudication is on the record and the
    numbers can be reproduced rather than adjusted by hand."""
    if not DECISIONS.exists():
        return {}
    return {(r["tool"].strip(), r["pmid"].strip()): r["decision"].strip().lower() == "accept"
            for r in csv.DictReader(open(DECISIONS))}


def wilson(k: int, n: int, z: float = 1.645) -> Tuple[float, float]:
    """90% interval by default, matching the paper's confidence intervals."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - half) * 100, min(1.0, c + half) * 100)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact test on the discordant pairs."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m)
    return min(1.0, 2 * tail)


def load_protocolnerd() -> Dict[str, bool]:
    return {r["pmid"].strip(): r["hit"].strip().lower() == "true"
            for r in csv.DictReader(open(PUBLISHED))}


def load_tool(path: Path) -> List[dict]:
    return list(csv.DictReader(open(path)))


def summarise(label: str, rows: List[dict], ours: Dict[str, bool],
              decisions: Optional[Dict[Tuple[str, str], bool]] = None) -> Optional[dict]:
    if not rows:
        return None
    decisions = decisions or {}
    n = len(rows)
    confirmed, pending, reached, pub_hits, adjudicated = {}, 0, 0, 0, 0
    for r in rows:
        pmid = r["pmid"].strip()
        is_hit = str(r.get("is_hit", "")).strip().lower() == "true"
        needs = str(r.get("needs_review", "")).strip().lower() == "true"
        if is_hit and not needs:
            confirmed[pmid] = True
            if r.get("tier") == "doi":
                pub_hits += 1
        elif is_hit and needs:
            ruling = decisions.get((label, pmid))
            confirmed[pmid] = bool(ruling)
            if ruling:
                adjudicated += 1
            elif ruling is None:
                pending += 1
        else:
            confirmed[pmid] = False
        try:
            urls = json.loads(r.get("candidate_urls") or "[]")
        except Exception:
            urls = []
        if any("protocols.io" in (u or "").lower() for u in urls):
            reached += 1

    hits = sum(confirmed.values())
    lo, hi = wilson(hits, n)
    # paired against ProtocolNerd on the papers both cover
    both = [p for p in confirmed if p in ours]
    b = sum(1 for p in both if ours[p] and not confirmed[p])      # only ProtocolNerd
    c = sum(1 for p in both if confirmed[p] and not ours[p])      # only the tool
    p_val = mcnemar_exact(b, c)
    return {"label": label, "n": n, "hits": hits, "lo": lo, "hi": hi, "pending": pending,
            "adjudicated": adjudicated,
            "reached": reached, "pub_hits": pub_hits, "paired_n": len(both),
            "ours": sum(1 for p in both if ours[p]), "b": b, "c": c, "p": p_val}


def main() -> int:
    ours = load_protocolnerd()
    decisions = load_decisions()
    print(f"  ProtocolNerd published vector: {sum(ours.values())}/{len(ours)}\n")

    found = sorted(Path(RESULTS_DIR).glob("competitor_*_100.csv"))
    if not found:
        print("  no competitor result files yet")
        return 0

    rows_out = []
    for path in found:
        label = path.stem.replace("competitor_", "").replace("_100", "")
        s = summarise(label, load_tool(path), ours, decisions)
        if s:
            rows_out.append(s)

    print(f"  {'tool':<34}{'found':>8}{'90% CI':>16}{'reached p.io':>14}{'via DOI':>9}{'by title':>10}{'pending':>9}")
    print("  " + "-" * 92)
    for s in rows_out:
        ci = f"[{s['lo']:.0f}%, {s['hi']:.0f}%]"
        print(f"  {s['label']:<34}{s['hits']:>4}/{s['n']:<3}{ci:>16}"
              f"{s['reached']:>14}{s['pub_hits']:>9}{s['adjudicated']:>10}{s['pending']:>9}")

    print(f"\n  Paired against ProtocolNerd (same papers, exact McNemar):")
    for s in rows_out:
        print(f"    {s['label']:<32} ProtocolNerd {s['ours']:>3}/{s['paired_n']:<4} "
              f"tool {s['hits']:>3}/{s['paired_n']:<4} "
              f"only ours {s['b']:>3}, only tool {s['c']:>3}, p = {s['p']:.4f}"
              f"{'  significant' if s['p'] < 0.05 else ''}")

    out = Path(RESULTS_DIR) / "competitor_summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
