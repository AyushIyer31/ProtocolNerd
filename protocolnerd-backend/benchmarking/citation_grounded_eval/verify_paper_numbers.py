#!/usr/bin/env python3
"""Check every find rate the paper reports, offline, against committed artifacts.

Three things are verified, with no network calls and no API key:

1. The three baselines, from the per-paper hit vectors their own scripts write into
   paired_vectors.json.
2. The published ProtocolNerd number, from the per-paper record in
   results/accuracy_final_100.csv.
3. The reproduction, from results/citation_ground_truth_report_FINAL_100.csv --
   the report written by a single fresh run of the pipeline over the single
   canonical input file.

(3) is expected to sit within a paper or two of (2) rather than match it exactly.
The re-ranker is an LLM and PubMed is searched live, so the pipeline is not
deterministic across runs; TOLERANCE below is what we treat as agreement.

    python verify_paper_numbers.py

Exits non-zero if a baseline disagrees, if the published record disagrees, or if
the reproduction drifts beyond TOLERANCE.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent

# Importing _bootstrap seeds cache/ from the committed replay_cache/ on a fresh
# clone, which is what makes this check work with no API key and no prior run.
sys.path.insert(0, str(BENCH_DIR))
import _bootstrap  # noqa: F401,E402

PAIRED_VECTORS = SCRIPT_DIR / "paired_vectors.json"

TOLERANCE = 2          # papers of run-to-run drift treated as agreement

RESULTS_DIR = BENCH_DIR / "results"
PUBLISHED_CSV = RESULTS_DIR / "accuracy_final_100.csv"
REPRO_CSV = RESULTS_DIR / "citation_ground_truth_report_FINAL_100.csv"

# What the paper reports in Section 5.2.
PAPER = {
    "protocolnerd": 46,
    "method_m": 22,
    "web_search_llm": 34,
    "web_search_llm_shortquery": 19,
}
LABELS = {
    "method_m": "DP keyword baseline",
    "web_search_llm": "LLM + web search (abstract)",
    "web_search_llm_shortquery": "LLM + web search (short query)",
}


def _row(label, got, expected, ok, extra=""):
    print(f"  {label:<34} {got:>3}/100   paper says {expected:>3}   "
          f"{'OK' if ok else 'MISMATCH'}{extra}")


def check_baselines():
    problems = []
    vectors = json.loads(PAIRED_VECTORS.read_text())
    for name in ("method_m", "web_search_llm", "web_search_llm_shortquery"):
        expected = PAPER[name]
        vec = vectors.get(name)
        if vec is None:
            problems.append(f"{name}: missing from paired_vectors.json")
            print(f"  {LABELS[name]:<34}  --          paper says {expected:>3}   MISSING")
            continue
        got = sum(vec)
        _row(LABELS[name], got, expected, got == expected)
        if got != expected:
            problems.append(f"{name}: vector has {got}, paper says {expected}")
    return problems


def check_published():
    expected = PAPER["protocolnerd"]
    if not PUBLISHED_CSV.exists():
        return [f"missing {PUBLISHED_CSV.name}"]
    rows = list(csv.DictReader(open(PUBLISHED_CSV)))
    got = sum(1 for r in rows if r["hit"].strip().lower() == "true")
    _row("ProtocolNerd (published record)", got, expected, got == expected)
    return [] if got == expected else [
        f"accuracy_final_100.csv has {got}, paper says {expected}"]


def check_reproduction():
    expected = PAPER["protocolnerd"]
    if not REPRO_CSV.exists():
        print(f"  {'ProtocolNerd (fresh run)':<34}  --          "
              f"paper says {expected:>3}   NOT RUN")
        print("       run run_citation_ground_truth_test.py to produce it "
              "(see README)")
        return []
    rows = list(csv.DictReader(open(REPRO_CSV)))
    got = sum(1 for r in rows if int(r["exact_match_position"]) > 0)
    drift = abs(got - expected)
    _row("ProtocolNerd (fresh run)", got, expected, drift <= TOLERANCE,
         f"  (drift {got - expected:+d}, tolerance +/-{TOLERANCE})")
    return [] if drift <= TOLERANCE else [
        f"fresh run drifted {drift} papers from the paper's {expected}"]


def main() -> int:
    print("Baselines, from the hit vectors their own scripts wrote:")
    problems = check_baselines()
    print("\nProtocolNerd:")
    problems += check_published()
    problems += check_reproduction()

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll reported find rates verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
