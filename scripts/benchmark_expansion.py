#!/usr/bin/env python3
"""
Benchmark: original vs. expanded protocols.io search.

Runs Prof. Dennis Shasha's three test queries through two pipelines and writes
a side-by-side comparison to data/benchmark_results.md:

  BASELINE  — the raw natural-language query sent straight to the protocols.io
              `key` parameter (what the user would type).
  EXPANDED  — extract biological concepts -> expand each with grounded
              synonyms (NCBI Taxonomy + Europe PMC, Ollama/static fallback) ->
              fire many short probes -> merge -> multi-signal re-rank.

For each query the report shows the expanded terms, the best protocols found,
why each one matches, and the limitations observed.

Usage:
    python benchmark_expansion.py
    python benchmark_expansion.py --no-external   # skip NCBI/EuropePMC/Ollama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the backend modules importable
sys.path.insert(0, str(Path(__file__).parent.parent / "protocolnerd-backend"))

from concept_expansion import (  # noqa: E402
    extract_concepts, expand_concepts, build_search_probes,
)
from protocolsio_client import search_live, multi_probe_search  # noqa: E402
from protocol_ranker import rank_protocols  # noqa: E402

SHASHA_QUERIES = [
    "Find protocols that can allow in-planta tests in which genes are modified",
    "Find protocols that allow more than one transcription factor to be modified at the same time for mice",
    "Find protocols that test for drought tolerance in rice",
]


def run_one(query: str, use_external: bool) -> dict:
    print(f"\n{'='*78}\nQUERY: {query}\n{'='*78}")

    # --- Baseline: raw query straight to the API ---
    base_total, base_results = search_live(query, max_results=5)
    print(f"[baseline] raw query -> total_results={base_total}, returned={len(base_results)}")

    # --- Expanded pipeline ---
    concepts = extract_concepts(query)
    print(f"[concepts] organisms={concepts['organisms']} methods={concepts['methods']} "
          f"goals={concepts['goals']} actions={concepts['actions']}")
    expansions = expand_concepts(concepts, use_external=use_external)
    for k, v in expansions.items():
        print(f"   expand  {k!r} -> {v}")
    probes = build_search_probes(concepts, expansions)
    print(f"[probes] {probes}")

    merged, hit_map, probe_totals = multi_probe_search(probes, per_probe=8, cap=60)
    print(f"[search] {len(merged)} unique protocols from {len(probes)} probes")
    ranked = rank_protocols(concepts, expansions, merged, hit_map, top_k=5)
    for r in ranked:
        print(f"   [{r['score']:>5}] {r['title'][:62]}")
        print(f"           why: {r['why']}")

    return {
        "query": query,
        "concepts": concepts,
        "expansions": expansions,
        "probes": probes,
        "probe_totals": probe_totals,
        "baseline_total": base_total,
        "baseline_results": base_results,
        "expanded": ranked,
    }


def _limitations(run: dict) -> str:
    notes = []
    if run["baseline_total"] == 0:
        notes.append("raw query returns 0 (protocols.io needs short adjacent phrases)")
    if not run["expanded"]:
        notes.append("no protocols matched even after expansion — corpus gap")
    else:
        top = run["expanded"][0]
        if top["signals"]["organism"] == 0:
            notes.append("top hit lacks an explicit organism match")
        if top["score"] < 3:
            notes.append("low overall score — matches are partial/keyword-level")
    if not notes:
        notes.append("expansion recovered relevant protocols the raw query missed")
    return "; ".join(notes)


def write_report(runs: list, path: Path):
    L = []
    L.append("# Protocol Search: Original vs. Expanded — Benchmark Results\n")
    L.append("Benchmark of Prof. Dennis Shasha's three test queries against the live "
             "protocols.io API, comparing the raw query against the concept-expansion "
             "pipeline (NCBI Taxonomy + Europe PMC term grounding, multi-signal re-rank).\n")

    L.append("## protocols.io search-syntax findings (empirical)\n")
    L.append("Probed directly against `/api/v3/protocols` (see "
             "`protocolnerd-backend/probe_protocolsio_syntax.py`):\n")
    L.append("| Syntax | Supported? | Evidence |")
    L.append("|---|---|---|")
    L.append("| Single keyword | ✅ yes | `rice`→209, `CRISPR`→346, `multiplex`→400 |")
    L.append("| Adjacent 2-word phrase | ✅ only if the phrase occurs verbatim | "
             "`in planta`→10, `gene editing`→36, but `drought tolerance`→0 |")
    L.append("| Quotes `\"...\"` | ❌ matched literally, not as operator | "
             "`\"drought tolerance\"`→0 |")
    L.append("| `OR` / `AND` | ❌ matched literally | `drought OR rice`→0 |")
    L.append("| Parentheses | ❌ matched literally | `(drought OR water deficit) rice`→0 |")
    L.append("| Pagination | `page_id` is **0-indexed** | first page = `page_id=0` |")
    L.append("\n**Design consequence:** send many *short* single/adjacent-phrase probes "
             "and merge + re-rank client-side; do not use quotes/OR/parentheses.\n")

    # Summary comparison table
    L.append("## Summary: original vs. expanded\n")
    L.append("| Query | Raw query hits | Best protocol after expansion | Why it matches | Limitations |")
    L.append("|---|---|---|---|---|")
    for r in runs:
        q = r["query"]
        raw_hits = r["baseline_total"]
        if r["expanded"]:
            best = r["expanded"][0]
            best_title = best["title"][:60].replace("|", "\\|")
            why = best["why"][:90].replace("|", "\\|")
        else:
            best_title, why = "— none —", "no match even after expansion"
        lim = _limitations(r).replace("|", "\\|")
        L.append(f"| {q[:60]} | {raw_hits} | {best_title} | {why} | {lim} |")
    L.append("")

    # Per-query detail
    for r in runs:
        L.append(f"## Query: {r['query']}\n")
        c = r["concepts"]
        L.append(f"**Extracted concepts** — organisms: `{c['organisms']}`, "
                 f"methods: `{c['methods']}`, goals: `{c['goals']}`, actions: `{c['actions']}`\n")

        L.append("**Concept expansion (grounded synonyms / related terms):**\n")
        if r["expansions"]:
            L.append("| Concept | Expanded to | Source |")
            L.append("|---|---|---|")
            for k, v in r["expansions"].items():
                src = "NCBI Taxonomy" if any(ch.isupper() for ch in " ".join(v)) and k in c["organisms"] else "Europe PMC / LLM / map"
                L.append(f"| {k} | {', '.join(v)} | {src} |")
        else:
            L.append("_No expansions generated._")
        L.append("")

        L.append(f"**Search probes fired** (`total_results` per probe): ")
        probe_bits = [f"`{p}`→{r['probe_totals'].get(p, '?')}" for p in r["probes"]]
        L.append(", ".join(probe_bits) + "\n")

        L.append(f"**Baseline (raw query):** `total_results={r['baseline_total']}`"
                 + (" — no usable results\n" if r["baseline_total"] == 0 else "\n"))

        L.append("**Expanded results (top 5, multi-signal ranked):**\n")
        if r["expanded"]:
            L.append("| # | Score | Protocol | Why it matches | Signals (T/O/M/D/S) |")
            L.append("|---|---|---|---|---|")
            for i, x in enumerate(r["expanded"], 1):
                title = x["title"][:55].replace("|", "\\|")
                link = x.get("url") or x.get("uri")
                if link:
                    title = f"[{title}]({link})"
                why = x["why"][:80].replace("|", "\\|")
                s = x["signals"]
                sig = f"{s['title']}/{s['organism']}/{s['method']}/{s['description']}/{s['synonym']}"
                L.append(f"| {i} | {x['score']} | {title} | {why} | {sig} |")
        else:
            L.append("_No protocols matched even after expansion._")
        L.append("")
        L.append(f"**Limitations:** {_limitations(r)}\n")

    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nReport written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-external", action="store_true",
                    help="Skip NCBI/EuropePMC/Ollama; use offline fallbacks only.")
    ap.add_argument("--query", action="append",
                    help="Run a custom query instead of the Shasha set (repeatable).")
    args = ap.parse_args()

    queries = args.query or SHASHA_QUERIES
    use_external = not args.no_external

    runs = [run_one(q, use_external) for q in queries]

    out = Path(__file__).parent.parent / "data" / "benchmark_results.md"
    out.parent.mkdir(exist_ok=True)
    write_report(runs, out)


if __name__ == "__main__":
    main()
