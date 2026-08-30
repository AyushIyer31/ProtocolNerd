#!/usr/bin/env python3
"""Methods-confirm the chemistry citation-grounded set.

`citation_ground_truth_chemistry.csv` is citation-LINKED: a paper cites a
protocol and the two are topically close (embedding relevance filter). That is
weaker ground truth than the biology set, whose headline find rate required an
LLM to confirm, from the citing paper's own METHODS section, that the authors
actually USED the protocol.

This applies that same gate to chemistry so the two numbers are comparable, and
writes a FINAL file with the same schema the runner already consumes:

    python confirm_methods_chemistry.py

The evaluated chemistry set is FROZEN at citation_ground_truth_Chemistry_100.csv
(n=100, matching biology). This script regenerates the confirmed pool it was derived
from and writes to a separate file; it does not reproduce the frozen set, and the
selection that produced it is recorded in CHEMISTRY_SET_PROVENANCE.md. To run the
find rate on the frozen set:

    python run_citation_ground_truth_test.py \
      --queries-csv citation_ground_truth_Chemistry_100.csv \
      --out ../results/citation_ground_truth_report_chemistry_100.csv \
      --cache-name rerank_finalchem_sonnet

Rows whose full text is not open-access (no fetchable Methods section) cannot be
confirmed either way and are dropped, exactly as in the biology pipeline. Rerun
any time; it overwrites its output and caches verdicts so a rerun is cheap.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401
from llm_providers import call_llm  # type: ignore
from pubmed_client import fetch_fulltext, extract_methods  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent

# Identical to the biology gate (gen_multi_target_ground_truth.METHODS_JUDGE_PROMPT)
# except the role line, which names the domain being judged. Same specialization
# we apply to the re-ranking prompt: the criterion is unchanged, only the expert.
METHODS_JUDGE_PROMPT = (
    "You are an expert chemist. Given the METHODS section and ABSTRACT of a "
    "research paper and the title and description of one specific protocol, judge "
    "whether the methods section describes the paper's authors actually USING that "
    "protocol to run part of their experiment, not just citing it elsewhere in the "
    "paper (introduction, discussion) without using it, and that the abstract does "
    "not directly name the protocol. Answer with exactly one word: YES or NO."
)

LLM_TIMEOUT_SEC = 45
# MUST match the judge biology's ground truth was built with. That pass
# (gen_multi_target_ground_truth.py) passes no model to call_llm, so it resolves
# to the provider default, claude-sonnet-4-6. Judging chemistry with a different
# model would mean a different bar for "actually used", and the two find rates
# would no longer be comparable -- which is the entire point of this gate.
MODEL = os.getenv("METHODS_JUDGE_MODEL", "claude-sonnet-4-6")
# Full-text fetch hits Europe PMC / PMC / Unpaywall. Keep concurrency low: those
# are shared public services and Europe PMC throttles without documenting a limit.
WORKERS = 4

FIELDS = ["pmid", "protocol_id", "protocol_title", "protocol_doi", "protocol_link",
          "citing_pmid", "citing_title", "citing_abstract", "relevance_score",
          "query", "used_in_methods_protocol_ids"]

_lock = threading.Lock()
_done = [0]


def _judge(methods: str, abstract: str, p_title: str) -> Optional[bool]:
    user = (f"PROTOCOL TITLE: {p_title}\n\n"
            f"PAPER ABSTRACT:\n{abstract[:2500]}\n\n"
            f"PAPER METHODS SECTION:\n{methods[:9000]}")
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(call_llm,
                          messages=[{"role": "system", "content": METHODS_JUDGE_PROMPT},
                                    {"role": "user", "content": user}],
                          temperature=0.0, provider="claude", model=MODEL)
        raw = (fut.result(timeout=LLM_TIMEOUT_SEC) or "").strip().upper()
    except (FutureTimeoutError, Exception):  # noqa: BLE001
        return None
    finally:
        pool.shutdown(wait=False)
    if raw.startswith("YES"):
        return True
    if raw.startswith("NO"):
        return False
    return None


def _process(row: Dict[str, Any], cache: Dict[str, Any], total: int) -> Dict[str, Any]:
    pmid = (row.get("citing_pmid") or "").strip()
    # Model is part of the key: a verdict is only valid for the judge that made
    # it, so changing MODEL must re-judge rather than replay stale verdicts.
    key = f"{MODEL}:{pmid}:{row.get('protocol_id')}"
    if key in cache:
        verdict, reason = cache[key]["verdict"], cache[key]["reason"]
    else:
        verdict, reason = None, ""
        try:
            fulltext = fetch_fulltext(pmid, "")
            methods = extract_methods(fulltext) if fulltext else ""
        except Exception:  # noqa: BLE001
            methods = ""
        if len(methods) < 200:
            reason = "no open-access methods section"
        else:
            verdict = _judge(methods, row.get("citing_abstract") or "",
                             row.get("protocol_title") or "")
            reason = "judged" if verdict is not None else "judge failed"
        cache[key] = {"verdict": verdict, "reason": reason}

    with _lock:
        _done[0] += 1
        tag = {True: "CONFIRMED", False: "rejected", None: "unusable"}[verdict]
        print(f"  [{_done[0]:>3}/{total}] {tag:<10} {reason:<28} "
              f"{(row.get('protocol_title') or '')[:44]}", flush=True)
    return {"row": row, "verdict": verdict, "reason": reason}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(SCRIPT_DIR / "citation_ground_truth_chemistry.csv"))
    # NOT the frozen benchmark. citation_ground_truth_Chemistry_100.csv is the
    # curated n=100 set the paper reports, and re-running this script would
    # regenerate a differently-sized confirmed pool. Writing that over the frozen
    # file would silently change the evaluated set, so this defaults to a separate
    # regeneration output. See CHEMISTRY_SET_PROVENANCE.md.
    ap.add_argument("--out", default=str(SCRIPT_DIR / "citation_ground_truth_chemistry_confirmed.csv"))
    ap.add_argument("--cache", default=str(SCRIPT_DIR / ".methods_confirm_chemistry_cache.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src)))
    cache: Dict[str, Any] = {}
    cache_path = Path(args.cache)
    if cache_path.exists():
        try:
            cache = json.load(open(cache_path))
        except Exception:  # noqa: BLE001
            cache = {}
    print(f"Methods-confirming {len(rows)} chemistry pairs "
          f"({len(cache)} cached) with {MODEL}, {WORKERS} workers...\n", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(lambda r: _process(r, cache, len(rows)), rows))

    json.dump(cache, open(cache_path, "w"), indent=2)

    confirmed = [o["row"] for o in out if o["verdict"] is True]
    rejected = [o for o in out if o["verdict"] is False]
    unusable = [o for o in out if o["verdict"] is None]

    final_rows = []
    for r in confirmed:
        merged = {k: r.get(k, "") for k in FIELDS}
        merged["pmid"] = r.get("citing_pmid", "")
        # One protocol per row here, so the confirmed set is that protocol.
        merged["used_in_methods_protocol_ids"] = r.get("protocol_id", "")
        final_rows.append(merged)

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(final_rows)

    n = len(rows)
    print("\n" + "=" * 72)
    print(f"input pairs                       : {n}")
    print(f"  CONFIRMED used in Methods       : {len(confirmed)}  ({len(confirmed)/n:.0%})")
    print(f"  rejected (cited, not used)      : {len(rejected)}")
    print(f"  unusable (no OA methods / fail) : {len(unusable)}")
    print("=" * 72)
    print(f"\nwrote {len(final_rows)} rows -> {out_path.name}")

    # Composition of the confirmed set, so clustering is visible rather than implicit.
    import re, collections
    adc = sum(1 for r in final_rows
              if re.search(r"\bADC\b|antibody.drug|drug-to-antibody|conjugat", r["protocol_title"], re.I))
    rev = sum(1 for r in final_rows
              if re.search(r"\bReview\b|Recent Advances|Basic Principles|A Guide to", r["protocol_title"], re.I))
    if final_rows:
        print(f"  antibody-drug-conjugate cluster : {adc} ({adc/len(final_rows):.0%})")
        print(f"  review/overview titles          : {rev} ({rev/len(final_rows):.0%})")
        print(f"  distinct protocols              : {len({r['protocol_id'] for r in final_rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
