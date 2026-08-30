"""
Expand the existing 100-query citation-grounded set with MULTI-target ground
truth, per Prof. Shasha's follow-up feedback: for a given
citing paper P1, we need to know ALL protocols P1 cites, not just the single
one we happened to pick via a relevance filter, split into two tiers:

  - "mentioned anywhere": any protocols.io protocol in our corpus that appears
    in P1's own reference list, verified via Semantic Scholar's citation
    graph queried from P1's side this time (paper/PMID:{pmid}/references),
    the mirror image of how X's citing papers were found originally.
  - "used in methods": the subset of the above that an LLM judges, from P1's
    actual Methods section text (existing production fetch_fulltext /
    extract_methods plumbing -- PMC / Europe PMC open-access only), to be
    genuinely used rather than just cited in passing elsewhere in the paper.

This also removes a bias in the original single-target design: picking the
ONE highest-relevance citing paper per protocol implicitly favored "easy",
topically-obvious pairs. Capturing every protocol a paper cites removes that
cherry-pick from the ground truth itself.

Usage: python gen_multi_target_ground_truth.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarking/ -- for _bootstrap, systems, etc.
import _bootstrap  # noqa: F401
from systems import get_index  # type: ignore
from protocol_rag import _draftjs_to_text  # type: ignore
from llm_providers import call_llm  # type: ignore
from pubmed_client import fetch_fulltext, extract_methods  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
S2_API = "https://api.semanticscholar.org/graph/v1"
LLM_TIMEOUT_SEC = 45
METHODS_JUDGE_PROMPT = (
    "You are an expert biologist. Given the METHODS section and ABSTRACT of a "
    "research paper and the title and description of one specific protocol, judge "
    "whether the methods section describes the paper's authors actually USING that "
    "protocol to run part of their experiment, not just citing it elsewhere in the "
    "paper (introduction, discussion) without using it, and that the abstract does "
    "not directly name the protocol. Answer with exactly one word: YES or NO."
)

ap = argparse.ArgumentParser()
ap.add_argument("--queries-csv", default=str(SCRIPT_DIR / "citation_ground_truth_Biology_100.csv"))
ap.add_argument("--cache-name", default="rerank_final100_haiku")
ap.add_argument("--out", default=str(SCRIPT_DIR / "multi_target_confirmed.csv"))
_args, _ = ap.parse_known_args()

SOURCE_CSVS = [(Path(_args.queries_csv), _args.cache_name)]
OUT_CSV = Path(_args.out)
FIELDNAMES = ["pmid", "query", "cache_name", "original_x_ids",
             "mentioned_protocol_ids", "original_x_confirmed_in_references",
             "fulltext_available", "methods_text_len", "used_in_methods_protocol_ids"]


def _call_llm_with_timeout(**kwargs) -> Optional[str]:
    """Fresh single-use executor per call -- see gen_citation_ground_truth.py for why:
    a shared pool lets one slow call delay every call queued behind it even after we've
    given up waiting on it."""
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(call_llm, **kwargs)
    try:
        return fut.result(timeout=LLM_TIMEOUT_SEC)
    except FutureTimeoutError:
        return None
    except Exception:
        return None
    finally:
        pool.shutdown(wait=False)


_rate_limit_hits = 0


def _curl_json(url: str) -> Optional[Dict[str, Any]]:
    global _rate_limit_hits
    try:
        out = subprocess.run(
            ["curl", "-s", "--max-time", "10", "-w", "\n%{http_code}", url],
            capture_output=True, text=True, timeout=12)
    except Exception:
        return None
    body, _, code = out.stdout.rpartition("\n")
    if code.strip() == "429":
        _rate_limit_hits += 1
        if _rate_limit_hits <= 5 or _rate_limit_hits % 20 == 0:
            print(f"  Semantic Scholar 429 (hit #{_rate_limit_hits}); backing off.", flush=True)
        time.sleep(min(2.0 + _rate_limit_hits * 0.5, 20.0))
        return None
    if code.strip() == "404":
        return None
    if code.strip() and not code.strip().startswith("2"):
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


def _norm_doi(doi: str) -> str:
    doi = ((doi or "").strip().lower()
          .replace("dx.doi.org/", "").replace("https://doi.org/", "").replace("http://doi.org/", ""))
    return re.sub(r"/v\d+$", "", doi)


def _s2_references(pmid: str) -> List[Dict[str, Any]]:
    url = f"{S2_API}/paper/PMID:{pmid}/references?fields=externalIds,title&limit=1000"
    d = _curl_json(url)
    return [r.get("citedPaper", {}) for r in (d or {}).get("data") or [] if r.get("citedPaper")]


def main():
    protocols = get_index()["protocols"]
    doi_to_protocol: Dict[str, Dict[str, Any]] = {}
    id_to_protocol: Dict[str, Dict[str, Any]] = {}
    for p in protocols:
        doi = _norm_doi(p.get("doi") or "")
        if doi:
            doi_to_protocol[doi] = p
        id_to_protocol[str(p.get("id"))] = p
    print(f"{len(doi_to_protocol)} protocols indexed by normalized DOI.", flush=True)

    # Dedupe P1s across both source sets by PMID, keeping every X each was originally
    # paired with (a P1 could in principle have been the strongest match for more than
    # one protocol, though that's rare).
    p1s: Dict[str, Dict[str, Any]] = {}
    for path, cache_name in SOURCE_CSVS:
        for row in csv.DictReader(open(path)):
            pmid = row["citing_pmid"]
            if pmid not in p1s:
                p1s[pmid] = {
                    "pmid": pmid, "query": row["query"], "cache_name": cache_name,
                    "original_x_ids": set(),
                    "abstract": row.get("citing_abstract", ""),
                }
            p1s[pmid]["original_x_ids"].add(str(row["protocol_id"]))
    print(f"{len(p1s)} unique P1 papers.", flush=True)

    already_done = set()
    n_done = 0
    if OUT_CSV.exists():
        with OUT_CSV.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add(row["pmid"])
                n_done += 1
        if n_done:
            print(f"Resuming: {n_done} rows already done, skipping.", flush=True)

    out_f = OUT_CSV.open("a" if n_done else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not n_done:
        writer.writeheader()
    out_f.flush()

    items = list(p1s.items())
    for i, (pmid, info) in enumerate(items, 1):
        if pmid in already_done:
            continue

        # 1. Mentioned anywhere: P1's own reference list, cross-checked against corpus.
        refs = _s2_references(pmid)
        time.sleep(1.0)
        mentioned_ids: Set[str] = set()
        for r in refs:
            doi = _norm_doi((r.get("externalIds") or {}).get("DOI", ""))
            if doi and doi in doi_to_protocol:
                mentioned_ids.add(str(doi_to_protocol[doi]["id"]))
        original_confirmed = bool(mentioned_ids & info["original_x_ids"])
        # Always include the original X(s): that citation edge is already verified
        # (it's how P1 was found, via the reverse citations endpoint), even if this
        # independent references-side lookup missed it due to DOI formatting or S2
        # graph asymmetry.
        mentioned_ids |= info["original_x_ids"]

        # 2. Used in methods: LLM judgment over the actual Methods section, OA only.
        fulltext = fetch_fulltext(pmid, "")
        methods_text = extract_methods(fulltext) if fulltext else ""
        used_in_methods: Set[str] = set()
        if methods_text and len(methods_text) >= 200:
            for pid in mentioned_ids:
                proto = id_to_protocol.get(pid)
                if not proto:
                    continue
                proto_desc = ((proto.get("title") or "") + ". " +
                             _draftjs_to_text(proto.get("description")))[:500]
                resp = _call_llm_with_timeout(
                    messages=[{"role": "system", "content": METHODS_JUDGE_PROMPT},
                              {"role": "user",
                               "content": f"METHODS SECTION:\n{methods_text}\n\n"
                                          f"ABSTRACT:\n{info.get('abstract', '')}\n\n"
                                          f"PROTOCOL:\n{proto_desc}"}],
                    temperature=0.0,
                )
                if (resp or "").strip().upper().startswith("YES"):
                    used_in_methods.add(pid)

        row = {
            "pmid": pmid,
            "query": info["query"],
            "cache_name": info["cache_name"],
            "original_x_ids": "|".join(sorted(info["original_x_ids"])),
            "mentioned_protocol_ids": "|".join(sorted(mentioned_ids)),
            "original_x_confirmed_in_references": original_confirmed,
            "fulltext_available": bool(methods_text),
            "methods_text_len": len(methods_text),
            "used_in_methods_protocol_ids": "|".join(sorted(used_in_methods)),
        }
        writer.writerow(row)
        out_f.flush()
        n_done += 1
        if n_done % 10 == 0 or i == len(items):
            print(f"  [{n_done}/{len(items)}] pmid={pmid} mentioned={len(mentioned_ids)} "
                  f"fulltext_available={bool(methods_text)} used_in_methods={len(used_in_methods)} "
                  f"rate_limit_hits={_rate_limit_hits}", flush=True)

    out_f.close()
    print(f"\nwrote {n_done} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
