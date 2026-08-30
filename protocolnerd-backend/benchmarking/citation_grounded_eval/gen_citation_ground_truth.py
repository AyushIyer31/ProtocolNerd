"""
Build a REAL citation-grounded test set (addresses Prof. Shasha's Q1/Q2):
sample protocols.io protocols that have at least one real citing paper
(verified via Semantic Scholar's citation graph, not our own guess), pick one
citing paper per protocol, and generate a query from ONLY that paper's abstract.

This replaces the earlier self-retrieval proxy (query built from item X's own
abstract, checking if X comes back) with actual ground truth: paper P1 really
does cite protocol X in its reference list, and the query never sees X's
title, materials, or steps -- only P1's abstract.

Originally built on OpenAlex; switched to Semantic Scholar after OpenAlex
started rate-limiting us into an effective standstill (a single isolated test
request got 429'd well after the run stopped, pointing to a sustained IP-level
block from today's cumulative usage, not just request pacing). Semantic
Scholar is separate infrastructure and, as a bonus, returns each citing
paper's PMID AND abstract in the SAME response as the citation list -- no
separate PubMed efetch needed, fewer total calls than the OpenAlex version.

Filters on citing papers:
  - has a PMID (so results stay comparable to the rest of this project's
    PubMed-based methodology)
  - real abstract, >200 chars

One citing paper is kept per protocol (the one with the strongest relevance
score, see below), so a handful of extremely-cited "mega" protocols can't
dominate the sample -- each protocol contributes at most one query.

Usage: python gen_citation_ground_truth.py --n 100 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarking/ -- for _bootstrap, systems, etc.
import _bootstrap  # noqa: F401
from systems import get_index  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
from protocol_rag import _draftjs_to_text  # type: ignore
from dense_index import embed  # type: ignore
from llm_providers import call_llm  # type: ignore
from pubmed_client import fetch_fulltext, extract_methods  # type: ignore

S2_API = "https://api.semanticscholar.org/graph/v1"
# Below this, a citing paper's title is treated as topically unrelated to the protocol
# (calibrated against real examples: unrelated pairs scored -0.06 to 0.10; genuinely
# related pairs scored 0.33-0.67+ using protocol title+description vs. paper title).
RELEVANCE_THRESHOLD = 0.20
QUERY_PROMPT = (
    "You are a bench biologist. Given a lab protocol or a research paper's title and "
    "abstract, write ONE natural-language sentence that a scientist would type into a "
    "search tool when they need this exact item. Describe the experimental goal, "
    "organism, and technique in your own words, in a single sentence. Do NOT copy the "
    "title verbatim, paraphrase and use everyday phrasing. Sound like a real request a "
    "scientist would type. Exactly one sentence, under 30 words. No quotes, no preamble. "
    "Return only the sentence."
)
LLM_TIMEOUT_SEC = 45  # the Anthropic SDK's own retry/backoff has no upper bound we
                      # control here, so a hung or repeatedly-retried call must be
                      # abandoned by US rather than block the whole scan indefinitely.


def _call_llm_with_timeout(**kwargs) -> Optional[str]:
    """A FRESH single-use executor per call, on purpose: a shared/reused pool with
    max_workers=1 means one slow call (past our timeout, but not actually dead --
    e.g. stuck in the SDK's own retry/backoff for minutes) keeps the one worker busy,
    so the NEXT call queued behind it inherits that same delay even though we already
    gave up waiting on the first one. A disposable executor means a slow call can only
    ever delay itself, never a later, unrelated call."""
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(call_llm, **kwargs)
    try:
        return fut.result(timeout=LLM_TIMEOUT_SEC)
    except FutureTimeoutError:
        print(f"  LLM call exceeded {LLM_TIMEOUT_SEC}s, abandoning (its thread is left "
              f"running in the background but can no longer block anything else).")
        return None
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None
    finally:
        pool.shutdown(wait=False)


_rate_limit_hits = 0


def _curl_json(url: str) -> Optional[Dict[str, Any]]:
    """Fetches JSON, checking the HTTP status explicitly -- a silently-parsed error
    body must not be misread as a legitimate empty result."""
    global _rate_limit_hits
    try:
        out = subprocess.run(
            ["curl", "-s", "--max-time", "10", "-w", "\n%{http_code}", url],
            capture_output=True, text=True, timeout=12)
    except Exception as e:
        print(f"  curl failed: {e}")
        return None
    body, _, code = out.stdout.rpartition("\n")
    if code.strip() == "429":
        _rate_limit_hits += 1
        if _rate_limit_hits <= 5 or _rate_limit_hits % 20 == 0:
            print(f"  Semantic Scholar 429 rate-limited (hit #{_rate_limit_hits}); backing off.")
        time.sleep(min(2.0 + _rate_limit_hits * 0.5, 20.0))
        return None
    if code.strip() == "404":
        return None
    if code.strip() and not code.strip().startswith("2"):
        print(f"  Semantic Scholar non-200 ({code.strip()}) for {url[:80]}")
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


def _s2_doi(doi: str) -> str:
    """Semantic Scholar doesn't recognize protocols.io's version-suffixed DOIs
    (e.g. trailing /v1, /v2) -- strip them, same as the base DOI without a version."""
    doi = doi.strip().replace("dx.doi.org/", "").replace("https://doi.org/", "")
    return re.sub(r"/v\d+$", "", doi)


def _s2_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    url = f"{S2_API}/paper/DOI:{_s2_doi(doi)}?fields=citationCount,title"
    return _curl_json(url)


def _s2_citing_works(doi: str, limit: int = 30) -> List[Dict[str, Any]]:
    url = f"{S2_API}/paper/DOI:{_s2_doi(doi)}/citations?fields=title,externalIds,abstract&limit={limit}"
    d = _curl_json(url)
    return [c.get("citingPaper", {}) for c in (d or {}).get("data", []) if c.get("citingPaper")]


_STOP = set(("a an the of for in on to and or with from how do i need get using use is are this "
             "that my me can could into out at by so it well test find looking way step protocol "
             "protocols measure make want").split())


def _ctoks(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}


def _title_overlap(query: str, title: str) -> float:
    q, t = _ctoks(query), _ctoks(title)
    if not t:
        return 0.0
    return len(q & t) / len(t)


FIELDNAMES = ["protocol_id", "protocol_title", "protocol_doi", "protocol_link", "citing_pmid",
             "citing_title", "citing_abstract", "relevance_score", "query"]


def generate(n: int, seed: int, out_path: Path, exclude_csvs: Optional[List[Path]] = None,
            require_fulltext: bool = False, only_ids: Optional[set] = None,
            max_per_protocol: int = 1) -> None:
    protocols = get_index()["protocols"]
    with_doi = [p for p in protocols if (p.get("doi") or "").strip()]
    # Restrict the pool to a specific set of protocol IDs. Used to build a
    # benchmark for a single domain (e.g. the chemistry subset of the corpus)
    # with the SAME methodology, rather than sampling the whole corpus.
    if only_ids:
        before = len(with_doi)
        with_doi = [p for p in with_doi if str(p.get("id")) in only_ids]
        print(f"Restricted pool to {len(with_doi)} of {before} DOI-bearing protocols "
              f"({len(only_ids)} IDs supplied).", flush=True)
    rng = random.Random(seed)
    rng.shuffle(with_doi)
    print(f"{len(with_doi)} protocols have a DOI; scanning for citations (Semantic Scholar)...", flush=True)

    # Resume support: if the output file already has rows (e.g. a prior run was
    # stopped), keep them and skip their protocol IDs on this pass instead of
    # rediscovering (and re-billing) the same ones.
    # With max_per_protocol > 1 a protocol that already contributed can still
    # contribute another paper, so track the COUNT per protocol rather than a
    # flat skip-set, plus the citing PMIDs already used so no paper repeats.
    from collections import Counter
    per_protocol = Counter()
    already_used_pmids = set()
    already_found_ids = set()
    n_found = 0
    if out_path.exists():
        with out_path.open("r", newline="", encoding="utf-8") as existing:
            for row in csv.DictReader(existing):
                per_protocol[row["protocol_id"]] += 1
                already_used_pmids.add(str(row.get("citing_pmid", "")).strip())
                n_found += 1
        already_found_ids = {pid for pid, c in per_protocol.items() if c >= max_per_protocol}
        if n_found:
            print(f"Resuming: {n_found} rows already in {out_path}; "
                  f"{len(already_found_ids)} protocols are at the "
                  f"max_per_protocol={max_per_protocol} cap and will be skipped.", flush=True)

    # Exclusion support: skip protocol IDs already used by EARLIER, separate
    # ground-truth sets (e.g. the 89-query and 100-query sets), so multiple sets
    # built for cross-validation never share a protocol.
    for exclude_csv in (exclude_csvs or []):
        if not exclude_csv.exists():
            continue
        excluded = set()
        with exclude_csv.open("r", newline="", encoding="utf-8") as ex_f:
            for row in csv.DictReader(ex_f):
                excluded.add(row["protocol_id"])
        already_found_ids |= excluded
        print(f"Excluding {len(excluded)} protocol IDs already used in {exclude_csv}.", flush=True)

    out_f = out_path.open("a" if n_found else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not n_found:
        writer.writeheader()
    out_f.flush()

    checked = 0
    for p in with_doi:
        if n_found >= n:
            break
        if str(p.get("id")) in already_found_ids:
            continue
        checked += 1
        # Heartbeat: prints even when nothing is found, so a long silent stretch (rate
        # limiting, a slow call, anything) is visible in the log instead of looking
        # identical to "still making normal progress" until the next actual hit.
        if checked % 25 == 0:
            print(f"  ...heartbeat: {checked} scanned, {n_found}/{n} found, "
                  f"{_rate_limit_hits} rate-limit hits so far", flush=True)
        doi = p.get("doi", "")
        work = _s2_paper_by_doi(doi)
        # S2 unauthenticated tier is 100 req/5min = 1 req/3s sustained. ~93% of scans
        # make exactly 1 call (no citations); ~7% make a 2nd. 3.5s here keeps the
        # AVERAGE comfortably under the limit without needing backoff at all -- the
        # 0.35s version this replaced averaged out well above the sustainable rate,
        # which is exactly why rate-limit hits climbed steadily over the first run.
        time.sleep(3.5)
        if not work or (work.get("citationCount") or 0) < 1:
            continue

        citing = _s2_citing_works(doi)
        time.sleep(0.5)
        candidates = [c for c in citing
                     if (c.get("externalIds") or {}).get("PubMed")
                     and (c.get("title") or "").strip()
                     and len((c.get("abstract") or "")) > 200]
        if not candidates:
            continue

        # Relevance pre-filter: score each candidate's TITLE against this protocol's
        # title+description, using the same embedding model as production. A citation
        # in a paper's reference list doesn't guarantee the paper actually USED the
        # protocol; this is a cheap check that it's at least topically about the same
        # thing, and lets us prefer the strongest match instead of a random one.
        protocol_context = ((p.get("title") or "") + ". " + _draftjs_to_text(p.get("description")))[:500]
        cand_titles = [c["title"].strip() for c in candidates]
        try:
            v_protocol = embed([protocol_context])
            v_cands = embed(cand_titles)
            sims = np.sum(v_protocol * v_cands, axis=1)
        except Exception:
            sims = np.zeros(len(candidates))
        scored = sorted(zip(candidates, sims), key=lambda x: -x[1])
        relevant = [(c, s) for c, s in scored if s >= RELEVANCE_THRESHOLD]
        if not relevant:
            continue

        # Full-text gate (require_fulltext=True runs): walk the relevance-ranked
        # candidates and take the FIRST one whose Methods section is actually
        # fetchable (PMC / Europe PMC open-access), instead of unconditionally
        # taking the top relevance match. This gates on availability only, not on
        # whether the LLM later confirms genuine usage -- gating on confirmed
        # usage would reintroduce the same cherry-picking bias the relevance
        # filter itself is already a known, accepted compromise on.
        # A protocol may contribute up to `max_per_protocol` citing papers. The
        # original design took exactly one so a heavily-cited "mega" protocol
        # could not dominate the sample; a small cap (2) keeps that guard while
        # roughly doubling the pairs available from a limited protocol pool.
        # Every extra pair is still an independent paper that must pass the same
        # relevance filter and the same Methods-section confirmation, so this
        # widens the sample without weakening the ground truth.
        taken = per_protocol.get(str(p.get("id")), 0)
        for cand, s in relevant:
            if taken >= max_per_protocol or n_found >= n:
                break
            cand_pmid = str(cand["externalIds"]["PubMed"])
            if cand_pmid in already_used_pmids:
                continue
            if require_fulltext:
                ft = fetch_fulltext(cand_pmid, "")
                if not ft or len(extract_methods(ft)) < 200:
                    continue

            abstract = (cand.get("abstract") or "").strip()
            citing_title = (cand.get("title") or "").strip()
            relevance_score = float(s)

            raw_query = _call_llm_with_timeout(
                messages=[{"role": "system", "content": QUERY_PROMPT},
                          {"role": "user", "content": f"Abstract: {abstract}"}],
                temperature=0.7,
            )
            query = (raw_query or "").strip().strip('"').strip()
            if not query or len(query.split()) < 3:
                continue
            if _title_overlap(query, citing_title) > 0.85:
                continue

            uri = (p.get("uri") or "").strip()
            protocol_link = f"https://www.protocols.io/view/{uri}" if uri else f"https://doi.org/{doi}"
            row = {
                "protocol_id": str(p.get("id")),
                "protocol_title": (p.get("title") or "").replace("\n", " ").strip(),
                "protocol_doi": doi,
                "protocol_link": protocol_link,
                "citing_pmid": cand_pmid,
                "citing_title": citing_title.replace("\n", " "),
                "citing_abstract": abstract.replace("\n", " "),
                "relevance_score": round(relevance_score, 4),
                "query": query.replace("\n", " ").strip(),
            }
            writer.writerow(row)
            out_f.flush()
            already_used_pmids.add(cand_pmid)
            n_found += 1
            taken += 1
            print(f"  [{n_found:>3}/{n}] (scanned {checked}) protocol={p.get('id')} "
                  f"citations={work.get('citationCount')} relevance={relevance_score:.2f} "
                  f"-> {query[:55]}", flush=True)

    out_f.close()
    print(f"\nwrote {n_found} rows (scanned {checked} protocols) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(SCRIPT_DIR / "citation_ground_truth_Biology_100.csv"))
    ap.add_argument("--exclude-csv", default=None,
                    help="Comma-separated CSV path(s), each with a protocol_id column, whose "
                         "IDs should be skipped (e.g. earlier ground-truth sets, to keep "
                         "multiple sets disjoint).")
    ap.add_argument("--require-fulltext", action="store_true",
                    help="Only accept a citing paper if its Methods section is actually "
                         "fetchable (PMC / Europe PMC open-access), so every row in the "
                         "output set has usable methods-tier ground truth.")
    ap.add_argument("--only-ids", default=None,
                    help="JSON file restricting the protocol pool. Either a list of IDs or "
                         "a list of objects with an 'id' field. Used to build a "
                         "single-domain benchmark (e.g. the chemistry subset).")
    ap.add_argument("--max-per-protocol", type=int, default=1,
                    help="How many citing papers one protocol may contribute (default 1, "
                         "the original behaviour). A small cap above 1 widens a limited "
                         "protocol pool while still preventing a heavily-cited protocol "
                         "from dominating; every pair passes the same filters.")
    args = ap.parse_args()
    exclude_paths = [Path(p.strip()) for p in args.exclude_csv.split(",")] if args.exclude_csv else None

    only_ids = None
    if args.only_ids:
        raw = json.load(open(args.only_ids))
        only_ids = {str(r.get("id") if isinstance(r, dict) else r) for r in raw}

    generate(args.n, args.seed, Path(args.out), exclude_paths, args.require_fulltext,
             only_ids, args.max_per_protocol)


if __name__ == "__main__":
    main()
