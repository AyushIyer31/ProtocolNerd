"""Reviewer 1.1: run the AI research tools the reviewers named against the same
100 biology papers, and score them by the same rule as everything else.

Prof. Shasha's instruction was to issue the same queries to these products, even
though they are not built for exactly this purpose, and then do the same
pairwise comparison. This script covers the two with usable API access:

  perplexity   POST https://api.perplexity.ai/v1/responses (the Agent API that
               replaced Sonar's /chat/completions). The response carries a
               search_results item whose entries have a title and url, which is
               the ranked candidate list we score.
  elicit       POST https://elicit.com/api/v2/search/papers. Returns a ranked
               papers array, each with title, doi and urls. The DOI is reported
               directly, which the matcher prefers over parsing a URL.

Consensus sells API access only to enterprises, so it is run by hand instead:
see open_web_matcher.py --template / --score.

Scoring is open_web_matcher, already verified to reproduce the paper's published
22, 34 and 19 unchanged, so every system in the comparison is judged the same
way: the confirmed protocol in the top 10, counted whether it is returned as a
Protocols.io entry, as its DOI at Springer or Nature, or as a Protocol Exchange
page under the protocol's own title.

Raw responses are cached per (engine, input, pmid), so a rerun costs nothing and
the exact answers behind the paper's numbers stay on disk.

Usage:
    python run_competitor_baselines.py --engine perplexity --input short
    python run_competitor_baselines.py --engine elicit --input short --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarking/
import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import open_web_matcher as M  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR.parents[1] / ".env"          # protocolnerd-backend/.env
CACHE_DIR = Path(_bootstrap.CACHE_DIR)
TOP_K = 10
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _env(name: str) -> str:
    """Read a key from protocolnerd-backend/.env without importing dotenv."""
    if not ENV_FILE.exists():
        raise SystemExit(f"no {ENV_FILE}")
    m = re.search(rf"^{name}=(.*)$", ENV_FILE.read_text(), re.M)
    if not m or not m.group(1).strip():
        raise SystemExit(f"{name} is not set in {ENV_FILE}")
    return m.group(1).strip().strip('"').strip("'")


# Cloudflare sits in front of elicit.com and rejects a request that sends no
# User-Agent with "error code: 1010", a browser-signature ban rather than an auth
# failure. Sending an ordinary UA is enough.
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")


def _get(url: str, headers: Dict[str, str], timeout: int = 120, retries: int = 5) -> Dict[str, Any]:
    """GET with the same backoff as _post, for key-in-header APIs."""
    delay = 5.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if not (e.code == 429 or e.code >= 500) or attempt == retries:
                raise
            wait = float(e.headers.get("Retry-After") or 0) or delay
            print(f"        {e.code}, retrying in {wait:.0f}s (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


def _post(url: str, payload: Dict[str, Any], key: str, timeout: int = 180,
          retries: int = 5) -> Dict[str, Any]:
    """POST with backoff on 429 and 5xx.

    Perplexity rate-limits a back-to-back run ("Request rate limit exceeded")
    and intermittently reports "upstream model is overloaded". Both are
    transient and both arrive as 429, so they are retried rather than counted as
    a miss, which would silently understate the tool."""
    delay = 5.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            transient = e.code == 429 or e.code >= 500
            if not transient or attempt == retries:
                raise
            wait = float(e.headers.get("Retry-After") or 0) or delay
            print(f"        {e.code}, retrying in {wait:.0f}s (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- engines

def ask_perplexity(query: str, preset: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    d = _post("https://api.perplexity.ai/v1/responses",
              {"preset": preset, "input": query}, _env("PERPLEXITY_API_KEY"))
    candidates: List[Dict[str, str]] = []
    for item in d.get("output") or []:
        if "search" in str(item.get("type", "")):
            for s in (item.get("results") or [])[:TOP_K]:
                candidates.append({"url": s.get("url") or "", "title": s.get("title") or ""})
            break
    cost = ((d.get("usage") or {}).get("cost") or {}).get("total_cost", 0.0)
    return candidates, {"cost_usd": cost, "raw": d}


# Elicit rejects a query over 2000 characters with HTTP 400. 12 of the 100
# abstracts are longer (the longest is 3081), so those are truncated at a word
# boundary and the truncation is recorded per paper and reported in the paper.
ELICIT_MAX_QUERY = 2000


def _truncate(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.8 else cut), True


def ask_elicit(query: str, corpus: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    query, truncated = _truncate(query, ELICIT_MAX_QUERY)
    d = _post("https://elicit.com/api/v2/search/papers",
              {"query": query, "corpus": corpus, "maxResults": TOP_K}, _env("ELICIT_API_KEY"))
    candidates: List[Dict[str, str]] = []
    for p in (d.get("papers") or [])[:TOP_K]:
        # The docs describe a urls array; live responses carry fullTextUrl, doi
        # and pmid instead, so fall back through those in order of usefulness.
        urls = p.get("urls") or []
        doi = (p.get("doi") or "").strip()
        url = (urls[0] if urls else p.get("fullTextUrl") or "")
        if not url and doi:
            url = f"https://doi.org/{doi}"
        if not url and p.get("pmid"):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"
        candidates.append({"url": url or "", "title": p.get("title") or "", "doi": doi})
    return candidates, {"cost_usd": 0.0, "truncated": truncated, "raw": d}


def ask_consensus(query: str, page_size: int = TOP_K) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """GET https://api.consensus.app/v1/search, authenticated with x-api-key.

    The url field is a consensus.app landing page rather than the publisher, so
    where a DOI is reported we use the doi.org link as the candidate url. That
    keeps the identifier visible to the matcher and, for a Protocols.io DOI,
    makes the host visible too."""
    params = urllib.parse.urlencode({"query": query, "page_size": page_size})
    d = _get(f"https://api.consensus.app/v1/search?{params}",
             {"x-api-key": _env("CONSENSUS_API_KEY")})
    candidates: List[Dict[str, str]] = []
    for p in (d.get("results") or [])[:TOP_K]:
        doi = (p.get("doi") or "").strip()
        candidates.append({"url": f"https://doi.org/{doi}" if doi else (p.get("url") or ""),
                           "title": p.get("title") or "", "doi": doi})
    return candidates, {"cost_usd": 0.0, "raw": d}


ENGINES = {"perplexity": "PERPLEXITY_API_KEY", "elicit": "ELICIT_API_KEY",
           "consensus": "CONSENSUS_API_KEY"}


# --------------------------------------------------------------------------- run

def load_papers(limit: int, input_kind: str) -> List[Tuple[str, str]]:
    """(pmid, text to send). Short query is what ProtocolNerd received; abstract
    is what DP and the web-search baseline received."""
    rows = list(csv.DictReader(open(M.GROUND_TRUTH)))
    rows.sort(key=lambda r: int(r["pmid"]))           # stated rule, so a subset is not cherry-picked
    out, seen = [], set()
    for r in rows:
        pmid = r["pmid"].strip()
        if pmid in seen:
            continue
        seen.add(pmid)
        text = ((r.get("query") or "").replace("Search query: ", "").strip() if input_kind == "short"
                else (r.get("citing_abstract") or "").strip())
        if text:
            out.append((pmid, text))
        if limit and len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=sorted(ENGINES), required=True)
    ap.add_argument("--input", choices=["short", "abstract"], default="short",
                    help="short: the query ProtocolNerd received. abstract: the full abstract the baselines received.")
    ap.add_argument("--limit", type=int, default=0, help="first N papers by pmid (0 = all 100)")
    ap.add_argument("--preset", default="fast", help="perplexity research intensity: fast, low, medium, high, xhigh")
    ap.add_argument("--corpus", default="elicit", choices=["elicit", "pubmed"], help="elicit corpus")
    ap.add_argument("--sleep", type=float, default=None,
                    help="seconds between calls (default: 2s for perplexity, 0 for elicit)")
    args = ap.parse_args()
    if args.sleep is None:
        args.sleep = 2.0 if args.engine == "perplexity" else (0.5 if args.engine == "consensus" else 0.0)

    tag = f"{args.engine}_{args.input}"
    if args.engine == "perplexity":
        tag += f"_{args.preset}"
    elif args.engine == "elicit":
        tag += f"_{args.corpus}"
    cache_path = CACHE_DIR / f"competitor_{tag}.json"
    cache: Dict[str, Any] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    out_path = Path(RESULTS_DIR) / f"competitor_{tag}_100.csv"

    targets = M.load_targets()
    resolve_cache = M._load_resolve_cache()
    papers = load_papers(args.limit, args.input)
    print(f"  {args.engine} | input={args.input} | {len(papers)} papers | cache {cache_path.name}")

    rows, hits, review, spend = [], 0, 0, 0.0
    for i, (pmid, text) in enumerate(papers, 1):
        if pmid in cache:
            cands, meta = cache[pmid]["candidates"], cache[pmid]["meta"]
        else:
            try:
                if args.engine == "perplexity":
                    cands, meta = ask_perplexity(text, args.preset)
                elif args.engine == "consensus":
                    cands, meta = ask_consensus(text)
                else:
                    cands, meta = ask_elicit(text, args.corpus)
            except urllib.error.HTTPError as e:
                print(f"  [{i:>3}/{len(papers)}] pmid={pmid} HTTP {e.code}: {e.read()[:160].decode()}")
                continue
            except Exception as e:  # noqa: BLE001 -- one bad call must not lose the run
                print(f"  [{i:>3}/{len(papers)}] pmid={pmid} FAILED: {type(e).__name__}: {e}")
                continue
            meta = {"cost_usd": meta.get("cost_usd", 0.0), "truncated": meta.get("truncated", False),
                    "raw": meta.get("raw")}
            cache[pmid] = {"candidates": cands, "meta": meta}
            cache_path.write_text(json.dumps(cache))       # checkpoint every call
            if args.sleep:
                time.sleep(args.sleep)

        spend += float(meta.get("cost_usd") or 0.0)
        tgt = targets.get(pmid)
        got = M.hit_rank(cands, tgt, resolve_cache) if tgt else None
        if got:
            hits += 1
            if got["needs_review"]:
                review += 1
        rows.append({
            "pmid": pmid, "input": args.input, "engine": args.engine,
            "is_hit": bool(got), "hit_rank": got["rank"] if got else "",
            "tier": got["tier"] if got else "", "matched_by": got["by"] if got else "",
            "needs_review": got["needs_review"] if got else "",
            "n_candidates": len(cands), "query_truncated": bool(meta.get("truncated")),
            "candidate_urls": json.dumps([c.get("url", "") for c in cands]),
            "candidate_titles": json.dumps([c.get("title", "") for c in cands]),
        })
        mark = (f"HIT rank {got['rank']} via {got['tier']}" + (" NEEDS REVIEW" if got["needs_review"] else "")) if got else "miss"
        print(f"  [{i:>3}/{len(papers)}] pmid={pmid} {len(cands):>2} results  {mark}   running {hits}/{len(rows)}")

    if rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    M._save_resolve_cache(resolve_cache)
    auto = hits - review
    print(f"\n  {args.engine} ({args.input}): {auto}/{len(rows)} confirmed"
          f"{f', plus {review} title matches awaiting confirmation' if review else ''}")
    if spend:
        print(f"  spend: ${spend:.4f}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
