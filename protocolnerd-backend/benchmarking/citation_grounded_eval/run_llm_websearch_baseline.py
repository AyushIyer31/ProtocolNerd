"""
Baseline: LLM (Claude) with a real web search tool, scoped to protocols.io via
allowed_domains, as a second "some other attempt to find protocols" comparison
against DP (method M), per Prof. Shasha's follow-up request for another
attempt beyond the keyword-prompt baseline.

Unlike DP, which sends a fixed one-shot prompt and queries protocols.io's own
search API directly, this baseline lets the LLM decide its own search queries
and iterate (up to MAX_USES searches per paper) against a general web search
index filtered to protocols.io, the way a researcher using a search-enabled
assistant might. Runs on the same 100 papers as the citation-grounded find-rate
test and DP (citation_ground_truth_Biology_100.csv), using each paper's abstract.

Matching: a candidate URL is scored against ALL of a paper's confirmed target
protocol ids (some papers have more than one). A raw DOI/URI substring match is
tried first. If that fails, the candidate's /view/<slug> is resolved against
protocols.io's OWN search API (search_live, same one DP uses to validate its
own hits) rather than compared as a string, so a legacy or duplicate slug for
the same underlying protocol still counts as a hit. This was confirmed
necessary during a 5-paper pilot: for protocol id 121741 (a Nature Protocol
Exchange protocol mirrored into protocols.io), the corpus stores
uri=...-d25m8g46, the web search tool returned the identical title under
...-3byl4w4wrvo5, and protocols.io's own search API for that exact title
returns only id 121741 -- confirming the two links are the same protocol, not
two different ones, something no amount of string comparison between the two
raw URLs could ever show. The resolution step reuses that same official search
API to confirm identity rather than guessing from the URLs alone, and only
accepts a resolution when the returned item's own uri stem exactly matches the
candidate's, so a short, generic word-dropped query can't accidentally borrow
an unrelated protocol's id.

Usage: python run_llm_websearch_baseline.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarking/ -- for _bootstrap, systems, etc.
import _bootstrap  # noqa: F401
from _bootstrap import CACHE_DIR, RESULTS_DIR, BACKEND_DIR
from systems import get_index  # type: ignore

sys.path.insert(0, str(BACKEND_DIR))
from protocolsio_client import search_live  # type: ignore

import anthropic

SCRIPT_DIR = Path(__file__).resolve().parent
# One canonical input for the whole experiment: paper metadata, the query, and the
# Methods-confirmed protocol IDs in a single file.
CANONICAL = SCRIPT_DIR / "citation_ground_truth_Biology_100.csv"
OUT_CSV = RESULTS_DIR / "websearch_llm_baseline_protocolsio_100.csv"
CACHE_PATH = CACHE_DIR / "websearch_llm_baseline.json"
RESOLVE_CACHE_PATH = CACHE_DIR / "websearch_llm_baseline_slug_resolution.json"
PAIRED_VECTORS_PATH = SCRIPT_DIR / "paired_vectors.json"

MODEL = "claude-sonnet-4-6"
MAX_USES = 5
CALL_TIMEOUT_SEC = 90
K = 10

SEARCH_PROMPT = (
    "You are helping locate the specific experimental protocol a paper used. "
    "Given the abstract below, search protocols.io to find the actual published "
    "protocol that this paper's methods most likely follow. "
    "Return ONLY a JSON array of up to 10 candidate protocols.io URLs, ranked "
    "most-likely-first. No prose, no explanation, just the JSON array."
)


def _call_with_timeout(client, **kwargs):
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(client.messages.create, **kwargs)
    try:
        return fut.result(timeout=CALL_TIMEOUT_SEC)
    except FutureTimeoutError:
        return None
    except Exception as e:
        print(f"    LLM call failed: {e}", flush=True)
        return None
    finally:
        pool.shutdown(wait=False)


def _doi_slug(doi: str) -> str:
    m = re.search(r"protocols\.io\.([a-z0-9]+)", doi or "", re.IGNORECASE)
    return m.group(1).lower() if m else ""


def _slug_from_url(url: str) -> str:
    m = re.search(r"protocols\.io/view/([^/?#]+)", url or "", re.IGNORECASE)
    slug = m.group(1).lower() if m else ""
    return re.sub(r"\.(html|pdf)$", "", slug)


def _title_stem(slug: str) -> str:
    """Strip protocols.io's trailing random token ("<title>-<token>") so two
    links for the same title with different tokens compare equal."""
    parts = slug.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else slug


def _resolve_via_protocolsio_search(slug: str, resolve_cache: Dict[str, Any]) -> Optional[str]:
    if slug in resolve_cache:
        return resolve_cache[slug]
    stem = _title_stem(slug)
    query_text = stem.replace("-", " ").strip()
    resolved: Optional[str] = None
    if query_text:
        words = query_text.split()
        for n in range(len(words), 0, -1):
            attempt = " ".join(words[:n])
            try:
                _total, items = search_live(attempt, max_results=5)
            except Exception:
                items = []
            if items:
                for it in items:
                    if _title_stem((it.get("uri") or "").lower()) == stem:
                        resolved = str(it.get("id"))
                        break
                break
    resolve_cache[slug] = resolved
    return resolved


def _cheap_match(url: str, target_ids: Set[str], target_lookup: Dict[str, Tuple[str, str]]) -> bool:
    url_l = (url or "").lower()
    for tid in target_ids:
        doi_slug, uri = target_lookup.get(tid, ("", ""))
        if doi_slug and doi_slug in url_l:
            return True
        if uri and uri in url_l:
            return True
    return False


_STOPWORDS = {"of", "the", "and", "for", "with", "in", "on", "to", "a", "an", "using", "from"}


def _meaningful_words(stem: str) -> Set[str]:
    return {w for w in stem.split("-") if w and w not in _STOPWORDS and len(w) > 2}


def _worth_resolving(slug: str, target_lookup: Dict[str, Tuple[str, str]]) -> bool:
    """Cheap prefilter before hitting protocols.io's live search: only bother
    resolving a candidate slug if it shares at least one non-trivial word with
    a target's title, since a genuinely unrelated protocol (most candidates,
    for most misses) shares none. Without this, a paper with 10 wrong
    candidates triggers 10 full progressive-word-drop searches for nothing --
    this is what made the first run of this script far too slow to finish."""
    cand_words = _meaningful_words(_title_stem(slug))
    if not cand_words:
        return False
    for _tid, (_doi, uri) in target_lookup.items():
        if uri and cand_words & _meaningful_words(_title_stem(uri)):
            return True
    return False


def _determine_hit_rank(candidate_urls: List[str], target_ids: Set[str],
                        target_lookup: Dict[str, Tuple[str, str]],
                        resolve_cache: Dict[str, Any]) -> Optional[int]:
    # Tier 1/2: cheap exact DOI/URI substring match, in rank order -- covers
    # the common case (including every hit DP would also recognize) with zero
    # network calls.
    for rank, url in enumerate(candidate_urls[:K], 1):
        if _cheap_match(url, target_ids, target_lookup):
            return rank

    # Tier 3: only for candidates that share real title words with a target,
    # resolve concurrently against protocols.io's own search API.
    to_resolve = [(rank, _slug_from_url((url or "").lower()))
                 for rank, url in enumerate(candidate_urls[:K], 1)]
    to_resolve = [(rank, slug) for rank, slug in to_resolve
                 if slug and _worth_resolving(slug, target_lookup)]
    if not to_resolve:
        return None

    resolved: Dict[int, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_resolve_via_protocolsio_search, slug, resolve_cache): rank
               for rank, slug in to_resolve}
        for fut, rank in futs.items():
            try:
                resolved[rank] = fut.result()
            except Exception:
                resolved[rank] = None

    hit_ranks = [rank for rank, rid in resolved.items() if rid in target_ids]
    return min(hit_ranks) if hit_ranks else None


def _extract_json_array(content_blocks) -> List[str]:
    text = "".join(b.text for b in content_blocks if getattr(b, "type", None) == "text")
    text = text.strip()
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(u) for u in arr]
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(u) for u in arr]
        except Exception:
            pass
    return []


def main():
    protocols = get_index()["protocols"]
    id_to_protocol = {str(p.get("id")): p for p in protocols}

    abstracts = {row["pmid"]: row.get("citing_abstract", "")
                 for row in csv.DictReader(open(CANONICAL))}

    papers = list(csv.DictReader(open(CANONICAL)))
    print(f"Running LLM web-search baseline (scoped to protocols.io) for {len(papers)} papers...", flush=True)

    cache: Dict[str, Any] = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    resolve_cache: Dict[str, Any] = (json.loads(RESOLVE_CACHE_PATH.read_text())
                                    if RESOLVE_CACHE_PATH.exists() else {})
    client = anthropic.Anthropic()

    hits = 0
    total = 0
    for i, r in enumerate(papers, 1):
        pmid = r["pmid"]
        target_ids = set(r["used_in_methods_protocol_ids"].split("|"))
        target_lookup = {tid: (_doi_slug(id_to_protocol.get(tid, {}).get("doi", "")),
                               id_to_protocol.get(tid, {}).get("uri", "").lower())
                         for tid in target_ids}

        if pmid in cache and "candidate_urls" in cache[pmid]:
            result = cache[pmid]
        else:
            abstract = abstracts.get(pmid, "")
            if not abstract:
                print(f"  [{i:>3}/{len(papers)}] pmid={pmid} NO ABSTRACT, skipping", flush=True)
                continue
            resp = _call_with_timeout(
                client, model=MODEL, max_tokens=1024, system=SEARCH_PROMPT,
                messages=[{"role": "user", "content": f"Abstract: {abstract}"}],
                tools=[{"type": "web_search_20250305", "name": "web_search",
                       "max_uses": MAX_USES, "allowed_domains": ["protocols.io"]}],
            )
            if resp is None:
                print(f"  [{i:>3}/{len(papers)}] pmid={pmid} CALL FAILED, skipping", flush=True)
                continue
            queries = [b.input.get("query") for b in resp.content if getattr(b, "type", None) == "server_tool_use"]
            candidate_urls = _extract_json_array(resp.content)
            result = {"queries_issued": queries, "candidate_urls": candidate_urls,
                     "stop_reason": resp.stop_reason}
            cache[pmid] = result

        candidate_urls = result.get("candidate_urls", [])
        hit_rank = _determine_hit_rank(candidate_urls, target_ids, target_lookup, resolve_cache)

        total += 1
        is_hit = hit_rank is not None
        hits += is_hit
        print(f"  [{i:>3}/{len(papers)}] pmid={pmid}  candidates={len(candidate_urls)}  hit={is_hit}"
              + (f" (rank {hit_rank})" if is_hit else ""), flush=True)

        result["pmid"] = pmid
        result["target_protocol_ids"] = sorted(target_ids)
        result["is_hit"] = is_hit
        result["hit_rank"] = hit_rank
        cache[pmid] = result

        if i % 5 == 0:
            CACHE_PATH.write_text(json.dumps(cache, indent=1))
            RESOLVE_CACHE_PATH.write_text(json.dumps(resolve_cache, indent=1))

    CACHE_PATH.write_text(json.dumps(cache, indent=1))
    RESOLVE_CACHE_PATH.write_text(json.dumps(resolve_cache, indent=1))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "target_protocol_ids", "queries_issued",
                                         "candidate_urls", "is_hit", "hit_rank"])
        w.writeheader()
        for r in papers:
            pmid = r["pmid"]
            row = cache.get(pmid, {})
            if "is_hit" not in row:
                continue
            w.writerow({"pmid": pmid, "target_protocol_ids": "|".join(row.get("target_protocol_ids", [])),
                       "queries_issued": json.dumps(row.get("queries_issued", [])),
                       "candidate_urls": json.dumps(row.get("candidate_urls", [])),
                       "is_hit": row.get("is_hit"), "hit_rank": row.get("hit_rank")})
    print(f"wrote results -> {OUT_CSV}")
    print(f"\nLLM web-search (scoped) find rate: {hits}/{total} = {hits/total:.1%}")

    pv = json.loads(PAIRED_VECTORS_PATH.read_text())
    order = pv["pmid_order"]
    vec = [1 if cache.get(pmid, {}).get("is_hit") else 0 for pmid in order]
    pv["web_search_llm"] = vec
    PAIRED_VECTORS_PATH.write_text(json.dumps(pv, indent=1))
    print(f"appended 'web_search_llm' vector ({sum(vec)}/{len(vec)}) -> {PAIRED_VECTORS_PATH}")


if __name__ == "__main__":
    main()
