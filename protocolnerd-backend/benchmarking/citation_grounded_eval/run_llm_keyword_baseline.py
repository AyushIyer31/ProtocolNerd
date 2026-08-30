"""
Baseline method M, per Prof. Shasha: take the same paper abstract used to build
ProtocolNerd's query, have an LLM convert it into a keyword/Boolean search
query (not a natural-language sentence), and run that directly against PubMed
and protocols.io -- no profile, no re-ranking, just what a scientist would get
from typing LLM-generated keywords into each source's own search box.

The Boolean-query half of the prompt is the verbatim "Detailed Prompt" (q2)
published in Wang, Scells, Koopman & Zuccon, "Can ChatGPT Write a Good
Boolean Query for Systematic Review Literature Search?" (SIGIR 2023) -- the
established system found via the search Prof. Shasha specified ("LLM prompt
to turn prose into a boolean keyword search"). It is used unmodified except
for substituting a paper abstract for a review title as the "information
need", plus a JSON output wrapper so the query can be parsed.

protocols.io's own search API takes a plain keyword string (no Boolean
syntax), and no published system exists for turning prose into a
protocols.io-style query (protocols.io is not a systematic-review search
target in that literature). The "keywords" field is therefore a small,
disclosed, non-literature-derived addition to the same LLM call, not part of
the Wang et al. method.

Runs on the same 100 papers as the citation-grounded find-rate test
(citation_ground_truth_Biology_100.csv), using each paper's abstract (not the
already-paraphrased NL query). Produces two result files: one for PubMed
search results, one for protocols.io search results, plus a summary find-rate
against the same used_in_methods_protocol_ids ground truth (only meaningful
for the protocols.io side, since PubMed can't contain a protocols.io
protocol).

Usage: python run_llm_keyword_baseline.py
"""
from __future__ import annotations

import csv
import re
import json
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarking/ -- for _bootstrap, systems, etc.
import _bootstrap  # noqa: F401
from _bootstrap import CACHE_DIR, RESULTS_DIR, BACKEND_DIR
from systems import get_index  # type: ignore
from llm_providers import call_llm  # type: ignore
from pubmed_client import _EUTILS, _with_key, _http_json, _http_text, _parse_pubmed_xml  # type: ignore

sys.path.insert(0, str(BACKEND_DIR))
from protocolsio_client import search_live  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
# One canonical input for the whole experiment: paper metadata, the query, and the
# Methods-confirmed protocol IDs in a single file.
CANONICAL = SCRIPT_DIR / "citation_ground_truth_Biology_100.csv"
PUBMED_OUT = RESULTS_DIR / "keyword_baseline_pubmed_100.csv"
PROTOCOLSIO_OUT = RESULTS_DIR / "keyword_baseline_protocolsio_100.csv"
KEYWORD_CACHE_PATH = CACHE_DIR / "keyword_baseline_queries.json"
LLM_TIMEOUT_SEC = 45
K = 10

# Verbatim Wang et al. 2023 (SIGIR) "Detailed Prompt" (q2), information-need
# slot substituted: "{review_title}" -> "the paper abstract provided below".
_WANG_ET_AL_Q2 = (
    "You are an information specialist who develops Boolean queries for systematic reviews. "
    "You have extensive experience developing highly effective queries for searching the "
    "medical literature. Your specialty is developing queries that retrieve as few irrelevant "
    "documents as possible and retrieve all relevant documents for your information need. Now "
    "you have your information need to conduct research on the paper abstract provided below. "
    "Please construct a highly effective systematic review Boolean query that can best serve "
    "your information need."
)

KEYWORD_PROMPT = (
    _WANG_ET_AL_Q2 + "\n\n"
    "Separately -- protocols.io has no published Boolean-query system, so also produce a "
    "SHORT plain-text keyword phrase (2-3 words MAXIMUM) for a simple keyword search box that "
    "requires every word to match (no relevance ranking, no partial credit) -- include ONLY "
    "the general technique or method name plus at most one broad qualifier (e.g. organism "
    "class, sample type). NEVER include a specific gene symbol, protein name, chemical name, "
    "disease name, or any other narrow/rare term -- these make an exact-match search return "
    "zero results. No Boolean operators, no quotes.\n\n"
    "Return ONLY a JSON object with exactly two string fields. Both values MUST be valid "
    "JSON strings: the entire value wrapped in double quotes, with any double quotes that "
    "appear inside the value escaped as \\\". Do not output raw, unquoted Boolean syntax as "
    "the value -- it must be one JSON string.\n\n"
    "\"boolean_query\": your Boolean query, with \\\"escaped quotes\\\" around multi-word "
    "phrases. Example value (note it is one JSON string, quotes inside are escaped): "
    "\"(\\\"CRISPR knockout\\\" OR \\\"CRISPR-Cas9\\\") AND (\\\"zebrafish\\\") AND (\\\"embryo\\\")\"\n"
    "\"keywords\": the short plain-text keyword phrase described above.\n\n"
    "Example full response: {\"boolean_query\": \"(\\\"gene knockout\\\") AND (\\\"mouse\\\")\", "
    "\"keywords\": \"knockout mouse\"}"
)


def _call_llm_with_timeout(**kwargs) -> Optional[str]:
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


def _regex_fallback_parse(resp: str) -> Optional[Dict[str, str]]:
    """Recover from a model that emitted the boolean_query value as raw unquoted
    syntax instead of a proper JSON string (breaks json.loads, but the two fields
    are still locatable by their key names)."""
    kw_m = re.search(r'"keywords"\s*:\s*"([^"]*)"', resp)
    if not kw_m:
        return None
    keywords = kw_m.group(1).strip()
    bq_m = re.search(r'"boolean_query"\s*:\s*"(.*?)"\s*,\s*"keywords"', resp, re.DOTALL)
    if not bq_m:
        # value wasn't quoted at all -- grab everything between the key and ", "keywords"
        bq_m = re.search(r'"boolean_query"\s*:\s*(.*?)\s*,\s*"keywords"', resp, re.DOTALL)
    boolean_query = bq_m.group(1).strip().strip('"') if bq_m else ""
    if not keywords or not boolean_query:
        return None
    return {"boolean_query": boolean_query, "keywords": keywords}


def _keywordify(pmid: str, abstract: str, cache: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if pmid in cache:
        return cache[pmid]
    resp = _call_llm_with_timeout(
        messages=[{"role": "system", "content": KEYWORD_PROMPT},
                  {"role": "user", "content": f"Abstract: {abstract}"}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    if not resp:
        return None
    try:
        obj = json.loads(resp)
        result = {"boolean_query": obj.get("boolean_query", "").strip(),
                 "keywords": obj.get("keywords", "").strip()}
    except Exception:
        result = _regex_fallback_parse(resp)
    if not result or not result["boolean_query"] or not result["keywords"]:
        return None
    cache[pmid] = result
    return result


def _search_pubmed_raw(query: str, retmax: int = K) -> List[Dict[str, str]]:
    """Direct esearch + esummary, no query sanitizing/relaxation -- tests our exact
    Boolean string as a real user would paste it into PubMed's search box."""
    esearch = f"{_EUTILS}/esearch.fcgi?" + _with_key(
        {"db": "pubmed", "term": query, "retmax": str(retmax), "retmode": "json", "sort": "relevance"})
    data = _http_json(esearch)
    pmids = (((data or {}).get("esearchresult") or {}).get("idlist")) or []
    if not pmids:
        return []
    efetch = f"{_EUTILS}/efetch.fcgi?" + _with_key({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    xml = _http_text(efetch, timeout=10)
    try:
        articles = _parse_pubmed_xml(xml) if xml else []
    except Exception:
        articles = []
    if not articles:
        articles = [{"pmid": p, "title": ""} for p in pmids]
    return [{"pmid": a.get("pmid", ""), "title": a.get("title", "")} for a in articles[:retmax]]


def main():
    protocols = get_index()["protocols"]
    id_to_protocol = {str(p.get("id")): p for p in protocols}

    abstracts = {row["pmid"]: row.get("citing_abstract", "")
                 for row in csv.DictReader(open(CANONICAL))}

    papers = list(csv.DictReader(open(CANONICAL)))
    print(f"Running LLM keyword baseline for {len(papers)} papers...", flush=True)

    kw_cache: Dict[str, Any] = (json.loads(KEYWORD_CACHE_PATH.read_text())
                               if KEYWORD_CACHE_PATH.exists() else {})

    pubmed_rows: List[Dict[str, Any]] = []
    pio_rows: List[Dict[str, Any]] = []
    pio_hits = 0
    pio_total = 0

    for i, r in enumerate(papers, 1):
        pmid = r["pmid"]
        abstract = abstracts.get(pmid, "")
        target = set(r["used_in_methods_protocol_ids"].split("|"))

        kw = _keywordify(pmid, abstract, kw_cache)
        if kw is None:
            print(f"  [{i:>3}/{len(papers)}] pmid={pmid} keyword-ification FAILED, skipping", flush=True)
            continue
        if i % 10 == 0:
            KEYWORD_CACHE_PATH.write_text(json.dumps(kw_cache, indent=1))

        pm_results = _search_pubmed_raw(kw["boolean_query"], K)
        for rank, res in enumerate(pm_results, 1):
            pubmed_rows.append({"pmid": pmid, "boolean_query": kw["boolean_query"],
                               "rank": rank, "result_pmid": res["pmid"], "result_title": res["title"]})

        # protocols.io's search requires every word to match (no relevance ranking), so a
        # multi-word query that's too specific returns nothing -- exactly what a real user
        # hitting a dead end would do: drop the last word and try again, down to one word.
        words = kw["keywords"].split()
        items: List[Dict[str, Any]] = []
        used_query = kw["keywords"]
        for n in range(len(words), 0, -1):
            attempt = " ".join(words[:n])
            try:
                _total, items = search_live(attempt, max_results=K)
            except Exception as e:
                items = []
                print(f"  protocols.io search failed for pmid={pmid}: {e}", flush=True)
                break
            used_query = attempt
            if items:
                break
        found_ids = {str(it.get("id")) for it in items}
        is_hit = bool(found_ids & target)
        pio_total += 1
        pio_hits += is_hit
        for rank, it in enumerate(items, 1):
            pio_rows.append({"pmid": pmid, "keywords": kw["keywords"], "used_query": used_query, "rank": rank,
                            "result_protocol_id": it.get("id"), "result_title": it.get("title", ""),
                            "is_confirmed_target": str(it.get("id")) in target})

        print(f"  [{i:>3}/{len(papers)}] pmid={pmid}  pubmed_hits={len(pm_results)}  "
              f"pio_hits={len(items)} (query: {used_query!r})  target_found={is_hit}", flush=True)

    KEYWORD_CACHE_PATH.write_text(json.dumps(kw_cache, indent=1))

    with PUBMED_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "boolean_query", "rank", "result_pmid", "result_title"])
        w.writeheader()
        w.writerows(pubmed_rows)
    print(f"wrote {len(pubmed_rows)} rows -> {PUBMED_OUT}")

    with PROTOCOLSIO_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "keywords", "used_query", "rank", "result_protocol_id",
                                         "result_title", "is_confirmed_target"])
        w.writeheader()
        w.writerows(pio_rows)
    print(f"wrote {len(pio_rows)} rows -> {PROTOCOLSIO_OUT}")

    print(f"\nprotocols.io find rate for method M: {pio_hits}/{pio_total} = {pio_hits/pio_total:.1%}")


if __name__ == "__main__":
    main()
