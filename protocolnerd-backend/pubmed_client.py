"""
Native PubMed client for blended protocol/literature search.

No MCP, no Node — reuses the same NCBI E-utilities infrastructure already used
for taxonomy lookups in concept_expansion.py, and adds article search plus a
full-text fallback chain (PMC -> Europe PMC -> Unpaywall).

Pipeline:
  search_pubmed()   esearch(db=pubmed) -> PMIDs, then efetch -> title + abstract
                    + metadata. Cheap; used for ranking and display.
  fetch_fulltext()  PMC BioC -> Europe PMC fullTextXML -> Unpaywall PDF/landing.
                    Expensive; called ONLY for results that surface in the blend.
  extract_methods() pull the Materials and Methods / Methods / Experimental
                    Procedures section out of fetched full text.

Every result is shaped like a protocols.io result (title/url/doi/description/
authors/keywords) plus source="pubmed" so the frontend renders both identically.
Everything is best-effort: any failure returns empty so the caller can degrade
to protocols.io-only.
"""

from __future__ import annotations

import logging
import os
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "variables.env", override=False)

log = logging.getLogger(__name__)

# Use certifi's CA bundle so HTTPS verification is consistent across hosts and
# proxies (the system trust store can miss roots NCBI/EBI chain to). Falls back
# to the default context if certifi isn't installed.
try:
    import certifi
    _SSL_CTX: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_UNPAYWALL = "https://api.unpaywall.org/v2"
_HTTP_UA = "Mozilla/5.0 (compatible; ProtocolsNerdBot/1.0)"

# Section headings that mark a methods section in full text.
_METHODS_HEADINGS = (
    "materials and methods",
    "methods",
    "experimental procedures",
    "materials & methods",
    "methodology",
    "experimental section",
)


def _api_key() -> str:
    return os.getenv("NCBI_API_KEY", "").strip('"').strip()


def _unpaywall_email() -> str:
    return os.getenv("UNPAYWALL_EMAIL", "").strip('"').strip()


def _with_key(params: Dict[str, str]) -> str:
    key = _api_key()
    if key:
        params = {**params, "api_key": key}
    return urllib.parse.urlencode(params)


def _http_text(url: str, timeout: int = 8) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _HTTP_UA})
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        log.debug(f"HTTP text failed for {url}: {e}")
        return None


def _http_json(url: str, timeout: int = 8) -> Optional[Any]:
    raw = _http_text(url, timeout=timeout)
    if raw is None:
        return None
    try:
        import json
        return json.loads(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Search: esearch -> efetch (title + abstract + metadata)
# ---------------------------------------------------------------------------

# Imperative/scaffolding tokens that mean nothing to PubMed's boolean AND search
# and only shrink the result set. protocols.io's TF-IDF treats these as
# stopwords; PubMed does not, so we strip them before querying.
_PUBMED_STOP = {
    "find", "search", "searching", "get", "locate", "show", "list", "want",
    "wanted", "need", "needed", "looking", "please", "protocol", "protocols",
    "method", "methods", "methodology", "technique", "techniques", "procedure",
    "procedures", "paper", "papers", "publication", "publications", "study",
    "studies", "article", "articles", "for", "using", "that", "can", "allow",
    "allows", "allowing", "i", "we", "me", "my", "to",
}


def _sanitize_for_pubmed(query: str) -> str:
    """Strip natural-language scaffolding so biology keywords reach PubMed."""
    tokens = re.split(r"\s+", query.strip())
    kept = [t for t in tokens if t.lower().strip(".,") not in _PUBMED_STOP]
    cleaned = " ".join(kept).strip()
    # If sanitizing removed almost everything, fall back to the raw query.
    return cleaned if len(cleaned) >= 3 else query.strip()


def search_pubmed(query: str, retmax: int = 5) -> List[Dict[str, Any]]:
    """
    Search PubMed and return up to `retmax` normalized article dicts with
    title, abstract (as `description`), authors, doi, url, pmid, source.
    Returns [] on any failure (caller degrades to protocols.io-only).
    """
    import time
    t_start = time.time()
    query = _sanitize_for_pubmed(query or "")
    if not query:
        return []

    esearch = (
        f"{_EUTILS}/esearch.fcgi?"
        + _with_key({"db": "pubmed", "term": query, "retmax": str(retmax), "retmode": "json", "sort": "relevance"})
    )
    data = _http_json(esearch)
    pmids = (((data or {}).get("esearchresult") or {}).get("idlist")) or []
    if not pmids:
        t_end = time.time()
        log.info(f"PubMed search (no results): {t_end - t_start:.2f}s")
        return []

    efetch = (
        f"{_EUTILS}/efetch.fcgi?"
        + _with_key({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    )
    xml = _http_text(efetch)
    if not xml:
        t_end = time.time()
        log.info(f"PubMed search (fetch failed): {t_end - t_start:.2f}s")
        return []

    results = _parse_pubmed_xml(xml)
    t_end = time.time()
    log.info(f"PubMed search: {len(results)} results in {t_end - t_start:.2f}s")
    return results


# ---------------------------------------------------------------------------
# LLM-built PubMed query (Haiku)
#
# The old token-trim core reduced a request to organism+method+goal tokens from a
# FIXED concept vocabulary, capped at 6. On realistic queries that collapsed to 1-2
# GENERIC words ("mouse", "extraction", "sequencing") — every discriminating term
# (Eucalyptus, nanopore, hippocampal, SARS-CoV-2) was dropped, so PubMed's AND
# search returned broad, off-topic papers (86% graded 0 on the held-out set).
#
# Haiku instead writes a real search query that KEEPS the discriminators. PubMed
# accepts boolean AND/OR/parens natively and `_sanitize_for_pubmed` passes them
# through untouched, so synonym groups work. A precise query can also return ZERO
# hits on a rare topic, hence the relaxation ladder in search_pubmed_smart().
# ---------------------------------------------------------------------------
PUBMED_QUERY_PROVIDER = os.getenv("PUBMED_QUERY_PROVIDER", "claude")
PUBMED_QUERY_MODEL = os.getenv("PUBMED_QUERY_MODEL", "claude-haiku-4-5")

# Haiku writes TWO queries per angle in ONE call:
#   precise — the best-targeted query (3-5 concepts, keeps every discriminator)
#   broad   — a purpose-built backup (2-3 concepts, OR-groups over stacked ANDs)
# On the held-out set the precise query returned 0 hits ~45% of the time (PubMed ANDs
# everything, so each extra concept makes a miss likelier). Rather than mechanically
# truncating it, we ask the model for a *deliberately broader, still on-topic* query up
# front — a better rung 2 than a chopped rung 1. The mechanical truncation survives as
# rung 3, the token core as rung 4.
_QUERY_RULES = (
    "You are a biomedical search librarian. Turn a scientist's lab-PROTOCOL request into "
    "PubMed queries that retrieve METHODS/PROTOCOL papers for it.\n"
    "\n"
    "FIRST, check for an ALTERNATION. If the request offers two or more alternatives for the "
    "same slot — organisms, targets, samples ('rice OR potato', 'mouse or rat', 'liver or "
    "kidney') — you MUST produce a SEPARATE query variant for EACH alternative. Do NOT fold "
    "them into one OR'd query: PubMed sorts by relevance, so the alternative with the larger "
    "literature crowds the smaller one out of the results entirely (a 'rice OR potato' query "
    "returns only rice). One variant per alternative guarantees each is represented.\n"
    "Alternation applies ONLY to genuine either/or choices. Synonyms for the SAME thing "
    "('HMW DNA OR high molecular weight DNA', 'nanopore OR long-read') are NOT alternation — "
    "keep those as an OR-group inside a single query, in parentheses.\n"
    "\n"
    "For EACH variant write TWO queries:\n"
    '  "precise" — the best-targeted query. Keep the MOST SPECIFIC discriminating terms: '
    "organism/species, technique/method, sample or tissue type, molecule/target/gene. "
    "3-5 concepts. OR close synonyms in parentheses, e.g. (nanopore OR long-read).\n"
    '  "broad" — a deliberately BROADER backup, used only if "precise" finds nothing. Use '
    "only the 2-3 concepts that matter most (usually technique + organism, or technique + "
    "sample). PREFER OR-groups over stacked ANDs; drop narrow qualifiers. It must stay "
    "ON-TOPIC — broader, not vaguer. Never reduce it to one generic word.\n"
    "Drop filler ('protocol', 'standard', 'I need', clinical framing) unless it IS the point.\n"
    "PubMed ANDs every term, so each extra AND makes zero hits likelier. Always PARENTHESIZE "
    "any OR-group — PubMed evaluates booleans left-to-right, so a bare OR changes the meaning.\n"
)

# Output-shape instructions, appended to whichever domain's query RULES apply.
# Shape is domain-independent, so only the rules half varies.
_SINGLE_SUFFIX = (
    'Return ONLY a JSON object: {"variants": [{"precise": "...", "broad": "..."}, ...]}. '
    "Give ONE variant normally; give one variant PER ALTERNATIVE when the request alternates "
    "(e.g. rice OR potato -> two variants). No explanation."
)

_BATCH_SUFFIX = (
    "You are given a NUMBERED list of related search angles on the SAME experiment. Handle each "
    "angle independently, keeping its own emphasis.\n"
    'Return ONLY a JSON object: {"queries": [{"variants": [{"precise": "...", "broad": "..."}, '
    "...]}, ...]} — one entry per input angle, SAME ORDER, SAME LENGTH as the input list. Each "
    "angle's `variants` holds one pair normally, or one pair PER ALTERNATIVE when that angle "
    "alternates. No explanation."
)


def _query_rules() -> str:
    """The active domain's PubMed query rules, falling back to _QUERY_RULES
    (biology's, and the historical default). Deferred import so this module stays
    importable independently of the domains package."""
    try:
        from domains import current_domain
        return current_domain().pubmed_query_rules() or _QUERY_RULES
    except Exception:  # noqa: BLE001 -- prompt lookup must never break a search
        return _QUERY_RULES


_PUBMED_QUERY_SYSTEM = _QUERY_RULES + _SINGLE_SUFFIX
_PUBMED_QUERY_BATCH_SYSTEM = _QUERY_RULES + _BATCH_SUFFIX

# A built pair: {"precise": str, "broad": str}. An angle yields a LIST of these — one per
# alternative when the request offers a genuine either/or (rice OR potato), otherwise one.
PubMedQueryPair = Dict[str, str]

# Hard cap on PubMed searches per request, so a request that alternates across several angles
# can't fan out into a rate-limit-tripping storm of E-utilities calls.
PUBMED_MAX_SEARCHES = int(os.getenv("PUBMED_MAX_SEARCHES", "6") or 6)


def _clean_pair(obj: Any) -> Optional[PubMedQueryPair]:
    """Normalize one model-emitted pair; None when there's nothing usable."""
    if not isinstance(obj, dict):
        return None
    precise = str(obj.get("precise") or "").strip().strip('"').strip()
    broad = str(obj.get("broad") or "").strip().strip('"').strip()
    if not precise and not broad:
        return None
    return {"precise": precise or broad, "broad": broad or precise}


def _clean_variants(obj: Any) -> List[PubMedQueryPair]:
    """Normalize one angle's variant list. Tolerates the model returning a bare pair
    instead of {"variants": [...]}, so a prompt slip degrades to the old behavior."""
    if isinstance(obj, dict) and "variants" in obj:
        raw = obj.get("variants") or []
    elif isinstance(obj, list):
        raw = obj
    else:
        raw = [obj]                       # bare {"precise":..,"broad":..}
    out = [p for p in (_clean_pair(x) for x in raw) if p]
    return out


def _call_query_model(system: str, user: str) -> str:
    from llm_providers import call_llm  # lazy: keeps pubmed_client import-light
    return call_llm(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
        provider=PUBMED_QUERY_PROVIDER,
        model=PUBMED_QUERY_MODEL,
    ) or ""


def build_pubmed_queries(nl_queries: List[str]) -> List[List[PubMedQueryPair]]:
    """Batched: ONE Haiku call builds the PubMed query variants for EVERY selected search
    angle. The angles are facets of the same experiment, so a round-trip each would be waste.

    Returns a list aligned 1:1 with the input; each entry is that angle's VARIANTS — normally
    one precise+broad pair, but one pair PER ALTERNATIVE when the request offers a genuine
    either/or ("rice OR potato" -> a rice variant and a potato variant). Searching each
    alternative separately is what stops the larger literature from crowding out the smaller.
    An empty list means the model gave nothing usable for that angle, so the caller falls back
    to its token core for just that one."""
    qs = [(q or "").strip() for q in (nl_queries or [])]
    if not any(qs):
        return [[] for _ in qs]
    if len(qs) == 1:
        return [build_pubmed_query(qs[0])]
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(qs, 1) if q)
    try:
        import json as _json
        raw = _call_query_model(_query_rules() + _BATCH_SUFFIX, numbered)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        items = _json.loads(m.group(0)).get("queries", []) if m else []
        out = [_clean_variants(x) for x in items]
    except Exception as e:  # noqa: BLE001
        log.warning(f"Batched PubMed query build failed ({e}); falling back to token cores.")
        return [[] for _ in qs]
    out = (out + [[] for _ in qs])[:len(qs)]          # pad/trim so callers can zip by index
    return [o if q else [] for o, q in zip(out, qs)]


def build_pubmed_query(nl_query: str) -> List[PubMedQueryPair]:
    """Build one angle's PubMed query variants (1 normally; one per alternative when the
    request alternates). Used by the eval, which has a single query per item."""
    nl_query = (nl_query or "").strip()
    if not nl_query:
        return []
    try:
        import json as _json
        raw = _call_query_model(_query_rules() + _SINGLE_SUFFIX, nl_query)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return _clean_variants(_json.loads(m.group(0))) if m else []
    except Exception as e:  # noqa: BLE001
        log.warning(f"PubMed query build failed ({e}); falling back to token core.")
        return []


def _relax_boolean(query: str) -> Optional[str]:
    """Broaden a boolean PubMed query by dropping its LAST AND-group.
    'A AND (B OR C) AND D' -> 'A AND (B OR C)'. None when there's nothing to drop."""
    parts = re.split(r"\s+AND\s+", (query or "").strip(), flags=re.IGNORECASE)
    if len(parts) <= 1:
        return None
    return " AND ".join(p.strip() for p in parts[:-1] if p.strip()) or None


def search_pubmed_smart(nl_query: str, retmax: int = 5,
                        fallback_core: Optional[str] = None,
                        info: Optional[Dict[str, Any]] = None,
                        prebuilt: Optional[PubMedQueryPair] = None) -> List[Dict[str, Any]]:
    """PubMed search driven by the Haiku-built query pair, with a 0-hit ladder:

        1. `precise` — Haiku's best-targeted query (3-5 concepts, all discriminators)
        2. `broad`   — Haiku's purpose-built broader query (2-3 concepts, OR-groups).
                       A designed fallback, NOT a truncation of rung 1.
        3. `precise` with its last AND-group mechanically dropped (last-ditch broadening)
        4. `fallback_core` — the caller's old token core
        5. the raw request (search_pubmed sanitizes it)

    Returns the first rung that yields hits, so a precise query never costs us the result
    entirely on a rare topic. Empty list only if every rung returns nothing.
    """
    # `prebuilt` lets a caller batch the LLM step (see build_pubmed_queries) and pay
    # ONE Haiku call for all its selected queries instead of one per query.
    pair = prebuilt if prebuilt else (build_pubmed_query(nl_query) or [{}])[0]
    precise, broad = pair.get("precise"), pair.get("broad")
    ladder, seen = [], set()
    for cand in (precise, broad, _relax_boolean(precise or ""), fallback_core, nl_query):
        c = (cand or "").strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            ladder.append(c)
    for i, cand in enumerate(ladder):
        hits = search_pubmed(cand, retmax) or []
        if hits:
            if i:
                log.info(f"PubMed relaxed to rung {i + 1}/{len(ladder)}: '{cand[:60]}'")
            if info is not None:
                info["query"], info["rung"] = cand, i + 1
            return hits
    log.info(f"PubMed: no hits on any rung for '{nl_query[:60]}'")
    if info is not None:
        info["query"], info["rung"] = (ladder[0] if ladder else nl_query), 0
    return []


def _dedup_key(r: Dict[str, Any]) -> Any:
    return r.get("pmid") or r.get("doi") or r.get("title")


def search_pubmed_fanout(nl_query: str, retmax: int = 5,
                         fallback_core: Optional[str] = None,
                         info: Optional[Dict[str, Any]] = None,
                         prebuilt: Optional[List[PubMedQueryPair]] = None) -> List[Dict[str, Any]]:
    """Search PubMed for ONE request, FANNING OUT across its alternatives.

    When a request offers a genuine either/or ("drought tolerance in rice OR potato"), a single
    OR'd query is not enough: PubMed sorts by relevance, so the alternative with the bigger
    literature (rice) takes every slot and the other (potato) never appears. Haiku therefore
    emits one query variant per alternative, and we search EACH and merge — so both are
    represented and the re-ranker gets to judge them side by side.

    Each variant still walks the full 0-hit ladder (see search_pubmed_smart). Results are
    merged in variant order and de-duplicated by pmid/doi/title.
    """
    variants = prebuilt if prebuilt is not None else build_pubmed_query(nl_query)
    if not variants:
        variants = [{}]                       # no LLM query -> ladder falls to the token core
    merged, seen = [], set()
    fired = []
    for vi, v in enumerate(variants[:PUBMED_MAX_SEARCHES]):
        sub: Dict[str, Any] = {}
        for r in search_pubmed_smart(nl_query, retmax, fallback_core, sub, prebuilt=v or None):
            k = _dedup_key(r)
            if k and k not in seen:
                seen.add(k)
                r["_pm_variant"] = vi         # which alternative produced this — see balanced_trim
                merged.append(r)
        if sub.get("query"):
            fired.append(sub)
    if len(variants) > 1:
        log.info(f"PubMed fan-out: {len(variants)} alternatives -> {len(merged)} merged hits "
                 f"({[f['query'][:38] for f in fired]})")
    if info is not None:
        info["variants"] = len(variants)
        info["queries"] = [f.get("query") for f in fired]
        info["query"] = " | ".join(f.get("query", "") for f in fired) or nl_query
        info["rung"] = fired[0].get("rung") if fired else 0
    return merged


def balanced_trim(results: List[Dict[str, Any]], keep: int,
                  rank_within=None) -> List[Dict[str, Any]]:
    """Trim a (possibly fanned-out) PubMed pool to `keep`, ROUND-ROBIN across alternatives.

    Fanning out "rice OR potato" into two searches only helps if the trim that follows keeps
    both. A plain relevance trim re-introduces the very imbalance the fan-out removed: score
    all 10 together, keep 5, and one organism can still take every slot. So we group by the
    alternative that produced each result, order WITHIN each group via `rank_within` (profile
    relevance), then take round-robin across groups.

    `rank_within(list) -> list` is optional; with a single alternative this degenerates to
    exactly the old behavior: rank_within(results)[:keep].
    """
    if keep <= 0 or not results:
        return []
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(int(r.get("_pm_variant", 0) or 0), []).append(r)
    for k in list(groups):
        try:
            groups[k] = rank_within(groups[k]) if rank_within else groups[k]
        except Exception as e:  # noqa: BLE001 — ranking is best-effort; keep fetch order
            log.warning(f"PubMed rank-within failed ({e}); keeping fetch order.")
    picked: List[Dict[str, Any]] = []
    while len(picked) < keep:
        progressed = False
        for k in sorted(groups):
            if groups[k] and len(picked) < keep:
                picked.append(groups[k].pop(0))
                progressed = True
        if not progressed:                    # every group exhausted
            break
    if len(groups) > 1:
        spread = {k: sum(1 for r in picked if int(r.get("_pm_variant", 0) or 0) == k)
                  for k in sorted(groups)}
        log.info(f"PubMed balanced trim -> {len(picked)} kept across {len(groups)} alternatives {spread}")
    return picked


def _parse_pubmed_xml(xml: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        log.debug(f"PubMed XML parse failed: {e}")
        return out

    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//PMID"))
        title = _text(art.find(".//ArticleTitle"))
        if not pmid or not title:
            continue
        abstract = " ".join(
            _text(a) for a in art.findall(".//Abstract/AbstractText") if _text(a)
        ).strip()
        authors: List[str] = []
        for a in art.findall(".//AuthorList/Author"):
            last = _text(a.find("LastName"))
            initials = _text(a.find("Initials"))
            name = (f"{last} {initials}".strip() if last else "").strip()
            if name:
                authors.append(name)
        keywords = [
            _text(k) for k in art.findall(".//KeywordList/Keyword") if _text(k)
        ]
        doi = ""
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        out.append({
            "id": f"pubmed:{pmid}",
            "pmid": pmid,
            "title": title,
            "uri": "",
            "url": url,
            "doi": doi,
            "description": abstract[:400],
            "abstract": abstract,
            "authors": authors[:6],
            "keywords": keywords[:8],
            "source": "pubmed",
            "score": 0.0,
        })
    return out


def _text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    # itertext() flattens nested markup (e.g. <i>, <sup>) inside titles/abstracts.
    return "".join(node.itertext()).strip()


# ---------------------------------------------------------------------------
# Full text: PMC -> Europe PMC -> Unpaywall (only for surfaced results)
# ---------------------------------------------------------------------------

def fetch_fulltext(pmid: str, doi: str = "") -> str:
    """Best-effort full text via PMC -> Europe PMC -> Unpaywall. "" if none."""
    for fetch in (_fulltext_europepmc, _fulltext_pmc, lambda *_: _fulltext_unpaywall(doi)):
        try:
            text = fetch(pmid, doi)
            if text and len(text) > 500:
                return text
        except Exception as e:
            log.debug(f"fulltext source failed for pmid={pmid}: {e}")
    return ""


def _fulltext_europepmc(pmid: str, doi: str = "") -> str:
    """Europe PMC fullTextXML (open-access subset)."""
    url = f"{_EUROPEPMC}/MED/{pmid}/fullTextXML"
    xml = _http_text(url)
    if not xml:
        return ""
    try:
        root = ET.fromstring(xml)
        return " ".join(t.strip() for t in root.itertext() if t.strip())
    except ET.ParseError:
        return ""


def _fulltext_pmc(pmid: str, doi: str = "") -> str:
    """Resolve PMID -> PMCID, then fetch the BioC/PMC text."""
    link = (
        f"{_EUTILS}/elink.fcgi?"
        + _with_key({"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"})
    )
    data = _http_json(link)
    pmcid = ""
    try:
        linksets = (data or {}).get("linksets") or []
        for ls in linksets:
            for db in ls.get("linksetdbs") or []:
                if db.get("dbto") == "pmc" and db.get("links"):
                    pmcid = str(db["links"][0])
                    break
    except Exception:
        pmcid = ""
    if not pmcid:
        return ""
    efetch = (
        f"{_EUTILS}/efetch.fcgi?"
        + _with_key({"db": "pmc", "id": pmcid, "retmode": "xml"})
    )
    xml = _http_text(efetch)
    if not xml:
        return ""
    try:
        root = ET.fromstring(xml)
        return " ".join(t.strip() for t in root.itertext() if t.strip())
    except ET.ParseError:
        return ""


def _fulltext_unpaywall(doi: str) -> str:
    """Last resort: Unpaywall open-access location (landing/PDF URL note only)."""
    doi = (doi or "").strip()
    email = _unpaywall_email()
    if not doi or not email:
        return ""
    data = _http_json(f"{_UNPAYWALL}/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}")
    loc = ((data or {}).get("best_oa_location")) or {}
    # Unpaywall returns a URL, not the article body. We surface the OA URL so the
    # methods extractor has something to point at; we do not scrape arbitrary PDFs.
    return loc.get("url_for_pdf") or loc.get("url") or ""


def extract_methods(fulltext: str) -> str:
    """
    Pull the Materials and Methods section from full text. Returns the section
    body (capped) or "" if no recognizable methods heading is found.
    """
    if not fulltext or len(fulltext) < 200:
        return ""
    low = fulltext.lower()
    start = -1
    for heading in _METHODS_HEADINGS:
        idx = low.find(heading)
        if idx != -1:
            start = idx
            break
    if start == -1:
        return ""
    # End at the next major section heading after the methods block.
    rest = fulltext[start:]
    end_markers = ("\nresults", "\ndiscussion", "\nconclusion", "\nreferences", "\nacknowledg")
    low_rest = rest.lower()
    end = len(rest)
    for m in end_markers:
        i = low_rest.find(m, 200)
        if i != -1:
            end = min(end, i)
    return re.sub(r"\s+", " ", rest[:end]).strip()[:3000]
