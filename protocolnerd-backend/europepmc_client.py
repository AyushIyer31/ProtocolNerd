"""
Europe PMC search client.

Europe PMC is used elsewhere in this codebase already (pubmed_client fetches
full text from it, concept_expansion pulls related terms), but only as a
support service. This module exposes it as a first-class SEARCH source.

Why it matters: Protocols.io holds almost no synthetic chemistry. Measured
across ten core techniques it returns 21 results in total, and zero for Suzuki
coupling, Grignard, RAFT polymerization, Schlenk line, catalyst preparation and
MOF synthesis. Europe PMC's Methods-section field query returns 104,565 for the
same terms, because chemistry procedures live in papers' Methods sections
rather than in protocol repositories.

The `METHODS:"..."` field query is the point of this source: it searches INSIDE
the methods section, so a hit is a paper that actually performed the technique
rather than one that merely mentions it.

Every failure returns an empty list. A source outage must degrade the search,
never break it.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Same CA handling as pubmed_client: the system trust store can miss roots EBI
# chains to, so prefer certifi's bundle when it is available.
try:
    import certifi
    _SSL_CTX: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = None

_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_HTTP_UA = "Mozilla/5.0 (compatible; ProtocolsNerdBot/1.0)"
_TIMEOUT = 12

# Profile fields that make good METHODS: clauses, most discriminating first.
# Chemistry-shaped, but read generically so another domain can reuse this.
_METHODS_FIELDS = ("reaction_type", "compound", "catalyst", "purification", "characterization")
_SKIP = {"", "not specified", "none", "unknown", "not sure", "null"}


def _empty(v: Any) -> bool:
    return v is None or str(v).strip().lower() in _SKIP


def _methods_clauses(profile: Optional[Dict[str, Any]]) -> List[str]:
    """METHODS: clauses from the profile, most discriminating first."""
    clauses: List[str] = []
    for f in _METHODS_FIELDS:
        v = (profile or {}).get(f)
        if _empty(v):
            continue
        # Keep only the first comma-separated value; "NMR, mass spectrometry"
        # as one quoted phrase would never match.
        term = str(v).split(",")[0].strip()
        if term and len(term) > 2:
            clauses.append(f'METHODS:"{term}"')
    return clauses


def build_methods_query(profile: Optional[Dict[str, Any]], fallback: str, max_clauses: int = 2) -> str:
    """Turn a profile into a Europe PMC METHODS: query.

    Deliberately deterministic -- no LLM call, so this adds no latency, no token
    cost and no extra prompt to maintain.
    """
    clauses = _methods_clauses(profile)[:max_clauses]
    return " AND ".join(clauses) if clauses else (fallback or "").strip()


def _terms_from_query(query: str) -> List[str]:
    """The METHODS: terms in a query we built, e.g. 'Suzuki coupling'."""
    import re
    return re.findall(r'METHODS:"([^"]+)"', query or "")


def _normalize(rec: Dict[str, Any], methods_terms: Optional[List[str]] = None) -> Dict[str, Any]:
    pmid = (rec.get("pmid") or "").strip()
    doi = (rec.get("doi") or "").strip()
    pmcid = (rec.get("pmcid") or "").strip()
    if pmid:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    elif pmcid:
        url = f"https://europepmc.org/article/PMC/{pmcid}"
    elif doi:
        url = f"https://doi.org/{doi}"
    else:
        url = "https://europepmc.org/"
    abstract = (rec.get("abstractText") or "").strip()
    # Why this paper is a candidate at all. A METHODS: hit means the paper
    # PERFORMED the technique, which its abstract often does not advertise --
    # so ranking it on the abstract alone systematically discards it. We built
    # the query, so we already know what matched; state it up front where both
    # the re-ranker (which reads description[:180]) and the user can see it.
    # Same idea as the `matched_queries` protocols.io results already carry.
    marker = f"[Methods section uses: {', '.join(methods_terms)}] " if methods_terms else ""
    # A stable, unique id is MANDATORY, not cosmetic. reranker.rerank builds its
    # shortlist with `if rid is not None`, and _llm_order maps the model's answer
    # back through the same field, so a result with no id is dropped before the
    # LLM ever sees it and can never be ranked. PubMed does the same thing with
    # "pubmed:{pmid}"; the re-rank pool already mixes string ids, so this matches.
    ident = pmid or pmcid or doi or (rec.get("title") or "")[:60]
    return {
        "id": f"europepmc:{ident}",
        "title": (rec.get("title") or "").strip(),
        "description": (marker + abstract).strip(),
        "abstract": abstract,
        "methods_match": list(methods_terms or []),
        "source": "europepmc",
        "url": url,
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "journal": (rec.get("journalTitle") or "").strip(),
        "year": (rec.get("pubYear") or "").strip(),
        "is_open_access": (rec.get("isOpenAccess") or "") == "Y",
    }


def search_europepmc(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search Europe PMC. Returns [] on any failure, by design."""
    query = (query or "").strip()
    if not query:
        return []
    url = (f"{_SEARCH}?query={urllib.parse.quote(query)}"
           f"&format=json&pageSize={max(1, min(k, 25))}&resultType=core")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _HTTP_UA})
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_SSL_CTX) as resp:
            data = json.load(resp)
    except Exception as e:  # noqa: BLE001 -- a source outage must not break search
        log.warning(f"Europe PMC search failed for {query[:60]!r}: {e}")
        return []

    rows = ((data.get("resultList") or {}).get("result") or [])[:k]
    terms = _terms_from_query(query)
    out = [_normalize(r, terms) for r in rows]
    log.info(f"Europe PMC '{query[:70]}' -> {len(out)} results "
             f"(of {data.get('hitCount', '?')} hits)")
    return [r for r in out if r["title"]]


def search_with_fallback(profile: Optional[Dict[str, Any]], fallback_query: str,
                         k: int = 5) -> List[Dict[str, Any]]:
    """Relaxation ladder, mirroring the precise -> broad approach in pubmed_client.

    Europe PMC ANDs the METHODS clauses, so a 3-clause query is usually
    over-constrained: in testing, `METHODS:"Suzuki coupling" AND METHODS:"biaryl"
    AND METHODS:"palladium catalyst"` returned 2 hits, too few to survive the
    joint re-rank. Stopping at the first NON-EMPTY rung was therefore not enough;
    we descend until a rung returns a usable number of results.

    Rungs: 2 METHODS clauses -> 1 clause -> the plain query.
    """
    clauses = _methods_clauses(profile)
    rungs: List[str] = []
    if len(clauses) >= 2:
        rungs.append(" AND ".join(clauses[:2]))
    if clauses:
        rungs.append(clauses[0])
    plain = (fallback_query or "").strip()
    if plain:
        rungs.append(plain)

    # "Usable" = at least half the requested slots; below that the source cannot
    # compete in the joint re-rank and a broader rung is worth the extra call.
    want = max(2, (k + 1) // 2)
    best: List[Dict[str, Any]] = []
    for i, q in enumerate(rungs):
        results = search_europepmc(q, k)
        if len(results) > len(best):
            best = results
        if len(results) >= want:
            return results
        if results and i < len(rungs) - 1:
            log.info(f"Europe PMC rung {i+1} returned {len(results)} (<{want}); relaxing.")
    return best
