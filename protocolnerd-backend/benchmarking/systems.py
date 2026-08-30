"""
The three retrieval systems under comparison. Each returns a ranked list of
protocol result dicts (best first); every dict has at least `id`, `title`,
`description`.

  plain_keyword       -- raw TF-IDF over the same corpus. No profile, no re-rank.
                         This is the ABLATION baseline.
  nerd_full           -- the full Protocol Nerd local pipeline: LLM profile ->
                         profile-derived candidate queries -> wide TF-IDF pool ->
                         profile-aware ranking (closeness_rank). Clarifications are
                         auto-skipped: the profile is whatever the LLM extracts from
                         the single input sentence (no human in the loop).
  protocolsio_native  -- the live protocols.io search API (best effort; needs a
                         token + network). Optional.

Results are cached to benchmarking/cache/<system>.json keyed by "<k>||<query>" so reruns
are free and don't re-bill the LLM. Pass use_cache=False to force recompute.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401  (sets sys.path + loads .env)
from _bootstrap import CACHE_DIR, INDEX_PATH

# Backend imports (available thanks to _bootstrap).
from protocol_rag import (  # type: ignore
    load_protocol_index,
    search_protocols,
    multi_search_protocols,
    expand_query,
    _draftjs_to_text,
)
from field_ranking import closeness_rank  # type: ignore
from claude_client import is_available  # type: ignore
from domains import current_domain, set_current_domain  # type: ignore
from experiment_profile import _is_generic, profile_to_search_query  # type: ignore
from concept_expansion import extract_concepts  # type: ignore
from retrievers import RETRIEVERS  # type: ignore
from blend_ranking import blend_sources  # type: ignore

# PubMed results have no integer id; namespace their pmid into a disjoint range
# so they never collide with protocol ids (which are < ~200k) when pooled/judged.
_PUBMED_ID_BASE = 1_000_000_000


# ---------------------------------------------------------------------------
# Index (loaded once)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_index() -> Dict[str, Any]:
    return load_protocol_index(INDEX_PATH)


def _slim(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what the judge / metrics need; keeps caches small."""
    return {
        "id": result.get("id"),
        "title": result.get("title") or "",
        "description": (result.get("description") or "")[:400],
        "source": result.get("source") or "protocols.io",
    }


def _slim_blended(result: Dict[str, Any]) -> Dict[str, Any]:
    """Like _slim but assigns PubMed hits a collision-free integer id."""
    source = result.get("source") or "protocols.io"
    if source == "pubmed":
        pmid = result.get("pmid")
        try:
            rid = _PUBMED_ID_BASE + int(pmid)
        except Exception:
            rid = _PUBMED_ID_BASE + (abs(hash(result.get("title") or "")) % 10 ** 8)
    else:
        rid = result.get("id")
    return {
        "id": rid,
        "title": result.get("title") or "",
        "description": (result.get("description") or "")[:400],
        "source": source,
    }


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _cache_path(system: str) -> Path:
    return CACHE_DIR / f"{system}.json"


def _load_cache(system: str) -> Dict[str, Any]:
    p = _cache_path(system)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(system: str, cache: Dict[str, Any]) -> None:
    _cache_path(system).write_text(json.dumps(cache, indent=1), encoding="utf-8")


def _cache_key(query: str, k: int) -> str:
    return f"{k}||{query.strip()}"


# ---------------------------------------------------------------------------
# System 1: plain keyword (ablation baseline)
# ---------------------------------------------------------------------------
def plain_keyword(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results = search_protocols(get_index(), query, top_k=k)
    return [_slim(r) for r in results]


def dense_only(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Pure semantic search over the same protocols.io corpus: embed the raw query,
    rank by cosine. The exact mirror of plain_keyword -- no profile, no expansion, no
    re-ranking -- so it represents the MODERN naive alternative a user could reach for.
    Without it, Nerd's margin could be read as 'they just have embeddings'."""
    mat = _dense_matrix()
    if mat is None:
        return []
    from dense_index import dense_candidates  # type: ignore
    cands = dense_candidates(mat, get_index()["protocols"], query, None, k)
    return [_slim(c) for c in cands]


# ---------------------------------------------------------------------------
# System 2: full Protocol Nerd local pipeline
# ---------------------------------------------------------------------------
def _nerd_profile(query: str) -> Tuple[Dict[str, Any], List[str]]:
    """Extract the experiment profile from the single sentence (no clarification)
    and derive the profile-based candidate search queries. Returns (profile, cqs).

    The domain comes from BENCHMARK_DOMAIN (default biology, so every existing
    biology run is unchanged). The planner is then called through the domain
    rather than through biology's analyze_experiment_request directly, so a
    second domain is profiled with its OWN fields and prompt."""
    if not is_available():
        raise RuntimeError(
            "No LLM available (set ANTHROPIC_API_KEY / OPENAI_API_KEY in "
            "protocolnerd-backend/.env). nerd_full needs the planner."
        )
    set_current_domain(os.getenv("BENCHMARK_DOMAIN", "biology").strip() or "biology")
    domain = current_domain()
    analysis = domain.analyze_request(user_query=query) or {}
    profile = analysis.get("experiment_profile") or {}
    cqs: List[str] = []
    try:
        if profile:
            cqs = list(domain.candidate_queries(profile, fallback_query=query, max_queries=5))
    except Exception:
        cqs = []
    # Also honour any queries the planner proposed directly.
    for q in analysis.get("candidate_search_queries") or []:
        if q and q not in cqs:
            cqs.append(q)
    if not cqs:
        cqs = [query]
    return profile, cqs


def _nerd_profile_cached(query: str):
    """Same as _nerd_profile but disk-cached (cache/profiles.json), so re-ranking
    sweeps don't re-bill the (non-deterministic) LLM profile step and every weight
    re-ranks the IDENTICAL profile -> a clean, noise-free comparison."""
    cache = _load_cache("profiles")
    key = query.strip()
    if key in cache:
        c = cache[key]
        return c["profile"], c["cqs"]
    profile, cqs = _nerd_profile(query)
    cache[key] = {"profile": profile, "cqs": cqs}
    _save_cache("profiles", cache)
    return profile, cqs


# Dense fusion mirrors the app exactly by calling the SAME helpers in dense_index --
# the balanced_trim episode showed how fast a re-implemented eval path drifts from prod.
DENSE_FUSION = os.getenv("DENSE_FUSION", "0") == "1"
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "60") or 60)
_dense_mat = None
_dense_tried = False


def _dense_matrix():
    """Lazily load the dense matrix once, regardless of the fusion flag -- the
    `dense_only` baseline is a METHOD we compare against, not part of the feature."""
    global _dense_mat, _dense_tried
    if not _dense_tried:
        _dense_tried = True
        from dense_index import load_dense_index  # type: ignore
        _dense_mat = load_dense_index(INDEX_PATH.parent / "protocol_dense.npy")
        if _dense_mat is None:
            print("  (dense index missing - run scripts/build_dense_index.py)")
    return _dense_mat


def _get_dense():
    """Matrix for Nerd's RRF fusion -- only when the feature flag is on."""
    return _dense_matrix() if DENSE_FUSION else None


def _nerd_pool(profile, cqs, query: str, k: int,
               use_dense: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Deterministic retrieval only (no LLM): expand -> wide TF-IDF pool ->
    organism-merge. Returns the candidate pool BEFORE profile ranking.

    With DENSE_FUSION=1 the pool is the RRF fusion of the TF-IDF list and an embedding
    list, so semantically-right protocols that share no words with the request still get
    into the pool -- the re-ranker can only reorder what retrieval fetched."""
    expanded: List[str] = []
    for q in [c for c in cqs if c.strip()][:5]:
        expanded.extend(expand_query(q))
    seen_q, uniq = set(), []
    for q in expanded:
        key = q.strip().lower()
        if key and key not in seen_q:
            seen_q.add(key)
            uniq.append(q.strip())

    pool_size = max(k * 24, 120)
    results = multi_search_protocols(get_index(), uniq, pool_size)

    organism = str((profile or {}).get("organism") or "").strip()
    if organism and not _is_generic(organism):
        seen = {r["id"] for r in results}
        for oq in [organism] + [f"{organism} {q}" for q in cqs[:3] if q.strip()]:
            for r in search_protocols(get_index(), oq, top_k=10):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    results.append(r)

    # use_dense=None -> follow the feature flag; True/False -> force it, so a single
    # process can measure the dense and no-dense arms of the ablation side by side.
    want_dense = DENSE_FUSION if use_dense is None else use_dense
    mat = _dense_matrix() if want_dense else None
    if mat is not None:
        from dense_index import dense_candidates, fuse_pool  # type: ignore
        dense = dense_candidates(mat, get_index()["protocols"], query, cqs,
                                 min(DENSE_TOP_K, pool_size))
        results = fuse_pool(results, dense)
    return results


def _nerd_local_ranked(query: str, k: int):
    """The full Protocol Nerd LOCAL (protocols.io) pipeline. Mirrors backend
    main.run_local_candidate_searches. Returns (profile, candidate_queries, ranked_full)."""
    profile, cqs = _nerd_profile(query)
    pool = _nerd_pool(profile, cqs, query, k)
    ranked = closeness_rank(profile, pool, max(k, 10), raw_query=query)
    return profile, cqs, ranked


def nerd_full(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Protocol Nerd, protocols.io corpus ONLY (no PubMed). The clean same-corpus
    system used in both the backbone and graded tracks."""
    _profile, _cqs, ranked = _nerd_local_ranked(query, k)
    return [_slim(r) for r in ranked[:k]]


_STOP = {
    "i", "want", "to", "see", "where", "a", "an", "the", "of", "for", "in", "on",
    "inside", "into", "how", "do", "does", "need", "get", "getting", "using", "use",
    "with", "and", "or", "is", "are", "this", "that", "my", "me", "can", "could",
    "so", "it", "at", "by", "from", "out", "up", "some", "any", "well", "test",
}


def _pubmed_core(q: str) -> str:
    """Trim a natural-language query to key terms for a PubMed/API keyword search.
    PubMed ANDs every term, so the raw sentence starves; we take organism+method+
    goal concept tokens (cap 6). If concept extraction finds nothing (vague
    queries), fall back to stopword-stripped content words rather than AND-ing the
    whole sentence -- degenerate query prep would guarantee 0 hits and unfairly
    cripple the baseline."""
    c = extract_concepts(q)
    toks, seen = [], set()
    for group in ("organisms", "methods", "goals"):
        for term in c.get(group, []):
            for t in str(term).lower().split():
                if t not in seen:
                    seen.add(t)
                    toks.append(t)
    if toks:
        return " ".join(toks[:6])
    words = [w for w in re.findall(r"[a-z0-9-]+", q.lower())
             if w not in _STOP and len(w) > 2]
    return " ".join(words[:6]) or q


def nerd_full_blended(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Protocol Nerd AS SHIPPED: protocols.io + PubMed blended on one relevance
    axis (mirrors backend main.py Step 4.5). GRADED TRACK ONLY -- PubMed returns
    papers (PMIDs), which can't id-match a protocol gold, so this is meaningless
    for the known-item backbone. Compared against nerd_full, the delta is PubMed's
    contribution."""
    profile, cqs, ranked = _nerd_local_ranked(query, k)
    protocols_top = ranked[:k]
    for r in protocols_top:
        r.setdefault("source", "protocols.io")

    # PubMed: one trimmed core query per candidate query, sequential, de-duped.
    pubmed_ret = RETRIEVERS["pubmed"]
    pm_queries = [q for q in cqs if q.strip()] or [query]
    pm_results, seen = [], set()
    for q in pm_queries[:5]:
        core = _pubmed_core(q)
        try:
            hits = pubmed_ret.search(core, k) or []
        except Exception as e:  # noqa: BLE001
            print(f"    (pubmed core '{core[:40]}' failed: {e})")
            hits = []
        for r in hits:
            key = r.get("pmid") or r.get("doi") or r.get("title")
            if key and key not in seen:
                seen.add(key)
                r.setdefault("source", "pubmed")
                pm_results.append(r)

    # Keep best per_provider from the full PubMed pool by profile relevance.
    if profile and len(pm_results) > k:
        pm_results = current_domain().rank(profile, pm_results, top_k=k)
    else:
        pm_results = pm_results[:k]

    structured = profile_to_search_query(profile, query)
    blended = blend_sources(structured, {"protocols.io": protocols_top, "pubmed": pm_results})
    return [_slim_blended(r) for r in blended[:k]]


# ---------------------------------------------------------------------------
# System 3: live protocols.io native search (optional)
# ---------------------------------------------------------------------------
def protocolsio_native(query: str, k: int = 5) -> List[Dict[str, Any]]:
    from protocolsio_client import search_live  # imported lazily; needs token
    _total, items = search_live(query, max_results=k)
    return [_slim(r) for r in items[:k]]


# ---------------------------------------------------------------------------
# System 4: live PubMed native search (standalone baseline)
# ---------------------------------------------------------------------------
def pubmed_native(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Standalone PubMed API baseline. PubMed ANDs every term, so the raw sentence
    starves; the query is built by Haiku (keeps species/technique/sample/target) with a
    0-hit relaxation ladder down to the old token core -- sensible query prep any PubMed
    user would do -- but NO profile and NO re-ranking, keeping PubMed's own relevance
    order. Mirrors the app (pubmed_client.search_pubmed_smart)."""
    from pubmed_client import search_pubmed_fanout  # lazy; needs network
    hits = search_pubmed_fanout(query, k, fallback_core=_pubmed_core(query)) or []
    for r in hits:
        r.setdefault("source", "pubmed")
    return [_slim_blended(r) for r in hits[:k]]


def pubmed_combined_candidates(query: str, keep: int) -> List[Dict[str, Any]]:
    """PubMed candidates for the JOINT re-rank pool -- mirrors main.py exactly:
    search_pubmed_fanout(...) -> balanced_trim(..., keep). The standalone
    pubmed_native baseline keeps PubMed's own relevance order (a plain [:k] slice),
    but the combined pool must ROUND-ROBIN across fan-out alternatives so a
    multi-organism query ('rice OR potato') isn't one-sided before the re-ranker
    ever sees it. (The app also profile-ranks WITHIN each variant group; the eval
    has no profile here, so we keep fetch order within a group -- same balance,
    same cross-variant spread, which is what this fix is about.)"""
    from pubmed_client import search_pubmed_fanout, balanced_trim  # lazy; needs network
    hits = search_pubmed_fanout(query, max(keep, 5), fallback_core=_pubmed_core(query)) or []
    for r in hits:
        r.setdefault("source", "pubmed")
    return [_slim_blended(r) for r in balanced_trim(hits, keep)]


# ---------------------------------------------------------------------------
# Uniform dispatch + caching
# ---------------------------------------------------------------------------
SYSTEMS = {
    "plain_keyword": plain_keyword,
    "dense_only": dense_only,
    "nerd_full": nerd_full,
    "protocolsio_native": protocolsio_native,
    "pubmed_native": pubmed_native,
    # Optional: Protocol Nerd as shipped (protocols.io + PubMed blended). Graded
    # track only. Compare vs nerd_full to isolate PubMed's contribution.
    "nerd_full_blended": nerd_full_blended,
}


def run_system(system: str, query: str, k: int = 5,
               use_cache: bool = True) -> List[Dict[str, Any]]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {list(SYSTEMS)}")
    cache = _load_cache(system) if use_cache else {}
    key = _cache_key(query, k)
    if use_cache and key in cache:
        return cache[key]
    results = SYSTEMS[system](query, k)
    if use_cache:
        cache[key] = results
        _save_cache(system, cache)
    return results


def result_ids(results: List[Dict[str, Any]]) -> List[int]:
    return [int(r["id"]) for r in results if r.get("id") is not None]
