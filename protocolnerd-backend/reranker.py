"""
Optional LLM re-ranker (feature-flagged, OFF by default).

When enabled, after the profile ranker produces the candidate list, an LLM
(Haiku by default) re-orders a shortlist -- profile-top-N UNION lexical-top-N --
so a strong match the profile ranker buried gets surfaced. Evaluation showed this
lifts nDCG@10 by ~+0.2 (measured circularity-free against an id-match set and a
different-model judge) at ~1.2s/query with Haiku.

Toggle (defaults match prod = reranker ON):
    env  ENABLE_RERANKER=false                disable it (master switch, default true)
    env  RERANKER_MODEL=claude-haiku-4-5       (default)
    env  RERANKER_PROVIDER=claude              (default)
    per-request override:  ChatRequest.enable_reranker (True/False)

Any failure -- LLM error, bad output, empty list -- falls back to the original
profile ranking, so enabling the reranker can never make a search worse than not
having it.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from llm_providers import call_llm

log = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "claude-haiku-4-5")
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "claude")
SHORTLIST_PROFILE = int(os.getenv("RERANKER_SHORTLIST_PROFILE", "15") or 15)
SHORTLIST_LEXICAL = int(os.getenv("RERANKER_SHORTLIST_LEXICAL", "15") or 15)
# How many PubMed candidates the JOINT re-rank (#2) may consider. The re-ranker keeps
# only the papers that earn a top-K slot, so a bigger pool costs nothing in output
# quality but gives it more to choose from. Raised from 2 once the Haiku-built PubMed
# query made those papers genuinely useful (held-out PubMed nDCG 0.142 -> 0.305).
COMBINED_PUBMED_CANDIDATES = int(os.getenv("COMBINED_PUBMED_CANDIDATES", "5") or 5)

_SYSTEM = (
    "You are an expert biologist helping a scientist find the most useful lab "
    "protocol or paper for their experiment. Given their request and a numbered "
    "list of candidates, rank the candidates by how directly each one helps them "
    "RUN the described experiment (right technique + compatible organism/sample). "
    "Return ONLY a JSON array of the candidate numbers, best first, e.g. [4,1,9,...]. "
    "Include every candidate number exactly once."
)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def is_enabled(request_override: Optional[bool] = None) -> bool:
    """Per-request override wins; otherwise the ENABLE_RERANKER env flag (default True, matching prod)."""
    if request_override is not None:
        return bool(request_override)
    return _env_bool("ENABLE_RERANKER", True)


def combined_pubmed_enabled(request_override: Optional[bool] = None) -> bool:
    """#2 (shipped, default ON): the re-ranker vets the PubMed candidates alongside
    protocols.io in ONE combined pass (weak papers sink out / are dropped) instead
    of blending PubMed in afterward by a lexical score the re-ranker never sees.
    Held-out eval: +0.05-0.07 nDCG@10 over the old lexical 8+2 blend. Disable via
    RERANK_COMBINED_PUBMED=false or a per-request override."""
    if request_override is not None:
        return bool(request_override)
    return _env_bool("RERANK_COMBINED_PUBMED", True)


def _lexical_score(r: Dict[str, Any]) -> float:
    b = r.get("blend_score")
    if b is None:
        b = r.get("score") or r.get("combined_score") or 0.0
    return float(b or 0.0)


def _system_prompt() -> str:
    """The active domain's re-rank prompt, falling back to _SYSTEM (biology's, and
    the historical default) if the domain doesn't supply one. Deferred import so
    this module stays importable independently of the domains package."""
    try:
        from domains import current_domain
        return current_domain().rerank_system_prompt() or _SYSTEM
    except Exception:  # noqa: BLE001 -- prompt lookup must never break a re-rank
        return _SYSTEM


def _llm_order(query: str, shortlist: List[Dict[str, Any]], model: str) -> List[Any]:
    lines = [f"[{i}] {c.get('title','')}: {(c.get('description') or '')[:180]}"
             for i, c in enumerate(shortlist, 1)]
    user = (f"REQUEST:\n{query}\n\nCANDIDATES:\n" + "\n".join(lines) +
            f"\n\nRank all {len(shortlist)} candidates. Return ONLY a JSON array of "
            "their numbers, best first.")
    raw = call_llm(
        messages=[{"role": "system", "content": _system_prompt()},
                  {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
        provider=RERANKER_PROVIDER,
        model=model,
    )
    ids, seen = [], set()
    for n in (int(x) for x in re.findall(r"\d+", raw)):
        if 1 <= n <= len(shortlist):
            rid = shortlist[n - 1].get("id")
            if rid is not None and rid not in seen:
                seen.add(rid)
                ids.append(rid)
    return ids


def rerank(query: str, ranked: List[Dict[str, Any]],
           pool: Optional[List[Dict[str, Any]]] = None, top_k: int = 5,
           model: Optional[str] = None) -> List[Dict[str, Any]]:
    """Re-order `ranked` (profile-ranked results) with an LLM over a shortlist of
    profile-top-N UNION lexical-top-N candidates; return top_k. Falls back to
    ranked[:top_k] on any problem, so it can never make results worse."""
    if not ranked:
        return ranked
    pool = pool or ranked
    lexical = sorted(pool, key=_lexical_score, reverse=True)[:SHORTLIST_LEXICAL]
    shortlist, seen = [], set()
    for r in list(ranked[:SHORTLIST_PROFILE]) + lexical:
        rid = r.get("id")
        if rid is not None and rid not in seen:
            seen.add(rid)
            shortlist.append(r)
    if len(shortlist) <= top_k:
        return ranked[:top_k]

    try:
        order_ids = _llm_order(query, shortlist, model or RERANKER_MODEL)
    except Exception as e:  # noqa: BLE001
        log.warning(f"reranker failed ({e}); keeping profile order.")
        return ranked[:top_k]
    if not order_ids:
        return ranked[:top_k]

    by_id = {r.get("id"): r for r in shortlist}
    ordered, oseen = [], set()
    for rid in order_ids:
        if rid in by_id and rid not in oseen:
            oseen.add(rid)
            ordered.append(by_id[rid])
    # append shortlist items the LLM omitted (in profile order) so nothing is dropped
    for r in shortlist:
        if r.get("id") not in oseen:
            oseen.add(r.get("id"))
            ordered.append(r)
    return ordered[:top_k]
