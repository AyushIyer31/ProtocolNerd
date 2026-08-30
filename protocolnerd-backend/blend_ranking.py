"""
Blend protocols.io and PubMed results into one ranked, source-tagged list.

Two rules, both from the product spec:

  1. Unified relevance weights TITLE over BODY (0.67 / 0.33), computed identically
     for both sources so the scores are directly comparable.
  2. The protocols.io list keeps its existing internal order untouched. The
     unified score is used ONLY as the common axis to interleave PubMed papers
     into that list — a PubMed paper is placed ahead of the first protocols.io
     entry it out-scores. We never re-sort the protocols.io entries among
     themselves.

So protocols.io ranking is unchanged; PubMed slots in by relevance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ranking_config import BLEND_TITLE_WEIGHT as _TITLE_WEIGHT, BLEND_BODY_WEIGHT as _BODY_WEIGHT

log = logging.getLogger(__name__)


def _body_text(r: Dict[str, Any]) -> str:
    # protocols.io: description; pubmed: abstract (falls back to description).
    return str(r.get("abstract") or r.get("description") or "")


def _title_text(r: Dict[str, Any]) -> str:
    return str(r.get("title") or "")


def _unified_scores(query: str, results: List[Dict[str, Any]]) -> List[float]:
    """
    Cosine of the query against each result's title and body, blended
    0.67*title + 0.33*body. One shared vocabulary across all candidates +
    the query so the numbers are comparable across both sources.
    """
    if not results:
        return []
    titles = [_title_text(r) for r in results]
    bodies = [_body_text(r) for r in results]
    corpus = [query] + titles + bodies
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=4096)
        matrix = vec.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary (all-stopword/empty corpus) — nothing to score.
        return [0.0] * len(results)

    qv = matrix[0]
    n = len(results)
    title_m = matrix[1 : 1 + n]
    body_m = matrix[1 + n : 1 + 2 * n]
    title_cos = cosine_similarity(qv, title_m).flatten()
    body_cos = cosine_similarity(qv, body_m).flatten()
    return [
        round(_TITLE_WEIGHT * float(title_cos[i]) + _BODY_WEIGHT * float(body_cos[i]), 4)
        for i in range(n)
    ]


def blend_results(
    query: str,
    protocols_results: List[Dict[str, Any]],
    pubmed_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return one list: protocols.io entries in their original order, with PubMed
    papers interleaved by the unified title>body score. Every entry carries a
    `source` tag and a `blend_score` for transparency.
    """
    for r in protocols_results:
        r.setdefault("source", "protocols.io")
    for r in pubmed_results:
        r["source"] = "pubmed"

    if not pubmed_results:
        scores = _unified_scores(query, protocols_results)
        for r, s in zip(protocols_results, scores):
            r["blend_score"] = s
        return list(protocols_results)
    if not protocols_results:
        scores = _unified_scores(query, pubmed_results)
        for r, s in zip(pubmed_results, scores):
            r["blend_score"] = s
        return sorted(pubmed_results, key=lambda r: r["blend_score"], reverse=True)

    # Score both sets on one shared axis.
    all_results = list(protocols_results) + list(pubmed_results)
    scores = _unified_scores(query, all_results)
    for r, s in zip(all_results, scores):
        r["blend_score"] = s

    # Stable interleave: walk protocols.io IN ORDER; before each anchor, emit any
    # remaining PubMed paper (highest first) that out-scores that anchor.
    papers = sorted(pubmed_results, key=lambda r: r["blend_score"], reverse=True)
    blended: List[Dict[str, Any]] = []
    pi = 0
    for anchor in protocols_results:
        while pi < len(papers) and papers[pi]["blend_score"] >= anchor["blend_score"]:
            blended.append(papers[pi])
            pi += 1
        blended.append(anchor)
    # Any lower-scored papers trail at the end.
    blended.extend(papers[pi:])
    return blended


def blend_sources(
    query: str,
    results_by_source: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Blend results from N named sources onto one comparable relevance axis.

    For the canonical protocols.io + PubMed pair this delegates to `blend_results`
    to preserve its exact legacy ordering (the profile re-ranker downstream sorts
    by profile_score anyway, but single-source / no-profile paths still rely on it).
    For any other combination (e.g. a third source added via a new Retriever), it
    assigns a uniform `blend_score` across every result and returns one
    relevance-sorted list — no source is structurally privileged.
    """
    for src, rows in results_by_source.items():
        for r in rows or []:
            r.setdefault("source", src)

    present = {src: rows for src, rows in results_by_source.items() if rows}
    if set(present) <= {"protocols.io", "pubmed"}:
        return blend_results(
            query,
            present.get("protocols.io", []),
            present.get("pubmed", []),
        )

    all_results = [r for rows in present.values() for r in rows]
    scores = _unified_scores(query, all_results)
    for r, s in zip(all_results, scores):
        r["blend_score"] = s
    return sorted(all_results, key=lambda r: r.get("blend_score", 0.0), reverse=True)
