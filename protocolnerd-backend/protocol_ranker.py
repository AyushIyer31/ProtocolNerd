"""
Multi-signal re-ranker for merged protocols.io results.

Implements the ranking signals Prof. Shasha asked for. Each candidate protocol
is scored on five weighted signals, plus a small frequency bonus for protocols
surfaced by multiple probes:

    title_match       core query terms appearing in the TITLE      (weight 3.0)
    organism_match    organism (common OR scientific name) present  (weight 2.0)
    method_match      a method/technique term present               (weight 1.5)
    description_match core query terms appearing in the DESCRIPTION (weight 1.0)
    synonym_match     an EXPANSION synonym present (vocabulary the
                      original query did not contain)               (weight 1.0)

Each result carries a per-signal breakdown and a human-readable "why it matches"
string so the benchmark table and the chatbot can explain the ranking.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_WEIGHTS = {
    "title": 3.0,
    "organism": 2.0,
    "method": 1.5,
    "description": 1.0,
    "synonym": 1.0,
}
_FREQ_BONUS = 0.25  # per extra probe that returned this protocol
_MISSING_MOLECULE_PENALTY = 0.25  # multiplier when a required molecule is absent

# Concept-coverage: reward protocols that match MANY of the query's distinct
# concepts (organism + molecule + method + goal), not just one. This extends
# Prof. Shasha's AND-logic from the molecule alone to ALL key concepts.
_COVERAGE_WEIGHT = 3.0   # scales with the fraction of concepts covered
_ALL_COVERED_BONUS = 2.0  # extra reward when a protocol covers every concept


def _present(terms: List[str], text: str) -> List[str]:
    """
    Return the terms that appear in `text` as whole words/phrases.

    Word-boundary matching avoids false positives like "rice" matching inside
    "price" or "Patrice" — which previously gave irrelevant protocols a spurious
    organism match.
    """
    tl = text.lower()
    hits = []
    for t in terms:
        t2 = (t or "").lower().strip()
        if not t2 or t in hits:
            continue
        if re.search(rf"\b{re.escape(t2)}\b", tl):
            hits.append(t)
    return hits


def _ratio(terms: List[str], text: str) -> float:
    if not terms:
        return 0.0
    return len(_present(terms, text)) / len(terms)


def rank_protocols(
    concepts: Dict[str, Any],
    expansions: Dict[str, List[str]],
    merged: Dict[int, Dict[str, Any]],
    hit_map: Dict[int, List[str]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    core_terms = concepts.get("core_terms", [])

    organism_terms: List[str] = list(concepts.get("organisms", []))
    for org in concepts.get("organisms", []):
        organism_terms += expansions.get(org, [])

    method_terms: List[str] = list(concepts.get("methods", []))
    for m in concepts.get("methods", []):
        method_terms += expansions.get(m, [])

    goal_terms: List[str] = list(concepts.get("goals", []))
    for g in concepts.get("goals", []):
        goal_terms += expansions.get(g, [])

    # Synonyms = every expansion term (the added vocabulary)
    synonym_terms: List[str] = []
    for exp in expansions.values():
        synonym_terms += exp

    # Required molecule terms (DNA/RNA/protein) + their synonyms. When the user
    # names a target molecule it MUST appear — a "DNA extraction" query should not
    # rank a protein/compound extraction protocol (Prof. Shasha's AND-logic point).
    molecules = concepts.get("molecules", [])
    molecule_terms: List[str] = list(molecules)
    for mol in molecules:
        molecule_terms += expansions.get(mol, [])

    # The distinct concept facets the query actually contains. Coverage is
    # measured against these — e.g. "DNA RNA mice" has 3 facet groups to cover.
    facet_terms = {
        "organism": organism_terms,
        "molecule": molecule_terms,
        "method": method_terms,
        "goal": goal_terms,
    }
    present_facets = [name for name, terms in facet_terms.items() if terms]

    ranked: List[Dict[str, Any]] = []
    for pid, p in merged.items():
        title = p.get("title") or ""
        desc = p.get("description") or ""
        kw = " ".join(str(k) for k in (p.get("keywords") or []))
        body = f"{desc} {kw}"
        full = f"{title} {body}"

        title_hits = _present(core_terms, title)
        desc_ratio = _ratio(core_terms, body)
        organism_hits = _present(organism_terms, full)
        method_hits = _present(method_terms, full)
        synonym_hits = _present(synonym_terms, full)
        molecule_hits = _present(molecule_terms, full)

        s_title = _ratio(core_terms, title) * _WEIGHTS["title"]
        s_org = (1.0 if organism_hits else 0.0) * _WEIGHTS["organism"]
        s_method = (1.0 if method_hits else 0.0) * _WEIGHTS["method"]
        s_desc = desc_ratio * _WEIGHTS["description"]
        s_syn = min(1.0, len(synonym_hits) / 2.0) * _WEIGHTS["synonym"]

        probes = hit_map.get(pid, [])
        freq_bonus = max(0, len(probes) - 1) * _FREQ_BONUS

        # Concept coverage: which of the query's facets does this protocol hit?
        facet_hits = {
            "organism": organism_hits,
            "molecule": molecule_hits,
            "method": method_hits,
            "goal": _present(goal_terms, full),
        }
        covered = [name for name in present_facets if facet_hits[name]]
        coverage_frac = (len(covered) / len(present_facets)) if present_facets else 0.0
        s_coverage = coverage_frac * _COVERAGE_WEIGHT
        all_covered = bool(present_facets) and len(covered) == len(present_facets) and len(present_facets) > 1
        coverage_bonus = _ALL_COVERED_BONUS if all_covered else 0.0

        raw_score = (s_title + s_org + s_method + s_desc + s_syn
                     + freq_bonus + s_coverage + coverage_bonus)

        # AND-logic penalty: if the query named a molecule and this protocol does
        # not mention it (or a synonym), it is almost certainly off-target.
        molecule_missing = bool(molecules) and not molecule_hits
        penalty = _MISSING_MOLECULE_PENALTY if molecule_missing else 1.0
        score = round(raw_score * penalty, 3)

        why_parts = []
        if title_hits:
            why_parts.append(f"title matches {title_hits}")
        if molecule_hits:
            why_parts.append(f"molecule match ({', '.join(molecule_hits[:2])})")
        if organism_hits:
            why_parts.append(f"organism match ({', '.join(organism_hits[:2])})")
        if method_hits:
            why_parts.append(f"method match ({', '.join(method_hits[:2])})")
        if facet_hits["goal"]:
            why_parts.append(f"goal match ({', '.join(facet_hits['goal'][:2])})")
        if present_facets:
            why_parts.append(f"covers {len(covered)}/{len(present_facets)} concepts"
                             + (" (ALL)" if all_covered else ""))
        if synonym_hits:
            why_parts.append(f"synonym match ({', '.join(synonym_hits[:2])})")
        if len(probes) > 1:
            why_parts.append(f"surfaced by {len(probes)} probes")
        if molecule_missing:
            why_parts.append(f"penalized: missing {molecules[0]}")
        if not why_parts:
            why_parts.append("description/keyword overlap only")

        ranked.append({
            **p,
            "score": score,
            "signals": {
                "title": round(s_title, 3),
                "organism": round(s_org, 3),
                "method": round(s_method, 3),
                "description": round(s_desc, 3),
                "synonym": round(s_syn, 3),
                "coverage": round(s_coverage + coverage_bonus, 3),
                "frequency_bonus": round(freq_bonus, 3),
                "molecule_penalty": penalty,
            },
            "concepts_covered": f"{len(covered)}/{len(present_facets)}" if present_facets else "0/0",
            "matched_probes": probes,
            "why": "; ".join(why_parts),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]
