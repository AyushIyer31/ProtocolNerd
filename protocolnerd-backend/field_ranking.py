"""
Closeness-tier ranking layer.

Implements the design rule we settled on:

  * Each profile field has a ROLE — a hard *anchor* (must match, by concept),
    a graded *preference*, or a soft sorter.
  * Anchor fields are scored by CLOSENESS (exact > near > broad > unrelated),
    not literal keyword presence.
  * "like X / such as / similar to" relaxes the field it modifies from a hard
    anchor to a graded preference; the other fields stay as they were.

The existing per-field matchers in experiment_profile.py already do graded,
concept-level scoring (and produce the result labels). The bug was only in the
ORDERING — off-organism protocols out-scored organism-correct ones. So this
layer reuses apply_profile_ranking for scoring + labels, then re-sorts so that
the named-organism (and named-method) closeness tier dominates.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from experiment_profile import (
    apply_profile_ranking,
    _result_text,
    _is_generic,
    _has_any_term,
    _method_match_score,
    _PLANT_CONTEXT_TERMS,
)

# --------------------------------------------------------------------------
# Organism closeness model — KINGDOM-SYMMETRIC (plants, animals, microbes):
#   exact species -> same family -> same kingdom -> none
# --------------------------------------------------------------------------

# Real taxonomic families (NOT grab-bags). Used only for the same-family (0.7) tier.
_FAMILIES: Dict[str, List[str]] = {
    # plant families
    "solanaceae": ["tomato", "solanum lycopersicum", "potato", "solanum tuberosum",
                   "pepper", "capsicum", "chili", "chilli", "tobacco", "nicotiana",
                   "nicotiana benthamiana", "eggplant", "aubergine", "petunia"],
    "brassicaceae": ["arabidopsis", "arabidopsis thaliana", "brassica", "canola",
                     "rapeseed", "oilseed rape", "mustard", "camelina", "cabbage",
                     "brassica rapa", "brassica napus"],
    "poaceae": ["rice", "oryza", "oryza sativa", "maize", "corn", "zea mays",
                "wheat", "triticum", "barley", "hordeum", "sorghum", "setaria",
                "brachypodium", "millet", "sugarcane"],
    "fabaceae": ["soybean", "soya", "glycine max", "glycine", "medicago", "lotus",
                 "common bean", "phaseolus", "pea", "cowpea", "chickpea"],
    # animal families
    "rodent": ["mouse", "mice", "mus musculus", "murine", "rat", "rattus"],
    "fish": ["zebrafish", "danio rerio", "medaka", "oryzias"],
}

_KINGDOM_OF_FAMILY = {
    "solanaceae": "plant", "brassicaceae": "plant", "poaceae": "plant", "fabaceae": "plant",
    "rodent": "animal", "fish": "animal",
}

# Organisms in a kingdom that aren't in a tracked family — used to assign kingdom.
_KINGDOM_MEMBERS = {
    "plant": ["banana", "musa", "mimulus", "poplar", "populus", "cassava", "grape",
              "vitis", "cotton", "gossypium", "strawberry", "moss", "physcomitrium",
              "physcomitrella", "marchantia", "liverwort", "fern"],
    "animal": ["human", "homo sapiens", "xenopus", "frog", "drosophila", "fruit fly",
               "c. elegans", "caenorhabditis", "worm", "nematode", "rabbit", "pig",
               "porcine", "bovine", "cattle", "chicken", "monkey", "primate", "insect"],
    "microbe": ["e. coli", "escherichia coli", "escherichia", "yeast", "saccharomyces",
                "cerevisiae", "bacteria", "bacterium", "salmonella", "pseudomonas",
                "bacillus", "streptococcus", "staphylococcus", "fungus", "aspergillus",
                "candida", "mycobacterium", "chlamydomonas", "algae"],
}

# Context terms that confirm a protocol is about a given kingdom. Kept kingdom-
# specific (no generic 'cell'/'tissue', which all kingdoms share).
_KINGDOM_CONTEXT = {
    "plant": _PLANT_CONTEXT_TERMS,
    "animal": ["mouse", "mice", "murine", "rat", "rattus", "rodent", "zebrafish",
               "danio", "xenopus", "frog", "tadpole", "drosophila", "fruit fly",
               "c. elegans", "caenorhabditis", "nematode", "mammalian", "mammal",
               "in vivo", "embryo", "larva", "oocyte", "rabbit", "porcine", "bovine",
               "chicken", "primate", "monkey", "human", "patient", "ipsc", "neuron",
               "retina", "cardiac", "endothelial"],
    "microbe": ["bacteria", "bacterial", "escherichia", "e. coli", "coli", "salmonella",
                "pseudomonas", "bacillus", "yeast", "saccharomyces", "cerevisiae",
                "fungal", "fungus", "mycelium", "colony", "strain", "broth", "agar"],
}

# species -> synonyms used for the EXACT tier
_ORG_SYNONYMS: Dict[str, List[str]] = {
    "tomato": ["tomato", "solanum lycopersicum"],
    "potato": ["potato", "solanum tuberosum"],
    "rice": ["rice", "oryza sativa", "oryza"],
    "maize": ["maize", "corn", "zea mays"],
    "wheat": ["wheat", "triticum"],
    "barley": ["barley", "hordeum"],
    "banana": ["banana", "musa"],
    "arabidopsis": ["arabidopsis", "arabidopsis thaliana"],
    "arabidopsis thaliana": ["arabidopsis", "arabidopsis thaliana"],
    "nicotiana benthamiana": ["nicotiana benthamiana", "n benthamiana", "n. benthamiana"],
    "tobacco": ["tobacco", "nicotiana"],
    "soybean": ["soybean", "soya", "glycine max"],
    "mouse": ["mouse", "mice", "mus musculus", "murine"],
    "rat": ["rat", "rattus"],
    "zebrafish": ["zebrafish", "danio rerio"],
    "human": ["human", "homo sapiens"],
    "yeast": ["yeast", "saccharomyces", "s. cerevisiae", "cerevisiae"],
    "e. coli": ["e. coli", "escherichia coli"],
    "drosophila": ["drosophila", "fruit fly"],
    "c. elegans": ["c. elegans", "caenorhabditis elegans"],
}


def _word_in(text: str, term: str) -> bool:
    """Word-ish boundary match (case-insensitive); term is already lowercase."""
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def _synonyms_for(org: str) -> List[str]:
    return _ORG_SYNONYMS.get(org, [org])


def _org_family(org: str) -> "str | None":
    for fam, members in _FAMILIES.items():
        if org in members:
            return fam
    return None


def _org_kingdom(org: str) -> "str | None":
    fam = _org_family(org)
    if fam:
        return _KINGDOM_OF_FAMILY.get(fam)
    for kingdom, members in _KINGDOM_MEMBERS.items():
        if any(org == m or _word_in(org, m) for m in members):
            return kingdom
    return None


def organism_closeness(organism: Any, text: str) -> float:
    """
    Kingdom-symmetric organism relevance in [0, 1] — same logic for plants,
    animals, and microbes:
      exact species ... 1.0
      same family ..... 0.7   (rat for mouse; pepper for tomato)
      same kingdom .... 0.4   (any animal for a mouse query; any plant for banana)
      different/none .. 0.0

    A match only counts when the text carries the organism's KINGDOM context
    (animal terms for an animal query, plant terms for a plant query), matched
    on WORD boundaries. This symmetrically blocks homonym false positives:
    'tomato' inside 'tdTomato' (no plant context), or 'plant' inside 'implant'.
    """
    o = str(organism or "").strip().lower()
    if not o or _is_generic(o):
        return 0.0

    exact = any(_word_in(text, s) for s in _synonyms_for(o))
    kingdom = _org_kingdom(o)

    if kingdom is None:
        # Unknown kingdom: only a direct word-boundary match counts.
        return 1.0 if exact else 0.0

    if not any(_word_in(text, t) for t in _KINGDOM_CONTEXT.get(kingdom, [])):
        return 0.0  # protocol isn't about this kingdom -> not a match

    if exact:
        return 1.0
    fam = _org_family(o)
    if fam and any(_word_in(text, m) for m in _FAMILIES.get(fam, [])):
        return 0.7
    return 0.4  # same kingdom


# --------------------------------------------------------------------------
# "like / such as / similar to" relaxation detection
# --------------------------------------------------------------------------

_METHOD_HINTS = ["crispr", "cas9", "cas12", "talen", "zfn", "base editing",
                 "prime editing", "genome editing", "gene editing", "knockout",
                 "knock-out", "knockdown", "knock-down", "overexpression",
                 "mutagenesis", "rnai", "transformation", "pcr", "qpcr",
                 "sequencing", "western", "cloning", "transfection"]
_ORG_HINTS = ["tomato", "rice", "maize", "wheat", "barley", "arabidopsis",
              "tobacco", "soybean", "potato", "pepper", "plant", "plants",
              "mouse", "mice", "rat", "zebrafish", "yeast", "human",
              "drosophila", "bacteria", "e. coli", "worm", "c. elegans"]
_LIKE_TRIGGER = re.compile(r"\b(like|such as|similar to|or similar|e\.?g\.?)\b")


# --------------------------------------------------------------------------
# Gene-modification relevance (concept-level, not literal)
# --------------------------------------------------------------------------
# When the intent is gene editing/modification, a protocol is "on-topic" if it
# does editing OR the transformation/delivery that carries it. A tomato DNA-
# extraction or metabolite protocol is off-topic even if it matches the tissue.

_GENE_MOD_INTENT_TERMS = [
    "crispr", "editing", "modification", "knockout", "knockdown", "knock-out",
    "knock-down", "knockin", "knock-in", "overexpression", "mutagenesis",
    "mutation", "transformation", "transgenic", "silencing", "rnai",
    "base editing", "prime editing",
]
_GENE_MOD_STRONG = [
    "crispr", "cas9", "cas12", "cas13", "talen", "zinc finger", "zfn",
    "genome editing", "gene editing", "base editing", "prime editing",
    "knockout", "knock-out", "knockin", "knock-in", "knockdown", "knock-down",
    "mutagenesis", "gene modification", "genome modification", "overexpression",
    "gene silencing", "rnai",
]
_GENE_MOD_DELIVERY = [
    "transform", "transgenic", "transformant", "agrobacterium", "floral dip",
    "biolistic", "particle bombardment", "gene gun", "electroporation",
]


def _is_gene_mod_intent(profile: Dict[str, Any]) -> bool:
    blob = " ".join(
        str((profile or {}).get(k) or "")
        for k in ("modification_type", "sub_intent", "experimental_method", "intent_family")
    ).lower()
    return any(t in blob for t in _GENE_MOD_INTENT_TERMS)


def gene_mod_relevance(text: str) -> float:
    """Graded: does the protocol actually do gene editing/modification (1.0) or
    the transformation/delivery that carries it (0.8), or neither (0.0)?"""
    if _has_any_term(text, _GENE_MOD_STRONG):
        return 1.0
    if _has_any_term(text, _GENE_MOD_DELIVERY):
        return 0.8
    return 0.0


# --------------------------------------------------------------------------
# Title relevance — boost protocols whose TITLE carries the core intent.
# A title is a curated summary of purpose, so a discriminating concept in the
# title ('multiplex', 'embryo', 'CRISPR') is strong evidence the protocol is
# about it — unlike the same word buried in an abstract's motivation. Used as a
# within-organism tiebreaker: a boost, never a filter (a plain title earns no
# boost but is not penalized).
# --------------------------------------------------------------------------

_TISSUE_TITLE_SYNONYMS = {
    "embryo": ["embryo", "embryos", "zygote", "pronuclear"],
    "leaf": ["leaf", "leaves"],
    "leaf tissue": ["leaf", "leaves"],
    "protoplast": ["protoplast", "protoplasts"],
    "protoplasts": ["protoplast", "protoplasts"],
    "callus": ["callus"],
    "callus / tissue culture": ["callus", "tissue culture"],
    "in planta / whole plant": ["in planta", "whole plant"],
    "primary cells": ["primary cell", "primary cells"],
}
_EDITING_TITLE_TERMS = ["crispr", "cas9", "cas12", "cas13", "talen", "zfn",
                        "base editing", "prime editing", "genome editing", "gene editing"]
_TRANSFORM_TITLE_TERMS = ["transgenic", "transformation", "transformant",
                          "knock-in", "knockin", "knockout", "genetically modified"]


def _title_concepts(profile: Dict[str, Any]):
    """Discriminating (weight, terms) groups from the profile, weighted by how
    much each separates relevant from irrelevant protocols. Organism is excluded
    (it's already the dominant tier); readouts/generic qualifiers are excluded."""
    p = profile or {}
    blob = " ".join(str(p.get(k) or "") for k in
                    ("sub_intent", "experimental_method", "modification_type")).lower()
    groups = []  # (weight, terms)

    if "multiplex" in blob or "combinatorial" in blob:
        groups.append((1.0, ["multiplex", "multiplexed", "combinatorial"]))
    if _is_gene_mod_intent(p) or "crispr" in blob or "editing" in blob:
        groups.append((0.8, _EDITING_TITLE_TERMS))
    deliv = (str(p.get("modification_type") or "") + " " + str(p.get("delivery_method") or "")).lower()
    if any(t in deliv for t in ("transform", "transgenic", "stable", "knock")):
        groups.append((0.6, _TRANSFORM_TITLE_TERMS))
    tissue = str(p.get("tissue_or_cell_type") or p.get("sample_type") or "").strip().lower()
    if tissue and not _is_generic(tissue):
        groups.append((0.7, _TISSUE_TITLE_SYNONYMS.get(tissue, [tissue])))
    target = str(p.get("target") or "").strip().lower()
    if target and not _is_generic(target) and target not in ("editing", "modification", "gene"):
        terms = (["transcription factor", "transcription factors"]
                 if "transcription factor" in target else [target])
        groups.append((0.6, terms))
    return groups


def title_relevance(profile: Dict[str, Any], title: str) -> float:
    """Fraction (0-1) of the profile's weighted title-concepts present in the
    title, matched on word boundaries."""
    groups = _title_concepts(profile)
    if not groups:
        return 0.0
    t = (title or "").lower()
    hit = sum(w for w, terms in groups if any(_word_in(t, term) for term in terms))
    total = sum(w for w, _ in groups)
    return hit / total if total else 0.0


def detect_like_relaxations(raw_query: str) -> Set[str]:
    """Return which anchor fields the user relaxed via 'like / such as / e.g.'."""
    q = str(raw_query or "").lower()
    relaxed: Set[str] = set()
    for m in _LIKE_TRIGGER.finditer(q):
        after = q[m.end():m.end() + 45]
        meth_pos = min([after.find(h) for h in _METHOD_HINTS if h in after] or [999])
        org_pos = min([after.find(h) for h in _ORG_HINTS if h in after] or [999])
        if meth_pos < org_pos:
            relaxed.add("method")
        elif org_pos < meth_pos:
            relaxed.add("organism")
    return relaxed


# --------------------------------------------------------------------------
# Public entry: score with the existing engine, then re-order by closeness
# --------------------------------------------------------------------------

def closeness_rank(
    profile: Dict[str, Any],
    results: List[Dict[str, Any]],
    top_k: int,
    raw_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Reuse apply_profile_ranking for per-field scoring + labels, then re-sort so
    the named organism's closeness tier dominates (unless relaxed by "like"),
    with the named method as a secondary tier, and the existing blended score as
    the within-tier tiebreaker. Falls back to the existing order when no anchor
    is specified or when nothing matches the anchor.
    """
    if not results:
        return results

    # Score + label everything (don't truncate yet — we re-rank below).
    ranked = apply_profile_ranking(profile, results, top_k=len(results))

    relaxed = detect_like_relaxations(raw_query)
    organism = str((profile or {}).get("organism") or "").strip()
    org_specific = bool(organism) and not _is_generic(organism)

    # Organism closeness is the dominant tier whenever a SPECIFIC organism is
    # named. Graded tiers (exact > family > plant > unrelated) serve both the
    # strict intent ("CRISPR in tomato": tomato dominates) and the flexible one
    # ("plants like tomato": tomato preferred, relatives next, then other
    # plants, then non-plants) — so "like" on the organism needs no special
    # handling here. When organism is vague ("plant"), keep the existing
    # balanced order (it still rewards the named organism via its score bonus).
    if not org_specific:
        return ranked[:top_k]

    method = str(
        (profile or {}).get("experimental_method")
        or (profile or {}).get("modification_type")
        or ""
    ).strip()
    # Within-organism tiebreaker (never a dominant tier). For gene-modification
    # intents use the broad concept relevance — so a tomato *transformation*
    # protocol (which delivers editing) ranks above a tomato DNA-extraction or
    # metabolite protocol that only matches the tissue. This also naturally
    # handles "like CRISPR" (transformation counts, not just literal CRISPR).
    # For other intents fall back to the existing per-method matcher, relaxed by
    # "like".
    gene_mod = _is_gene_mod_intent(profile)
    use_method_tiebreak = (
        bool(method) and not _is_generic(method) and "method" not in relaxed
    )

    def sort_key(r: Dict[str, Any]):
        text = _result_text(r)
        org_tier = organism_closeness(organism, text)
        # Title match is the strongest within-organism signal: a discriminating
        # concept in the title means the protocol is *about* it, not just
        # mentioning it (which is what let an RNA-extraction protocol outrank a
        # multiplex-CRISPR one when both merely contained "CRISPR").
        title_score = title_relevance(profile, r.get("title") or "")
        if gene_mod:
            rel = gene_mod_relevance(text)
        elif use_method_tiebreak:
            rel = _method_match_score(method, text)
        else:
            rel = 0.0
        base = float(r.get("profile_score", r.get("score", 0.0)) or 0.0)
        # organism tier -> title concept match -> body relevance -> blended score
        return (round(org_tier, 1), round(title_score, 1), round(rel, 1), base)

    ranked.sort(key=sort_key, reverse=True)
    return ranked[:top_k]
