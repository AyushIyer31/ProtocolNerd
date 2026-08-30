"""
Declarative profile schema (increment 4a — ranking).

The experiment profile is described here as DATA rather than hard-coded in the
ranking/clarification logic. Each ranking signal is one record: which key it
reads from `_profile_match_signals`, and its additive weight. The ranker iterates
this list instead of naming fields inline, so adding / removing / reweighting a
signal is a one-line schema edit (weights still come from `ranking_config`, so
env overrides keep working).

This is the foundation the domain plugin (#5) builds on: a domain supplies its
own schema, and the engine consumes it generically. Currently only the ranking
consumer is wired; clarification, query-building, and the LLM planner prompt are
the next increments to migrate onto the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import ranking_config as rank_cfg


@dataclass(frozen=True)
class Clarification:
    question: str
    options: tuple = ()


@dataclass(frozen=True)
class FieldSpec:
    """A single profile field's declarative metadata.

    name          — schema field name (e.g. "organism")
    signal_key    — key produced by `_profile_match_signals` for this field
    rank_weight   — additive weight applied to that signal in the profile bonus
    composite     — True if the signal combines several fields (not one field)
    profile_field — the experiment_profile key this field reads (per-field only)
    query_priority— order/inclusion in profile_to_search_query (lower = earlier;
                    None = excluded from the structured search query)
    clarification — canonical question/options for this field (content only; the
                    intent-conditional ordering stays domain logic, see #5)
    """
    name: str
    signal_key: str
    rank_weight: float
    composite: bool = False
    profile_field: str = ""
    query_priority: Optional[int] = None
    clarification: Optional[Clarification] = None


# ---------------------------------------------------------------------------
# Per-field matcher registry (late-bound).
#
# The matcher callables (`_organism_match_score`, …) live in experiment_profile
# alongside the biology term lists they depend on. To keep the schema free of a
# circular import, experiment_profile registers them here at import time via
# `register_matcher`. `field_scores()` then computes each per-field signal by
# looking up the registered matcher — the field→matcher→profile_field wiring is
# declared by the schema, the matcher logic stays in the domain module.
# ---------------------------------------------------------------------------

MATCHERS: Dict[str, Callable[[Any, str], float]] = {}


def register_matcher(signal_key: str, fn: Callable[[Any, str], float]) -> None:
    MATCHERS[signal_key] = fn


# ---------------------------------------------------------------------------
# Biology ranking schema — single source of truth for the profile-bonus formula.
# Order matches the original inline sum (float addition is commutative, but we
# keep it stable for readability and exact reproduction).
# ---------------------------------------------------------------------------

RANKING_SCHEMA: List[FieldSpec] = [
    # Per-field matches (each reads one profile field via a registered matcher)
    FieldSpec("organism", "organism_match", rank_cfg.W_ORGANISM, profile_field="organism"),
    FieldSpec("method", "method_match", rank_cfg.W_METHOD, profile_field="experimental_method"),
    FieldSpec("readout", "readout_match", rank_cfg.W_READOUT, profile_field="readout_assay"),
    FieldSpec("expression", "expression_match", rank_cfg.W_EXPRESSION, profile_field="expression_type"),
    FieldSpec("tissue", "tissue_match", rank_cfg.W_TISSUE, profile_field="tissue_or_cell_type"),
    # Composite / cross-field signals (computed from several fields, not one)
    FieldSpec("combined", "combined_match", rank_cfg.W_COMBINED, composite=True),
    FieldSpec("coverage", "profile_coverage", rank_cfg.W_COVERAGE, composite=True),
    FieldSpec("required_concept", "required_concept_match", rank_cfg.W_REQUIRED_CONCEPT, composite=True),
    FieldSpec("title_context", "title_context_match", rank_cfg.W_TITLE_CONTEXT, composite=True),
    FieldSpec("completeness", "completeness", rank_cfg.W_COMPLETENESS, composite=True),
    FieldSpec("recency", "recency", rank_cfg.W_RECENCY, composite=True),
    FieldSpec("community", "community", rank_cfg.W_COMMUNITY, composite=True),
]

# The per-field matchers, in signal order (subset of RANKING_SCHEMA with a matcher).
FIELD_SIGNALS: List[FieldSpec] = [s for s in RANKING_SCHEMA if s.profile_field]


# Ordered profile fields that build the structured search query. Mirrors the
# previous hard-coded tuple in profile_to_search_query exactly (order preserved).
QUERY_FIELD_ORDER: List[str] = [
    "organism", "method", "experimental_method", "target", "modification_type",
    "delivery_method", "expression_type", "sample_type", "tissue_or_cell_type",
    "readout", "readout_assay", "condition", "gene_or_construct", "timeline",
    "difficulty", "protocol_difficulty",
]


# Context-independent per-field clarifications (declarative content). Fields whose
# clarification varies by intent — readout options, tissue phrasing, the
# organism-aware question — stay as conditional domain logic in next_clarification;
# they are not flat field data. (In #5 the whole clarification flow moves into the
# Domain plugin; this schema holds the parts that are genuinely field-level.)
CLARIFICATIONS: Dict[str, Clarification] = {
    "modification_type": Clarification(
        "What kind of gene modification do you mean?",
        ("CRISPR / genome editing", "overexpression", "knockdown / silencing",
         "mutation / mutagenesis", "stable transformation", "not sure"),
    ),
    "delivery_method": Clarification(
        "How should the modification be delivered or introduced?",
        ("stable transformation", "transient expression", "Agrobacterium-mediated delivery",
         "biolistic / gene gun", "not sure"),
    ),
}


def field_clarification(field: str) -> Optional[Dict[str, Any]]:
    """The declarative {field, question, options} for a context-independent field,
    or None if the field's clarification is intent-conditional (domain logic)."""
    c = CLARIFICATIONS.get(field)
    if not c:
        return None
    return {"field": field, "question": c.question, "options": list(c.options)}


# Full profile field set + default shape (scalar->None, list->[], dict->{}).
# Single source of truth for the profile's fields; drives the LLM planner's
# expected `experiment_profile` contract so extraction matches the schema.
# Order preserved to reproduce the planner prompt exactly.
PROFILE_FIELD_DEFAULTS: Dict[str, Any] = {
    "intent_family": None,
    "sub_intent": None,
    "organism": None,
    "sample_type": None,
    "tissue_or_cell_type": None,
    "target": None,
    "gene_or_construct": None,
    "modification_type": None,
    "method": None,
    "experimental_method": None,
    "delivery_method": None,
    "expression_type": None,
    "readout": None,
    "readout_assay": None,
    "condition": None,
    "timeline": None,
    "equipment": [],
    "required_equipment": [],
    "difficulty": None,
    "protocol_difficulty": None,
    "constraints": [],
    "intent_specific": {},
}


def llm_profile_template() -> Dict[str, Any]:
    """Fresh copy of the profile field template for the LLM planner's JSON schema."""
    import copy
    return copy.deepcopy(PROFILE_FIELD_DEFAULTS)


def ranking_bonus(signals: Dict[str, float]) -> float:
    """Weighted sum of the schema's signals — the profile-match bonus.

    Reproduces the previous inline formula exactly; driven by RANKING_SCHEMA so
    the ranking weights/signals live in one declarative place.
    """
    return sum(spec.rank_weight * signals.get(spec.signal_key, 0.0) for spec in RANKING_SCHEMA)


def field_scores(profile: Dict[str, Any], text: str) -> Dict[str, float]:
    """Per-field match scores, computed by each field's registered matcher.

    Returns {signal_key: score} for every per-field signal in the schema. The
    matcher for a field must be registered (experiment_profile does this at
    import); an unregistered field scores 0.0.
    """
    out: Dict[str, float] = {}
    for spec in FIELD_SIGNALS:
        matcher = MATCHERS.get(spec.signal_key)
        out[spec.signal_key] = matcher(profile.get(spec.profile_field), text) if matcher else 0.0
    return out
