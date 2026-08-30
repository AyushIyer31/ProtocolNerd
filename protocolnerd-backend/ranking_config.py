"""
Centralized, tunable ranking configuration.

All ranking weights and thresholds live here as named constants, each read from
an environment variable with the current hard-coded value as its default. This
keeps every knob in one place and lets a deployment override any of them (e.g.
on Cloud Run) without a code change or redeploy of source.

Naming: RANK_W_* are additive/multiplier weights; RANK_T_* are thresholds.
Defaults exactly reproduce the pre-config behavior.
"""

from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip('"').strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Cross-source relevance (blend_score = title-weighted cosine)
# ---------------------------------------------------------------------------
BLEND_TITLE_WEIGHT = _f("RANK_W_BLEND_TITLE", 0.67)
BLEND_BODY_WEIGHT = _f("RANK_W_BLEND_BODY", 0.33)

# How strongly the comparable cross-source text-relevance (blend_score) counts in
# the final profile re-ranking, relative to the field-match bonuses below.
TEXT_RELEVANCE_WEIGHT = _f("RANK_W_TEXT_RELEVANCE", 18.0)


# ---------------------------------------------------------------------------
# Profile field-match bonus weights (apply_profile_ranking)
# ---------------------------------------------------------------------------
W_ORGANISM = _f("RANK_W_ORGANISM", 2.4)
W_METHOD = _f("RANK_W_METHOD", 1.8)
W_READOUT = _f("RANK_W_READOUT", 1.2)
W_EXPRESSION = _f("RANK_W_EXPRESSION", 2.2)
W_TISSUE = _f("RANK_W_TISSUE", 1.8)
W_COMBINED = _f("RANK_W_COMBINED", 3.2)
W_COVERAGE = _f("RANK_W_COVERAGE", 2.0)
W_REQUIRED_CONCEPT = _f("RANK_W_REQUIRED_CONCEPT", 2.8)
W_TITLE_CONTEXT = _f("RANK_W_TITLE_CONTEXT", 1.0)
W_COMPLETENESS = _f("RANK_W_COMPLETENESS", 0.5)
W_RECENCY = _f("RANK_W_RECENCY", 0.3)
W_COMMUNITY = _f("RANK_W_COMMUNITY", 0.3)


# ---------------------------------------------------------------------------
# Thresholds & penalty multipliers
# ---------------------------------------------------------------------------
# A signal at or above this counts as a "strong" match (used by combined-match,
# title-context, and the organism gate).
T_STRONG_MATCH = _f("RANK_T_STRONG_MATCH", 0.75)

# Results whose false-positive penalty is at or below this are treated as severely
# off-target and pushed to the bottom (only used to backfill if too few remain).
T_SEVERE_OFF_TARGET = _f("RANK_T_SEVERE_OFF_TARGET", 0.4)

# When the profile defines >=2 required concept groups and coverage is below this,
# multiply the penalty by P_REQUIRED_CONCEPT.
T_REQUIRED_CONCEPT_MIN = _f("RANK_T_REQUIRED_CONCEPT_MIN", 0.5)
P_REQUIRED_CONCEPT = _f("RANK_P_REQUIRED_CONCEPT", 0.55)

# Strict overexpression/readout profiles: graduated penalty by required-concept
# coverage.
P_STRICT_OX_LOW = _f("RANK_P_STRICT_OX_LOW", 0.25)   # coverage < 0.6
P_STRICT_OX_MID = _f("RANK_P_STRICT_OX_MID", 0.55)   # coverage < 0.8
