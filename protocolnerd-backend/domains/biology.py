"""
Biology domain — the original protocol-search domain.

This adapter delegates to the existing biology functions in `experiment_profile`,
`claude_client`, and `retrievers`, so routing the engine through it is exactly
behavior-preserving. (Phase 2 can relocate the biology internals into this
package; the interface stays identical either way.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import experiment_profile as ep
import claude_client
from biology_intents import controlled_intent_payload, normalize_sub_intent
from retrievers import pubmed_core_query
from .base import Domain
from . import biology_prompts as bp
# NOTE: current_domain is imported lazily (inside _controlled_intent_name) rather
# than at module scope -- registry.py imports THIS module to register
# BiologyDomain, so a top-level "from .registry import current_domain" here
# would be a circular import.

# Relocated from main.py verbatim (was mis-placed there: main.py is meant to be
# domain-agnostic, but this logic only makes sense against biology's own
# controlled sub-intent vocabulary). A non-biology domain's plan used to be run
# through this too, which is why chemistry's sub_intents ("organic_synthesis",
# etc.) collapsed to "unknown" -- they don't match anything below.

_INTENT_ALIASES = {
    "gene modification": "gene_modification",
    "gene_modification": "gene_modification",
    "gene editing": "gene_modification",
    "gene_editing": "gene_modification",
    "genome editing": "gene_modification",
    "genome_editing": "gene_modification",
    "crispr": "gene_modification",
    "gene overexpression": "gene_overexpression",
    "gene_overexpression": "gene_overexpression",
    "overexpression": "gene_overexpression",
    "gene knockdown": "gene_knockdown",
    "gene_knockdown": "gene_knockdown",
    "knockdown": "gene_knockdown",
    "protein purification": "protein_purification",
    "protein_purification": "protein_purification",
    "pcr": "pcr_qpcr",
    "qpcr": "pcr_qpcr",
    "pcr_qpcr": "pcr_qpcr",
    "transformation": "transformation",
    "microscopy": "microscopy",
    "sequencing prep": "sequencing_prep",
    "sequencing_prep": "sequencing_prep",
    "chitchat": "chitchat",
    "unknown": "unknown",
}


def _controlled_intent_name(value: Any, source_query: str = "") -> Optional[str]:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    if not text:
        return None
    normalized = normalize_sub_intent(text)
    if normalized != "unknown":
        return normalized
    compact = text.replace(" ", "_")
    if compact in ep.INTENT_LABELS or compact in {"chitchat"}:
        return compact
    if text in _INTENT_ALIASES:
        return _INTENT_ALIASES[text]
    for phrase, intent_name in _INTENT_ALIASES.items():
        if phrase not in {"unknown", "chitchat"} and phrase in text:
            return intent_name
    from .registry import current_domain  # deferred: see note above imports
    detected = current_domain().detect_intent(" ".join([source_query, text]).strip())
    if detected.get("intent") != "unknown":
        return detected.get("intent")
    return None


def _intent_response(intent_name: str, confidence: Any = 0.7, alternatives: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if intent_name not in ep.INTENT_LABELS and intent_name != "chitchat":
        intent_name = "unknown"
    return controlled_intent_payload(intent_name, confidence=confidence, alternatives=alternatives)


def _normalize_intent(plan: Dict[str, Any], fallback: Dict[str, Any], source_query: str = "") -> Dict[str, Any]:
    raw = plan.get("intent") if isinstance(plan, dict) else {}
    raw_sub_intent = plan.get("sub_intent") if isinstance(plan, dict) else None
    raw_text = ""
    if raw_sub_intent:
        name = _controlled_intent_name(raw_sub_intent, source_query)
        if (
            name == "gene_overexpression"
            and fallback.get("intent") == "gene_modification"
            and "overexpress" not in source_query.lower()
            and "overexpression" not in source_query.lower()
        ):
            return fallback
        if (not name or name == "unknown") and fallback.get("intent") != "unknown":
            return fallback
        return controlled_intent_payload(
            name or "unknown",
            intent_family=plan.get("intent_family"),
            confidence=plan.get("confidence", 0.7),
        )
    if isinstance(raw, str):
        raw_text = raw
        name = _controlled_intent_name(raw, source_query)
        if (
            name == "gene_overexpression"
            and fallback.get("intent") == "gene_modification"
            and "overexpress" not in source_query.lower()
            and "overexpression" not in source_query.lower()
        ):
            return fallback
        if (not name or name == "unknown") and fallback.get("intent") != "unknown":
            return fallback
        return _intent_response(name or "unknown")
    if isinstance(raw, dict):
        raw_text = " ".join(str(raw.get(key) or "") for key in ("sub_intent", "name", "label"))
        name = _controlled_intent_name(raw_text, source_query)
        if (
            name == "gene_overexpression"
            and fallback.get("intent") == "gene_modification"
            and "overexpress" not in source_query.lower()
            and "overexpression" not in source_query.lower()
        ):
            return fallback
        if (not name or name == "unknown") and fallback.get("intent") != "unknown":
            return fallback
        return controlled_intent_payload(
            name or "unknown",
            intent_family=plan.get("intent_family") or raw.get("intent_family"),
            confidence=raw.get("confidence", 0.7),
            alternatives=raw.get("alternatives", []),
        )
    return fallback


# Profile fields eligible for the non-answer-skip / clarification-misassignment
# safety nets (main.py: _apply_nonanswer_skips, _fix_clarification_misassignment).
# Relocated from main.py's _SKIPPABLE_FIELDS verbatim.
_SKIPPABLE_FIELDS = (
    "organism", "modification_type", "sub_intent", "experimental_method",
    "method", "target", "gene_or_construct", "delivery_method", "expression_type",
    "readout", "readout_assay", "condition", "tissue_or_cell_type", "sample_type",
    "timeline", "difficulty",
)


class BiologyDomain(Domain):
    name = "biology"
    description = (
        "molecular and cell biology: genes, proteins, cells, tissues and whole organisms; "
        "DNA/RNA extraction, PCR/qPCR, cloning, CRISPR and genome editing, transformation, "
        "western blots, microscopy and imaging, flow cytometry, cell culture, sequencing "
        "library prep, microbiology, and plant or animal phenotyping"
    )

    is_default = True

    # Signature terms for the keyword fallback router (see registry).
    keywords = (
        "gene", "crispr", "protein", "cell", "organism", "rna", "dna", "pcr", "plant",
        "mouse", "tissue", "expression", "transform", "genome", "antibody", "bacteria",
        "enzyme", "clone", "arabidopsis", "rice", "drought", "phenotype",
    )

    skippable_fields = list(_SKIPPABLE_FIELDS)

    def canonical_field(self, field: str) -> str:
        return ep._canonical_profile_field(field)

    # -- domain-specific prompts (see domains/biology_prompts.py) ------------
    def rerank_system_prompt(self) -> Optional[str]:
        return bp.RERANK_SYSTEM

    def pubmed_query_rules(self) -> Optional[str]:
        return bp.PUBMED_QUERY_RULES

    def candidate_query_prompt(self, n: int) -> Optional[str]:
        return bp.candidate_query_system(n)

    def clarification_explanation_prompt(self, field: str) -> Optional[str]:
        return bp.CLARIFICATION_EXPLANATION_SYSTEM

    def search_query_fields(self, profile: Dict[str, Any]) -> List[Tuple[str, Any]]:
        return bp.search_query_fields(profile)

    def normalize_llm_intent(self, plan: Dict[str, Any], fallback: Dict[str, Any], source_query: str = "") -> Dict[str, Any]:
        return _normalize_intent(plan, fallback, source_query)

    # -- intent + profile ----------------------------------------------------
    def detect_intent(self, query: str) -> Dict[str, Any]:
        return ep.detect_experiment_intent(query)

    def build_profile(self, query: str, previous_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return ep.build_experiment_profile(query, previous_profile=previous_profile)

    def validate(self, source_query, intent, profile) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return ep.validate_biology_profile(source_query, intent, profile)

    def normalize(self, source_query, intent, profile) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return ep.normalize_experiment_goal(source_query, intent, profile)

    def surface_extras(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return ep.surface_growth_stage(profile)

    # -- clarification -------------------------------------------------------
    def next_clarification(self, profile, intent) -> Optional[Dict[str, Any]]:
        return ep.next_clarification(profile, intent)

    def needs_clarification(self, profile, field) -> bool:
        return ep.needs_clarification(profile, field)

    # -- search queries ------------------------------------------------------
    def can_generate_search_queries(self, profile) -> bool:
        return ep.can_generate_search_queries(profile)

    def candidate_queries(self, profile, fallback_query="", max_queries=5) -> List[str]:
        return ep.generate_candidate_search_queries(profile, fallback_query=fallback_query, max_queries=max_queries)

    def build_search_query(self, profile, fallback_query) -> str:
        return ep.profile_to_search_query(profile, fallback_query)

    def source_query(self, source, profile, fallback) -> str:
        if source == "pubmed":
            return pubmed_core_query(profile, fallback)
        return ep.profile_to_search_query(profile, fallback)

    # -- ranking -------------------------------------------------------------
    def rank(self, profile, results, top_k) -> List[Dict[str, Any]]:
        return ep.apply_profile_ranking(profile, results, top_k=top_k)

    # -- LLM planner ---------------------------------------------------------
    def analyze_request(self, **kwargs) -> Dict[str, Any]:
        return claude_client.analyze_experiment_request(**kwargs)
