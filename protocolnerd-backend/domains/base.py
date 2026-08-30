"""
Domain plugin interface.

A `Domain` packages everything the engine needs that is domain-specific:
intent detection, profile validation/normalization, clarification flow, search-
query building, per-source query shaping, result-match signals + ranking, and
the LLM planner. The orchestrator (`main.py`) and planner call the *active*
domain through this interface and never name a domain (biology, chemistry, …)
directly, so adding a domain is a new `Domain` subclass + registration.

Every method mirrors an existing engine call site 1:1, so a domain that simply
delegates to the current functions is behavior-preserving.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class Domain:
    name: str = "base"

    # One-line scope statement. The router shows this to the LLM classifier, so a
    # new domain announces its own coverage instead of the router hard-coding
    # anything about it.
    description: str = ""

    # Profile fields eligible for the non-answer-skip / clarification-misassignment
    # safety nets in main.py (_apply_nonanswer_skips, _fix_clarification_misassignment).
    # A domain that doesn't override this just gets no safety net, not a crash.
    skippable_fields: List[str] = []

    # Signature terms for the KEYWORD router -- the fallback used only when the LLM
    # classifier is unavailable or returns something unrecognized. Every registered
    # domain is scored by how many of its own terms appear in the request, so adding
    # a domain needs no router change (see registry._classify_domain_keywords).
    keywords: Tuple[str, ...] = ()

    # True for the domain the router falls back to when no domain scores. Exactly
    # one registered domain should set this; `DEFAULT_DOMAIN` env var wins over it.
    is_default: bool = False

    # The literature lane(s) searched alongside protocols.io, by Retriever
    # name. protocols.io itself is always on; the paper sources are per-domain
    # so each discipline pairs the protocol corpus with the literature source
    # that actually covers it (biology: PubMed; chemistry: Europe PMC), and a
    # source chosen for one domain cannot change another domain's result set.
    paper_sources: Tuple[str, ...] = ("pubmed",)

    # Back-compat alias: older code called these "extra" sources on top of an
    # always-on PubMed lane. Nothing declares it any more; kept as an empty
    # default so stale readers see "no extras" rather than crashing.
    extra_sources: List[str] = []

    def canonical_field(self, field: str) -> str:
        """Map a field alias to its canonical name (e.g. biology's "system" ->
        "tissue_or_cell_type"). Default: identity, i.e. no aliases."""
        return field

    # -- intent + profile ----------------------------------------------------
    def detect_intent(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError

    def build_profile(self, query: str, previous_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rule-based profile extraction (LLM-failure fallback)."""
        raise NotImplementedError

    def validate(self, source_query: str, intent: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    def normalize(self, source_query: str, intent: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    def surface_extras(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Post-clarification display touch-ups (e.g. surfacing growth_stage)."""
        return profile

    # -- clarification -------------------------------------------------------
    def next_clarification(self, profile: Dict[str, Any], intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def needs_clarification(self, profile: Dict[str, Any], field: str) -> bool:
        raise NotImplementedError

    # -- search queries ------------------------------------------------------
    def can_generate_search_queries(self, profile: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def candidate_queries(self, profile: Dict[str, Any], fallback_query: str = "", max_queries: int = 5) -> List[str]:
        raise NotImplementedError

    def build_search_query(self, profile: Dict[str, Any], fallback_query: str) -> str:
        raise NotImplementedError

    def source_query(self, source: str, profile: Dict[str, Any], fallback: str) -> str:
        """Per-source query shaping (e.g. the PubMed core query). Default: the
        structured search query."""
        return self.build_search_query(profile, fallback)

    # -- ranking -------------------------------------------------------------
    def rank(self, profile: Dict[str, Any], results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    # -- LLM planner ---------------------------------------------------------
    def analyze_request(
        self,
        *,
        user_query: str,
        conversation_query: str = "",
        previous_profile: Optional[Dict[str, Any]] = None,
        pending_field: Optional[str] = None,
        max_queries: int = 5,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    # -- domain-specific prompts ---------------------------------------------
    # Each returns None to mean "use the shared module's own default", so a
    # domain only overrides the stages it actually cares about. The shared
    # modules (reranker.py, pubmed_client.py, claude_client.py) keep their
    # existing constants as that fallback.

    def rerank_system_prompt(self) -> Optional[str]:
        """System prompt for the LLM re-ranker (reranker.py)."""
        return None

    def pubmed_query_rules(self) -> Optional[str]:
        """Query-shaping rules for the PubMed query writer (pubmed_client.py)."""
        return None

    def candidate_query_prompt(self, n: int) -> Optional[str]:
        """System prompt for natural-language candidate query generation."""
        return None

    def clarification_explanation_prompt(self, field: str) -> Optional[str]:
        """System prompt for explaining why a clarification question is asked."""
        return None

    def search_query_fields(self, profile: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Labeled (label, value) profile pairs that feed the candidate-query
        prompt's structured block. Empty list -> the caller keeps its default."""
        return []

    def normalize_llm_intent(self, plan: Dict[str, Any], fallback: Dict[str, Any], source_query: str = "") -> Dict[str, Any]:
        """Turn analyze_request()'s raw `plan` into the engine's intent shape.

        Default: trust whatever `plan.get("intent")` already is if present,
        otherwise fall back to detect_intent(source_query). A domain whose
        analyze_request() already returns a well-formed intent dict (chemistry
        does) doesn't need to override this.
        """
        raw = plan.get("intent") if isinstance(plan, dict) else None
        if isinstance(raw, dict) and raw.get("intent_family"):
            return raw
        return self.detect_intent(source_query) if source_query else fallback
