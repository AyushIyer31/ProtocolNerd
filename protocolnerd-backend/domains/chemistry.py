"""
Chemistry domain — a second protocol-search domain, added purely as a Domain
plugin (no engine edits).

It defines its own profile fields rather than overloading biology's names
(a "target" in biology is a gene/protein; in chemistry it's a compound, a
different enough concept that reusing the name was more confusing than it
was worth).

    -- what is being made / from what --
    compound           -> the product / compound / material of interest
    starting_material  -> substrate(s) the reaction starts from

    -- how --
    reaction_type      -> the reaction or technique (e.g. Suzuki coupling, TLC)
    catalyst           -> catalyst or key reagent system (e.g. Pd(PPh3)4)
    solvent            -> reaction solvent (e.g. THF, DMF, water)
    temperature        -> thermal conditions (e.g. reflux, 0 degC, room temperature)
    timeline           -> reaction time (e.g. overnight, 2 h, until TLC shows consumption)

    -- what comes out / how it is checked --
    purification       -> how the product is isolated (column, recrystallization, distillation)
    characterization   -> how the result is verified (NMR, MS, HPLC, IR, yield)

    -- practical --
    scale              -> scale / phase (e.g. milligram, solution)
    equipment          -> required apparatus (Schlenk line, glovebox, rotovap)
    constraints        -> anhydrous / air-free, inert atmosphere, safety limits
    difficulty         -> rough difficulty, if stated

An earlier version squeezed solvent + temperature + catalyst into one
`reaction_conditions` string. That under-modelled the domain (the planner kept
trying to return it as a nested object, which is the model asking for more
fields) and made those facets individually unrankable, so they are now separate.

intent_family, sub_intent and intent_specific are shared with biology (every
domain sets them the same way). Only the domain-specific behavior differs:
intent taxonomy, clarification questions, query shaping, the LLM planner
prompt, and result matching. Retrieval (protocols.io + PubMed) is shared —
both hold chemistry methods — so no new corpus is required.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import ranking_config as rank_cfg
from .base import Domain
from . import chemistry_prompts as cp
from .chemistry_prompts import planner_system_prompt

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

SUB_INTENTS = {
    "organic_synthesis": "Organic Synthesis",
    "purification": "Purification / Separation",
    "characterization": "Characterization / Analysis",
    "extraction": "Extraction / Isolation",
    "electrochemistry": "Electrochemistry",
    "materials_synthesis": "Materials / Nanomaterial Synthesis",
    "general_chemistry_search": "General Chemistry Search",
}

_INTENT_KEYWORDS = [
    ("organic_synthesis", ("synthesis", "synthesize", "coupling", "reaction", "reflux", "grignard", "suzuki", "amide bond", "esterification")),
    ("purification", ("purify", "purification", "chromatography", "recrystalli", "distillation", "column")),
    ("characterization", ("nmr", "mass spec", "characteriz", "hplc", "gc-ms", "ir spectroscopy", "xrd", "titration")),
    ("extraction", ("extraction", "isolate", "liquid-liquid", "soxhlet", "solvent extraction")),
    ("electrochemistry", ("electroch", "cyclic voltammetry", "electrolysis", "electrode")),
    ("materials_synthesis", ("nanoparticle", "sol-gel", "mof", "polymer synthesis", "thin film", "catalyst preparation")),
]

# Every substantive profile field, in the order they are shown and searched.
PROFILE_FIELDS = [
    "compound", "starting_material", "reaction_type", "catalyst", "solvent",
    "temperature", "timeline", "purification", "characterization", "scale",
    "equipment", "constraints", "difficulty",
]

# Fields that carry real retrieval signal, in priority order. Practical fields
# (scale/equipment/constraints/difficulty) describe how to run a protocol rather
# than what it is about, so they are profiled and displayed but kept out of the
# generated query text, where they would only add noise.
QUERY_FIELD_ORDER = [
    "compound", "starting_material", "reaction_type", "catalyst", "solvent",
    "temperature", "purification", "characterization",
]

# Display labels for the UI's "Still missing" line.
_HUMAN_FIELD = {
    "compound": "product / compound",
    "starting_material": "starting material",
    "reaction_type": "reaction / technique",
    "catalyst": "catalyst / reagent",
    "solvent": "solvent",
    "temperature": "temperature",
    "timeline": "reaction time",
    "purification": "purification method",
    "characterization": "characterization method",
    "scale": "scale",
    "equipment": "equipment",
    "constraints": "constraints",
    "difficulty": "difficulty",
}

# Chemistry ranking signals + weights (reusing the shared weight config).
# What the protocol MAKES and HOW it makes it anchor the ranking; the specific
# catalyst/solvent/temperature refine it; characterization and purification are
# supporting signals.
_RANK_SIGNALS = [
    ("compound_match", "compound", rank_cfg.W_ORGANISM),          # primary anchor
    ("reaction_match", "reaction_type", rank_cfg.W_METHOD),
    ("starting_material_match", "starting_material", rank_cfg.W_TISSUE),
    ("catalyst_match", "catalyst", rank_cfg.W_EXPRESSION),
    ("purification_match", "purification", rank_cfg.W_READOUT),
    ("characterization_match", "characterization", rank_cfg.W_READOUT),
    ("solvent_match", "solvent", rank_cfg.W_TITLE_CONTEXT),
    ("temperature_match", "temperature", rank_cfg.W_TITLE_CONTEXT),
]

_SKIP = {"", "not specified", "none", "unknown", "not sure", "null"}


def _empty(v: Any) -> bool:
    return v is None or str(v).strip().lower() in _SKIP


def _stringify(v: Any) -> str:
    """Flattens a field value into plain text. The planner prompt asks for short
    strings, but an LLM occasionally nests a field anyway (e.g. reaction_conditions
    as {solvent, temperature, catalyst}); without this, str(dict) leaks a raw
    Python-repr fragment straight into the search query text."""
    if isinstance(v, dict):
        return ", ".join(str(x) for x in v.values() if x and not _empty(x))
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v if x and not _empty(x))
    return str(v)


def _detect_sub_intent(text: str) -> str:
    low = text.lower()
    for sub, kws in _INTENT_KEYWORDS:
        if any(k in low for k in kws):
            return sub
    return "general_chemistry_search"


def _term_match(value: Any, text: str) -> float:
    """1.0 if any token of the (possibly multi-value) field appears in text."""
    if _empty(value):
        return 0.0
    low = text.lower()
    parts = re.split(r"\s+or\s+|\s+and\s+|,|/", _stringify(value).lower())
    for p in (p.strip() for p in parts if p.strip()):
        # match on the most specific word (>=4 chars) or the whole phrase
        if p in low or any(w in low for w in p.split() if len(w) >= 4):
            return 1.0
    return 0.0


class ChemistryDomain(Domain):
    name = "chemistry"
    description = (
        "chemistry: synthesizing, purifying or characterizing compounds and materials; "
        "reactions, reagents, solvents and catalysts; chromatography, distillation and "
        "recrystallization; NMR, mass spectrometry, HPLC, IR and titration analysis; "
        "electrochemistry; polymer and nanoparticle synthesis"
    )

    # Chemistry's literature lane is Europe PMC instead of PubMed. Protocols.io
    # holds almost no synthetic chemistry (21 results across ten core
    # techniques, zero for Suzuki coupling / Grignard / RAFT / Schlenk line),
    # while Europe PMC's METHODS: field search reaches procedures published
    # inside papers, which is where chemistry methods actually live. Biology
    # keeps the base default ("pubmed",), so its result set is unchanged.
    paper_sources = ("europepmc",)

    # Signature terms for the keyword fallback router (see registry).
    keywords = (
        "synthesis", "synthesize", "nmr", "chromatography", "reagent", "solvent",
        "catalyst", "reaction", "distillation", "recrystalli", "suzuki", "grignard",
        "mass spec", "hplc", "reflux", "titration", "electrochem", "nanoparticle",
        "polymer", "stoichiom", "organic compound", "esterification", "amide bond",
    )

    skippable_fields = PROFILE_FIELDS + ["sub_intent"]
    # canonical_field: no aliases needed yet, inherit Domain's identity default.

    # -- domain-specific prompts (see domains/chemistry_prompts.py) ----------
    def rerank_system_prompt(self) -> Optional[str]:
        return cp.RERANK_SYSTEM

    def pubmed_query_rules(self) -> Optional[str]:
        return cp.PUBMED_QUERY_RULES

    def candidate_query_prompt(self, n: int) -> Optional[str]:
        return cp.candidate_query_system(n)

    def clarification_explanation_prompt(self, field: str) -> Optional[str]:
        return cp.CLARIFICATION_EXPLANATION_SYSTEM

    def search_query_fields(self, profile: Dict[str, Any]) -> List[Tuple[str, Any]]:
        p = profile or {}
        return [
            (_HUMAN_FIELD[f], _stringify(p.get(f)) if not _empty(p.get(f)) else None)
            for f in PROFILE_FIELDS
        ]

    def normalize_llm_intent(self, plan: Dict[str, Any], fallback: Dict[str, Any], source_query: str = "") -> Dict[str, Any]:
        # analyze_request() already builds a well-formed intent dict
        # (data.setdefault("intent", {...})), so trust it directly rather than
        # running it through biology's controlled-vocabulary normalizer.
        raw = plan.get("intent") if isinstance(plan, dict) else None
        if isinstance(raw, dict) and raw.get("name"):
            return {
                "intent": raw.get("name"), "label": raw.get("label", "Chemistry"),
                "intent_family": "chemistry", "intent_family_label": "Chemistry",
                "sub_intent": raw.get("name"), "sub_intent_label": raw.get("label", "Chemistry"),
                "confidence": raw.get("confidence", 0.7), "alternatives": raw.get("alternatives", []),
            }
        return self.detect_intent(source_query) if source_query else fallback

    # -- intent + profile ----------------------------------------------------
    def detect_intent(self, query: str) -> Dict[str, Any]:
        sub = _detect_sub_intent(query)
        label = SUB_INTENTS[sub]
        return {
            "intent": sub, "name": sub, "label": label,
            "intent_family": "chemistry", "intent_family_label": "Chemistry",
            "sub_intent": sub, "sub_intent_label": label,
            "confidence": 0.6, "alternatives": [],
        }

    def build_profile(self, query: str, previous_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rule-based fallback profile, used when the LLM planner did not run.

        Deliberately does NOT seed reaction_type from the sub-intent label.
        SUB_INTENTS values are display strings ("Organic Synthesis"), and
        reaction_type is a SEARCHED field: seeding it produced queries like
        METHODS:"Organic Synthesis" instead of METHODS:"Suzuki coupling", which
        is noise in both the protocol search and the literature search. Leaving
        it empty lets the raw request be used instead, which is far better.
        """
        prof = dict(previous_profile or {})
        sub = _detect_sub_intent(query)
        prof.setdefault("intent_family", "chemistry")
        prof["sub_intent"] = prof.get("sub_intent") or sub
        return prof

    def _clamp_sub_intent(self, value: Any, source_query: str = "") -> str:
        """Force sub_intent into the controlled vocabulary.

        The planner occasionally echoes the request back in this field, which
        then leaks into the UI's Sub-intent tile and breaks the
        _REQUIRED_BY_SUB_INTENT lookup. Biology clamps via normalize_sub_intent;
        this is chemistry's equivalent. Falls back to keyword detection on the
        original request rather than a bare default, so a mislabelled request
        still lands on a sensible intent.
        """
        raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if raw in SUB_INTENTS:
            return raw
        for name, label in SUB_INTENTS.items():
            if raw == label.lower().replace(" / ", "_").replace(" ", "_"):
                return name
        return _detect_sub_intent(source_query or str(value or ""))

    def validate(self, source_query, intent, profile) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        profile = dict(profile or {})
        intent = dict(intent or {})
        intent.setdefault("intent_family", "chemistry")
        sub = self._clamp_sub_intent(
            intent.get("sub_intent") or intent.get("intent") or profile.get("sub_intent"),
            source_query,
        )
        intent["intent"] = sub
        intent["sub_intent"] = sub
        intent["label"] = SUB_INTENTS.get(sub, "Chemistry")
        intent["sub_intent_label"] = intent["label"]
        profile.setdefault("intent_family", "chemistry")
        profile["sub_intent"] = sub
        return intent, profile

    def normalize(self, source_query, intent, profile) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return intent, dict(profile or {})

    # -- clarification -------------------------------------------------------
    # One question per field, asked only when that field is BOTH empty and
    # required for the current sub-intent (see _REQUIRED_BY_SUB_INTENT).
    _QUESTIONS = {
        "compound": {
            "field": "compound",
            "question": "What compound or material are you working with?",
            "options": [],
        },
        "reaction_type": {
            "field": "reaction_type",
            "question": "What reaction or technique do you want to run?",
            "options": ["organic synthesis", "purification / chromatography", "extraction",
                        "characterization (NMR / MS / HPLC)", "not sure"],
        },
        "starting_material": {
            "field": "starting_material",
            "question": "What are you starting from?",
            "options": [],
        },
        "catalyst": {
            "field": "catalyst",
            "question": "What catalyst or key reagent are you using?",
            "options": ["palladium catalyst", "acid catalyst", "base", "enzyme", "not sure"],
        },
        "solvent": {
            "field": "solvent",
            "question": "What solvent are you running this in?",
            "options": ["THF", "DMF", "dichloromethane", "toluene", "water", "not sure"],
        },
        "purification": {
            "field": "purification",
            "question": "How do you plan to purify or isolate the product?",
            "options": ["column chromatography", "recrystallization", "distillation",
                        "extraction", "not sure"],
        },
        "characterization": {
            "field": "characterization",
            "question": "How will you characterize or measure the result?",
            "options": ["NMR", "mass spectrometry", "HPLC / GC", "yield / purity", "not sure"],
        },
    }

    # Which fields materially change WHICH protocol fits, per sub-intent, in the
    # order they are worth asking. Mirrors biology's per-intent required fields:
    # a synthesis request should be asked what it starts from and what catalyses
    # it, while a characterization request should not be.
    _REQUIRED_BY_SUB_INTENT = {
        "organic_synthesis":        ["compound", "reaction_type", "starting_material", "catalyst"],
        "purification":             ["compound", "purification"],
        "characterization":         ["compound", "characterization"],
        "extraction":               ["compound", "reaction_type"],
        "electrochemistry":         ["compound", "reaction_type", "solvent"],
        "materials_synthesis":      ["compound", "reaction_type", "starting_material"],
        "general_chemistry_search": ["compound", "reaction_type"],
    }
    _DEFAULT_REQUIRED = ["compound", "reaction_type", "characterization"]

    def needs_clarification(self, profile, field) -> bool:
        return _empty((profile or {}).get(field))

    def required_fields(self, profile, intent) -> List[str]:
        sub = (profile or {}).get("sub_intent") or (intent or {}).get("sub_intent")
        return self._REQUIRED_BY_SUB_INTENT.get(sub, self._DEFAULT_REQUIRED)

    def next_clarification(self, profile, intent) -> Optional[Dict[str, Any]]:
        profile = profile or {}
        skipped = set(profile.get("_skipped_fields") or [])
        pending = [(f, self._QUESTIONS[f])
                   for f in self.required_fields(profile, intent) if f in self._QUESTIONS]
        for field, clar in pending:
            if field in skipped:
                continue
            if self.needs_clarification(profile, field):
                return dict(clar)
        return None

    # -- search queries ------------------------------------------------------
    def can_generate_search_queries(self, profile) -> bool:
        profile = profile or {}
        return any(not _empty(profile.get(f)) for f in QUERY_FIELD_ORDER)

    def build_search_query(self, profile, fallback_query) -> str:
        profile = profile or {}
        terms: List[str] = []
        for f in QUERY_FIELD_ORDER:
            v = profile.get(f)
            if not _empty(v):
                terms.append(_stringify(v))
        # de-dup tokens preserving order
        seen, out = set(), []
        for t in terms:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                out.append(t)
        q = " ".join(out).strip()
        return q if len(q.split()) >= 2 else fallback_query

    def candidate_queries(self, profile, fallback_query="", max_queries=5) -> List[str]:
        """Rule-based angles, used only when the LLM generator is unavailable.
        Each angle emphasises a different facet so the set collectively covers
        the profile, mirroring what the LLM prompt asks for."""
        base = self.build_search_query(profile, fallback_query)
        profile = profile or {}

        def g(field: str) -> str:
            v = profile.get(field)
            return _stringify(v) if not _empty(v) else ""

        compound, start = g("compound"), g("starting_material")
        reaction, catalyst = g("reaction_type"), g("catalyst")
        solvent, temperature = g("solvent"), g("temperature")
        purification, characterization = g("purification"), g("characterization")

        cands = [
            base,
            # what is made, by what reaction
            " ".join(x for x in [compound, reaction, "protocol"] if x).strip(),
            # the reaction under its specific catalyst/solvent/temperature
            " ".join(x for x in [reaction, catalyst, solvent, temperature] if x).strip(),
            # substrate -> product framing
            " ".join(x for x in [start, "to", compound, reaction] if x).strip() if start else "",
            # downstream workup and verification
            " ".join(x for x in [compound, purification, characterization] if x).strip(),
            fallback_query,
        ]
        seen, out = set(), []
        for c in cands:
            c = " ".join(str(c).split())
            if len(c) > 2 and c.lower() not in seen:
                seen.add(c.lower())
                out.append(c)
        return out[:max_queries]

    def source_query(self, source, profile, fallback) -> str:
        # PubMed ANDs terms; keep the query compact (compound + technique).
        profile = profile or {}
        if source == "pubmed":
            parts = [profile.get("compound"), profile.get("reaction_type")]
            q = " ".join(_stringify(p) for p in parts if not _empty(p)).strip()
            return q if len(q.split()) >= 2 else fallback
        return self.build_search_query(profile, fallback)

    # -- ranking -------------------------------------------------------------
    def rank(self, profile, results, top_k) -> List[Dict[str, Any]]:
        profile = profile or {}
        ranked = []
        for r in results:
            text = f"{r.get('title','')} {r.get('description') or r.get('abstract') or ''}".lower()
            signals = {sig: _term_match(profile.get(field), text) for sig, field, _ in _RANK_SIGNALS}
            base = float(r.get("blend_score") or r.get("score") or 0.0)
            bonus = sum(w * signals[sig] for sig, _, w in _RANK_SIGNALS)
            annotated = dict(r)
            annotated["profile_score"] = round(rank_cfg.TEXT_RELEVANCE_WEIGHT * base + bonus, 3)
            annotated["profile_signals"] = signals
            matched = [field for sig, field, _ in _RANK_SIGNALS if signals[sig] >= 0.75 and not _empty(profile.get(field))]
            missing = [field for sig, field, _ in _RANK_SIGNALS if signals[sig] < 0.75 and not _empty(profile.get(field))]
            # Contract note: why_it_matches is a STRING, but may_not_fit /
            # assumptions / missing_information are LISTS -- the frontend calls
            # .join() on those three (see renderMatchNotes in chat.html), so a
            # string here throws at render time.
            annotated["why_it_matches"] = "; ".join(f"matches {f}: {_stringify(profile.get(f))}" for f in matched) or "keyword match"
            annotated["may_not_fit"] = [
                f"May not cover {_HUMAN_FIELD.get(f, f)}: {_stringify(profile.get(f))}." for f in missing
            ]
            annotated["assumptions"] = []
            # Profile fields the user never supplied -- surfaced as "Still missing"
            # in the UI, mirroring biology's apply_profile_ranking().
            annotated["missing_information"] = [
                _HUMAN_FIELD.get(f, f.replace("_", " "))
                for f in PROFILE_FIELDS if _empty(profile.get(f))
            ][:4]
            ranked.append(annotated)
        ranked.sort(key=lambda x: x.get("profile_score", 0.0), reverse=True)
        return ranked[:top_k]

    # -- LLM planner ---------------------------------------------------------
    def analyze_request(self, *, user_query, conversation_query="", previous_profile=None, pending_field=None, max_queries=5) -> Dict[str, Any]:
        import claude_client
        if not claude_client.is_available():
            return {}
        system = planner_system_prompt(SUB_INTENTS)
        user = json.dumps({
            "current_user_message": user_query,
            "conversation_query": conversation_query,
            "previous_profile": previous_profile or {},
            "pending_field": pending_field,
        })
        raw = claude_client._call(system=system, user=user, response_format={"type": "json_object"}, temperature=0.2)
        data = claude_client._safe_json_object(raw)
        if not data:
            return {}
        prof = data.get("experiment_profile") or {}
        prof.setdefault("intent_family", "chemistry")
        sub = data.get("sub_intent") or _detect_sub_intent(user_query)
        prof["sub_intent"] = sub
        data["experiment_profile"] = prof
        data["intent_family"] = "chemistry"
        data["sub_intent"] = sub
        data.setdefault("intent", {"name": sub, "label": SUB_INTENTS.get(sub, "Chemistry"), "confidence": 0.7})
        return data
