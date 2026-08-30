"""
LLM client for the protocol search chatbot.

The module name is kept for compatibility with existing imports. Calls are
dispatched through the pluggable provider layer (`llm_providers`), which resolves
the active provider from LLM_PROVIDER. The canonical default is Claude:

LLM_PROVIDER=claude
CLAUDE_MODEL=claude-sonnet-4-6      # general calls; the re-ranker uses Haiku

Set LLM_PROVIDER=openai / gemini / ollama to switch providers.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from biology_intents import INTENT_FAMILIES, SUB_INTENTS
import profile_schema
from ollama_executions import (
    _get_ollama_model,
    _get_claude_model,
    _get_openai_model,
    _get_gemini_model,
    _retryable_ollama_call,
    active_provider,
    claude_available,
    gemini_available,
)

# Load .env first (authoritative for keys), then variables.env as a fallback for
# any settings not in .env (protocols.io, NCBI, etc.). override=False => the
# first-loaded value wins, so .env's keys take precedence.
load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(Path(__file__).parent / "variables.env", override=False)

log = logging.getLogger(__name__)

# Global provider override for the current request (set by main.py chat endpoint)
_current_provider: Optional[str] = None

def set_provider(provider: Optional[str]) -> None:
    """Set the LLM provider for this request. Call before LLM operations."""
    global _current_provider
    _current_provider = provider

def get_provider() -> Optional[str]:
    """Get the current provider override."""
    return _current_provider


def _native_ollama_base_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip('"').strip()
    if not base_url:
        base_url = "http://localhost:11434"
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    return base_url


def active_model() -> str:
    """Model id for the resolved provider (respects the per-request override)."""
    prov = active_provider(override=get_provider())
    if prov == "openai":
        return _get_openai_model()
    if prov == "claude":
        return _get_claude_model()
    if prov == "gemini":
        return _get_gemini_model()
    return _get_ollama_model()


def current_llm_info() -> Dict[str, Any]:
    """Provider, model id, and availability for the current request — surfaced in
    the /chat response so the UI can show which model produced the results."""
    prov = active_provider(override=get_provider())
    return {
        "provider": prov,
        "model": active_model(),
        "available": is_available(),
    }


def is_available(provider: Optional[str] = None) -> bool:
    """True when the active LLM provider is configured and reachable."""
    from ollama_executions import openai_available
    resolved_provider = provider or get_provider()
    resolved = active_provider(override=resolved_provider)
    if resolved == "openai":
        return openai_available()
    if resolved == "claude":
        return claude_available()
    if resolved == "gemini":
        return gemini_available()
    try:
        with urllib.request.urlopen(f"{_native_ollama_base_url()}/api/tags", timeout=1.5):
            return True
    except Exception:
        return False


def _call(
    system: str,
    user: str,
    max_tokens: int = 512,
    response_format: Optional[Dict[str, str]] = None,
    temperature: float = 0.2,
    provider: Optional[str] = None,
) -> str:
    """LLM call (Ollama/Claude/OpenAI). Returns empty string on failure."""
    resolved_provider = provider or get_provider()
    if not is_available(provider=resolved_provider):
        return ""
    log.info(f"🤖 LLM CALL [{resolved_provider}] - User: {user[:100]}...")
    raw = _retryable_ollama_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format=response_format,
        provider=resolved_provider,
    )
    result = (raw or "").strip()
    log.info(f"🤖 LLM RESPONSE [{resolved_provider}]: {result[:200]}...")
    return result


def _safe_json_object(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_json_array(raw: str) -> List[Any]:
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def analyze_experiment_request(
    *,
    user_query: str,
    conversation_query: str = "",
    previous_profile: Optional[Dict[str, Any]] = None,
    pending_field: Optional[str] = None,
    max_queries: int = 5,
) -> Dict[str, Any]:
    """
    LLM-first planner for the protocol chatbot.

    Returns the structured JSON contract used by /chat:
      intent_family, sub_intent, experiment_profile, missing_fields, next_action,
      clarifying_question, candidate_search_queries.
    """
    if not is_available():
        return {}
    log.info(f"📋 ANALYZE_EXPERIMENT_REQUEST - Query: {user_query[:80]}... [pending_field={pending_field}]")

    schema = {
        "intent_family": f"one of: {', '.join(sorted(INTENT_FAMILIES))}",
        "sub_intent": f"one of: {', '.join(sorted(SUB_INTENTS))}",
        "intent": {
            "name": "same value as sub_intent for compatibility",
            "label": "human-readable label",
            "confidence": 0.0,
        },
        "experiment_profile": profile_schema.llm_profile_template(),
        "missing_fields": [],
        "next_action": "ask_clarification|generate_search_queries|respond_chitchat",
        "user_declined_to_answer": False,
        "clarifying_question": {
            "field": "field_name_or_null",
            "question": "short question or null",
            "options": [],
        },
        "candidate_search_queries": [],
        "reply": None,
    }

    raw = _call(
        system=(
            "You are a biology protocol-search planning agent running locally in Ollama. "
            "Your job is to turn a scientist's natural-language request and prior chat "
            "state into a structured experiment profile and a next action.\n\n"
            "Return ONLY valid JSON. No markdown. No commentary.\n\n"
            "Allowed next_action values:\n"
            "- ask_clarification: use when a required search field is missing.\n"
            "- generate_search_queries: use when enough information exists to propose protocol-search queries.\n"
            "- respond_chitchat: use only for greetings or non-protocol conversation.\n\n"
            "Answering a specific question:\n"
            "- When 'pending_field' is set, the current_user_message is the ANSWER to that "
            "specific field. Assign the answer to THAT field — do not place it in a different "
            "field. E.g. if pending_field is 'readout_assay' and the user says 'phenotype', set "
            "readout_assay='phenotype' (NOT organism). If the message clearly also specifies "
            "other fields, you may fill those too, but the pending_field is the primary target.\n\n"
            "Decline detection:\n"
            "- Set user_declined_to_answer=true ONLY when the user's latest message is the "
            "answer to your pending clarification AND it declines to specify (e.g. 'not sure', "
            "'I don't know', 'no clue', 'no preference', 'you choose', 'whatever', 'doesn't "
            "matter', 'skip it', 'either is fine'). For any concrete answer, set it false.\n\n"
            "Field-value rules:\n"
            "- PRESERVE conditional operators VERBATIM in field values. If the user writes "
            "'rice or tomato', store the field as 'rice or tomato' (do NOT pick one or "
            "generalize to 'plant'). Likewise keep 'X and Y', 'like X', 'similar to X', and "
            "'such as X' exactly as written in the relevant field.\n\n"
            "Controlled intent rules:\n"
            "- Use exactly one allowed intent_family and one allowed sub_intent from the schema.\n"
            "- gene_modification means the user wants genes modified but did not specify the mechanism.\n"
            "- multiplex_gene_modification means more than one gene/target is modified at the same time.\n"
            "- gene_overexpression requires explicit words like overexpress, overexpression, or transgene expression.\n"
            "- genome_editing requires explicit CRISPR/Cas/base-editing/prime-editing language.\n"
            "- stress_tolerance_assay is for drought, salt, heat, cold, or other stress tolerance tests.\n\n"
            "Clarification rules:\n"
            "- Ask exactly one question at a time.\n"
            "- ALWAYS fill the clarifying_question 'options' array with 3-5 short, "
            "concrete example answers the user can click (single words or brief "
            "phrases). Never leave 'options' empty, and do NOT put the examples only "
            "inside the question text. Example: field 'organism' -> options "
            "['E. coli', 'human', 'mouse', 'Arabidopsis', 'yeast'].\n"
            "- If the user says genes are modified, gene modification, gene editing, or genome editing without specifying how, use intent gene_modification and ask for modification_type first.\n"
            "- Do not treat ambiguous 'modified genes' as overexpression unless the user explicitly says overexpress or overexpression.\n"
            "- For multiplex gene modification, ask modification_type before searching.\n"
            "- For drought/stress tolerance, ask organism first, then growth_stage/sample/treatment if missing.\n"
            "- For plant gene overexpression, prioritize missing fields in this order: "
            "organism, expression_type, tissue_or_cell_type, readout_assay.\n\n"
            "Search-query rules:\n"
            f"- If next_action is generate_search_queries, provide 3-{max_queries} candidate_search_queries.\n"
            "- Candidate queries should be short phrases suitable for protocols.io, not full sentences.\n"
            "- Generate variants using common/scientific organism names, method synonyms, tissue/sample, and readout/assay terms.\n"
            "- Preserve organism + sub_intent + critical intent_specific concepts in every candidate query.\n"
            "- Do not invent gene names, equipment, or constraints the user did not provide.\n\n"
            "Use this exact schema shape:\n"
            f"{json.dumps(schema, indent=2)}"
        ),
        user=json.dumps(
            {
                "current_user_message": user_query,
                "conversation_query": conversation_query,
                "previous_experiment_profile": previous_profile or {},
                "pending_field": pending_field or None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        max_tokens=900,
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return _safe_json_object(raw)


def generate_search_queries(query: str, n_probes: int = 5) -> List[str]:
    """
    Convert a natural-language query into compact search probes for protocols.io.
    """
    raw = _call(
        system=(
            "You are a biomedical search expert. Convert the scientist's question into "
            f"{n_probes} short, precise search phrases for a protocol database. "
            "Rules:\n"
            "- Each phrase must be 1-6 words, no punctuation\n"
            "- Use proper scientific terms when useful\n"
            "- Cover different angles: technique, organism, molecule, goal\n"
            "- Most specific phrases first\n"
            f"Return ONLY a JSON array of {n_probes} strings, no explanation."
        ),
        user=query,
        max_tokens=256,
        temperature=0.2,
    )
    return [str(x).strip() for x in _safe_json_array(raw) if str(x).strip()][:n_probes]


def classify_intent(query: str) -> Dict[str, Any]:
    """
    Decide whether query is a protocol search or general conversation.
    Returns {"intent": "search"|"chitchat", "reply": str|None}
    """
    raw = _call(
        system=(
            "Classify the user's message for a lab protocol search assistant.\n\n"
            "If it is a request to FIND or SEARCH for a lab protocol, experiment method, "
            "or scientific procedure: reply exactly with: SEARCH\n\n"
            "If it is general conversation, a greeting, a question about you, or anything "
            "unrelated to lab protocols: reply with: CHITCHAT | <1-2 sentence response "
            "that mentions what you can help with>\n\n"
            "Examples:\n"
            "  'RNA extraction from plant tissue' -> SEARCH\n"
            "  'hi there' -> CHITCHAT | Hi. Describe an experiment and I will find matching protocols.\n"
            "  'western blot protocol' -> SEARCH\n"
            "  'thanks' -> CHITCHAT | You're welcome. Send another experimental goal when ready."
        ),
        user=query,
        max_tokens=128,
        temperature=0.0,
    )
    if raw:
        raw = raw.strip()
        if raw.upper().startswith("SEARCH"):
            return {"intent": "search", "reply": None}
        if raw.upper().startswith("CHITCHAT"):
            parts = raw.split("|", 1)
            reply = parts[1].strip() if len(parts) > 1 else (
                "I'm a lab protocol search assistant. Describe an experiment and I will find matching protocols from protocols.io."
            )
            return {"intent": "chitchat", "reply": reply}
    return _keyword_fallback(query)


_LAB_KEYWORDS = {
    "protocol", "extraction", "isolation", "purification", "assay", "pcr", "rna", "dna",
    "protein", "cell", "tissue", "buffer", "gel", "blot", "western", "elisa", "crispr",
    "transfection", "sequencing", "microscopy", "culture", "staining", "antibody", "primer",
    "cloning", "transformation", "centrifuge", "incubate", "lysis", "pellet", "supernatant",
    "plasmid", "enzyme", "reagent", "sample", "experiment", "lab", "bacteria", "yeast",
    "arabidopsis", "mouse", "human", "plant", "mammalian", "fluorescence", "overexpress",
    "overexpression", "knockdown", "gene",
}


def _keyword_fallback(query: str) -> Dict[str, Any]:
    words = set(query.lower().split())
    if words & _LAB_KEYWORDS:
        return {"intent": "search", "reply": None}
    return {
        "intent": "chitchat",
        "reply": "I'm a lab protocol search assistant. Describe an experiment you need to run and I will find matching protocols from protocols.io.",
    }


def explain_matches(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Plain-English explanation of why the top protocols match the query.
    """
    protocol_summaries = ""
    for i, result in enumerate(results[:3], 1):
        protocol_summaries += (
            f"\nMatch #{i}: {result.get('title', '')}\n"
            f"  Description: {(result.get('description') or '')[:200]}\n"
            f"  Materials: {(result.get('materials_text') or '')[:120]}\n"
            f"  Why it ranked: {result.get('why', '')}\n"
        )

    return _call(
        system=(
            "You are a helpful lab assistant for bench scientists. A scientist asked "
            "a question and the system retrieved matching protocols from protocols.io. "
            "Explain in 2-4 plain sentences which protocols are most relevant and why. "
            "Refer to each protocol by the label it was given (\"Match #1\", \"Match #2\", "
            "etc.) — never as \"Protocol 1\" or \"Protocol 2\". "
            "Do not invent steps or materials not mentioned."
        ),
        user=f"Scientist's request: {query}\n\nTop matching protocols:{protocol_summaries}",
        max_tokens=300,
        temperature=0.2,
    )


def get_synonyms(term: str, max_terms: int = 3) -> List[str]:
    """
    Generate biomedical synonyms for a search concept.
    """
    raw = _call(
        system=(
            "You are a biomedical search assistant. Given one concept, reply with ONLY "
            "a JSON array of up to 3 short synonyms or closely related search terms. "
            "No explanation."
        ),
        user=term,
        max_tokens=128,
        temperature=0.2,
    )
    return [str(x).strip() for x in _safe_json_array(raw) if str(x).strip()][:max_terms]


def get_sentence_variants(query: str, n: int = 10) -> List[str]:
    """
    Generate reworded full-sentence versions of the query.
    """
    raw = _call(
        system=(
            f"Rewrite the user's protocol search request as {n} alternative full-sentence "
            "search queries that mean the same thing, using different scientific phrasing "
            "and synonyms. Reply with ONLY a JSON array of strings, no explanation."
        ),
        user=query,
        max_tokens=512,
        temperature=0.3,
    )
    return [str(x).strip() for x in _safe_json_array(raw) if str(x).strip()][:n]


def is_new_search_topic(current_goal: str, new_message: str) -> bool:
    """
    True when the new message starts a clearly different search rather than
    answering or refining the current one. Used to reset the experiment profile
    mid-conversation when the user switches topics.

    Conservative: returns False (stay in the current search) whenever the LLM is
    unavailable or unsure — it never resets on doubt, so a clarification answer
    like "banana" or "stable transformation" is never mistaken for a new search.
    """
    current_goal = (current_goal or "").strip()
    new_message = (new_message or "").strip()
    if not current_goal or not new_message or not is_available():
        return False
    raw = _call(
        system=(
            "A scientist is building ONE protocol-search request over a conversation. "
            "Given their CURRENT search goal and their NEW message, decide whether the new "
            "message belongs to the SAME search or starts a NEW one. Answer with EXACTLY one "
            "word: SAME or NEW.\n"
            "SAME — the message answers the pending question, or briefly refines ONE detail "
            "of the current search (e.g. 'banana', 'use mouse instead', 'how about rice', "
            "'stable transformation', 'leaf tissue', 'look at qPCR'). A short tweak that keeps "
            "the same overall experiment is SAME, even if it changes the organism.\n"
            "NEW — the message is a COMPLETE, standalone protocol-search request (e.g. it "
            "starts like 'Find protocols for ...' / 'I want protocols that ...' and specifies "
            "an experiment), OR it switches to a different technique/experiment type (e.g. "
            "genome editing -> western blot, or -> a drought-tolerance assay). A fully "
            "re-stated request is NEW even if its technique overlaps the current one.\n"
            "When unsure, answer SAME."
        ),
        user=f"CURRENT GOAL: {current_goal}\nNEW MESSAGE: {new_message}",
        max_tokens=8,
        temperature=0.0,
    )
    return raw.strip().upper().startswith("NEW")


def classify_domain(query: str, choices: Dict[str, str], default: str = "biology") -> str:
    """
    Route a request to a scientific domain, given the registered domains and the
    one-line scope each one declares.

    Conservative: returns "" whenever the LLM is unavailable, or answers with
    anything that is not a registered domain name. The caller then keeps its own
    fallback (the keyword router) instead of acting on an unusable answer.
    """
    query = (query or "").strip()
    if not query or len(choices) < 2 or not is_available():
        return ""
    menu = "\n".join(f"- {name}: {desc}" for name, desc in choices.items())
    raw = _call(
        system=(
            "You route a scientist's protocol-search request to the right scientific "
            "domain. Given the request and the domains below, answer with EXACTLY one "
            "domain name from this list and nothing else.\n"
            f"{menu}\n"
            "Judge the experiment the scientist wants to run, not individual words. A "
            "biological experiment that uses a chemical reagent is still biology, and a "
            "chemical procedure applied to a biological sample is still chemistry. "
            f"When the request fits more than one domain, or you are unsure, answer {default}."
        ),
        user=query[:600],
        max_tokens=8,
        temperature=0.0,
    )
    answer = raw.strip().strip(".").lower()
    for name in choices:
        if answer == name.lower():
            return name
    return ""


def _domain_search_query_fields(profile: Dict[str, Any]):
    """The active domain's labeled profile fields for the candidate-query prompt.
    Empty/failed -> caller keeps its own default. Deferred import keeps this
    module importable independently of the domains package."""
    try:
        from domains import current_domain
        return current_domain().search_query_fields(profile)
    except Exception:  # noqa: BLE001 -- must never break query generation
        return []


def _domain_prompt(method: str, *args) -> Optional[str]:
    """Ask the active domain for a named prompt; None -> caller's own default."""
    try:
        from domains import current_domain
        return getattr(current_domain(), method)(*args)
    except Exception:  # noqa: BLE001
        return None


def generate_natural_search_queries(
    profile: Dict[str, Any],
    original_query: str,
    n: int = 5,
) -> List[str]:
    """
    Generate natural-language protocol-search queries from the structured
    profile + the user's original request.

    Each query is a focused angle; across the set every populated field value is
    used at least once (collective coverage); the original query is included; and
    only concepts from the profile/original request are used (no invented terms
    like "in planta" for a non-plant). Returns [] when the LLM is unavailable.
    """
    if not is_available():
        return []

    p = profile or {}
    # The labeled field block is DOMAIN-SPECIFIC: a chemistry profile has no
    # "organism"/"expression type", so using biology's list for it produced an
    # empty block and the model generated queries blind to the profile.
    field_labels = _domain_search_query_fields(p) or [
        ("organism", p.get("organism")),
        ("system / sample type", p.get("tissue_or_cell_type") or p.get("sample_type")),
        ("target", p.get("target")),
        ("modification / technique", p.get("modification_type")),
        ("method / approach", p.get("sub_intent") or p.get("experimental_method")),
        ("delivery method", p.get("delivery_method")),
        ("expression type", p.get("expression_type")),
        ("readout", p.get("readout_assay") or p.get("readout")),
        ("condition", p.get("condition")),
    ]
    skip = {"", "not specified", "none", "unknown", "not sure", "null"}
    fields = [
        f"- {label}: {str(val).strip()}"
        for label, val in field_labels
        if str(val or "").strip().lower() not in skip
    ]
    intent_specific = p.get("intent_specific")
    if isinstance(intent_specific, dict) and intent_specific:
        extras = "; ".join(
            f"{k}: {v}" for k, v in intent_specific.items()
            if v not in (None, "", False) and str(v).strip().lower() not in skip
        )
        if extras:
            fields.append(f"- additional details: {extras}")
    fields_block = "\n".join(fields)

    raw = _call(
        system=_domain_prompt("candidate_query_prompt", n) or (
            "You write search queries for a biology protocol database (protocols.io). "
            f"Given a structured experiment profile and the scientist's original request, "
            f"write {n} search queries as NATURAL-LANGUAGE phrases.\n"
            "Rules:\n"
            "- Each query is a short, readable phrase — not a keyword dump and not a question.\n"
            "- Each query focuses on a DIFFERENT combination of the fields (a different angle).\n"
            "- ACROSS the whole set, use EVERY field value at least once — don't ignore any field.\n"
            "- Vary phrasing with natural synonyms (scientific organism names, 'simultaneous' for "
            "multiplex, 'western blot' for protein level, etc.).\n"
            "- Include the scientist's ORIGINAL request as one of the queries, essentially unchanged.\n"
            "- PRESERVE conditional operators VERBATIM. If a field value contains 'or' "
            "('rice or tomato'), 'and' ('rice and maize'), or 'like'/'similar to'/'such as' "
            "('like tomato'), keep that exact phrasing in the queries — do NOT pick one side, "
            "drop the operator, or rephrase it (e.g. write '...in rice or tomato...').\n"
            "- Use ONLY concepts present in the profile or the original request. NEVER invent "
            "organisms, tissues, genes, or techniques. In particular, never write 'in planta' "
            "unless the system is literally a plant.\n"
            "Return ONLY a JSON array of strings, no explanation."
        ),
        user=f"ORIGINAL REQUEST: {original_query}\n\nEXPERIMENT PROFILE:\n{fields_block}",
        max_tokens=400,
        temperature=0.3,
    )
    return [str(x).strip() for x in _safe_json_array(raw) if str(x).strip()][:n]


def is_asking_about_clarification_reason(user_message: str) -> bool:
    """
    Simple check: does the message ask why we need something?
    Examples: "why", "what's the difference", "does it matter", "what's the point"
    """
    msg = (user_message or "").strip().lower()
    patterns = ["why", "what difference", "does it matter", "what's the point", "do i need"]
    return any(p in msg for p in patterns)


def generate_clarification_explanation(field: str, clarification_options: Optional[List[str]] = None) -> str:
    """
    Use LLM to generate a brief explanation of why a clarification field matters.
    Called when user asks "why do you need this?" during clarification.
    """
    if not is_available():
        return f"The {field.replace('_', ' ')} helps narrow down protocol options."

    log.info(f"📚 GENERATE_CLARIFICATION_EXPLANATION - Field: {field}")
    options_text = ""
    if clarification_options:
        options_text = f"\nThe options are: {', '.join(clarification_options)}"

    explanation = _call(
        system=_domain_prompt("clarification_explanation_prompt", field) or (
            "You are a helpful biology protocol assistant. A user is asking why a specific field "
            "matters for finding protocols. Provide a brief, friendly 1-2 sentence explanation of why "
            "this field is important and how different choices affect protocol selection. Be concise."
        ),
        user=f"Field: {field.replace('_', ' ')}{options_text}\n\nWhy does this field matter for protocol selection?",
        max_tokens=150,
        temperature=0.3,
    )
    return explanation


def generate_chitchat_response(user_message: str) -> str:
    """Generate a friendly LLM response for non-protocol-search chitchat."""
    if not is_available():
        return "I'm a lab protocol search assistant. Describe an experiment and I can help find relevant protocols."

    log.info(f"💬 GENERATE_CHITCHAT_RESPONSE - Message: {user_message[:80]}")
    response = _call(
        system=(
            "You are a friendly lab protocol search assistant. "
            "The user is asking a non-protocol question (like greeting or general chat). "
            "Respond warmly and briefly, then prompt them to describe an experiment they need help with. "
            "Keep response under 2 sentences."
        ),
        user=user_message,
        max_tokens=100,
        temperature=0.7,
    )
    return response or "I'm a lab protocol search assistant. Describe an experiment and I can help find relevant protocols."


_PIPELINE_FACTS = {
    "ranking": (
        "How results are ranked in this app (state ONLY these facts, grounded in the "
        "user's profile): protocols.io and PubMed results are scored on ONE shared "
        "cross-source relevance axis, so the most relevant wins regardless of source. "
        "profile_score = 18 x text-relevance (blend_score: title-weighted TF-IDF cosine "
        "of the query vs each result) + weighted field-match bonuses (organism, method, "
        "readout, tissue, and required-concept coverage), minus penalties for missing "
        "required concepts. Results are sorted highest score first, and each result shows "
        "its own 'why it matches / may not fit' breakdown. Do NOT invent any other factors."
    ),
    "query_generation": (
        "How the suggested search queries are generated (state ONLY these facts, grounded "
        "in the user's profile): an LLM writes ~5 natural-language search angles from the "
        "user's structured experiment profile so they collectively cover its fields "
        "(organism, technique, condition, readout, growth stage, ...); the user's original "
        "request is always included as one option; a rule-based generator is used as a "
        "fallback if the LLM is unavailable. The queries are de-duplicated and ordered "
        "most-specific-first (candidate #1 carries the fullest accumulated intent). "
        "Do NOT invent any other factors."
    ),
    "query_ordering": (
        "How the SUGGESTED SEARCH QUERIES (not the results) are ORDERED (state ONLY these "
        "facts): the candidate queries are sorted most-specific-first by a deterministic "
        "specificity count — how many of the user's profile field-values (organism, "
        "technique, condition, readout, growth stage) appear in each query. So candidate #1 "
        "covers the most of the profile and carries the fullest accumulated intent, while "
        "broader phrasings and the user's original request appear lower; ties keep the "
        "generator's order. This is a simple field-coverage count, NOT an LLM ranking and "
        "NOT the relevance scoring used for search results. Do NOT invent any other factors."
    ),
}

_META_SENTINEL = "NOT_META"


_VIEW_DESCRIPTIONS = {
    "query_selection": "the list of SUGGESTED SEARCH QUERIES (the search has NOT been run yet — no results are showing)",
    "results": "the RANKED SEARCH RESULTS from protocols.io and PubMed (a list of matched protocols/papers)",
    "clarification": "a clarification question",
}


def answer_session_message(
    user_message: str,
    profile: Optional[Dict[str, Any]] = None,
    candidate_queries: Optional[List[str]] = None,
    current_view: Optional[str] = None,
) -> Optional[str]:
    """One context-aware conversational handler for a mid-session message.

    Given the message + what the user is currently viewing + their profile, the
    LLM decides and responds:
      1. A question about how the app works / about the session (query generation,
         query ordering, result ranking, why a field was chosen, what a field
         means) -> an accurate answer grounded in the facts + profile.
      2. Chitchat / social (greeting, thanks, "who are you") -> a warm short reply.
      3. A new search, a refinement, or search input -> returns None (the caller
         falls through to the search pipeline).

    Ambiguous references ("these", "this", "ranked this way") are resolved to what
    the user is CURRENTLY VIEWING. No keyword logic — the model reads the sentence
    plus the context.
    """
    if not is_available():
        return None  # can't classify without the LLM; let the normal flow handle it
    facts = "\n".join(f"- {v}" for v in _PIPELINE_FACTS.values())
    view_desc = _VIEW_DESCRIPTIONS.get((current_view or "").strip(),
                                       "suggested search queries and/or ranked results")
    log.info(f"📚 ANSWER_SESSION_MESSAGE - view={current_view} msg={user_message[:60]}")
    system = (
        "You are a friendly lab protocol search assistant. The user is mid-session. RIGHT NOW they "
        f"are looking at: {view_desc}. Classify their message and respond accordingly:\n\n"
        "1. QUESTION about how the app works or about this session — e.g. how the suggested queries "
        "are generated or ordered, how the results are ranked, why a profile field was chosen, what "
        "a field means. Answer in 2-4 sentences of plain prose (no markdown), using ONLY the facts "
        "below plus the user's profile. CRITICAL: resolve 'these', 'this', 'the order', 'ranked "
        "this way' to WHAT THEY ARE CURRENTLY VIEWING (stated above) — the QUERIES if they're "
        "viewing the query list, the RESULTS if they're viewing results.\n\n"
        "2. CHITCHAT / social — a greeting, thanks, small talk, or 'who are you'. Reply warmly in "
        "1-2 sentences and gently point back to their current search.\n\n"
        "3. A NEW SEARCH, a REFINEMENT of the search (e.g. 'also add salt stress', 'search in "
        "maize instead'), or SEARCH INPUT — INCLUDING an experiment description that is UNRELATED "
        "to what's currently showing (a different organism, technique, or goal). Do NOT answer and "
        "do NOT explain any mismatch; reply with EXACTLY " + _META_SENTINEL + " and NOTHING ELSE, "
        "so the search pipeline handles it as a fresh search.\n\n"
        "FACTS you may use for case 1:\n" + facts
    )
    user = json.dumps({
        "message": user_message,
        "currently_viewing": current_view or "unknown",
        "profile": profile or {},
        "suggested_queries": candidate_queries or [],
    })
    resp = (_call(system=system, user=user, max_tokens=240, temperature=0.3) or "").strip()
    # The model may emit the sentinel alone OR (less ideally) tack it onto an
    # explanation ("...there's a mismatch... NOT_META"). Treat ANY occurrence as
    # "this is a new search / refinement" and fall through — so a new-topic query
    # is never swallowed as a meta explanation, and the tag never leaks to the user.
    if not resp or _META_SENTINEL in resp.upper():
        return None
    return resp
