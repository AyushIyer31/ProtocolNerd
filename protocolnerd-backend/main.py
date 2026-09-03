from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, Query, Form, File
from fastapi.responses import Response, RedirectResponse
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import asyncio
import uuid
import json
import logging
import os
import shutil
import time
from functools import lru_cache
from pathlib import Path

from helper_functions import (
    extract_text_from_upload,
    build_local_rag_index,
    retrieve_relevant_chunks,
    analyze_target_with_ollama,
    ensure_session_dirs,
    check_ollama_health,
    get_default_execution_strategy,
    normalize_execution_strategy,
)
from protocol_rag import (
    build_protocol_index, search_protocols, explain_matches, classify_intent,
    is_vague_query, get_clarification_question, expand_query, multi_search_protocols,
    load_protocol_index,
)
from claude_client import (
    analyze_experiment_request,
    is_available as llm_is_available,
    is_new_search_topic,
    generate_natural_search_queries,
    current_llm_info,
    is_asking_about_clarification_reason,
    generate_clarification_explanation,
)
from concept_expansion import (
    extract_concepts, expand_concepts, build_search_probes, generate_sentence_variants,
)
from protocolsio_client import multi_probe_search
from protocol_ranker import rank_protocols
from experiment_profile import (
    apply_profile_ranking,
    build_experiment_profile,
    can_generate_search_queries,
    candidate_query_preserves_required_concepts,
    detect_experiment_intent,
    generate_candidate_search_queries,
    merge_profiles,
    needs_clarification,
    next_biology_clarification,
    next_clarification,
    normalize_experiment_goal,
    profile_to_search_query,
    profile_source_query_for_request,
    should_respond_as_chitchat,
    surface_growth_stage,
    validate_biology_profile,
    _is_generic,
)
from field_ranking import closeness_rank
import reranker
import dense_index
import query_logger
from pubmed_client import (search_pubmed_fanout, build_pubmed_queries, balanced_trim,
                           fetch_fulltext, extract_methods)
from blend_ranking import blend_results, blend_sources
from retrievers import RETRIEVERS, RetrievalContext, pubmed_core_query as _pubmed_core_query
from domains import current_domain, set_current_domain, route, pinnable_domain_names
from query_operators import pubmed_terms, is_like, detect_operator

logging.basicConfig(level=logging.INFO)

app = FastAPI()
update_queues: Dict[str, asyncio.Queue] = {}
main_loop: Optional[asyncio.AbstractEventLoop] = None
executor = ThreadPoolExecutor()

# Protocol RAG index — loaded once at startup
PROTOCOL_INDEX: Optional[Dict[str, Any]] = None
PROTOCOLS_DATA_DIR = Path("../data/protocols")
# Prebuilt index baked into deploy images by scripts/build_index.py.
PROTOCOL_INDEX_CACHE = Path("../data/protocol_index.pkl")
DENSE_INDEX_CACHE = Path("../data/protocol_dense.npy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_STORAGE_DIR = Path("storage")
SESSIONS_DIR = BASE_STORAGE_DIR / "sessions"


@app.on_event("startup")
async def startup_event():
    global main_loop, PROTOCOL_INDEX
    main_loop = asyncio.get_event_loop()
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer a prebuilt index baked into the image (scripts/build_index.py runs
    # at Docker build time). Loading a pickle is fast and CPU-light, so it's safe
    # to do synchronously at startup — the index is ready the moment the server
    # accepts requests. This avoids Cloud Run's between-request CPU throttling,
    # which stalls a runtime build and leaves local search permanently 503-ing.
    if PROTOCOL_INDEX_CACHE.exists():
        try:
            PROTOCOL_INDEX = load_protocol_index(PROTOCOL_INDEX_CACHE)
            logging.info(f"Loaded prebuilt protocol index: {len(PROTOCOL_INDEX['protocols'])} protocols.")
            RETRIEVERS["protocols.io"].set_index(PROTOCOL_INDEX)
        except Exception as e:
            logging.warning(f"Could not load prebuilt index ({e}); will build at runtime.")

    if PROTOCOL_INDEX is None and PROTOCOLS_DATA_DIR.exists():
        # Local-dev fallback: no prebuilt pickle, so build in a background thread
        # (a dev machine has no CPU throttling, so the build completes fine).
        # Request handlers guard for `PROTOCOL_INDEX is None` until it's ready.
        def _build_index():
            global PROTOCOL_INDEX
            try:
                PROTOCOL_INDEX = build_protocol_index(PROTOCOLS_DATA_DIR)
                logging.info(f"Protocol index ready: {len(PROTOCOL_INDEX['protocols'])} protocols loaded.")
                RETRIEVERS["protocols.io"].set_index(PROTOCOL_INDEX)
            except Exception as e:
                logging.warning(f"Could not build protocol index: {e}")

        executor.submit(_build_index)
        logging.info("No prebuilt index found; building in background.")
    elif PROTOCOL_INDEX is None:
        logging.warning(f"Protocol data dir not found: {PROTOCOLS_DATA_DIR}. Run fetch_protocols.py first.")

    logging.info("RAG backend started.")


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    explain: bool = True
    # Set True when the user is responding to a clarification question,
    # so we don't ask for clarification a second time.
    skip_clarification: bool = False
    # "live"  -> concept-expansion pipeline against the live protocols.io API
    # "local" -> TF-IDF search over the cached protocol index
    search_mode: str = "live"
    # Client-maintained state for the current experimental goal. This lets a
    # short clarification answer like "stable transformation" update the
    # original request instead of replacing it.
    experiment_profile: Optional[Dict[str, Any]] = None
    conversation_query: Optional[str] = None
    # Search only runs after the user chooses/edits generated candidate queries.
    search_confirmed: bool = False
    selected_search_query: Optional[str] = None
    search_all: bool = False
    candidate_search_queries: Optional[List[str]] = None
    # True when this message answers a clarification the assistant just asked.
    # New-topic detection is skipped for these so a short answer (e.g. "CRISPR")
    # refines the current profile instead of being mistaken for a new search.
    is_clarification_answer: bool = False
    # Client-supplied conversation id, captured for query logging.
    session_id: Optional[str] = None
    # Set by non-user pings (e.g. the warmup/count request) to stay out of the query logs.
    no_log: bool = False
    # Field of the clarification the assistant last asked (echoed by the client).
    # Authoritative pending field — used to fill/skip the right field, especially
    # for LLM-driven clarifications the rule-based path can't infer.
    pending_field: Optional[str] = None
    # The exact question + option chips the client is currently showing for the
    # pending clarification. Echoed back so "why this question?" re-shows the SAME
    # chips (rule- or LLM-generated) instead of reconstructing them (which loses
    # LLM-origin options).
    pending_clarification_question: Optional[str] = None
    pending_clarification_options: Optional[List[str]] = None
    # LLM provider override: "claude", "openai", or "ollama"
    provider: Optional[str] = None
    # Claude model override (debug-only in the UI). Whitelisted to Sonnet/Haiku;
    # anything else (or None) falls back to the env default (Sonnet 4.6).
    model: Optional[str] = None
    # Per-request toggle for the optional LLM re-ranker. None -> use the
    # ENABLE_RERANKER env default (on by default, matching prod). True/False overrides.
    enable_reranker: Optional[bool] = None
    # Experiment #2 toggle: re-rank PubMed candidates jointly with protocols.io
    # (vs blending PubMed in afterward). None -> RERANK_COMBINED_PUBMED env (default off).
    rerank_combined: Optional[bool] = None
    # What the client is currently showing ("query_selection" | "results" |
    # "clarification"), so a meta-question's "these/this" resolves to the right thing.
    client_view: Optional[str] = None
    # How many results to pull from EACH source (protocols.io + PubMed) before
    # blending. UI-configurable; defaults to 5 per source. (Legacy symmetric knob.)
    results_per_provider: int = 5
    # Asymmetric blend mix "P+M" = P protocols.io + M PubMed in the top-10.
    # Default 8+2 (best grade-2 coverage + near-best nDCG in held-out eval).
    # Debug UI options: "10+0", "8+2", "5+5". None -> default 8+2.
    blend_mix: Optional[str] = None


def run_live_expansion_search(query: str, top_k: int, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Live protocols.io search. Searches the SAME 3 query variants as Local RAG mode
    (original, stopword-stripped, core phrase) via expand_query(), so Live and Local
    stay consistent. Concepts/expansions are still computed for ranking + display,
    but are no longer turned into keyword probes (which pulled in off-domain
    synonyms, e.g. "stress" -> "anxiety").

    expand_query -> 3 variants per query -> multi_probe_search (parallel) -> merge -> re-rank.
    Returns results plus intermediate variants and timing for display.
    """
    concepts = extract_concepts(query)
    expansions = expand_concepts(concepts, use_external=True)
    probes = expand_query(query)  # 3 clean variants, identical to Local RAG
    sentence_variants = generate_sentence_variants(concepts, expansions)
    merged, hit_map, probe_totals = multi_probe_search(probes, per_probe=6, cap=50)
    initial_k = max(top_k * 3, top_k)
    ranked = rank_protocols(concepts, expansions, merged, hit_map, top_k=initial_k)
    if profile:
        ranked = current_domain().rank(profile, ranked, top_k=top_k)
    else:
        ranked = ranked[:top_k]
    return {
        "results": ranked,
        "concepts": concepts,
        "expansions": expansions,
        "probes": probes,
        "probe_totals": probe_totals,
        "sentence_variants": sentence_variants,
    }


def run_live_candidate_searches(
    queries: List[str],
    top_k: int,
    profile: Optional[Dict[str, Any]] = None,
    raw_query: str = "",
) -> Dict[str, Any]:
    merged: Dict[Any, Dict[str, Any]] = {}
    expanded: List[str] = []
    concepts: Dict[str, Any] = {}
    expansions: Dict[str, List[str]] = {}
    sentence_variants: List[str] = []

    for query in [q for q in queries if q.strip()][:5]:
        live = run_live_expansion_search(query, max(top_k * 2, top_k), profile)
        expanded.extend(live.get("probes", []))
        concepts = concepts or live.get("concepts", {})
        expansions = expansions or live.get("expansions", {})
        sentence_variants.extend(live.get("sentence_variants", []))
        for result in live.get("results", []):
            pid = result.get("id") or result.get("url") or result.get("title")
            if not pid:
                continue
            existing = merged.get(pid)
            if not existing or result.get("profile_score", result.get("score", 0)) > existing.get("profile_score", existing.get("score", 0)):
                merged[pid] = {**result, "selected_query_matches": [query]}
            else:
                existing.setdefault("selected_query_matches", []).append(query)

    results = list(merged.values())
    if profile:
        results = closeness_rank(profile, results, top_k, raw_query=raw_query)
    else:
        results = sorted(results, key=lambda x: x.get("profile_score", x.get("score", 0)), reverse=True)[:top_k]

    return {
        "results": results,
        "expanded": _dedup_strings(expanded),
        "concepts": concepts,
        "expansions": expansions,
        "sentence_variants": _dedup_strings(sentence_variants),
    }


def run_local_candidate_searches(
    queries: List[str],
    top_k: int,
    profile: Dict[str, Any],
    raw_query: str = "",
    rerank: bool = False,
    rerank_query: str = "",
) -> Dict[str, Any]:
    expanded: List[str] = []
    for query in [q for q in queries if q.strip()][:5]:
        expanded.extend(expand_query(query))
    expanded = _dedup_strings(expanded)
    # Keep a wider candidate pool for the profile-aware ranker. Narrow TF-IDF
    # pools tend to preserve broad "protein/expression" hits before the ranker
    # can reward organism + expression type + tissue matches.
    candidate_pool_size = max(top_k * 24, 120)
    results = multi_search_protocols(PROTOCOL_INDEX, expanded, candidate_pool_size)

    # Organism-merge: the TF-IDF stage forces certain keywords (e.g. "crispr")
    # as required terms, which can penalize an organism-correct protocol out of
    # the pool when it lacks that literal word (e.g. a tomato transformation
    # protocol that doesn't spell out "crispr"). When the user named a SPECIFIC
    # organism, retrieve organism-focused matches directly and merge them in so
    # the profile ranker can evaluate and surface them. Skipped for vague
    # organisms ("plant", "cell") — searching those floods the pool with noise.
    organism = str((profile or {}).get("organism") or "").strip()
    if organism and not _is_generic(organism):
        seen = {r["id"] for r in results}
        organism_queries = _dedup_strings(
            [organism] + [f"{organism} {q}" for q in queries[:3] if q.strip()]
        )
        for oq in organism_queries:
            for r in search_protocols(PROTOCOL_INDEX, oq, top_k=10):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    results.append(r)

    # Fuse in embedding-retrieved candidates (RRF). TF-IDF only matches literal word
    # overlap, so a request phrased unlike the protocol ("how do plants cope without
    # water" vs "Drought stress tolerance assay") never enters the pool at all -- and the
    # re-ranker cannot rescue what retrieval never fetched. Same helpers the eval calls,
    # so the two paths can't drift. Silently a no-op if the index wasn't built.
    if dense_index.is_enabled():
        try:
            _dmat = dense_index.get_dense_matrix(DENSE_INDEX_CACHE)
            if _dmat is not None:
                _dense = dense_index.dense_candidates(
                    _dmat, PROTOCOL_INDEX["protocols"],
                    raw_query or (queries[0] if queries else ""),
                    queries, min(dense_index.DENSE_TOP_K, candidate_pool_size))
                _before = len(results)
                results = dense_index.fuse_pool(results, _dense)
                logging.info(f"🧬 Dense fusion: {_before} TF-IDF + {len(_dense)} dense "
                             f"-> {len(results)} fused candidates")
        except Exception as e:  # noqa: BLE001 — dense is an ENHANCEMENT, never a hard dep:
            # a missing model/index or an ONNX failure must degrade to TF-IDF, not 500 the search.
            logging.warning(f"Dense fusion failed ({e}); falling back to TF-IDF only.")

    # When the re-ranker is on, rank a wider depth so its shortlist (profile-top-N
    # UNION lexical-top-N) has enough candidates to rescue a buried match.
    pool = results
    depth = max(top_k, reranker.SHORTLIST_PROFILE) if rerank else top_k
    ranked = closeness_rank(profile, pool, depth, raw_query=raw_query)
    if rerank:
        ranked = reranker.rerank(rerank_query or raw_query, ranked, pool=pool, top_k=top_k)
    return {
        "results": ranked[:top_k],
        "expanded": expanded,
        "concepts": {},
        "expansions": {},
        "sentence_variants": [],
    }


def _parse_blend_mix(mix: Optional[str]) -> tuple:
    """Parse a blend-mix string "P+M" into (protocols_k, pubmed_k). Defaults to
    8+2. Accepts "10+0", "8+2", "5+5", etc.; falls back to 8+2 on anything odd."""
    if mix:
        try:
            p, m = mix.lower().replace(" ", "").split("+")
            pk, mk = int(p), int(m)
            if pk >= 0 and mk >= 0 and (pk + mk) > 0:
                return pk, mk
        except Exception:  # noqa: BLE001
            pass
    return 8, 2


def _dedup_strings(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        key = str(value).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(str(value).strip())
    return out


# _INTENT_ALIASES, _controlled_intent_name, _intent_response, and _normalize_intent
# moved to domains/biology.py (BiologyDomain.normalize_llm_intent) -- they only ever
# made sense against biology's controlled sub-intent vocabulary, and living here made
# every domain's request pass through them regardless (chemistry's sub-intents were
# silently collapsing to "unknown" because of it). Call site: domain.normalize_llm_intent(...).


def _clarification_from_plan(plan: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    raw = plan.get("clarifying_question") if isinstance(plan, dict) else None
    if not isinstance(raw, dict):
        return None
    question = str(raw.get("question") or "").strip()
    if not question or question.lower() in {"none", "null"}:
        return None
    field = str(raw.get("field") or "general").strip() or "general"
    if profile and _llm_clarification_is_stale(profile, field, question):
        return None
    options = raw.get("options") if isinstance(raw.get("options"), list) else []
    return {
        "field": field,
        "question": question,
        "options": [str(x) for x in options if str(x).strip()][:6],
    }


def _llm_clarification_is_stale(profile: Dict[str, Any], field: str, question: str) -> bool:
    fields = _clarification_candidate_fields(field, question)
    for candidate in fields:
        if candidate != "general" and not current_domain().needs_clarification(profile, candidate):
            return True
    return False


def _clarification_candidate_fields(field: str, question: str) -> List[str]:
    fields = []
    if field:
        fields.append(str(field).strip())
    text = str(question or "").strip().lower()
    phrase_fields = [
        ("modification_type", [
            "what type of gene modification",
            "kind of gene modification",
            "modification type",
            "overexpression, knockdown",
            "deletion",
            "insertion",
        ]),
        ("organism", ["organism", "species", "experimental system are you working", "plant species"]),
        ("delivery_method", ["delivery", "delivered", "introduced", "stable transformation", "transient expression"]),
        ("expression_type", ["expression type", "stable whole-plant", "transient expression"]),
        ("readout_assay", ["readout", "assay", "measure", "rna level", "qpcr", "phenotype", "protein level"]),
        ("tissue_or_cell_type", ["tissue", "cell type", "sample type", "where should", "what system"]),
        ("target", ["target", "gene targets", "target class", "protein or tag"]),
        ("condition", ["condition", "treatment", "stress treatment"]),
    ]
    for candidate_field, phrases in phrase_fields:
        if any(phrase in text for phrase in phrases):
            fields.append(candidate_field)
    return _dedup_strings(fields)


def _candidate_specificity(query: str, profile: Optional[Dict[str, Any]]) -> int:
    """Count how many populated profile field-values appear in the query.

    The vague original request ("find protocols that can allow ...") contains
    few/no profile terms and scores low; profile-derived queries (CRISPR, tomato,
    qPCR, ...) score high. Used to order candidates most-specific-first so the
    UI default (candidate #1) carries the full accumulated intent.
    """
    if not profile:
        return 0
    low = query.lower()
    skip = {"", "not specified", "none", "unknown", "not sure", "null"}
    hits = 0
    scalar_fields = (
        "organism", "tissue_or_cell_type", "sample_type", "target",
        "modification_type", "sub_intent", "experimental_method",
        "delivery_method", "expression_type", "readout_assay", "readout",
        "condition", "gene_or_construct",
    )
    seen_vals: set = set()
    for f in scalar_fields:
        v = str(profile.get(f) or "").strip().lower()
        if v and v not in skip and v not in seen_vals and v in low:
            seen_vals.add(v)
            hits += 1
    intent_specific = profile.get("intent_specific")
    if isinstance(intent_specific, dict):
        for v in intent_specific.values():
            sv = str(v or "").strip().lower()
            if sv and sv not in skip and sv not in seen_vals and sv in low:
                seen_vals.add(sv)
                hits += 1
    return hits


def _finalize_candidate_queries(
    queries: List[str],
    original_query: str,
    profile: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Light validation for natural-language query suggestions, and guarantee the
    user's original request is one of the options. Unlike the field-complete
    `candidate_query_preserves_required_concepts`, this allows focused queries
    (each covering a subset of fields) — collective coverage is handled by the
    generator's prompt.

    Candidates are ordered most-specific-first (by profile-field coverage) so the
    UI default (candidate #1) carries the accumulated intent rather than the vague
    original request. The original is still included, just lower in the list."""
    out = [
        " ".join(str(q).split())
        for q in (queries or [])
        if _is_valid_candidate_query(q)
    ]
    # No generated queries (e.g. a transient LLM failure) — return [] so the
    # caller falls through to the rule-based generator instead of presenting the
    # vague original query as the sole, default-selected candidate.
    if not out:
        return []
    oq = " ".join(str(original_query or "").split())
    if oq and _is_valid_candidate_query(oq) and oq.lower() not in {x.lower() for x in out}:
        out.append(oq)
    out = _dedup_strings(out)
    # Stable sort by descending specificity (ties keep generator order).
    out.sort(key=lambda q: _candidate_specificity(q, profile), reverse=True)
    return out[:5]


def _candidate_queries_from_plan(
    plan: Dict[str, Any],
    profile: Dict[str, Any],
    fallback_query: str,
) -> List[str]:
    if not current_domain().can_generate_search_queries(profile):
        return []
    raw = plan.get("candidate_search_queries") if isinstance(plan, dict) else []
    llm_queries = [
        str(q).strip()
        for q in raw
        if _is_valid_candidate_query(q)
    ] if isinstance(raw, list) else []
    rule_queries = [
        query for query in current_domain().candidate_queries(profile, fallback_query=fallback_query, max_queries=5)
        if _is_valid_candidate_query(query)
    ]
    preserved_llm_queries = [
        query for query in llm_queries
        if candidate_query_preserves_required_concepts(profile, query)
    ]
    if len(preserved_llm_queries) < len(llm_queries):
        return _dedup_strings(rule_queries + preserved_llm_queries)[:5]
    return _dedup_strings(preserved_llm_queries + rule_queries)[:5]


def _is_valid_candidate_query(query: Any) -> bool:
    text = " ".join(str(query or "").split())
    if len(text) < 4:
        return False
    lowered = text.lower()
    if lowered in {"protocol", "protocols", "method", "search"}:
        return False
    # A declined answer ("I don't know", "not sure") is not a search query.
    if _is_nonanswer(text):
        return False
    if "?" in text:
        return False
    question_starts = (
        "what ",
        "which ",
        "do you ",
        "does ",
        "should ",
        "can you ",
        "are you ",
        "is ",
    )
    return not lowered.startswith(question_starts)


def _profile_goal_summary(profile: Optional[Dict[str, Any]]) -> str:
    """A short natural-language summary of the current search, used as the goal
    for new-topic detection when conversation_query isn't sent by the client."""
    if not profile:
        return ""
    parts = []
    for key in ("sub_intent", "modification_type", "experimental_method",
                "organism", "tissue_or_cell_type", "readout_assay", "condition"):
        v = str(profile.get(key) or "").strip()
        if v and v.lower() not in ("not specified", "none", "unknown", "null"):
            parts.append(v)
    # de-dup while preserving order
    return ", ".join(dict.fromkeys(parts))


_OPERATOR_EMPTY = {"", "not specified", "none", "unknown", "not sure", "null"}

# Fast-path / offline fallback for decline detection. The LLM planner
# (`user_declined_to_answer`) is the primary interpreter and handles arbitrary
# phrasing ("no clue", "you decide", ...); this list just catches the obvious
# cases instantly and keeps the feature working when the LLM is unavailable.
# IMPORTANT: must NOT include empty-state markers ("not specified", "unknown",
# "none", "n/a") — those represent unfilled fields, so matching them would
# auto-skip every empty field (e.g. organism) and never ask.
_NONANSWER_VALUES = {
    "not sure", "unsure", "no idea", "idk", "i don't know", "i dont know",
    "dont know", "don't know", "do not know", "no preference", "doesn't matter",
    "does not matter", "doesnt matter", "whatever", "skip", "no opinion",
    "not sure / flexible", "either / not sure",
}
# Moved to each Domain's `skippable_fields` (domains/biology.py, domains/chemistry.py) --
# was a hardcoded biology-only tuple applied to every domain's profile regardless.


def _is_nonanswer(text: Any) -> bool:
    t = " ".join(str(text or "").split()).strip().lower().rstrip(".!?")
    return t in _NONANSWER_VALUES


def _apply_nonanswer_skips(
    new_profile: Dict[str, Any],
    prior_profile: Optional[Dict[str, Any]],
    user_text: str,
    intent: Dict[str, Any],
    llm_declined: bool = False,
    pending_field: Optional[str] = None,
) -> None:
    """Record skipped fields so 'not sure'-type answers don't block the flow.

    Whether the user declined is decided by the LLM planner
    (`user_declined_to_answer`, which handles arbitrary phrasing) OR a small
    static list (fast path / fallback when the LLM is unavailable).

    (1) If the user declined, skip the field that was being asked. The pending
        field is the client-echoed `pending_field` (authoritative, works for
        LLM-driven clarifications) or, failing that, the rule-based pending
        clarification on the prior profile.
    (2) Scrub stray non-answer values that leaked into other fields (clean only).
    Skips persist in profile['_skipped_fields'] (carried by the client), so
    needs_clarification() won't re-ask them.
    """
    # Seed from BOTH the rebuilt profile and the carried-over one — the per-turn
    # merge/validate/normalize rebuild drops '_skipped_fields', so without the
    # prior profile a skipped field would get re-asked on the next turn.
    skipped = set(new_profile.get("_skipped_fields") or [])
    skipped |= set((prior_profile or {}).get("_skipped_fields") or [])

    # (1) User declined the pending question -> skip the field that was just asked.
    if llm_declined or _is_nonanswer(user_text):
        field_to_skip = pending_field
        if not field_to_skip:
            try:
                pending = current_domain().next_clarification(prior_profile or new_profile, intent)
            except Exception:
                pending = None
            field_to_skip = (pending or {}).get("field")
        if field_to_skip:
            # Skip both the raw and canonical form so needs_clarification() —
            # which canonicalizes the incoming field but not the skip list —
            # recognizes the skip regardless of which alias the LLM asks about.
            skipped.add(str(field_to_skip))
            try:
                skipped.add(current_domain().canonical_field(str(field_to_skip)))
            except Exception:
                pass

    # (2) Clean stray non-answer values that leaked into fields (the LLM may
    #     mis-assign the message, e.g. "I don't know" landing in organism). Null
    #     them, but do NOT skip — only the pending field (step 1) is skipped, so a
    #     non-answer in the wrong field doesn't wrongly suppress that question.
    for field in current_domain().skippable_fields:
        val = new_profile.get(field)
        if isinstance(val, str) and _is_nonanswer(val):
            new_profile[field] = None

    if skipped:
        new_profile["_skipped_fields"] = sorted(skipped)


def _fix_clarification_misassignment(
    new_profile: Dict[str, Any],
    prior_profile: Optional[Dict[str, Any]],
    user_text: str,
    pending_field: Optional[str],
) -> None:
    """Deterministic safety net for clarification answers landing in the wrong
    field (e.g. a 'readout' answer the LLM dropped into organism).

    The user's answer belongs to `pending_field` (the question that was shown).
    Remove the verbatim answer from any OTHER field where it newly appeared, and
    ensure the pending field carries it if the LLM left it empty. Only exact
    full-answer matches are moved, so multi-concept answers are left intact.
    Caller gates this to concrete (non-declined) clarification answers.
    """
    if not pending_field:
        return
    ans = " ".join(str(user_text or "").split())
    ans_low = ans.strip().lower()
    if not ans_low:
        return
    prior = prior_profile or {}

    # Remove the answer from non-pending fields where it newly appeared.
    for field in current_domain().skippable_fields:
        if field == pending_field:
            continue
        val = new_profile.get(field)
        if (
            isinstance(val, str)
            and val.strip().lower() == ans_low
            and str(prior.get(field) or "").strip().lower() != ans_low
        ):
            new_profile[field] = prior.get(field)

    # Ensure the answered field carries the answer if the LLM left it empty.
    cur = new_profile.get(pending_field)
    cur_low = str(cur or "").strip().lower()
    if not cur_low or cur_low in ("not specified", "none", "unknown", "null") or _is_nonanswer(cur):
        new_profile[pending_field] = ans


# Profile keys that hold a controlled vocabulary rather than user text. The
# free-text safety nets below must leave these alone; the domain's validate()
# owns them.
_CONTROLLED_VOCAB_FIELDS = frozenset({"intent_family", "sub_intent"})


def _preserve_operator_fields(
    new_profile: Dict[str, Any],
    prior_profile: Optional[Dict[str, Any]],
    user_text: str,
) -> None:
    """Keep AND/OR/LIKE operator values verbatim through profile normalization.

    (1) If THIS turn's message is an operator phrase ("tomato or rice") and a
        field got collapsed to one operand ("tomato"), restore the full phrase.
    (2) Carry forward an operator value from the PRIOR profile when the rebuilt
        field collapsed to one operand or went empty — unless the user genuinely
        changed that field to an unrelated value.
    Mutates new_profile in place.
    """
    if not isinstance(new_profile, dict):
        return

    # (1) Restore from the current user message.
    op, operands, _ = detect_operator(user_text or "")
    if op and operands:
        opset = {o.strip().lower() for o in operands}
        phrase = " ".join((user_text or "").split())
        for field, val in list(new_profile.items()):
            # Controlled-vocabulary fields are never user phrases, so they must
            # not be rewritten to the raw request. Chemistry's single-word values
            # ("extraction") otherwise match an operand of a query like
            # "Extraction AND derivatization of ..." and the whole sentence
            # lands in sub_intent. Biology only escaped this because its values
            # are underscored ("western_blot" never equals "western blot").
            if field in _CONTROLLED_VOCAB_FIELDS:
                continue
            if isinstance(val, str) and val.strip().lower() in opset:
                new_profile[field] = phrase

    # (2) Carry forward prior-profile operator values that got collapsed/emptied.
    for field, pval in (prior_profile or {}).items():
        if not isinstance(pval, str):
            continue
        pop, poperands, _ = detect_operator(pval)
        if not pop:
            continue
        nval = new_profile.get(field)
        if not isinstance(nval, str):
            continue
        nlow = nval.strip().lower()
        poperset = {o.strip().lower() for o in poperands}
        if nlow in poperset or nlow in _OPERATOR_EMPTY or nlow == pval.strip().lower():
            new_profile[field] = pval


# _pubmed_core_query now lives in retrievers.py (imported above as the profile-
# aware PubMed query builder) so the PubMed source is fully encapsulated behind
# the Retriever interface. Imported alias kept for existing callers/tests.


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chatbot endpoint with clarification, multi-query expansion, and re-ranking.

    Flow:
      1. Classify intent (chitchat vs. protocol search).
      2. If the query is vague and clarification hasn't been skipped, ask a follow-up.
      3. Expand the query into 3-5 related variants.
      4. Run TF-IDF search on all variants, merge and re-rank results.
      5. Optionally generate a plain-English explanation via the local LLM.
      6. Return results with a feedback prompt.
    """
    # Set the LLM provider for this request
    from claude_client import set_provider
    set_provider(req.provider)

    # Per-request Claude model override (debug-only switch: Sonnet 4.6 <-> Haiku 4.5).
    # Whitelist server-side so a client can't request an arbitrary model; anything
    # unknown clears the override and falls back to the env default (Sonnet 4.6).
    from llm_providers import set_model_override
    ALLOWED_CLAUDE_MODELS = {"claude-sonnet-4-6", "claude-haiku-4-5-20251001"}
    _model = req.model if req.model in ALLOWED_CLAUDE_MODELS else None
    set_model_override("claude", _model)

    logging.info(
        f"💬 CHAT REQUEST - Provider: {req.provider}, Model: {_model or 'default'}, "
        f"Query: {req.query[:80]}..."
    )

    # Live mode searches protocols.io directly and does not need the local index;
    # only the legacy TF-IDF ("local") mode requires it.
    if req.search_mode == "local" and not PROTOCOL_INDEX:
        raise HTTPException(status_code=503, detail="Protocol index not loaded. Run fetch_protocols.py first.")

    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query = req.query.strip()
    conversation_query = (req.conversation_query or "").strip()
    has_active_experiment_context = bool(conversation_query or req.experiment_profile)
    loop = asyncio.get_event_loop()

    # Resolve the domain for this request: a carried chemistry profile pins the
    # domain across the conversation; otherwise classify from the text (biology
    # is the default). Engine helpers read it via current_domain().
    _carried_domain = (req.experiment_profile or {}).get("intent_family")
    _domain_override = _carried_domain if _carried_domain in pinnable_domain_names() else None
    set_current_domain(route(query, conversation_query or query, override=_domain_override).name)
    domain = current_domain()
    logging.info(f"🔬 DOMAIN: {domain.name}")

    # New-topic detection runs FIRST — before the conversational handler below — so a
    # clearly-new search is never swallowed as a meta/explain answer. `is_new_search_topic`
    # is a deterministic (temp 0) classifier; the conversational handler runs the LLM at a
    # warmer temperature and can occasionally misread a fresh experiment description as a
    # "how are these ranked?" question, so it must NOT get first crack at a new topic.
    #
    # If the user switches to a clearly different search mid-conversation (e.g. answering a
    # tomato-CRISPR clarification with "western blot in mouse", or typing a whole new
    # experiment), clear the carried-over profile so the new search starts clean instead of
    # merging into the old goal. Skipped during the search-confirmation step (clicking a
    # candidate query is not a new topic) and for clarification answers (a short answer like
    # "CRISPR" must never wipe the accumulated profile). The "current goal" comes from
    # conversation_query when present, else a summary of the carried-over profile (the
    # frontend doesn't always send conversation_query).
    new_search = False
    _is_search_action = req.search_confirmed or req.search_all
    _is_refinement = _is_search_action or req.is_clarification_answer
    _topic_goal = conversation_query or _profile_goal_summary(req.experiment_profile)
    logging.debug(f"New-topic check: has_profile={bool(req.experiment_profile)}, topic_goal={bool(_topic_goal)}, is_refinement={_is_refinement}")
    if req.experiment_profile and _topic_goal and not _is_refinement and llm_is_available():
        if await loop.run_in_executor(executor, is_new_search_topic, _topic_goal, query):
            logging.info(f"🔴 NEW SEARCH DETECTED: '{query[:50]}' vs goal '{_topic_goal[:50]}'")
            new_search = True
            req.experiment_profile = None
            conversation_query = ""
            has_active_experiment_context = False

    # --- Query logging (never raises) --------------------------------------------------
    # A search is a multi-step interaction; log each step the moment it happens, correlated
    # by session_id. Doing it HERE (before clarification/selection/search) guarantees a
    # record the instant a new query is detected, even if the user abandons before results.
    _is_search_continuation = req.search_confirmed or req.search_all or req.is_clarification_answer
    if req.no_log:
        pass                                          # warmup/non-user ping — skip logging
    elif req.is_clarification_answer:
        # The user answered a clarification. The client echoes the question + options it
        # showed, so this one record holds both the question and the selection.
        query_logger.log_event(
            "clarification", session_id=req.session_id,
            question=req.pending_clarification_question,
            field=req.pending_field,
            options=req.pending_clarification_options,
            selection=req.query,
        )
    elif new_search or not req.experiment_profile:
        # A fresh search request (first message, or a detected new topic) — not a
        # continuation of an existing one.
        query_logger.log_event(
            "new_query", session_id=req.session_id,
            original_query=req.query, search_mode=req.search_mode,
        )

    # Mid-session conversational handler: when the user STILL has an active experiment
    # context (i.e. this was NOT a new topic — new-topic detection above already reset the
    # profile for a fresh search) and this isn't a search action or clarification answer,
    # let the LLM decide — and respond — for ANY conversational message: a
    # how-does-this-work / about-the-session question (answered from grounded facts +
    # profile, "these/this" resolved to what they're viewing), or chitchat (a warm reply).
    # It returns None only for a real search / refinement, which falls through to the
    # pipeline. No keyword gate — fully LLM-driven.
    if (has_active_experiment_context and not new_search and not req.is_clarification_answer
            and not (req.search_confirmed or req.search_all)):
        from claude_client import answer_session_message
        _explain = await loop.run_in_executor(
            executor, answer_session_message, query,
            req.experiment_profile, req.candidate_search_queries, req.client_view,
        )
        if _explain:
            return {
                "query": query,
                "intent": "explanation",
                "domain": domain.name,
                "experiment_intent": {},
                "experiment_profile": req.experiment_profile,  # preserve context
                "new_search": False,
                "clarification": None,
                "conversation_query": conversation_query,
                "search_query": "",
                "candidate_search_queries": req.candidate_search_queries or [],
                "reply": _explain,
                "results": [],
                "explanation": "",
                "expanded_queries": [],
                "feedback_prompt": None,
                "feedback_options": [],
                "total_protocols_indexed": len(PROTOCOL_INDEX["protocols"]) if PROTOCOL_INDEX else 0,
                "llm_info": current_llm_info(),
                "suppress_profile": True,
            }

    profile_source_query = profile_source_query_for_request(
        query=query,
        conversation_query=conversation_query,
        search_confirmed=req.search_confirmed,
        experiment_profile=req.experiment_profile,
    )
    if not profile_source_query and not (req.search_confirmed and req.experiment_profile):
        profile_source_query = query

    total_indexed = len(PROTOCOL_INDEX["protocols"]) if PROTOCOL_INDEX else 0

    llm_plan: Dict[str, Any] = {}
    rule_intent = {}
    rule_profile = {}

    # The field the user is answering (the question shown last turn). Passed to
    # the planner so a clarification answer lands in the RIGHT field instead of
    # being guessed (e.g. "phenotype" -> readout_assay, not organism).
    pending_field = None
    pending_clarification = None
    if req.is_clarification_answer and req.experiment_profile:
        try:
            # Need rule_intent for next_clarification, so extract it first
            temp_intent = domain.detect_intent(profile_source_query)
            _pending = domain.next_clarification(req.experiment_profile, temp_intent)
            pending_field = (_pending or {}).get("field")
            pending_clarification = _pending
        except Exception:
            pending_field = None
            pending_clarification = None
    # The client echoes the field of the clarification it last displayed. This is
    # authoritative — the rule-based next_clarification above returns None for
    # LLM-driven clarifications (e.g. "general protocol search"), so without this
    # a declined answer ("I don't know") would never be recorded and the same
    # question would loop forever.
    if req.is_clarification_answer and req.pending_field and not pending_field:
        pending_field = req.pending_field

    # If user is asking why we need a clarification field, generate LLM explanation
    if req.is_clarification_answer and pending_field and is_asking_about_clarification_reason(query):
        try:
            # Prefer the exact question + chips the client is showing (works for
            # BOTH rule- and LLM-generated clarifications); fall back to the
            # rule-based reconstruction only if the client didn't echo them.
            q_text = (req.pending_clarification_question or "").strip()
            q_opts = [str(o) for o in (req.pending_clarification_options or []) if str(o).strip()]
            if not q_text and pending_clarification:
                q_text = pending_clarification.get("question", "")
            if not q_opts and pending_clarification:
                q_opts = list(pending_clarification.get("options", []))

            explanation = await loop.run_in_executor(
                executor,
                lambda: generate_clarification_explanation(pending_field, q_opts),
            )

            # Build response
            exp_prof = dict(req.experiment_profile) if req.experiment_profile else {}
            conv_query = conversation_query if conversation_query else ""

            return {
                "query": query,
                "intent": "clarification",
                "domain": domain.name,
                "experiment_intent": {},
                "experiment_profile": exp_prof,
                "new_search": False,
                "clarification": {
                    "field": pending_field,
                    "question": q_text,
                    "options": q_opts,
                },
                "conversation_query": conv_query,
                "search_query": "",
                "candidate_search_queries": [],
                "missing_fields": [pending_field],
                "reply": f"{explanation}\n\nYou can select \"Not sure\" to skip this, or choose an option above.",
                "results": [],
                "explanation": "",
                "expanded_queries": [],
                "feedback_prompt": None,
                "feedback_options": [],
                "total_protocols_indexed": len(PROTOCOL_INDEX["protocols"]) if PROTOCOL_INDEX else 0,
                # This is a side-explanation of why we ask a field, not a new
                # clarification turn — tell the client not to re-render the big
                # experiment profile card underneath it.
                "suppress_profile": True,
            }
        except Exception as e:
            logging.warning(f"Could not generate clarification explanation: {e}; continuing with normal flow")

    # LLM-first approach: try LLM, fall back to rule-based on failure
    if not req.search_confirmed and llm_is_available():
        llm_plan = await loop.run_in_executor(
            executor,
            lambda: domain.analyze_request(
                user_query=query,
                conversation_query=conversation_query,
                previous_profile=req.experiment_profile,
                pending_field=pending_field,
            ),
        )

    # If LLM succeeded, use LLM results; otherwise fall back to rule-based
    if llm_plan and llm_plan.get("intent_family"):
        # LLM succeeded: use LLM results as primary
        logging.debug(f"Using LLM profile extraction for query: {query[:60]}...")
        experiment_intent = domain.normalize_llm_intent(llm_plan, {}, profile_source_query)
        experiment_profile = llm_plan.get("experiment_profile", {})
    else:
        # LLM failed or not available: use rule-based
        logging.debug(f"Falling back to rule-based profile extraction for query: {query[:60]}...")
        rule_intent = domain.detect_intent(profile_source_query)
        rule_profile = domain.build_profile(
            profile_source_query,
            previous_profile=req.experiment_profile,
        )
        experiment_intent = rule_intent
        experiment_profile = rule_profile
    experiment_intent, experiment_profile = domain.validate(
        profile_source_query,
        experiment_intent,
        experiment_profile,
    )
    experiment_intent, experiment_profile = domain.normalize(
        profile_source_query,
        experiment_intent,
        experiment_profile,
    )
    experiment_intent, experiment_profile = domain.validate(
        profile_source_query,
        experiment_intent,
        experiment_profile,
    )
    # Rule-based safeguard: the LLM/normalizers sometimes collapse an operator
    # value ("tomato or rice" -> "tomato"). Re-inject the verbatim operator phrase
    # from the user's current message and carry forward operator values from the
    # prior profile, so AND/OR/LIKE survive the clarification flow.
    _preserve_operator_fields(experiment_profile, req.experiment_profile, query)
    # Non-answer handling: if the user answered a clarification with "not sure" /
    # "I don't know" / etc., skip that field (don't store it, don't re-ask) even
    # when it's required, then carry the skip forward so the flow advances.
    _declined = (bool(llm_plan.get("user_declined_to_answer")) if llm_plan else False) or _is_nonanswer(query)
    _apply_nonanswer_skips(
        experiment_profile, req.experiment_profile, query, experiment_intent,
        llm_declined=_declined, pending_field=pending_field,
    )
    # Safety net: a concrete clarification answer must land in the field that was
    # asked (pending_field), not be guessed into the wrong field by the LLM.
    if req.is_clarification_answer and not _declined:
        _fix_clarification_misassignment(
            experiment_profile, req.experiment_profile, query, pending_field,
        )
    # Surface a stress-assay growth_stage ("seedlings") into Sample type / System
    # so it shows in the grid. Runs AFTER the misassignment fixer, which otherwise
    # strips the value from those non-pending fields.
    domain.surface_extras(experiment_profile)
    structured_query = domain.build_search_query(experiment_profile, profile_source_query)
    conversation_query = conversation_query or query

    if not req.search_confirmed:
        next_action = str(llm_plan.get("next_action") or "").strip()
        logging.info(f"📊 NEXT_ACTION: {next_action} | Missing fields: {llm_plan.get('missing_fields', [])}")
        if should_respond_as_chitchat(has_active_experiment_context, experiment_intent, next_action):
            from claude_client import generate_chitchat_response
            chitchat_reply = generate_chitchat_response(query)
            return {
                "query": query,
                "intent": "chitchat",
                "domain": domain.name,
                "experiment_intent": {},
                "experiment_profile": None,  # Don't store profile for chitchat (prevents new_search false positives)
                "new_search": False,
                "clarification": None,
                "conversation_query": "",
                "search_query": "",
                "candidate_search_queries": [],
                "reply": chitchat_reply,
                "results": [],
                "explanation": "",
                "expanded_queries": [],
                "feedback_prompt": None,
                "feedback_options": [],
                "total_protocols_indexed": total_indexed,
                "llm_info": current_llm_info(),
        }

        clarification = None
        if not req.skip_clarification:
            clarification = domain.next_clarification(experiment_profile, experiment_intent)
        if not clarification and next_action == "ask_clarification":
            clarification = _clarification_from_plan(llm_plan, experiment_profile)
        elif clarification and not clarification.get("options") and next_action == "ask_clarification":
            # The rule-based pass knows WHICH field is missing, but for open-ended
            # fields it can only offer a static, option-less question ("What are you
            # starting from?"). The planner, seeing the actual request, can ask the
            # same thing with concrete clickable answers (for a gold-nanoparticle
            # synthesis: Turkevich / Brust-Schiffrin / seed-mediated / ...). Prefer
            # the richer question when the rule-based one has nothing to click.
            # No-op for biology, whose rule-based questions always carry options.
            _llm_clar = _clarification_from_plan(llm_plan, experiment_profile)
            if _llm_clar and _llm_clar.get("options"):
                clarification = _llm_clar
        if (
            not clarification
            and not req.skip_clarification
            and not has_active_experiment_context
            and not llm_plan
            and is_vague_query(query)
        ):
            clarification_text = get_clarification_question(query)
            clarification = {
                "field": "general",
                "question": clarification_text,
                "options": [],
            }
        if clarification:
            return {
                "query": query,
                "intent": "clarification",
                "domain": domain.name,
                "experiment_intent": experiment_intent,
                "experiment_profile": experiment_profile,
                "new_search": new_search,
                "clarification": clarification,
                "conversation_query": conversation_query,
                "search_query": structured_query,
                "candidate_search_queries": [],
                "missing_fields": llm_plan.get("missing_fields", []),
                "reply": clarification["question"],
                "results": [],
                "explanation": "",
                "expanded_queries": [],
                "feedback_prompt": None,
                "feedback_options": [],
                "total_protocols_indexed": total_indexed,
                "llm_info": current_llm_info(),
            }

        if not llm_plan and not has_active_experiment_context:
            intent = await loop.run_in_executor(executor, classify_intent, query)
            if intent["intent"] == "chitchat":
                from claude_client import generate_chitchat_response
                chitchat_reply = generate_chitchat_response(query)
                return {
                    "query": query,
                    "intent": "chitchat",
                    "domain": domain.name,
                    "experiment_intent": {},
                    "experiment_profile": None,  # Don't store profile for chitchat
                "new_search": False,
                    "clarification": None,
                    "conversation_query": "",
                    "search_query": "",
                    "candidate_search_queries": [],
                    "reply": chitchat_reply,
                    "results": [],
                    "explanation": "",
                    "expanded_queries": [],
                    "feedback_prompt": None,
                    "feedback_options": [],
                    "total_protocols_indexed": total_indexed,
                    "llm_info": current_llm_info(),
                }

        # Prefer Claude-generated natural-language suggestions (focused angles
        # that collectively cover every field + include the original query).
        # Fall back to the rule-based generator when Claude is unavailable.
        candidate_queries: List[str] = []
        if llm_is_available() and domain.can_generate_search_queries(experiment_profile):
            nl_queries = await loop.run_in_executor(
                executor,
                generate_natural_search_queries,
                experiment_profile,
                conversation_query or query,
                5,
            )
            candidate_queries = _finalize_candidate_queries(nl_queries, conversation_query or query, experiment_profile)
        if not candidate_queries:
            candidate_queries = _candidate_queries_from_plan(
                llm_plan,
                experiment_profile,
                structured_query,
            )
        if not candidate_queries:
            clarification = (
                domain.next_clarification(experiment_profile, experiment_intent)
                or _clarification_from_plan(llm_plan, experiment_profile)
            )
            # Hardcoded last-resort clarification — but ONLY if we haven't already
            # asked experimental_method. Re-asking a skipped field loops forever
            # when the user keeps declining, so once it's skipped we stop asking
            # and fall through to a broad search with whatever we have.
            _skipped = set(experiment_profile.get("_skipped_fields") or [])
            if not clarification and "experimental_method" not in _skipped:
                clarification = {
                    "field": "experimental_method",
                    "question": "What experimental task are you trying to run?",
                    "options": [
                        "gene overexpression",
                        "gene knockdown",
                        "PCR/qPCR",
                        "protein purification",
                        "microscopy",
                    ],
                }
            if clarification:
                return {
                    "query": query,
                    "intent": "clarification",
                    "domain": domain.name,
                "domain": domain.name,
                    "experiment_intent": experiment_intent,
                    "experiment_profile": experiment_profile,
                    "new_search": new_search,
                    "clarification": clarification,
                    "conversation_query": conversation_query,
                    "search_query": structured_query,
                    "candidate_search_queries": [],
                    "missing_fields": llm_plan.get("missing_fields", []),
                    "reply": clarification["question"],
                    "results": [],
                    "explanation": "",
                    "expanded_queries": [],
                    "feedback_prompt": None,
                    "feedback_options": [],
                    "total_protocols_indexed": total_indexed,
                }
            # No more questions to ask (user declined everything). Rather than
            # loop, do a best-effort broad search using the structured profile
            # terms, or the raw request as a last resort.
            fallback_q = structured_query or " ".join(str(conversation_query or query).split())
            candidate_queries = [fallback_q] if fallback_q else []
        # The system proposed candidate queries. Captured verbatim (no results yet); the
        # later search record shares this session_id.
        if not req.no_log:
          query_logger.log_event(
            "suggestions", session_id=req.session_id,
            original_query=req.query,
            suggested_queries=candidate_queries,
            profile=experiment_profile,
        )
        return {
            "query": query,
            "intent": "query_selection",
            "domain": domain.name,
            "experiment_intent": experiment_intent,
            "experiment_profile": experiment_profile,
            "new_search": new_search,
            "clarification": None,
            "conversation_query": conversation_query,
            "search_query": structured_query,
            "candidate_search_queries": candidate_queries,
            "missing_fields": llm_plan.get("missing_fields", []),
            "reply": "Choose a search query to run, edit one, or search all suggested queries.",
            "results": [],
            "explanation": "",
            "expanded_queries": [],
            "feedback_prompt": None,
            "feedback_options": [],
            "total_protocols_indexed": total_indexed,
        }

    selected_query = (req.selected_search_query or query).strip()
    candidate_queries = [q for q in (req.candidate_search_queries or []) if str(q).strip()]
    search_queries = candidate_queries if req.search_all and candidate_queries else [selected_query]
    search_queries = _dedup_strings([str(q) for q in search_queries if str(q).strip()])
    if not search_queries:
        search_queries = [structured_query]
    structured_query = " | ".join(search_queries) if req.search_all else search_queries[0]

    # Confirmed search. Either the live concept-expansion pipeline (default) or
    # the legacy local TF-IDF pipeline.
    # The user's natural-language request is used only for "like/such as"
    # relaxation detection in the closeness ranker — not for retrieval. Also feed
    # any LIKE-bearing profile field values (e.g. organism "like tomato") so the
    # ranker's existing relaxation logic sees them without changing the ranker.
    like_field_values = [
        str(v) for v in (experiment_profile or {}).values()
        if isinstance(v, str) and is_like(v)
    ]
    like_query = " ".join(filter(None, [conversation_query, query, *like_field_values])).strip()
    concepts: Dict[str, Any] = {}
    expansions: Dict[str, List[str]] = {}
    sentence_variants: List[str] = []
    # How many to surface per source (protocols.io + PubMed). The upstream
    # protocols.io search is bounded by the top_k we pass it, so fetch at least
    # `per_provider` here — otherwise selecting "10 per source" would still cap
    # protocols.io at top_k (5) while PubMed returned 10.
    # Asymmetric blend mix (default 8 protocols.io + 2 PubMed).
    protocols_k, pubmed_k = _parse_blend_mix(req.blend_mix)
    per_provider = protocols_k          # protocols.io slots + pool sizing
    fetch_k = max(req.top_k, protocols_k)
    # Optional LLM re-ranker (default ON, matching prod; per-request override via req.enable_reranker).
    _rerank_on = reranker.is_enabled(req.enable_reranker)
    _rerank_query = (conversation_query or query or structured_query or "").strip()
    if _rerank_on:
        logging.info(f"🔀 Re-ranker ON ({reranker.RERANKER_MODEL}) for local search")
    # Experiment #2 (default OFF): re-rank protocols.io + PubMed jointly so the LLM
    # vets PubMed. Needs a wider protocol pool than the final top-K so the joint
    # re-rank can drop weak PubMed and backfill protocols.
    _combined_on = _rerank_on and reranker.combined_pubmed_enabled(req.rerank_combined)
    if _combined_on:
        fetch_k = max(fetch_k, per_provider + pubmed_k + 2)
    if req.search_mode == "live":
        try:
            live = await loop.run_in_executor(
                executor,
                run_live_candidate_searches,
                search_queries,
                fetch_k,
                experiment_profile,
                like_query,
            )
            results = live["results"]
            concepts = live["concepts"]
            expansions = live["expansions"]
            expanded = live["expanded"]
            sentence_variants = live["sentence_variants"]
            logging.info(f"Live confirmed search '{structured_query}': {len(expanded)} probes -> {len(results)} ranked results")
        except Exception as e:
            logging.warning(f"Live search failed ({e}); falling back to local index.")
            if not PROTOCOL_INDEX:
                raise HTTPException(status_code=503, detail=f"Live search failed and no local index: {e}")
            local = await loop.run_in_executor(
                executor,
                run_local_candidate_searches,
                search_queries,
                fetch_k,
                experiment_profile,
                like_query,
                _rerank_on,
                _rerank_query,
            )
            results = local["results"]
            expanded = local["expanded"]
    else:
        local = await loop.run_in_executor(
            executor,
            run_local_candidate_searches,
            search_queries,
            fetch_k,
            experiment_profile,
            like_query,
            _rerank_on,
            _rerank_query,
        )
        results = local["results"]
        expanded = local["expanded"]
        logging.info(f"Local confirmed search '{structured_query}' expanded into {len(expanded)} queries")

    # Step 4.5: blend PubMed literature into the protocols.io results.
    # Pull `results_per_provider` from each source, then interleave by a unified
    # title>body relevance score (protocols.io internal order is preserved; see
    # blend_ranking.py). PubMed is best-effort — any failure leaves the
    # protocols.io results untouched. `per_provider`/`fetch_k` computed above.
    for r in results:
        r.setdefault("source", "protocols.io")
    protocols_top = results[:per_provider]
    # PubMed: ONE Haiku call writes a precise+broad query pair for EVERY selected angle
    # (they're facets of the same experiment, so a round-trip each would be waste). PubMed
    # ANDs every term, so the raw sentence starves and the old concept-token core collapsed
    # to generic words ("mouse", "extraction") — the Haiku query keeps the discriminators
    # (species, technique, sample, target). Each angle is searched SEQUENTIALLY to stay under
    # PubMed's rate limit, dropping down a 0-hit ladder (precise -> broad -> relaxed ->
    # token core) so a rare topic still returns something; then merged + de-duped.
    pubmed_queries = [q for q in (search_queries or []) if str(q).strip()] or [structured_query]

    def _pubmed_core_from_query(q: str) -> str:
        c = extract_concepts(q)
        toks, seen = [], set()
        for group in ("organisms", "methods", "goals"):
            for term in c.get(group, []):
                for t in str(term).lower().split():
                    if t not in seen:
                        seen.add(t)
                        toks.append(t)
        return " ".join(toks[:6]) or q

    pubmed_results, _pm_seen = [], set()
    pubmed_query_debug: List[Dict[str, Any]] = []  # what was actually sent to PubMed
    # The literature lane is per-domain (biology: PubMed; chemistry: Europe PMC,
    # fetched by the registry loop below). A domain that does not declare
    # "pubmed" skips this lane entirely rather than searching a source that
    # does not cover its discipline.
    _pubmed_lane_on = "pubmed" in (current_domain().paper_sources or ())
    if not _pubmed_lane_on and pubmed_k > 0:
        logging.info(f"PubMed lane off: domain '{current_domain().name}' pairs "
                     f"protocols.io with {list(current_domain().paper_sources)}.")
    if pubmed_k > 0 and _pubmed_lane_on:
        _pm_fetch = max(pubmed_k, 5)   # fetch a small pool per query, rank down to pubmed_k
        # ONE Haiku call builds the PubMed query variants for EVERY selected angle (they're
        # facets of the same experiment, so a round-trip each would be pure waste). An angle
        # that ALTERNATES ("rice OR potato") yields one variant per alternative — each is
        # searched separately and merged, because a single OR'd query lets the bigger
        # literature take every slot. Falls back to the token core per-angle if the model
        # gave nothing for it.
        try:
            _pm_built = await loop.run_in_executor(executor, build_pubmed_queries, pubmed_queries)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Batched PubMed query build failed ({e}); using token cores.")
            _pm_built = [[] for _ in pubmed_queries]
        for _q, _variants in zip(pubmed_queries, _pm_built):
            # Haiku's query keeps the species/technique/sample/target the old token core
            # dropped; a 0-hit result relaxes down the ladder to the token core.
            _info: Dict[str, Any] = {}
            try:
                _hits = await loop.run_in_executor(
                    executor, search_pubmed_fanout, _q, _pm_fetch,
                    _pubmed_core_from_query(_q), _info, _variants,
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(f"PubMed search failed for '{_q[:60]}' ({e}).")
                _hits = []
            for r in _hits:
                r.setdefault("source", "pubmed")
                key = r.get("pmid") or r.get("doi") or r.get("title")
                if key and key not in _pm_seen:
                    _pm_seen.add(key)
                    pubmed_results.append(r)
            _sent = _info.get("query", "")
            _nvar = _info.get("variants", 1)
            pubmed_query_debug.append({"selected": _q, "core": _sent,
                                       "variants": _nvar, "hits": len(_hits)})
            logging.info(f"PubMed query '{_sent}' ({_nvar} variant(s), rung {_info.get('rung')}, "
                         f"from '{_q[:45]}') -> {len(_hits)} hits")
        # Keep the best candidates from the FULL merged pool by profile relevance —
        # strongest across ALL selected queries, not just whichever was searched first.
        # When the joint re-rank (#2) is on we keep a WIDER PubMed pool than the blend's
        # pubmed_k: the re-ranker only promotes papers that earn a top-K slot, so giving
        # it more to choose from can only help. The lexical-blend fallback still uses
        # exactly pubmed_k (see results_by_source below).
        _pm_keep = max(pubmed_k, reranker.COMBINED_PUBMED_CANDIDATES) if _combined_on else pubmed_k
        _pm_pool = len(pubmed_results)
        for r in pubmed_results:
            r.setdefault("source", "pubmed")
        # Trim ROUND-ROBIN across the alternatives, ranking by profile relevance WITHIN each.
        # A plain relevance trim would undo the fan-out: score "rice OR potato" results all
        # together, keep 5, and one organism can still take every slot. With one alternative
        # (the common case) this is identical to the old profile-ranked trim.
        _rank_within = ((lambda rs: domain.rank(experiment_profile, rs, top_k=len(rs)))
                        if experiment_profile else None)
        pubmed_results = balanced_trim(pubmed_results, _pm_keep, _rank_within)
        logging.info(f"PubMed merged pool {_pm_pool} -> kept {len(pubmed_results)} "
                     f"(pubmed_k={pubmed_k}, rerank candidates={_pm_keep}).")
    elif _pubmed_lane_on:
        logging.info("PubMed disabled for this request (blend mix has 0 PubMed slots).")
    # Relevance axis for blending = the user's combined search intent.
    pubmed_query = structured_query
    # The lexical blend keeps its fixed pubmed_k slots; the joint re-rank below gets the
    # wider `pubmed_results` pool and decides for itself how many papers deserve a slot.
    results_by_source = {"protocols.io": protocols_top, "pubmed": pubmed_results[:pubmed_k]}
    # Registry-driven extra sources: the two incumbents (protocols.io, PubMed) have
    # bespoke fetch above; ANY other registered+enabled Retriever is fetched here
    # and folded into the blend automatically. Adding a third source is therefore a
    # Retriever class + register() — no edit to this orchestrator.
    ctx = RetrievalContext(
        queries=list(search_queries or []),
        structured_query=structured_query,
        profile=experiment_profile,
        k=per_provider,
        raw_query=conversation_query or query,
        search_mode=req.search_mode,
    )
    for name, retriever in RETRIEVERS.items():
        if name in results_by_source or not retriever.is_enabled():
            continue
        try:
            results_by_source[name] = await loop.run_in_executor(executor, retriever.retrieve, ctx)
        except Exception as e:
            logging.warning(f"Retriever '{name}' failed ({e}); skipping.")
    # N-source blend across every source on one comparable relevance axis.
    _combined_applied = False
    # Paper-bearing sources that the joint re-rank may vet alongside protocols.
    # PubMed is always present; europepmc appears ONLY for domains that enable it
    # (chemistry), so for a biology request `_paper_pool == pubmed_results` and the
    # subset test is the same one as before -- identical branch, identical pool.
    _paper_pool = list(pubmed_results) + list(results_by_source.get("europepmc") or [])
    if (_combined_on and _paper_pool
            and set(results_by_source) <= {"protocols.io", "pubmed", "europepmc"}):
        # #2: re-rank protocols.io (wide pool) + paper candidates JOINTLY so the LLM
        # vets the papers (weak ones get dropped) and backfills protocols,
        # instead of blending them in by a lexical score the re-ranker never sees.
        _combined_pool = list(results[:per_provider + pubmed_k]) + list(_paper_pool)
        logging.info(f"🔀 Combined re-rank (protocols + papers, {len(_combined_pool)} cands) ON")
        results = await loop.run_in_executor(
            executor,
            lambda: reranker.rerank(_rerank_query, _combined_pool,
                                    top_k=per_provider + pubmed_k,
                                    model=reranker.RERANKER_MODEL),
        )
        _combined_applied = True
    else:
        results = blend_sources(pubmed_query, results_by_source)

    # Apply profile ranking to ALL results to add why_it_matches explanations + safety gating.
    # Default path: profile_score sets the final order. When the #2 combined re-rank ran,
    # PERSIST its LLM order (what the eval measures) — use profile ranking only to annotate
    # and drop gated items, not to re-sort the results back to profile order.
    if experiment_profile:
        _combined_order = [r.get("id") for r in results] if _combined_applied else None
        results = domain.rank(experiment_profile, results, top_k=len(results))
        if _combined_order is not None:
            _by_id = {r.get("id"): r for r in results}
            results = [_by_id[i] for i in _combined_order if i in _by_id]

    # Fetch full text + extract the Methods section for the PubMed papers that
    # actually surfaced (bounded by per_provider), in parallel.
    surfaced_pubmed = [r for r in results if r.get("source") == "pubmed"]
    if surfaced_pubmed:
        def _attach_methods(paper: Dict[str, Any]) -> None:
            try:
                fulltext = fetch_fulltext(paper.get("pmid", ""), paper.get("doi", ""))
                if fulltext:
                    paper["methods"] = extract_methods(fulltext)
            except Exception as e:
                logging.debug(f"methods fetch failed for {paper.get('pmid')}: {e}")
        await asyncio.gather(*[
            loop.run_in_executor(executor, _attach_methods, p) for p in surfaced_pubmed
        ])

    # Step 5: Optionally explain the top results using the local LLM.
    # Skip entirely when Ollama is unreachable — otherwise explain_matches would
    # spend ~15s on doomed retries and stall the request.
    explanation = ""
    if req.explain and results and llm_is_available():
        explanation = await loop.run_in_executor(
            executor, explain_matches, structured_query, results[:3]
        )

    # The search ran: the queries the user selected, the PubMed queries built, and the
    # results in the order shown. Never raises — see query_logger.
    if not req.no_log:
      query_logger.log_event(
        "search", session_id=req.session_id,
        original_query=req.query,
        conversation_query=conversation_query,
        selected_queries=req.candidate_search_queries,
        pubmed_queries=pubmed_query_debug,
        profile=experiment_profile,
        search_mode=req.search_mode,
        results=results,
    )

    return {
        "query": query,
        "intent": "search",
        "domain": domain.name,
        "experiment_intent": experiment_intent,
        "experiment_profile": experiment_profile,
        "new_search": new_search,
        "clarification": None,
        "conversation_query": conversation_query,
        "search_query": structured_query,
        "candidate_search_queries": candidate_queries,
        "reply": None,
        "explanation": explanation,
        "results": results,
        "expanded_queries": expanded,
        "pubmed_queries": pubmed_query_debug,
        "concepts": concepts,
        "expansions": expansions,
        "sentence_variants": sentence_variants,
        "search_mode": req.search_mode,
        "llm_provider": (_llm_info := current_llm_info())["provider"],
        "llm_model": _llm_info["model"] if _llm_info["available"] else None,
        "feedback_prompt": "Were these results relevant? I can help narrow the search.",
        "feedback_options": [
            "Narrow by organism (e.g. plant, human, mouse)",
            "Narrow by technique (e.g. qPCR, western blot, CRISPR)",
            "Narrow by sample type (e.g. tissue, cell line, bacteria)",
            "Narrow by experimental goal (e.g. extraction, detection, quantification)",
        ],
        "total_protocols_indexed": total_indexed,
    }


@app.get("/")
async def root():
    # Open the chat UI on the bare URL instead of showing API JSON.
    return RedirectResponse(url="/chat.html")


# --- Query-log viewer (debug tool) -------------------------------------------------
# Reads back the query logs, which contain users' raw queries. Access is controlled by the
# server, so the viewer never prompts for anything:
#   * QUERY_LOG_VIEWER_OPEN=1  -> open, no token (frictionless; protect the app at the edge,
#                                 e.g. Cloudflare Access, since anyone who reaches it can read).
#   * QUERY_LOG_VIEWER_TOKEN=x -> require ?token=x (use if the app itself is public).
#   * neither set              -> 404, feature disabled (safe default).
def _check_log_token(token: Optional[str]) -> None:
    if os.getenv("QUERY_LOG_VIEWER_OPEN", "").strip() in ("1", "true", "True", "yes"):
        return
    expected = os.getenv("QUERY_LOG_VIEWER_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=404, detail="Not found.")   # feature disabled
    if not token or token != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing log viewer token.")


@app.get("/debug/query_logs")
async def query_log_dates(token: Optional[str] = None):
    """List available log dates (newest first)."""
    _check_log_token(token)
    return {"dates": query_logger.list_dates()}


@app.get("/debug/query_logs/{date}")
async def query_log_day(date: str, token: Optional[str] = None):
    """All records for one day, grouped into conversations by session_id (order preserved)."""
    _check_log_token(token)
    records = query_logger.read_records(date)
    sessions: Dict[str, Dict[str, Any]] = {}
    loose: List[Dict[str, Any]] = []
    order: List[str] = []
    for r in records:
        sid = r.get("session_id")
        if not sid:
            loose.append(r)
            continue
        if sid not in sessions:
            sessions[sid] = {"session_id": sid, "first_ts": r.get("ts"), "events": []}
            order.append(sid)
        sessions[sid]["events"].append(r)
        sessions[sid]["last_ts"] = r.get("ts")
    return {
        "date": date,
        "record_count": len(records),
        "sessions": [sessions[s] for s in order] + (
            [{"session_id": None, "events": loose}] if loose else []),
    }


# --- Deployment identity -----------------------------------------------------
# BUILD_SHA / BUILD_DATE are baked into the image at build time (see Dockerfile).
# Locally they're unset, so we fall back to the git checkout — that way the debug
# panel shows something truthful in dev instead of "unknown", and the `source`
# field says which one you're looking at.
BUILD_SHA = os.getenv("BUILD_SHA", "").strip()
BUILD_DATE = os.getenv("BUILD_DATE", "").strip()


@lru_cache(maxsize=1)
def get_build_info() -> Dict[str, Any]:
    sha, built_at, source = BUILD_SHA, BUILD_DATE, "image"
    if not sha:
        source = "git"
        try:
            import subprocess
            root = Path(__file__).resolve().parent.parent
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=root,
                stderr=subprocess.DEVNULL, timeout=3).decode().strip()
            if not built_at:
                built_at = subprocess.check_output(
                    ["git", "log", "-1", "--format=%cI"], cwd=root,
                    stderr=subprocess.DEVNULL, timeout=3).decode().strip()
        except Exception:  # noqa: BLE001 — never let version reporting break a request
            sha, source = "unknown", "unavailable"
    return {"sha": sha or "unknown", "built_at": built_at or "", "source": source}


@app.get("/protocol_text/{protocol_id}")
def protocol_text(protocol_id: int):
    """Full text of one corpus protocol, for the downloadable results report.

    The report includes each Protocols.io result's complete steps so a reader
    can interrogate the protocol (or hand it to an LLM) without leaving the
    PDF. Reads straight from the corpus JSON baked into the image; papers
    (PubMed / Europe PMC) have no entry here and the frontend skips them.
    """
    from protocol_rag import _draftjs_to_text
    path = PROTOCOLS_DATA_DIR / f"{protocol_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="protocol not found")
    with open(path) as f:
        p = json.load(f)
    return {
        "id": p.get("id"),
        "title": p.get("title") or "",
        "uri": p.get("uri") or "",
        "doi": p.get("doi") or "",
        "materials": _draftjs_to_text(p.get("materials_text"))[:4000],
        "guidelines": _draftjs_to_text(p.get("guidelines"))[:2000],
        "steps": [s for s in (p.get("steps") or []) if s],
    }


@app.get("/health")
def health_check():
    # Note: we intentionally do NOT probe Ollama here. Ollama is a local-dev-only
    # provider; on hosted deploys (Claude/OpenAI) there is no Ollama server, so the
    # probe would just block until it times out (~3.8s) on every page load.
    storage_ok = SESSIONS_DIR.exists()
    status = "healthy" if storage_ok else "degraded"

    return {
        "status": status,
        "storage_dir": str(SESSIONS_DIR),
        "storage_ready": storage_ok,
        "message": "Application is running",
        "build": get_build_info(),
    }


@app.get("/sse")
async def sse(session_id: str = Query(default=None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return EventSourceResponse(event_generator(session_id))


async def event_generator(session_id: str):
    # Reuse the queue created by the POST endpoint, or create one if SSE connects first.
    if session_id not in update_queues:
        update_queues[session_id] = asyncio.Queue()
    queue = update_queues[session_id]
    try:
        while True:
            data = await queue.get()
            if isinstance(data, dict) and "final_output" in data:
                yield {"event": "message", "data": json.dumps(data)}
                break
            elif isinstance(data, dict) and "error" in data:
                yield {"event": "message", "data": json.dumps(data)}
                break
            else:
                yield {"event": "message", "data": json.dumps({"update": data})}
    finally:
        update_queues.pop(session_id, None)


def _thread_safe_send_update(session_id: str, message: Any):
    """
    Send an update from a background thread to the SSE queue on the main event loop.
    Thread-safe: uses call_soon_threadsafe to schedule the put on the correct loop.
    """
    queue = update_queues.get(session_id)
    if queue and main_loop:
        main_loop.call_soon_threadsafe(queue.put_nowait, message)


def save_uploaded_file(file_info: Dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        f.write(file_info["content"])
    return destination


@app.get("/fetch_backend_mode")
async def fetch_backend_mode():
    return {
        "mode": "local_ollama_rag",
        "default_execution_strategy": get_default_execution_strategy(),
        "available_execution_strategies": ["agentic", "prompt_based"],
        "external_search_enabled": False,
        "providers": ["ollama"],
        "data_leaves_machine": False,
        "build": get_build_info(),
        "models": {
            "general": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            "reranker": reranker.RERANKER_MODEL,
            "pubmed_query": os.getenv("PUBMED_QUERY_MODEL", "claude-haiku-4-5"),
        },
    }


@app.get("/ollama_status")
async def ollama_status():
    return check_ollama_health()


# Serve the static frontend from the same origin for single-container deploys
# (Cloud Run, HF Spaces, Docker). All API routes above are registered first and
# take precedence; any other path falls through to the static files. The chat
# UI is at /chat.html. Harmless locally and on the Render split deploy (the
# frontend is a separate service there, but mounting its files here is fine).
from fastapi.staticfiles import StaticFiles
from pathlib import Path as _Path

_FRONTEND_DIR = _Path(__file__).resolve().parent.parent / "protocolnerd-website"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
