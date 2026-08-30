from __future__ import annotations

import html as _html
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import claude_client as llm_client

log = logging.getLogger(__name__)


def _draftjs_to_text(raw: Any) -> str:
    """Extract clean plain text from a rich-text field.

    Handles all three formats present in the corpus: Draft.js JSON (older
    protocols.io), HTML (newer protocols.io), and plain strings. This keeps the
    index consistent even though protocols.io changed its rich-text format over
    time — otherwise HTML fields would leak <p>/style/etc. tags into the TF-IDF.
    """
    if not raw:
        return ""
    if not isinstance(raw, str):
        return str(raw)
    stripped = raw.strip()
    # Draft.js JSON
    if stripped.startswith("{") and "blocks" in stripped:
        try:
            data = json.loads(stripped)
            texts = [b.get("text", "") for b in data.get("blocks", []) if b.get("text")]
            return " ".join(texts)
        except Exception:
            pass
    # HTML or plain: strip any tags, unescape entities, collapse whitespace.
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", stripped))).strip()


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _protocol_to_text(p: Dict[str, Any]) -> str:
    """Combine all searchable fields into one text blob. Title repeated for weight."""
    steps_text = " ".join(p.get("steps") or [])
    parts = [
        (p.get("title") or "") * 3,   # boost title weight
        _draftjs_to_text(p.get("description")),
        _draftjs_to_text(p.get("guidelines")),
        _draftjs_to_text(p.get("before_start")),
        _draftjs_to_text(p.get("materials_text")),
        steps_text,
        " ".join(p.get("keywords") or []),
    ]
    return " ".join(t for t in parts if t).strip()


def build_protocol_index(data_dir: Path) -> Dict[str, Any]:
    """
    Load all protocol JSON files and build a TF-IDF index.
    Returns a dict with protocols list, vectorizer, and matrix.
    """
    protocol_files = sorted(data_dir.glob("*.json"))
    if not protocol_files:
        raise ValueError(f"No protocol JSON files found in {data_dir}")

    protocols: List[Dict[str, Any]] = []
    corpus: List[str] = []

    for fpath in protocol_files:
        try:
            p = json.loads(fpath.read_text(encoding="utf-8"))
            protocols.append(p)
            corpus.append(_protocol_to_text(p))
        except Exception as e:
            log.warning(f"Skipping {fpath.name}: {e}")

    log.info(f"Building TF-IDF index over {len(protocols)} protocols...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=80000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)

    # Build a separate title-only index.
    # A keyword appearing in a protocol's title is a much stronger relevance
    # signal than it appearing in the steps or description — just like the
    # protocols.io browser ranks title matches first.
    title_corpus = [(p.get("title") or "") for p in protocols]
    title_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=40000,
        sublinear_tf=True,
    )
    title_matrix = title_vectorizer.fit_transform(title_corpus)
    log.info("Index ready.")

    return {
        "protocols": protocols,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "title_vectorizer": title_vectorizer,
        "title_matrix": title_matrix,
    }


def save_protocol_index(index: Dict[str, Any], path: Path) -> None:
    """Pickle a built index so it can be baked into a deploy image."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"Saved protocol index to {path}")


def load_protocol_index(path: Path) -> Dict[str, Any]:
    """Load a prebuilt index. Fast — no TF-IDF fitting, so safe at startup."""
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_protocols(
    index: Dict[str, Any],
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return top_k protocols most relevant to query.

    Scoring = full-text TF-IDF score + 0.5 * title-only TF-IDF score.

    The title boost means a protocol with the query term in its title
    ranks higher than one where the term only appears in the steps or
    description — matching how protocols.io's browser ranks results.
    """
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    title_vectorizer = index["title_vectorizer"]
    title_matrix = index["title_matrix"]
    protocols = index["protocols"]

    # Full-text score
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    # Title-only score — weighted 0.5x so title matches get a clear boost
    # without completely overriding full-text relevance
    try:
        title_vec = title_vectorizer.transform([query])
        title_scores = cosine_similarity(title_vec, title_matrix).flatten()
        combined_scores = scores + 0.5 * title_scores
    except Exception:
        combined_scores = scores

    top_indices = combined_scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        p = protocols[idx]
        desc = _draftjs_to_text(p.get("description"))
        materials = _draftjs_to_text(p.get("materials_text"))
        results.append({
            "id": p.get("id"),
            "title": p.get("title") or "",
            "uri": p.get("uri") or "",
            "url": p.get("url") or (f"https://{p.get('doi')}" if p.get("doi") else ""),
            "doi": p.get("doi") or "",
            "created_on": p.get("created_on"),
            "published_on": p.get("published_on"),
            "updated_on": p.get("updated_on"),
            "step_count": len(p.get("steps") or []),
            "stats": p.get("stats") or {},
            "description": desc[:400],
            "materials_text": materials[:300],
            "steps_preview": [s for s in (p.get("steps") or []) if s][:3],
            "authors": p.get("authors") or [],
            "keywords": p.get("keywords") or [],
            "score": round(float(combined_scores[idx]), 4),
        })

    return results


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

def rewrite_query(query: str) -> str:
    """
    Use the configured local LLM to convert a verbose natural language question
    into compact search terms. Falls back to the original query when unavailable.
    """
    if not llm_client.is_available():
        return query
    probes = llm_client.generate_search_queries(query, n_probes=1)
    rewritten = probes[0] if probes else ""
    if rewritten:
        log.info(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten
    return query


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


_CHITCHAT_STARTERS = {"hi", "hello", "hey", "thanks", "thank", "bye", "goodbye", "sup", "yo"}


def _keyword_classify(q: str) -> dict:
    words = set(q.lower().split())
    # Only flag as chitchat for clearly conversational short messages.
    # Default to search — better to return weak results than wrongly dismiss a lab query.
    if len(words) <= 3 and words & _CHITCHAT_STARTERS:
        return {
            "intent": "chitchat",
            "reply": "Hi! I'm a lab protocol search assistant. Describe an experiment and I'll find the right protocol for you.",
        }
    return {"intent": "search", "reply": None}


def classify_intent(query: str) -> dict:
    """
    Decide whether a query is a protocol search request or general conversation.
    Uses the configured local LLM if available, otherwise falls back to keyword heuristic instantly.
    Returns {"intent": "search" | "chitchat", "reply": str | None}
    """
    q = query.strip()
    if not llm_client.is_available():
        return _keyword_classify(q)
    result = llm_client.classify_intent(q)
    return result


# ---------------------------------------------------------------------------
# LLM explanation
# ---------------------------------------------------------------------------

def explain_matches(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Use the configured local LLM to explain which protocols best match the query.
    """
    return llm_client.explain_matches(query, results)


# ---------------------------------------------------------------------------
# Vague query detection
# ---------------------------------------------------------------------------

import re as _re

# Generic lab terms that, on their own, produce poor TF-IDF results
_VAGUE_SINGLE_TERMS = {
    "pcr", "cloning", "extraction", "blot", "western", "elisa", "sequencing",
    "transfection", "transformation", "gel", "electrophoresis", "microscopy",
    "assay", "culture", "protocol", "experiment", "analysis", "expression",
    "overexpression", "knockdown", "knockout", "crispr", "chip",
    "immunofluorescence", "immunoprecipitation", "lysis", "centrifugation",
    # Molecule names alone are too vague — need context (organism, technique, goal)
    "rna", "dna", "protein", "mrna", "cdna", "lipid", "metabolite",
}

# Words that make a short query specific enough to search without clarification
_SPECIFIC_CONTEXT_WORDS = {
    "plant", "mouse", "human", "yeast", "bacteria", "bacterial", "mammalian",
    "arabidopsis", "hela", "stem", "liver", "brain", "leaf", "root", "seed",
    "rna", "dna", "mrna", "cdna", "antibody", "primer", "tissue",
    "blood", "serum", "neuron", "fibroblast", "zebrafish", "drosophila",
}

# "I want to test X"-style phrases that signal vague intent
_VAGUE_INTENT_PATTERNS = [
    r"^(i want to|i need to|how do i|help me|i am trying to) (test|do|run|perform|try|conduct)\b",
    r"^(a |an |the )?(gene|protein|cell|tissue|bacterial) (experiment|protocol|assay|test|analysis)$",
    r"^(gene|protein) expression$",
    r"^cell (culture|protocol|experiment)$",
]


def is_vague_query(query: str) -> bool:
    """
    Return True if the query is too broad/vague for TF-IDF to return good results.

    Checks three conditions:
    1. Single generic lab term (e.g. "PCR", "overexpression")
    2. Matches a vague intent pattern (e.g. "I want to test X")
    3. Short query (≤4 words) with a generic term but no organism/sample specificity
    """
    q = query.lower().strip()
    words = q.split()

    # Single generic term
    if len(words) == 1 and q in _VAGUE_SINGLE_TERMS:
        return True

    # Matches "I want to test X"-style pattern
    for pattern in _VAGUE_INTENT_PATTERNS:
        if _re.search(pattern, q):
            return True

    # Short query with a vague term but nothing to anchor it (no organism, sample, etc.)
    if 2 <= len(words) <= 4:
        has_vague = any(w in _VAGUE_SINGLE_TERMS for w in words)
        has_specific = any(w in _SPECIFIC_CONTEXT_WORDS for w in words)
        if has_vague and not has_specific:
            return True

    return False


# ---------------------------------------------------------------------------
# Clarification questions
# ---------------------------------------------------------------------------

_CLARIFICATION_MAP = {
    "overexpression": (
        "Do you mean gene overexpression, protein overexpression, or overexpression "
        "in a specific system (e.g. mammalian cells, plants, bacteria)?"
    ),
    "expression": (
        "Are you looking for gene expression analysis (e.g. qPCR, RNA-seq), "
        "protein expression (e.g. western blot), or overexpression (e.g. transfection)?"
    ),
    "pcr": (
        "What type of PCR? Options include standard PCR, quantitative PCR (qPCR), "
        "RT-PCR, colony PCR, or digital PCR."
    ),
    "extraction": (
        "What are you extracting, and from what source? "
        "For example: RNA from plant tissue, DNA from bacteria, protein from cell lysate."
    ),
    "cloning": (
        "Are you doing molecular cloning (gene insertion into a vector), "
        "cell cloning, or positional cloning? What organism or cell type?"
    ),
    "culture": (
        "What type of cell culture? For example: mammalian, bacterial, yeast, "
        "plant cell culture, or primary cell culture."
    ),
    "transfection": (
        "Are you doing transient or stable transfection? "
        "What cell type — mammalian, insect, or plant?"
    ),
    "sequencing": (
        "What sequencing approach? For example: Sanger sequencing, RNA-seq, "
        "ChIP-seq, whole genome sequencing, or amplicon sequencing."
    ),
    "knockdown": (
        "Are you using siRNA, shRNA, or antisense oligonucleotides for knockdown? "
        "What gene and organism?"
    ),
    "knockout": (
        "Are you doing CRISPR knockout, homologous recombination, or another approach? "
        "What organism or cell type?"
    ),
    "crispr": (
        "What CRISPR application: knockout, knockin, base editing, or CRISPRi/CRISPRa? "
        "What organism or cell type are you working with?"
    ),
    "rna": (
        "What do you need to do with RNA? For example: extract RNA from a sample, "
        "sequence RNA (RNA-seq), measure gene expression (qPCR), or something else? "
        "What organism or tissue are you working with?"
    ),
    "dna": (
        "What do you need to do with DNA? For example: extract genomic DNA, "
        "clone a gene, sequence DNA, or something else? What organism or tissue?"
    ),
    "protein": (
        "What do you need to do with protein? For example: extract protein, "
        "detect by western blot, purify recombinant protein, or run a protein assay? "
        "What organism or cell type?"
    ),
    "assay": (
        "What type of assay? For example: cell viability, ELISA, reporter assay, "
        "enzyme activity, or binding assay. What are you measuring?"
    ),
    "microscopy": (
        "What type of microscopy: confocal, fluorescence, brightfield, electron, "
        "or live-cell imaging? What sample or cell type?"
    ),
}


def get_clarification_question(query: str) -> str:
    """
    Return a contextually relevant clarification question for a vague query.
    Matches on keywords in the query; falls back to a generic prompt.
    """
    q = query.lower()
    for keyword, question in _CLARIFICATION_MAP.items():
        # Match whole word to avoid partial hits (e.g. "expression" inside "overexpression")
        if _re.search(rf"\b{_re.escape(keyword)}\b", q):
            return question
    return (
        "Could you be more specific? For example: what organism or tissue are you "
        "working with, and what is your experimental goal or technique?"
    )


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

# Synonym substitutions — each key maps to a list of interchangeable terms
_SYNONYM_MAP: Dict[str, List[str]] = {
    "extraction": ["isolation", "purification", "preparation", "collection"],
    "isolation": ["extraction", "purification", "separation"],
    "purification": ["isolation", "extraction", "cleanup"],
    "expression": ["overexpression", "production", "regulation", "synthesis"],
    "overexpression": ["expression", "transfection", "transduction", "ectopic expression",
                       "protein production", "gene regulation", "forced expression",
                       "lentiviral", "plasmid expression"],
    "analysis": ["assay", "detection", "quantification", "measurement"],
    "assay": ["analysis", "detection", "test", "measurement"],
    "culture": ["growth", "maintenance", "propagation", "cultivation"],
    "protocol": ["method", "procedure", "technique", "workflow"],
    "knockdown": ["silencing", "siRNA", "shRNA", "gene silencing", "RNAi"],
    "knockout": ["deletion", "CRISPR", "gene disruption", "null mutant"],
    "transfection": ["transformation", "transduction", "gene delivery", "lipofection"],
    "sequencing": ["library preparation", "NGS", "next generation sequencing", "genome sequencing"],
    "staining": ["labeling", "immunostaining", "fluorescent labeling", "dyeing"],
    "microscopy": ["imaging", "visualization", "confocal", "fluorescence imaging"],
    "cloning": ["subcloning", "gene cloning", "molecular cloning", "vector cloning"],
    "lysis": ["homogenization", "disruption", "cell disruption", "tissue homogenization"],
    "amplification": ["PCR", "qPCR", "RT-PCR", "polymerase chain reaction"],
    "detection": ["quantification", "analysis", "measurement", "assay"],
    "transformation": ["transfection", "electroporation", "gene transfer"],
    "degradation": ["breakdown", "hydrolysis", "digestion", "enzymatic degradation",
                    "hydrolase", "hydrolases", "depolymerization", "biodegradation"],
    "binding": ["interaction", "affinity", "pull-down", "co-immunoprecipitation"],
}

# Common reagent/kit names associated with techniques — adds specificity to queries
_TECHNIQUE_REAGENTS: Dict[str, List[str]] = {
    "rna extraction": ["TRIzol RNA", "CTAB RNA", "RNeasy RNA", "total RNA"],
    "rna isolation": ["TRIzol", "CTAB", "phenol chloroform RNA", "total RNA isolation"],
    "dna extraction": ["CTAB DNA", "phenol chloroform DNA", "DNeasy", "genomic DNA"],
    "western blot": ["SDS-PAGE western", "immunoblot", "protein western blot"],
    "pcr": ["polymerase chain reaction", "Taq PCR", "PCR amplification"],
    "qpcr": ["quantitative PCR", "real-time PCR", "RT-qPCR", "gene expression qPCR"],
    "cell culture": ["mammalian cell culture", "tissue culture", "in vitro culture"],
    "crispr": ["CRISPR Cas9", "CRISPR knockout", "CRISPR guide RNA", "sgRNA design"],
    "flow cytometry": ["FACS", "cell sorting flow", "fluorescence flow cytometry"],
    "immunofluorescence": ["IF staining", "fluorescent antibody staining", "confocal IF"],
}

# Context suffixes to append for additional query coverage
_CONTEXT_SUFFIXES = ["protocol", "method", "procedure", "step by step", "kit", "workflow"]


def _expand_query_rule_based(query: str) -> List[str]:
    """
    Rule-based expansion generating up to 10 related queries via:
    1. Synonym substitution on individual words
    2. Technique-specific reagent/kit name variants
    3. Context suffix appending (protocol, method, kit, etc.)

    Always includes the original query as the first entry.
    """
    q = query.strip()
    q_lower = q.lower()
    words = q_lower.split()
    queries: List[str] = [q]

    # 1. Synonym substitution — swap each recognised word with all its synonyms
    for i, word in enumerate(words):
        if word in _SYNONYM_MAP:
            for synonym in _SYNONYM_MAP[word]:
                alt = " ".join(words[:i] + [synonym] + words[i + 1:])
                if alt not in queries:
                    queries.append(alt)
                if len(queries) >= 10:
                    break
        if len(queries) >= 10:
            break

    # 2. Technique-reagent variants — look for known two-word technique phrases
    for technique, reagent_variants in _TECHNIQUE_REAGENTS.items():
        if technique in q_lower:
            for variant in reagent_variants:
                candidate = q_lower.replace(technique, variant)
                if candidate not in queries:
                    queries.append(candidate)
                if len(queries) >= 10:
                    break
        if len(queries) >= 10:
            break

    # 3. Context suffixes — append protocol/method/kit if not already present
    for suffix in _CONTEXT_SUFFIXES:
        if suffix not in q_lower and len(queries) < 10:
            queries.append(q + " " + suffix)

    return queries[:10]


def _expand_query_with_llm(query: str) -> List[str]:
    """
    Use the configured local LLM to generate 8 alternative search queries.
    Falls back to rule-based expansion if the model is unavailable or call fails.
    """
    candidates = llm_client.generate_search_queries(query, n_probes=8)
    if candidates and len(candidates) >= 2:
        return ([query] + candidates)[:10]
    return _expand_query_rule_based(query)


def expand_query(query: str) -> List[str]:
    """
    Generate 3 search query variants: Initial (original), Level 1 (full), Level 2 (stripped).

    Levels:
    1. Initial: Original query as-is
    2. Level 1: Full with all core terms
    3. Level 2: Stripped query (stop words removed for tighter phrase match)

    This keeps queries at strictness levels: full → stripped → pairs.
    Returns 3 variants per query; for 5 candidate queries = 15 total probes.
    """
    # Start with strict-to-loose structural variants
    strict_queries = _generate_strict_to_loose_queries(query)

    # Keep Initial (original), Level 1 (full), and Level 2 (stripped)
    # Level 0/Initial: queries[0], Level 1: queries[1], Level 2: queries[2] if exists, else queries[1]
    result = []
    if len(strict_queries) > 0:
        result.append(strict_queries[0])  # Initial/Level 1: original full query
    if len(strict_queries) > 1:
        result.append(strict_queries[1])  # Level 2: stripped (stop words removed)
    if len(strict_queries) > 2:
        result.append(strict_queries[2])  # Level 3: pairs of core terms

    # Ensure we always return at least 3 variants (fallback to repeating if needed)
    while len(result) < 3 and len(result) > 0:
        result.append(result[-1])  # Repeat last variant if we have fewer than 3

    return result if result else [query]


# ---------------------------------------------------------------------------
# Relevance filtering — local AND logic
# ---------------------------------------------------------------------------

# Words to ignore when extracting core terms from a query
_FILTER_STOP_WORDS = {
    "a", "an", "the", "in", "of", "from", "with", "for", "to", "and", "or",
    "is", "are", "i", "me", "my", "want", "need", "looking", "find", "get",
    "how", "what", "do", "does", "can", "will", "would", "should", "could",
    "using", "use", "make", "create", "run", "perform", "test", "try", "about",
    "some", "this", "that", "its", "it", "on", "at", "by", "as", "be", "have",
}


def _extract_core_terms(query: str) -> List[str]:
    """
    Extract the meaningful biology terms from a query by removing stop words.
    These are the terms that MUST appear in a relevant result.

    Example: "gene overexpression in mammalian cells"
             → ["gene", "overexpression", "mammalian", "cells"]
    """
    words = query.lower().split()
    return [w for w in words if w not in _FILTER_STOP_WORDS and len(w) > 2]


def _relevance_penalty(
    result: Dict[str, Any],
    core_terms: List[str],
    key_term: Optional[str] = None,
) -> float:
    """
    Return a penalty multiplier (0.0–1.0) enforcing local AND logic.

    Two-stage check:
    1. Key term check: if the single most specific term (highest IDF / rarest word)
       is missing from the result, apply a heavy penalty regardless of other matches.
       This is the core of Prof. Shasha's suggestion — "overexpression" must appear,
       not just "mammalian" or "cells".
    2. Overall hit ratio: penalise results that match fewer than half the core terms.

    Penalty scale:
        key term present AND ≥ 60% core terms → no penalty (1.0)
        key term present, 30–60% core terms   → light penalty (0.7)
        key term missing                       → heavy penalty (0.2)
        < 30% core terms                       → heavy penalty (0.3)
    """
    if not core_terms:
        return 1.0

    title = (result.get("title") or "").lower()
    desc = (result.get("description") or "").lower()
    text = title + " " + desc

    # Stage 1: key term must be present
    if key_term and key_term not in text:
        return 0.2

    # Stage 2: overall hit ratio
    hit_ratio = sum(1 for t in core_terms if t in text) / len(core_terms)
    if hit_ratio >= 0.6:
        return 1.0
    if hit_ratio >= 0.3:
        return 0.7
    return 0.3


# ---------------------------------------------------------------------------
# Strictness-ordered query generation
# ---------------------------------------------------------------------------

def _generate_strict_to_loose_queries(query: str) -> List[str]:
    """
    Generate queries ordered from most strict to least strict, as suggested
    by Prof. Shasha: require all key terms first, then progressively relax.

    Strictness levels:
      1. Original query (all terms together)
      2. Core terms only (stop words removed — tighter TF-IDF match)
      3. Pairs of core terms (two-concept queries)
      4. Individual core terms (single-concept fallback)
    """
    core = _extract_core_terms(query)
    queries: List[str] = [query]

    # Level 2: stop-word-stripped version (tighter phrase match)
    stripped = " ".join(core)
    if stripped and stripped != query.lower() and stripped not in queries:
        queries.append(stripped)

    # Level 3: pairs of adjacent core terms
    for i in range(len(core) - 1):
        pair = f"{core[i]} {core[i + 1]}"
        if pair not in queries:
            queries.append(pair)

    # Level 4: individual core terms (most lenient)
    for term in core:
        if len(term) > 3 and term not in queries:
            queries.append(term)

    return queries[:10]


# ---------------------------------------------------------------------------
# Multi-query search with result merging, re-ranking, and relevance filter
# ---------------------------------------------------------------------------

def multi_search_protocols(
    index: Dict[str, Any],
    queries: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run TF-IDF search for each query, merge results, and re-rank using:
      1. Frequency bonus: protocols matched by more queries score higher
      2. Relevance penalty: protocols missing core terms from the original
         query are penalised (local AND logic — implements Prof. Shasha's
         suggestion that all key concepts must be present)

    Query list should be ordered strict → loose so stricter matches are
    preferred when scores tie.
    """
    best_results: Dict[int, Dict[str, Any]] = {}
    matched_by: Dict[int, List[str]] = {}

    # Search with a wider net internally — fetch 4x more candidates per query
    # so niche but highly relevant protocols aren't cut off before re-ranking.
    inner_k = max(top_k * 4, 20)
    for query in queries:
        for result in search_protocols(index, query, top_k=inner_k):
            pid = result["id"]
            if pid not in best_results or result["score"] > best_results[pid]["score"]:
                best_results[pid] = result
            if pid not in matched_by:
                matched_by[pid] = []
            if query not in matched_by[pid]:
                matched_by[pid].append(query)

    # Extract core terms from the original (first) query for relevance filtering
    original_query = queries[0] if queries else ""
    core_terms = _extract_core_terms(original_query)

    # Find the top-2 most specific terms using IDF weights.
    # Prof. Shasha's AND logic: both key concepts must appear, not just one.
    vectorizer = index["vectorizer"]
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    scored_terms = sorted(
        [(idf[vocab[t]], t) for t in core_terms if t in vocab],
        reverse=True,
    )
    required_terms = [t for _, t in scored_terms[:2]]

    # Override: specific molecule/target names mentioned explicitly by the user
    # must always be required, even if IDF ranks them as common.
    # This ensures "RNA extraction" never matches DNA-only protocols.
    _ALWAYS_REQUIRED = {"rna", "dna", "mrna", "cdna", "protein", "lipid",
                        "crispr", "plasmid", "antibody", "metabolite"}
    for term in core_terms:
        if term in _ALWAYS_REQUIRED and term not in required_terms:
            required_terms = [term] + required_terms[:1]
            break

    key_term = required_terms[0] if required_terms else None

    ranked = []
    for pid, result in best_results.items():
        match_count = len(matched_by[pid])
        combined = result["score"] + (match_count - 1) * 0.05

        title = (result.get("title") or "").lower()
        desc = (result.get("description") or "").lower()
        text = title + " " + desc

        # AND check: both required terms (or their synonyms) must appear.
        # This means a transfection protocol correctly satisfies an overexpression
        # query because transfection is in _SYNONYM_MAP["overexpression"].
        def _term_satisfied(term: str, text: str) -> bool:
            if term in text:
                return True
            for syn in _SYNONYM_MAP.get(term, []):
                if syn.lower() in text:
                    return True
            return False

        missing_required = [t for t in required_terms if not _term_satisfied(t, text)]
        if len(missing_required) == len(required_terms):
            penalty = 0.1
        elif missing_required:
            penalty = 0.2
        else:
            # Both required terms confirmed present (including via synonyms).
            # Ensure a minimum of 0.7 — the AND check already validated relevance,
            # so a heavy hit-ratio penalty would be too harsh.
            penalty = max(0.7, _relevance_penalty(result, core_terms, key_term=None))

        combined = round(combined * penalty, 4)

        ranked.append({
            **result,
            "combined_score": combined,
            "matched_queries": matched_by[pid],
            "match_count": match_count,
            "relevance_penalty": penalty,
            "key_term": key_term,
            "required_terms": required_terms,
        })

    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return ranked[:top_k]
