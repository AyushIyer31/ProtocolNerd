"""
Concept extraction + grounded term expansion for biomedical protocol search.

Pipeline (implements Prof. Shasha's suggestions):

  1. extract_concepts(query)
        Break a natural-language request into biological facets:
        organisms, methods/techniques, goals/phenotypes, and gene-modification
        actions. Rule-based detection, optionally refined by the local LLM.

  2. expand_concepts(concepts)
        Expand each concept into 2-3 high-quality synonyms / related terms.
        Term sources, in priority order ("DB primary, LLM fallback"):
          a. NCBI Taxonomy E-utilities  -> organism scientific + common names
                                           (rice -> Oryza sativa)
          b. Europe PMC                 -> related keywords/MeSH mined from the
                                           top matching papers (grounded synonyms)
          c. local Ollama LLM           -> fallback when the DBs return nothing
          d. static biomedical map      -> final offline fallback

  3. build_search_probes(concepts, expansions)
        Turn the expanded concepts into an ordered list of SHORT (1-2 word)
        keyword/phrase probes suitable for the protocols.io `key` parameter,
        which only matches adjacent phrases (see protocolsio_client.py).

External calls are bounded by short timeouts and degrade gracefully so the
pipeline still works fully offline.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional

import claude_client as llm_client

log = logging.getLogger(__name__)

_HTTP_UA = "Mozilla/5.0 (compatible; ProtocolSearchBot/1.0)"
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


# ---------------------------------------------------------------------------
# Facet vocabularies (used for rule-based extraction)
# ---------------------------------------------------------------------------

# Common-name organisms the user might mention -> hint for taxonomy lookup
_ORGANISM_WORDS = {
    "rice", "wheat", "maize", "corn", "arabidopsis", "tomato", "soybean",
    "barley", "tobacco", "potato", "mouse", "mice", "rat", "human", "humans",
    "yeast", "zebrafish", "drosophila", "fly", "worm", "elegans", "bacteria",
    "bacterial", "e. coli", "ecoli", "coli", "plant", "plants", "fungus",
    "fungal", "fish", "pig", "cow", "chicken", "monkey", "primate",
}

# Method / technique terms
_METHOD_WORDS = {
    "pcr", "qpcr", "rt-pcr", "crispr", "cas9", "transfection", "transformation",
    "transduction", "cloning", "sequencing", "rna-seq", "chip-seq", "western",
    "blot", "elisa", "electroporation", "agrobacterium", "biolistic",
    "microinjection", "knockout", "knockdown", "knockin", "knock-in",
    "overexpression", "mutagenesis", "editing", "phenotyping", "imaging",
    "assay", "staining", "microscopy", "extraction", "purification",
    "transgenic", "regeneration", "multiplex", "multiplexed", "screening",
}

# Words that signal a gene-modification action
_GENE_ACTION_WORDS = {
    "modified", "modify", "modification", "edited", "editing", "engineered",
    "engineering", "overexpress", "overexpressed", "overexpression",
    "knockout", "knockdown", "knockin", "knock-in", "mutated", "mutation",
    "transgenic", "transformed", "transformation", "silenced", "silencing",
}

# Phenotype / experimental-goal cue words
_GOAL_WORDS = {
    "tolerance", "resistance", "stress", "drought", "salinity", "salt", "cold",
    "heat", "yield", "growth", "expression", "viability", "proliferation",
    "differentiation", "apoptosis", "tolerant", "susceptibility",
}

# Target molecules. When the user names one, it is a REQUIRED key term for
# ranking — "DNA extraction" must not match a protein or compound extraction
# protocol (Prof. Shasha's AND-logic point).
_MOLECULE_WORDS = {
    "dna", "rna", "mrna", "cdna", "genomic", "plasmid", "protein", "proteins",
    "lipid", "lipids", "metabolite", "metabolites", "peptide", "chromatin",
}

# Very generic method words whose Europe PMC mining is too noisy (e.g.
# "extraction" pulls dental "tooth extraction"). For these we trust the curated
# map only and skip the external term-mining.
_GENERIC_METHOD = {
    "extraction", "isolation", "purification", "analysis", "expression",
    "assay", "preparation", "detection", "quantification",
}

# Grammatical / search-boilerplate words to drop. Note: domain words like
# "modified" or "genes" are deliberately NOT here — they carry biological intent.
_STOP = {
    "a", "an", "the", "in", "of", "from", "with", "for", "to", "and", "or",
    "is", "are", "i", "me", "my", "want", "need", "looking", "find", "get",
    "how", "what", "do", "does", "can", "will", "would", "should", "could",
    "that", "which", "more", "than", "one", "same", "time", "at", "be",
    "test", "tests", "testing", "allow", "allows", "allowing", "protocols",
    "protocol", "use", "using", "able", "via",
}

# Common-name organism variants kept together so a query saying "mice" still
# matches a protocol titled "... in mouse ...".
_ORGANISM_VARIANTS = {
    "mice": ["mouse"], "mouse": ["mice"], "rat": ["rats"],
    "human": ["humans"], "humans": ["human"], "plant": ["plants"],
    "plants": ["plant"], "bacteria": ["bacterial"], "fly": ["drosophila"],
}

# Curated biomedical synonyms (final offline fallback / supplement)
_BIOMED_SYNONYMS: Dict[str, List[str]] = {
    "drought": ["drought stress", "water deficit", "water stress"],
    "drought tolerance": ["drought stress", "water deficit", "drought resistance"],
    "tolerance": ["resistance", "stress response"],
    "salt": ["salinity", "salt stress", "NaCl stress"],
    "gene modification": ["genome editing", "transgenic", "gene editing"],
    "genes": ["genome editing", "transgenic", "gene editing"],
    "modified": ["genome editing", "transgenic", "CRISPR"],
    "gene editing": ["genome editing", "CRISPR", "gene knockout"],
    "transcription factor": ["transcription factor", "transcriptional regulator", "DNA-binding"],
    "in planta": ["plant transformation", "agrobacterium", "agroinfiltration"],
    "in-planta": ["plant transformation", "agrobacterium", "agroinfiltration"],
    "overexpression": ["ectopic expression", "transgene expression", "forced expression"],
    "multiplex": ["multiplex CRISPR", "multiplexed", "combinatorial"],
    "phenotyping": ["phenotype analysis", "trait measurement", "phenotypic assay"],
    # Generic methods + molecules — curated to stay in the molecular-biology sense
    "extraction": ["isolation", "purification", "DNA extraction"],
    "isolation": ["extraction", "purification"],
    "purification": ["isolation", "extraction"],
    "dna": ["genomic DNA", "DNA extraction", "DNA isolation"],
    "rna": ["total RNA", "RNA extraction", "RNA isolation"],
    "mrna": ["mRNA isolation", "poly-A RNA", "RNA extraction"],
    "protein": ["protein extraction", "protein purification", "western blot"],
}

# Organisms whose scientific name we know offline (fallback if NCBI is down)
_ORGANISM_FALLBACK = {
    "rice": ["Oryza sativa"],
    "wheat": ["Triticum aestivum"],
    "maize": ["Zea mays"],
    "corn": ["Zea mays"],
    "arabidopsis": ["Arabidopsis thaliana"],
    "tomato": ["Solanum lycopersicum"],
    "mouse": ["Mus musculus"],
    "mice": ["Mus musculus"],
    "human": ["Homo sapiens"],
    "yeast": ["Saccharomyces cerevisiae"],
    "zebrafish": ["Danio rerio"],
    "drosophila": ["Drosophila melanogaster"],
    "tobacco": ["Nicotiana benthamiana", "Nicotiana tabacum"],
}


# ---------------------------------------------------------------------------
# External term sources
# ---------------------------------------------------------------------------

def _http_json(url: str, timeout: int = 6) -> Optional[Any]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _HTTP_UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        log.debug(f"HTTP JSON failed for {url}: {e}")
        return None


def ncbi_scientific_names(common: str) -> List[str]:
    """Resolve a common organism name to scientific (+ common) names via NCBI Taxonomy."""
    s = _http_json(f"{_EUTILS}/esearch.fcgi?db=taxonomy&term={urllib.parse.quote(common)}&retmode=json")
    ids = (((s or {}).get("esearchresult") or {}).get("idlist")) or []
    if not ids:
        return list(_ORGANISM_FALLBACK.get(common.lower(), []))
    su = _http_json(f"{_EUTILS}/esummary.fcgi?db=taxonomy&id={ids[0]}&retmode=json")
    rec = (((su or {}).get("result") or {}).get(ids[0])) or {}
    names: List[str] = []
    if rec.get("scientificname"):
        names.append(rec["scientificname"])
    if rec.get("commonname") and rec["commonname"].lower() != common.lower():
        names.append(rec["commonname"])
    return names or list(_ORGANISM_FALLBACK.get(common.lower(), []))


def _stems(text: str) -> set:
    """Crude 4-char stems of the words in `text`, for relevance overlap checks."""
    return {w[:4] for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


def _is_relevant(mined: str, concept: str) -> bool:
    """
    Keep a mined term only if it is topically tied to the concept — it shares a
    word stem with the concept (e.g. 'drought stress' ~ 'drought'). This filters
    the noisy co-occurring MeSH terms Europe PMC returns (morphine, proteomics,
    artificial intelligence) for a concept like 'tolerance'.
    """
    cs, ms = _stems(concept), _stems(mined)
    return bool(cs & ms)


def europepmc_related_terms(term: str, max_terms: int = 4) -> List[str]:
    """
    Mine related keywords/MeSH from the top Europe PMC papers for `term`, keeping
    only terms that stay on-topic (share a stem with `term`).
    """
    url = f"{_EUROPEPMC}?" + urllib.parse.urlencode(
        {"query": term, "format": "json", "pageSize": 5, "resultType": "core"}
    )
    d = _http_json(url)
    results = (((d or {}).get("resultList") or {}).get("result")) or []
    counter: Counter = Counter()
    for r in results:
        for m in ((r.get("meshHeadingList") or {}).get("meshHeading") or []):
            name = (m.get("descriptorName") or "").strip()
            if name:
                counter[name.lower()] += 1
        for kw in ((r.get("keywordList") or {}).get("keyword") or []):
            kw = (kw or "").strip()
            if kw:
                counter[kw.lower()] += 1
    term_l = term.lower()
    out = []
    for t, _ in counter.most_common(max_terms * 4):
        if t == term_l or len(t) <= 2:
            continue
        if not (1 <= len(t.split()) <= 3):   # probe-friendly length
            continue
        if not _is_relevant(t, term):        # topical relevance filter
            continue
        out.append(t)
    return out[:max_terms]


def _ollama_synonyms(term: str, max_terms: int = 3) -> List[str]:
    return llm_client.get_synonyms(term, max_terms)


# ---------------------------------------------------------------------------
# 1. Concept extraction
# ---------------------------------------------------------------------------

def _normalize(query: str) -> str:
    """Lowercase and turn hyphens into spaces so 'in-planta' == 'in planta'."""
    return re.sub(r"[-/]", " ", query.lower())


def _core_terms(query: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z]+", _normalize(query))
    return [w for w in words if w not in _STOP and len(w) > 2]


def _dedup(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


# Natural-language cues that imply "do this to several targets at once"
_MULTIPLEX_CUES = ("more than one", "multiple", "at the same time",
                   "simultaneous", "simultaneously", "in parallel", "several")


def extract_concepts(query: str) -> Dict[str, Any]:
    """
    Decompose a query into biological facets.

    Returns:
      {
        "query": str,
        "organisms": [...],   # common-name organism mentions
        "molecules": [...],   # target molecules (DNA/RNA/protein) — REQUIRED terms
        "methods":   [...],   # technique / method terms (incl. phrases)
        "goals":     [...],   # phenotype / experimental-goal terms
        "actions":   [...],   # gene-modification action terms
        "core_terms":[...],   # all meaningful terms (de-duplicated)
      }
    """
    q = _normalize(query)
    words = _core_terms(query)

    organisms = [w for w in words if w in _ORGANISM_WORDS]
    if "e. coli" in q or "e.coli" in q:
        organisms.append("e. coli")

    molecules = [w for w in words if w in _MOLECULE_WORDS]

    methods = [w for w in words if w in _METHOD_WORDS]
    # adjacent method phrases worth keeping intact
    for phrase in ("transcription factor", "gene editing", "genome editing",
                   "in planta", "multiplex crispr", "plant transformation"):
        if phrase in q and phrase not in methods:
            methods.append(phrase)

    actions = [w for w in words if w in _GENE_ACTION_WORDS]
    # "genes ... modified" / "in which genes are modified" -> gene-modification intent
    if ("gene" in words or "genes" in words) and (
        actions or "modif" in q or "edit" in q or "engineer" in q
    ):
        actions.append("gene modification")

    goals = [w for w in words if w in _GOAL_WORDS]
    for phrase in ("drought tolerance", "salt tolerance", "stress tolerance"):
        if phrase in q and phrase not in goals:
            goals.append(phrase)

    # Multiplex intent: "more than one X at the same time"
    if any(cue in q for cue in _MULTIPLEX_CUES) and "multiplex" not in methods:
        methods.append("multiplex")

    return {
        "query": query,
        "organisms": _dedup(organisms),
        "molecules": _dedup(molecules),
        "methods": _dedup(methods),
        "goals": _dedup(goals),
        "actions": _dedup(actions),
        "core_terms": _dedup(words),
    }


# ---------------------------------------------------------------------------
# 2. Concept expansion
# ---------------------------------------------------------------------------

def expand_concepts(concepts: Dict[str, Any], use_external: bool = True) -> Dict[str, List[str]]:
    """
    Expand each concept into grounded synonyms / related terms.

    Returns a mapping {original_concept: [expansion terms]}. The original term
    is never repeated in its own expansion list.
    """
    expansions: Dict[str, List[str]] = {}

    # Organisms -> scientific names (NCBI) + common variants + offline fallback
    for org in concepts.get("organisms", []):
        terms: List[str] = []
        if use_external:
            terms += ncbi_scientific_names(org)
        terms += _ORGANISM_FALLBACK.get(org.lower(), [])
        terms += _ORGANISM_VARIANTS.get(org.lower(), [])
        if terms:
            expansions[org] = _dedup_against(terms, org)[:4]

    # Molecules + goals + methods + actions: curated map first (high precision),
    # then grounded Europe PMC terms (filtered), then Ollama — "DB primary, LLM fallback".
    concept_terms: List[str] = (
        concepts.get("molecules", []) + concepts.get("goals", [])
        + concepts.get("methods", []) + concepts.get("actions", [])
    )
    llm_up = llm_client.is_available() if use_external else False

    for term in concept_terms:
        if term in expansions:
            continue
        terms: List[str] = list(_BIOMED_SYNONYMS.get(term.lower(), []))
        # Skip external mining for very generic words — it drifts off-domain
        # (e.g. "extraction" -> dental "tooth extraction"). Trust the curated map.
        if use_external and term.lower() not in _GENERIC_METHOD:
            terms += [t for t in europepmc_related_terms(term) if t not in terms]
        if len(terms) < 2 and llm_up:
            terms += [t for t in _ollama_synonyms(term) if t not in terms]
        if terms:
            expansions[term] = _dedup_against(terms, term)[:4]

    return expansions


def _dedup_against(terms: List[str], original: str) -> List[str]:
    seen, out = {original.lower()}, []
    for t in terms:
        tl = t.lower().strip()
        if tl and tl not in seen:
            seen.add(tl)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# 3. Search-probe construction
# ---------------------------------------------------------------------------

def build_search_probes(
    concepts: Dict[str, Any],
    expansions: Dict[str, List[str]],
    max_probes: int = 14,
) -> List[str]:
    """
    Build an ordered list of SHORT probes for the protocols.io `key` parameter.

    protocols.io only matches adjacent phrases, so every probe is 1-2 words
    (occasionally a 2-word scientific name or known phrase). Ordered most
    specific -> most general so the early, high-precision probes dominate the
    merge when the result cap is hit.
    """
    probes: List[str] = []

    def add(p: str):
        p = p.strip()
        if p and p.lower() not in {x.lower() for x in probes}:
            probes.append(p)

    molecules = concepts.get("molecules", [])
    methods = concepts.get("methods", [])
    orgs = concepts.get("organisms", [])

    # 1. Molecule + method combos first — the most precise probe (e.g. "DNA extraction").
    for mol in molecules:
        for method in methods:
            add(f"{mol} {method}")
        for syn in expansions.get(mol, []):   # "genomic DNA", "DNA isolation"
            add(syn)

    # 2. Organism + molecule combo (e.g. "mouse DNA")
    for org in orgs:
        for mol in molecules:
            add(f"{org} {mol}")

    # 3. Organism: scientific names first (high precision), then the common word
    for org in orgs:
        for sci in expansions.get(org, []):
            add(sci)
        add(org)

    # 4. Goal synonyms (phenotype) — the terms that actually return hits
    for goal in concepts.get("goals", []):
        for syn in expansions.get(goal, []):
            add(syn)
        add(goal)

    # 5. Method / technique terms and their expansions
    for method in methods:
        add(method)
        for syn in expansions.get(method, []):
            add(syn)

    # 6. Action expansions (gene modification synonyms)
    for action in concepts.get("actions", []):
        for syn in expansions.get(action, []):
            add(syn)

    # 7. Other useful adjacent-phrase combos
    if orgs and methods:
        add(f"{methods[0]} {orgs[0]}")
    if "multiplex" in methods:
        add("multiplex CRISPR")

    # 8. Bare molecule terms last (broad)
    for mol in molecules:
        add(mol)

    # 9. Fall back to bare core terms if we still have very few probes
    if len(probes) < 3:
        for t in concepts.get("core_terms", []):
            add(t)

    # keep only short probes (<=2 words), trim to budget
    probes = [p for p in probes if len(p.split()) <= 2]
    return probes[:max_probes]


# ---------------------------------------------------------------------------
# 4. Full-sentence query variants
# ---------------------------------------------------------------------------

# Maps a detected gene-action word to a natural verb/noun for sentence building
_ACTION_VERB = {
    "modified": ("modify", "modification"),
    "modification": ("modify", "modification"),
    "modify": ("modify", "modification"),
    "gene modification": ("modify genes in", "gene modification"),
    "edited": ("edit", "editing"),
    "editing": ("edit", "editing"),
    "overexpression": ("overexpress", "overexpression"),
    "knockout": ("knock out", "knockout"),
    "knockdown": ("knock down", "knockdown"),
    "transformed": ("transform", "transformation"),
}


def generate_sentence_variants(
    concepts: Dict[str, Any],
    expansions: Dict[str, List[str]],
    max_variants: int = 10,
) -> List[str]:
    """
    Produce reworded FULL-SENTENCE versions of the query.

    Unlike the short protocols.io probes, these are natural-language rephrasings
    suitable for literature databases that understand sentences (PubMed / Europe
    PMC) and for showing the user how the query was understood. Uses the local
    LLM when available, otherwise builds them by combining the detected concepts
    (and their synonyms) across a set of sentence templates.
    """
    query = concepts.get("query", "")
    multiplex = "multiplex" in concepts.get("methods", [])

    # Prefer the local LLM for fluent rephrasings when available.
    if llm_client.is_available():
        llm = _ollama_sentence_variants(query, max_variants)
        if llm:
            return llm

    # --- Template path -----------------------------------------------------
    # "Subjects" = the thing being done (method/goal/molecule) + their synonyms,
    # ordered so the originals come first and synonyms add variety afterwards.
    subjects: List[str] = []
    for facet in ("methods", "goals", "molecules"):
        for term in concepts.get(facet, []):
            if term == "multiplex":
                continue
            subjects.append(term)
            subjects += expansions.get(term, [])
    subjects = _dedup(subjects)

    # Organism phrasings: common name(s) + scientific name(s) + variants
    org_phrases: List[str] = []
    for org in concepts.get("organisms", []):
        org_phrases.append(org)
        org_phrases += expansions.get(org, [])
    org_phrases = _dedup(org_phrases)

    verb, noun = ("", "")
    for a in concepts.get("actions", []):
        if a in _ACTION_VERB:
            verb, noun = _ACTION_VERB[a]
            break
    has_genes = "gene" in concepts.get("core_terms", []) or "genes" in concepts.get("core_terms", [])

    variants: List[str] = []

    def add(s: str):
        s = " ".join(s.split())
        if s and s.lower() != query.lower() and s.lower() not in {v.lower() for v in variants}:
            variants.append(s)

    mult = "multiple " if multiplex else ""
    simul = " simultaneously" if multiplex else ""

    # Templates combining each subject with each organism phrasing.
    def with_org(subj: str, org: str):
        if verb:
            add(f"{verb} {mult}{subj} in {org}{simul}")
            add(f"protocol to {verb} {subj} in {org}")
        if noun:
            add(f"{mult}{subj} {noun} in {org}")
        add(f"{subj} protocol in {org}")
        add(f"how to perform {subj} in {org}")
        add(f"{subj} assay in {org}")
        add(f"protocol to measure {subj} in {org}")
        add(f"{subj} in {org}")

    # Templates with no organism (e.g. "in-planta gene modification").
    def without_org(subj: str):
        if verb and has_genes:
            add(f"{verb} genes {subj}" if subj.startswith("in ") else f"{verb} genes using {subj}")
            add(f"protocol for {subj} gene {noun or 'modification'}")
        if has_genes:
            add(f"{subj} gene editing protocol")
        if verb:
            add(f"protocol to {verb} using {subj}")
        add(f"{subj} protocol")
        add(f"how to perform {subj}")
        add(f"step-by-step {subj} method")

    # Build breadth-first: one template family across all subject/org pairs,
    # so we collect varied sentences and reach max_variants reliably.
    if org_phrases:
        for org in org_phrases:
            for subj in subjects or [""]:
                if subj:
                    with_org(subj, org)
                if len(variants) >= max_variants:
                    break
            if len(variants) >= max_variants:
                break
    else:
        for subj in subjects:
            without_org(subj)
            if len(variants) >= max_variants:
                break

    return variants[:max_variants]


def _ollama_sentence_variants(query: str, n: int) -> List[str]:
    return llm_client.get_sentence_variants(query, n)
