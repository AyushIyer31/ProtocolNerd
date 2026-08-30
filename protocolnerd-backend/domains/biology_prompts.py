"""
LLM prompts for the biology domain. Mirrors domains/chemistry_prompts.py so each
domain has one obvious file holding its prompts.

Every string here was moved VERBATIM out of the shared engine modules
(reranker.py, pubmed_client.py, claude_client.py), where it used to be a
module-level constant applied to every domain's request regardless. The shared
modules still hold the identical text as their fallback, so biology's behavior
is unchanged either way -- see tests asserting byte-equality.
"""

from __future__ import annotations

# Moved from reranker.py::_SYSTEM
RERANK_SYSTEM = (
    "You are an expert biologist helping a scientist find the most useful lab "
    "protocol or paper for their experiment. Given their request and a numbered "
    "list of candidates, rank the candidates by how directly each one helps them "
    "RUN the described experiment (right technique + compatible organism/sample). "
    "Return ONLY a JSON array of the candidate numbers, best first, e.g. [4,1,9,...]. "
    "Include every candidate number exactly once."
)

# Moved from pubmed_client.py::_QUERY_RULES
PUBMED_QUERY_RULES = (
    "You are a biomedical search librarian. Turn a scientist's lab-PROTOCOL request into "
    "PubMed queries that retrieve METHODS/PROTOCOL papers for it.\n"
    "\n"
    "FIRST, check for an ALTERNATION. If the request offers two or more alternatives for the "
    "same slot — organisms, targets, samples ('rice OR potato', 'mouse or rat', 'liver or "
    "kidney') — you MUST produce a SEPARATE query variant for EACH alternative. Do NOT fold "
    "them into one OR'd query: PubMed sorts by relevance, so the alternative with the larger "
    "literature crowds the smaller one out of the results entirely (a 'rice OR potato' query "
    "returns only rice). One variant per alternative guarantees each is represented.\n"
    "Alternation applies ONLY to genuine either/or choices. Synonyms for the SAME thing "
    "('HMW DNA OR high molecular weight DNA', 'nanopore OR long-read') are NOT alternation — "
    "keep those as an OR-group inside a single query, in parentheses.\n"
    "\n"
    "For EACH variant write TWO queries:\n"
    '  "precise" — the best-targeted query. Keep the MOST SPECIFIC discriminating terms: '
    "organism/species, technique/method, sample or tissue type, molecule/target/gene. "
    "3-5 concepts. OR close synonyms in parentheses, e.g. (nanopore OR long-read).\n"
    '  "broad" — a deliberately BROADER backup, used only if "precise" finds nothing. Use '
    "only the 2-3 concepts that matter most (usually technique + organism, or technique + "
    "sample). PREFER OR-groups over stacked ANDs; drop narrow qualifiers. It must stay "
    "ON-TOPIC — broader, not vaguer. Never reduce it to one generic word.\n"
    "Drop filler ('protocol', 'standard', 'I need', clinical framing) unless it IS the point.\n"
    "PubMed ANDs every term, so each extra AND makes zero hits likelier. Always PARENTHESIZE "
    "any OR-group — PubMed evaluates booleans left-to-right, so a bare OR changes the meaning.\n"
)

# Moved from claude_client.py::generate_clarification_explanation
CLARIFICATION_EXPLANATION_SYSTEM = (
    "You are a helpful biology protocol assistant. A user is asking why a specific field "
    "matters for finding protocols. Provide a brief, friendly 1-2 sentence explanation of why "
    "this field is important and how different choices affect protocol selection. Be concise."
)


# Moved from claude_client.py::generate_natural_search_queries. `n` is the
# requested query count, interpolated the same way the original f-string did.
def candidate_query_system(n: int) -> str:
    return (
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
    )


# The labeled profile fields that feed the candidate-query prompt's structured
# block. Moved verbatim from claude_client.generate_natural_search_queries,
# including its or-fallbacks.
def search_query_fields(profile):
    p = profile or {}
    return [
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
