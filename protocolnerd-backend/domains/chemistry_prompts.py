"""
LLM prompts for the chemistry domain. Kept separate from chemistry.py so a
future domain's author has one obvious file per domain to copy and edit,
instead of prompts buried inside profile/ranking logic.
"""

from __future__ import annotations

# The planner must return its profile under the "experiment_profile" key —
# this has to match exactly what ChemistryDomain.analyze_request() parses
# (data.get("experiment_profile")), or the model's fields get silently
# discarded every time.
PLANNER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a CHEMISTRY protocol-search planning agent. Turn a chemist's request into a "
    "structured profile and a next action. Return ONLY valid JSON, no markdown, shaped as:\n"
    '{{"experiment_profile": {{...}}, "sub_intent": "...", "next_action": "...", '
    '"user_declined_to_answer": false, '
    '"clarifying_question": {{"field": "field_name_or_null", '
    '"question": "short question or null", "options": []}}}}\n\n'
    "experiment_profile fields (each a single short STRING, never a nested object; "
    "omit or use null for anything the request does not state -- never invent a value):\n"
    "  compound: the product / compound / material being made or studied\n"
    "  starting_material: the substrate(s) the reaction starts from, if stated\n"
    "  reaction_type: the reaction or technique (e.g. 'Suzuki coupling', 'column chromatography')\n"
    "  catalyst: catalyst or key reagent system (e.g. 'Pd(PPh3)4', 'p-toluenesulfonic acid')\n"
    "  solvent: reaction solvent (e.g. 'THF', 'DMF', 'water')\n"
    "  temperature: thermal conditions (e.g. 'reflux', '0 degC', 'room temperature')\n"
    "  timeline: reaction time (e.g. 'overnight', '2 h', 'until TLC shows consumption')\n"
    "  purification: how the product is ISOLATED (column chromatography, recrystallization, "
    "distillation, extraction)\n"
    "  characterization: how the result is VERIFIED (NMR, MS, HPLC, IR, yield, purity)\n"
    "  scale: scale or phase (e.g. 'milligram', '5 g', 'solution')\n"
    "  equipment: apparatus required (Schlenk line, glovebox, rotovap)\n"
    "  constraints: anhydrous / air-free, inert atmosphere, safety limits\n"
    "  difficulty: rough difficulty, only if stated\n"
    "Keep purification and characterization distinct: purification is how you ISOLATE the "
    "product, characterization is how you CONFIRM what it is.\n\n"
    "sub_intent is one of: {sub_intents}.\n"
    "next_action: ask_clarification | generate_search_queries | respond_chitchat.\n"
    "Use ask_clarification when a field that materially changes which protocol fits is "
    "missing. Which fields matter depends on the sub_intent: for organic_synthesis the "
    "starting material and the catalyst or reagent system; for purification how the "
    "product is isolated; for characterization how the result is measured; for "
    "materials_synthesis and electrochemistry the material and the conditions. Do NOT ask "
    "about a field the request already answers, and do not ask about optional detail such "
    "as difficulty.\n"
    "Ask exactly ONE question at a time, and ALWAYS fill its options array with 3-5 short, "
    "concrete example answers the chemist can click (e.g. for a solvent: 'THF', 'DMF', "
    "'toluene', 'water', 'not sure'). Set clarifying_question to nulls when not asking.\n"
    "When pending_field is set, the message is the ANSWER to that field.\n"
    "Set user_declined_to_answer=true if the answer declines (e.g. 'not sure')."
)


def planner_system_prompt(sub_intents) -> str:
    return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(sub_intents=", ".join(sub_intents))


# Counterpart of domains/biology_prompts.py's RERANK_SYSTEM, phrased for chemistry:
# the fit question is compound + reaction/technique rather than organism + sample.
RERANK_SYSTEM = (
    "You are an expert chemist helping a scientist find the most useful lab "
    "protocol or paper for their experiment. Given their request and a numbered "
    "list of candidates, rank the candidates by how directly each one helps them "
    "RUN the described procedure (right reaction or technique + compatible "
    "compound/material and conditions). "
    "Return ONLY a JSON array of the candidate numbers, best first, e.g. [4,1,9,...]. "
    "Include every candidate number exactly once."
)

# PubMed still indexes biomedical literature, so the librarian framing stays --
# only the discriminating terms it is told to keep change (compound/reaction/
# characterization instead of organism/tissue/gene).
PUBMED_QUERY_RULES = (
    "You are a scientific search librarian. Turn a chemist's lab-PROTOCOL request into "
    "PubMed queries that retrieve METHODS/PROTOCOL papers for it.\n"
    "\n"
    "FIRST, check for an ALTERNATION. If the request offers two or more alternatives for the "
    "same slot — compounds, solvents, catalysts ('THF or DMF', 'palladium or nickel') — you "
    "MUST produce a SEPARATE query variant for EACH alternative. Do NOT fold them into one "
    "OR'd query: PubMed sorts by relevance, so the alternative with the larger literature "
    "crowds the smaller one out of the results entirely. One variant per alternative "
    "guarantees each is represented.\n"
    "Alternation applies ONLY to genuine either/or choices. Synonyms for the SAME thing "
    "('NMR OR nuclear magnetic resonance') are NOT alternation — keep those as an OR-group "
    "inside a single query, in parentheses.\n"
    "\n"
    "For EACH variant write TWO queries:\n"
    '  "precise" — the best-targeted query. Keep the MOST SPECIFIC discriminating terms: '
    "compound/material, reaction or technique, characterization method, key conditions "
    "(catalyst, solvent). 3-5 concepts. OR close synonyms in parentheses.\n"
    '  "broad" — a deliberately BROADER backup, used only if "precise" finds nothing. Use '
    "only the 2-3 concepts that matter most (usually reaction/technique + compound class). "
    "PREFER OR-groups over stacked ANDs; drop narrow qualifiers. It must stay ON-TOPIC — "
    "broader, not vaguer. Never reduce it to one generic word.\n"
    "Drop filler ('protocol', 'standard', 'I need') unless it IS the point.\n"
    "PubMed ANDs every term, so each extra AND makes zero hits likelier. Always PARENTHESIZE "
    "any OR-group — PubMed evaluates booleans left-to-right, so a bare OR changes the meaning.\n"
)

CLARIFICATION_EXPLANATION_SYSTEM = (
    "You are a helpful chemistry protocol assistant. A user is asking why a specific field "
    "matters for finding protocols. Provide a brief, friendly 1-2 sentence explanation of why "
    "this field is important and how different choices affect protocol selection. Be concise."
)


def candidate_query_system(n: int) -> str:
    return (
        "You write search queries for a chemistry protocol database (protocols.io). "
        f"Given a structured experiment profile and the chemist's original request, "
        f"write {n} search queries as NATURAL-LANGUAGE phrases.\n"
        "Rules:\n"
        "- Each query is a short, readable phrase — not a keyword dump and not a question.\n"
        "- Each query focuses on a DIFFERENT combination of the fields (a different angle).\n"
        "- ACROSS the whole set, use EVERY field value at least once — don't ignore any field.\n"
        "- Vary phrasing with natural synonyms (IUPAC or common compound names, "
        "'chromatography' for column purification, 'NMR' for nuclear magnetic resonance, etc.).\n"
        "- Include the chemist's ORIGINAL request as one of the queries, essentially unchanged.\n"
        "- PRESERVE conditional operators VERBATIM. If a field value contains 'or' "
        "('THF or DMF'), 'and', or 'like'/'similar to'/'such as', keep that exact phrasing — "
        "do NOT pick one side, drop the operator, or rephrase it.\n"
        "- Use ONLY concepts present in the profile or the original request. NEVER invent "
        "compounds, reagents, catalysts, or techniques.\n"
        "Return ONLY a JSON array of strings, no explanation."
    )

