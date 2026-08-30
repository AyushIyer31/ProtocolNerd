"""
Conditional operators inside profile field values.

A user can type an operator into any field (via "Others: please specify"):

  OR    "rice or tomato"            -> match either
  AND   "rice and maize" / "x + y"  -> match both
  LIKE  "like tomato" /             -> soft preference (relax to similar);
        "similar to tomato" /          organism also expands to relatives
        "such as tomato"

One operator per field (flat). Word operators only — "/" and "," are NOT
operators because existing canonical values already use them as part of single
concept labels ("CRISPR / genome editing", "RNA level / qPCR").

The value is kept verbatim for display and the protocols.io query (TF-IDF
handles "rice or tomato" loosely); this module renders the operator semantics
for PubMed, whose esearch is strict boolean AND.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Leading LIKE markers (checked longest-first).
_LIKE_MARKERS = ("similar to", "such as", "like")
# Operator words to strip when reducing a value to bare PubMed tokens.
_OPERATOR_WORDS = {"or", "and", "like", "such", "as", "similar", "to"}

_OR_SPLIT = re.compile(r"\bor\b", re.IGNORECASE)
_AND_SPLIT = re.compile(r"\band\b|\+")
_OR_SEARCH = re.compile(r"\bor\b", re.IGNORECASE)
_AND_SEARCH = re.compile(r"\band\b|\+", re.IGNORECASE)


def detect_operator(value: str) -> Tuple[str, List[str], str]:
    """
    Classify a field value's conditional operator.

    Returns (operator, operands, term) where:
      operator ∈ {"OR", "AND", "LIKE", ""}
      operands = the parts (>=2 for OR/AND; the single term for LIKE/none)
      term     = the LIKE/plain term with the marker stripped (else the value)

    One operator per field: LIKE (leading marker) wins, then OR, then AND.
    """
    v = " ".join((value or "").split())
    if not v:
        return ("", [], "")
    low = v.lower()

    # LIKE — leading marker only ("like tomato", "similar to tomato").
    for marker in _LIKE_MARKERS:
        if low == marker or low.startswith(marker + " "):
            term = v[len(marker):].strip() or v
            return ("LIKE", [term], term)

    # OR
    if _OR_SEARCH.search(v):
        operands = [p.strip() for p in _OR_SPLIT.split(v) if p.strip()]
        if len(operands) >= 2:
            return ("OR", operands, v)

    # AND ("and" word or "+")
    if _AND_SEARCH.search(v):
        operands = [p.strip() for p in _AND_SPLIT.split(v) if p.strip()]
        if len(operands) >= 2:
            return ("AND", operands, v)

    return ("", [v], v)


def is_like(value: str) -> bool:
    """True when the value uses a LIKE marker (for ranker relaxation)."""
    return detect_operator(value)[0] == "LIKE"


def pubmed_terms(value: str) -> List[str]:
    """
    Render a field value as PubMed term fragment(s), AND-joined by the caller:

      OR   -> ["(rice OR tomato)"]          (real disjunction)
      AND  -> ["rice", "maize"]             (PubMed ANDs by default)
      LIKE -> ["tomato"]                    (bare term; "soft" handled in ranker)
      none -> ["crispr", "genome", "editing"] (deduped by the caller)

    Operator words are stripped from plain token output.
    """
    op, operands, term = detect_operator(value)
    if op == "OR" and len(operands) >= 2:
        return ["(" + " OR ".join(operands) + ")"]
    if op == "AND" and len(operands) >= 2:
        text = " ".join(operands)
    elif op == "LIKE":
        text = term
    else:
        text = value
    tokens = text.replace("/", " ").replace("+", " ").split()
    return [t for t in tokens if t.lower() not in _OPERATOR_WORDS]
