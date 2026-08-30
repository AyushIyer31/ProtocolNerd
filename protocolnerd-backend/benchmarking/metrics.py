"""
Standard information-retrieval metrics.

All functions take a `ranked_ids` list (system output, best-first) and evaluate
it against ground truth. Two shapes of ground truth are supported:

  * known-item / binary  -> `gold_ids`: a set of ids that count as correct.
  * graded relevance      -> `relevance`: {id: grade} where grade in {0,1,2}.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def _as_int_set(ids: Iterable) -> set:
    return {int(x) for x in ids if x is not None}


def hit_at_k(ranked_ids: Sequence, gold_ids: Iterable, k: int) -> int:
    """1 if any gold id appears in the top-k, else 0."""
    gold = _as_int_set(gold_ids)
    top = [int(x) for x in ranked_ids[:k]]
    return int(any(pid in gold for pid in top))


def reciprocal_rank(ranked_ids: Sequence, gold_ids: Iterable) -> float:
    """1 / rank of the first gold id (0 if none found)."""
    gold = _as_int_set(gold_ids)
    for i, pid in enumerate(ranked_ids, start=1):
        if int(pid) in gold:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_ids: Sequence, relevance: Dict[int, int], k: int,
                   relevant_grade: int = 1) -> float:
    """Fraction of the top-k whose graded relevance is >= relevant_grade."""
    if k <= 0:
        return 0.0
    top = [int(x) for x in ranked_ids[:k]]
    hits = sum(1 for pid in top if relevance.get(pid, 0) >= relevant_grade)
    return hits / k


def dcg_at_k(ranked_ids: Sequence, relevance: Dict[int, int], k: int) -> float:
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k], start=1):
        grade = relevance.get(int(pid), 0)
        if grade:
            dcg += (2 ** grade - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: Sequence, relevance: Dict[int, int], k: int) -> float:
    """
    Normalized DCG@k. `relevance` maps id -> grade (0/1/2). The ideal ranking is
    the graded relevances sorted descending — so nDCG rewards putting the most
    relevant results highest.
    """
    dcg = dcg_at_k(ranked_ids, relevance, k)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1)
               for i, g in enumerate(ideal_grades, start=1) if g)
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
