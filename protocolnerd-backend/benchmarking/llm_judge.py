"""
LLM-as-judge: graded relevance scoring.

Given a query and a single protocol result, return a relevance grade:
    2 = directly usable protocol for the described experiment
    1 = related / adaptable (right area, wrong specifics)
    0 = off-topic / not useful

The judge sees ONLY the query + result text -- it is blind to which system
produced the result, which removes system-favouring bias. Judgments are cached
by (query, result_id) so the same pair is never re-judged (and never re-billed).

Validate the judge before trusting it: have a human grade a sample of the same
pairs and compare (see README, "validating the judge").
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

import _bootstrap  # noqa: F401
from _bootstrap import CACHE_DIR
from llm_providers import call_llm  # type: ignore

_JUDGE_CACHE = CACHE_DIR / "judge.json"

# The judge is ALWAYS a local Llama (Ollama), independent of the app's LLM_PROVIDER.
# Using a different model from the Claude re-ranker removes the circularity where a
# ranker's own model grades its output. Overridable via JUDGE_PROVIDER for experiments.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "ollama")

SYSTEM_PROMPT = (
    "You are an expert biologist grading how relevant a search result is to a "
    "scientist's request. The result may be a step-by-step lab PROTOCOL or a "
    "research PAPER -- grade both on the same basis: does it help the scientist "
    "actually run the described experiment? Do NOT reward or penalize a result "
    "just for being a protocol vs a paper. Grade STRICTLY on this scale:\n"
    "  2 = directly usable: gives what you need to run this experiment (a protocol "
    "for it, or a paper whose methods let you reproduce it) with a compatible "
    "organism/sample.\n"
    "  1 = related/adaptable: same general area but wrong specifics (right "
    "technique different organism; only part of the goal; or a paper that mentions "
    "but doesn't detail the method).\n"
    "  0 = off-topic: different technique or unrelated purpose.\n"
    "Judge only on the text shown. Respond with ONLY a JSON object: "
    '{\"grade\": 0|1|2, \"reason\": \"<=12 words\"}.'
)


def _cache_file(name: str):
    return CACHE_DIR / f"{name}.json"


def _load(path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(path, cache: Dict[str, Any]) -> None:
    path.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def _parse_grade(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            g = int(obj.get("grade"))
            if g in (0, 1, 2):
                return {"grade": g, "reason": str(obj.get("reason", ""))[:80]}
        except Exception:
            pass
    # Fallback: first bare 0/1/2 in the text.
    m = re.search(r"[012]", text)
    return {"grade": int(m.group(0)) if m else 0, "reason": "parse-fallback"}


def judge(query: str, result: Dict[str, Any], use_cache: bool = True,
          provider: str = None, cache_name: str = "judge") -> int:
    """Grade a (query, result) pair 0/1/2. `provider` selects the judging model;
    None -> JUDGE_PROVIDER (local Llama by default), so the eval judge never falls
    back to the app's Claude default. `cache_name` picks the cache file so
    different judges never share grades."""
    provider = provider or JUDGE_PROVIDER
    path = _cache_file(cache_name)
    rid = result.get("id")
    key = f"{query.strip()}||{rid}"
    cache = _load(path) if use_cache else {}
    if use_cache and key in cache:
        return int(cache[key]["grade"])

    user = (
        f"SCIENTIST'S REQUEST:\n{query}\n\n"
        f"RESULT:\nTitle: {result.get('title','')}\n"
        f"Description: {(result.get('description') or '')[:400]}"
    )
    try:
        raw = call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            provider=provider,
        )
        verdict = _parse_grade(raw)
    except Exception as e:
        verdict = {"grade": 0, "reason": f"error:{e}"[:80]}

    if use_cache:
        cache[key] = verdict
        _save(path, cache)
    return int(verdict["grade"])
