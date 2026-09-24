"""Per-call instrumentation for the language-model workflow (Reviewers 1.6, 3.7).

Off unless PN_TRACE_LLM names a file, so the shipped path is unchanged in
production: every entry point returns immediately when tracing is off, and no
provider response is inspected.

When on, one JSON line per model call is appended to that file, carrying the
stage, model, wall-clock milliseconds, tokens in and out, how many attempts the
provider needed, and whether the call ultimately returned anything. Appending
rather than accumulating in memory means a run that dies part way still leaves
its measurements behind, and the server can be traced while a separate process
drives it.

The stage is taken from the calling frame rather than from an argument, so no
call site has to change and the labels line up with the stages already named in
the paper's LLM call table.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Optional

_PATH = os.getenv("PN_TRACE_LLM") or ""
_LOCK = threading.Lock()
_LOCAL = threading.local()


def enabled() -> bool:
    return bool(_PATH)


def note_usage(input_tokens: Optional[int], output_tokens: Optional[int],
               attempts: int) -> None:
    """Called by a provider once a request succeeds. Providers that do not
    report usage pass None, which is recorded as null rather than zero so the
    summary can say how much of the cost estimate is grounded."""
    if not _PATH:
        return
    _LOCAL.usage = (input_tokens, output_tokens, attempts)


# Naming the stage by "first frame outside this module" does not work: every call
# routes through at least two generic dispatchers (ollama_executions._retryable_
# ollama_call, claude_client._call), so every stage would carry the same label.
# Rather than maintain a list of shims and rediscover the next one by re-running,
# record the whole chain and pick from it: the stage is the shallowest frame whose
# function is not private, which lands on the named entry points
# (analyze_experiment_request, generate_search_queries, classify_intent) that
# correspond to the rows of the paper's LLM call table.
_PLUMBING = {"llm_trace", "llm_providers"}


def _stack(limit: int = 8) -> list:
    out, f = [], sys._getframe(1)
    while f and len(out) < limit:
        mod = f.f_globals.get("__name__", "").split(".")[-1]
        if mod not in _PLUMBING:
            out.append(f"{mod}.{f.f_code.co_name}")
        f = f.f_back
    return out


def _stage_from(stack: list) -> str:
    for frame in stack:
        if not frame.split(".", 1)[-1].startswith("_"):
            return frame
    return stack[0] if stack else "unknown"


class call:
    """Context manager wrapping one model call.

        with llm_trace.call(model) as c:
            text = provider.chat(...)
            c.result(text)
    """

    def __init__(self, model: str, provider: str = ""):
        self.model, self.provider = model or "", provider or ""
        self.text = ""

    def __enter__(self):
        if _PATH:
            _LOCAL.usage = None
            self.chain = _stack()
            self.stage = _stage_from(self.chain)
            self.t0 = time.perf_counter()
        return self

    def result(self, text: str) -> None:
        self.text = text or ""

    def __exit__(self, exc_type, exc, tb):
        if not _PATH:
            return False
        ms = (time.perf_counter() - self.t0) * 1000.0
        usage = getattr(_LOCAL, "usage", None) or (None, None, None)
        row = {
            "ts": time.time(),
            "stage": self.stage,
            "stack": self.chain,
            "provider": self.provider,
            "model": self.model,
            "ms": round(ms, 1),
            "input_tokens": usage[0],
            "output_tokens": usage[1],
            "attempts": usage[2],
            # A provider that exhausts its retries returns "" rather than raising,
            # so an empty result is the failure signal, not the exception.
            "ok": bool(self.text) and exc_type is None,
            "raised": exc_type.__name__ if exc_type else None,
        }
        line = json.dumps(row)
        with _LOCK:
            with open(_PATH, "a") as fh:
                fh.write(line + "\n")
        return False
