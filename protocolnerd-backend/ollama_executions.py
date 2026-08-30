"""
Backwards-compatibility shim over the pluggable provider layer (`llm_providers`).

Historically every LLM call funnelled through `_retryable_ollama_call` and a set
of `_get_*_model` / `*_available` helpers defined here. That logic now lives in
provider classes in `llm_providers.py`; this module simply re-exports thin
wrappers so existing imports (`claude_client`, `helper_functions`) keep working
unchanged. New code should prefer `llm_providers.call_llm` / `get_provider`.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from llm_providers import (
    MAX_RETRIES,
    BACKOFF_SECS,
    PROVIDER_REGISTRY,
    call_llm,
    get_provider,
    resolve_provider_name,
)

logging.basicConfig(level=logging.INFO)

__all__ = [
    "active_provider",
    "openai_available", "claude_available", "gemini_available",
    "_get_ollama_model", "_get_claude_model", "_get_openai_model", "_get_gemini_model",
    "_retryable_ollama_call", "reinitialize_ollama_client",
    "MAX_RETRIES", "BACKOFF_SECS",
]


def active_provider(override: Optional[str] = None) -> str:
    """Return provider: 'openai', 'claude', 'gemini', or 'ollama'. Accepts override."""
    return resolve_provider_name(override)


# --- model getters (delegate to each provider) -----------------------------

def _get_ollama_model() -> str:
    return PROVIDER_REGISTRY["ollama"].get_model()


def _get_claude_model() -> str:
    return PROVIDER_REGISTRY["claude"].get_model()


def _get_openai_model() -> str:
    return PROVIDER_REGISTRY["openai"].get_model()


def _get_gemini_model() -> str:
    return PROVIDER_REGISTRY["gemini"].get_model()


# --- availability ----------------------------------------------------------

def openai_available() -> bool:
    return PROVIDER_REGISTRY["openai"].is_available()


def claude_available() -> bool:
    return PROVIDER_REGISTRY["claude"].is_available()


def gemini_available() -> bool:
    return PROVIDER_REGISTRY["gemini"].is_available()


def reinitialize_ollama_client() -> None:
    PROVIDER_REGISTRY["ollama"].reinitialize()


# --- core call -------------------------------------------------------------

def _retryable_ollama_call(
    *,
    messages,
    temperature=0.3,
    top_p=1.0,
    response_format: Optional[Dict[str, str]] = None,
    provider: Optional[str] = None,
) -> str:
    """Dispatch an LLM call to the active provider (name kept for compatibility)."""
    return call_llm(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        response_format=response_format,
        provider=provider,
    )
