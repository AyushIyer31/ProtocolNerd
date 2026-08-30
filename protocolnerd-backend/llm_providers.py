"""
Pluggable LLM provider layer (strategy pattern).

Each provider is a small class implementing the `LLMProvider` interface:
one `chat()` method plus availability/model helpers. Providers self-register
in `PROVIDER_REGISTRY`, so adding a new backend is a matter of writing one
class and calling `register(...)` — no edits to any caller.

The high-level planner (`claude_client.py`) never names a provider; it calls
`call_llm(...)`, which resolves the active provider from `LLM_PROVIDER` (or a
per-request override) and dispatches. `ollama_executions.py` is kept as a thin
compatibility shim over this module so existing imports keep working.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env first (authoritative for keys), then variables.env as a fallback.
# override=False => the first-loaded value wins. Loaded here because this is the
# lowest layer that reads provider env vars.
load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(Path(__file__).parent / "variables.env", override=False)

MAX_RETRIES = 3
BACKOFF_SECS = 2

Messages = List[Dict[str, str]]

# Request-scoped per-provider model override (e.g. switch Claude to Haiku for one
# request). Set by the chat endpoint; a provider's get_model() prefers it over the
# env default. None/absent -> use the env default.
_MODEL_OVERRIDE: Dict[str, str] = {}


def set_model_override(provider_name: str, model: Optional[str]) -> None:
    if model:
        _MODEL_OVERRIDE[provider_name] = model
    else:
        _MODEL_OVERRIDE.pop(provider_name, None)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip('"').strip()


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class LLMProvider:
    """Common interface every provider implements."""

    name: str = "base"

    def get_model(self) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def chat(
        self,
        *,
        messages: Messages,
        temperature: float = 0.3,
        top_p: float = 1.0,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Return the model's text reply, or "" on failure. `model` overrides the
        provider's default model for this one call (used by the reranker)."""
        raise NotImplementedError


def _wants_json(response_format: Optional[Dict[str, str]]) -> bool:
    return bool(response_format and response_format.get("type") == "json_object")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        self._client = None

    def get_model(self) -> str:
        return _env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

    def _get_client(self):
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                logging.warning("OPENAI_API_KEY not set; OpenAI provider unavailable.")
                return None
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:  # noqa: BLE001
                logging.error(f"Failed to create OpenAI client: {e}")
                self._client = None
        return self._client

    def is_available(self) -> bool:
        return self._get_client() is not None

    def chat(self, *, messages, temperature=0.3, top_p=1.0, response_format=None, model=None) -> str:
        client = self._get_client()
        if client is None:
            return ""
        kwargs: Dict[str, Any] = dict(
            model=model or self.get_model(),
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=2048,
        )
        if _wants_json(response_format):
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None)
                retryable = status in (408, 429, 500, 503) or status is None
                logging.warning(f"[OpenAI attempt {attempt + 1}/{MAX_RETRIES}] {e}")
                if not retryable:
                    break
                time.sleep(BACKOFF_SECS * (2 ** attempt))
        return ""


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self) -> None:
        self._client = None

    def get_model(self) -> str:
        return _MODEL_OVERRIDE.get(self.name) or _env("CLAUDE_MODEL", "claude-sonnet-4-6") or "claude-sonnet-4-6"

    def _max_tokens(self) -> int:
        return int(_env("CLAUDE_MAX_TOKENS", "4096") or "4096")

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logging.warning("ANTHROPIC_API_KEY not set; Claude provider unavailable.")
                return None
            logging.info(f"✅ ANTHROPIC_API_KEY found: {api_key[:30]}...")
            try:
                import anthropic
            except ImportError:
                logging.error("anthropic package not installed. Run: pip install anthropic")
                return None
            try:
                self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
                logging.info("✅ Claude client created successfully")
            except Exception as e:  # noqa: BLE001
                logging.error(f"Failed to create Anthropic client: {e}")
                self._client = None
        return self._client

    def is_available(self) -> bool:
        return self._get_client() is not None

    def chat(self, *, messages, temperature=0.3, top_p=1.0, response_format=None, model=None) -> str:
        client = self._get_client()
        if client is None:
            return ""
        # Anthropic takes the system prompt separately from the turn messages.
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        system = "\n\n".join(p for p in system_parts if p)
        if _wants_json(response_format):
            system += "\n\nReturn ONLY a single valid JSON value. No markdown fences, no commentary."
        # Note: Opus 4.8 rejects temperature/top_p (400), so we omit sampling
        # params entirely — omitting is valid on every Claude model.
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.messages.create(
                    model=model or self.get_model(),
                    max_tokens=self._max_tokens(),
                    system=system or "You are a helpful assistant.",
                    messages=convo or [{"role": "user", "content": ""}],
                )
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None)
                retryable = status in (408, 409, 429, 500, 529) or status is None
                logging.warning(f"[Claude attempt {attempt + 1}/{MAX_RETRIES}] {e}")
                if not retryable:
                    break
                time.sleep(BACKOFF_SECS * (2 ** attempt))
        return ""


# ---------------------------------------------------------------------------
# Google Gemini (REST, no SDK dependency)
# ---------------------------------------------------------------------------

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
try:
    import certifi as _certifi
    import ssl as _ssl
    _GEMINI_SSL_CTX = _ssl.create_default_context(cafile=_certifi.where())
except Exception:  # noqa: BLE001
    _GEMINI_SSL_CTX = None


class GeminiProvider(LLMProvider):
    name = "gemini"

    def get_model(self) -> str:
        return _env("GEMINI_MODEL", "gemini-2.0-flash") or "gemini-2.0-flash"

    def _api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip('"').strip()

    def is_available(self) -> bool:
        if not self._api_key():
            logging.warning("GEMINI_API_KEY not set; Gemini provider unavailable.")
            return False
        return True

    def chat(self, *, messages, temperature=0.3, top_p=1.0, response_format=None, model=None) -> str:
        import json as _json
        import urllib.request as _ur

        key = self._api_key()
        if not key:
            return ""

        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_text = "\n\n".join(p for p in system_parts if p)
        if _wants_json(response_format):
            system_text += "\n\nReturn ONLY a single valid JSON value. No markdown fences, no commentary."

        contents = [
            {
                "role": "model" if m.get("role") == "assistant" else "user",
                "parts": [{"text": m.get("content", "")}],
            }
            for m in messages
            if m.get("role") in ("user", "assistant")
        ] or [{"role": "user", "parts": [{"text": ""}]}]

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "topP": top_p, "maxOutputTokens": 2048},
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if _wants_json(response_format):
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = f"{_GEMINI_ENDPOINT}/{model or self.get_model()}:generateContent?key={key}"
        data = _json.dumps(body).encode("utf-8")

        for attempt in range(MAX_RETRIES):
            try:
                req = _ur.Request(url, data=data, headers={"Content-Type": "application/json"})
                with _ur.urlopen(req, timeout=30, context=_GEMINI_SSL_CTX) as r:
                    payload = _json.loads(r.read().decode("utf-8", errors="ignore"))
                candidates = payload.get("candidates") or []
                if not candidates:
                    return ""
                parts = (candidates[0].get("content") or {}).get("parts") or []
                return "".join(p.get("text", "") for p in parts).strip()
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "code", None)
                retryable = status in (408, 429, 500, 503) or status is None
                logging.warning(f"[Gemini attempt {attempt + 1}/{MAX_RETRIES}] {e}")
                if not retryable:
                    break
                time.sleep(BACKOFF_SECS * (2 ** attempt))
        return ""


# ---------------------------------------------------------------------------
# Local Ollama (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self) -> None:
        self._client = None
        self._built = False

    def get_model(self) -> str:
        return _env("OLLAMA_MODEL", "llama3.2") or "llama3.2"

    def _base_url(self) -> str:
        base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
        if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
            base_url = base_url.rstrip("/") + "/v1/"
        return base_url

    def _get_client(self):
        if not self._built:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=self._base_url(), api_key="ollama")
            except Exception as e:  # noqa: BLE001
                logging.error(f"Failed to create Ollama client: {e}")
                self._client = None
            self._built = True
        return self._client

    def reinitialize(self) -> None:
        self._built = False
        self._client = None
        client = self._get_client()
        if client:
            logging.info(f"Ollama client reinitialized (model={self.get_model()}, url={self._base_url()})")
        else:
            logging.warning("Ollama client could not be initialized")

    def is_available(self) -> bool:
        return self._get_client() is not None

    def chat(self, *, messages, temperature=0.3, top_p=1.0, response_format=None, model=None) -> str:
        from openai import RateLimitError, APITimeoutError, APIConnectionError, NotFoundError
        client = self._get_client()
        if not client:
            logging.error("Ollama client not initialized.")
            return ""
        model = model or self.get_model()
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        if response_format:
            kwargs["response_format"] = response_format
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                logging.warning(f"[Ollama attempt {attempt + 1}/{MAX_RETRIES}] {e}")
                time.sleep(BACKOFF_SECS * (2 ** attempt))
            except NotFoundError:
                logging.error(f"Ollama model '{model}' not found. Run: ollama pull {model}")
                break
            except Exception as e:  # noqa: BLE001
                logging.exception(f"[Ollama fatal] {e}")
                break
        return ""


# ---------------------------------------------------------------------------
# Registry + resolution
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: Dict[str, LLMProvider] = {}

# Aliases map user-facing names to a canonical provider key. Anything unset or
# unknown falls back to Claude — the canonical default everywhere (Sonnet for
# general calls, Haiku for the re-ranker).
_ALIASES = {
    "openai": "openai", "gpt": "openai",
    "claude": "claude", "anthropic": "claude",
    "gemini": "gemini", "google": "gemini",
    "ollama": "ollama",
}


def register(provider: LLMProvider) -> None:
    PROVIDER_REGISTRY[provider.name] = provider


for _p in (OpenAIProvider(), ClaudeProvider(), GeminiProvider(), OllamaProvider()):
    register(_p)


def resolve_provider_name(override: Optional[str] = None) -> str:
    """Canonical provider key from an override or the LLM_PROVIDER env var."""
    raw = (override or os.getenv("LLM_PROVIDER", os.getenv("LLM", "claude")) or "").strip('"').strip().lower()
    return _ALIASES.get(raw, "claude")


def get_provider(name: Optional[str] = None) -> LLMProvider:
    return PROVIDER_REGISTRY[resolve_provider_name(name)]


def call_llm(
    *,
    messages: Messages,
    temperature: float = 0.3,
    top_p: float = 1.0,
    response_format: Optional[Dict[str, str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Resolve the active provider and dispatch a chat call. "" on failure.
    `model` overrides the provider's default model for this one call."""
    return get_provider(provider).chat(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        response_format=response_format,
        model=model,
    )
