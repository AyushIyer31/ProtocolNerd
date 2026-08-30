"""
Domain registry + router.

Domains register here; the engine calls `route(query, conversation)` to pick the
domain for a request. Phase 1 ships with biology as the sole, default domain
(router always returns it — zero behavior change). Phase 3 adds real
classification (rule/LLM) and additional domains (e.g. chemistry).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .base import Domain
from .biology import BiologyDomain
from .chemistry import ChemistryDomain

DOMAINS: Dict[str, Domain] = {}


def register(domain: Domain) -> None:
    DOMAINS[domain.name] = domain


register(BiologyDomain())
register(ChemistryDomain())


# Keyword domain classifier: the fallback used only when the LLM router is
# unavailable or returns something unrecognized.
#
# Each domain declares its own signature terms (Domain.keywords), so this scores
# every REGISTERED domain generically instead of naming any of them -- adding a
# domain needs no change here. The default domain wins ties and wins outright
# when nothing scores, so a new domain can only ever take a request another
# domain did not claim more strongly.
def _classify_domain_keywords(text: str) -> str:
    low = (text or "").lower()
    default = default_domain_name()
    scores = {
        name: sum(1 for t in (d.keywords or ()) if t in low)
        for name, d in DOMAINS.items()
    }
    if not scores:
        return default
    best = max(scores, key=lambda n: (scores[n], n == default))
    if scores[best] == 0 or scores[best] <= scores.get(default, 0):
        return default
    return best


# Routing decisions are cached by request text. `route()` runs once per chat
# request, but repeat queries and the eval harness would otherwise re-spend a
# call on text already classified.
_ROUTE_CACHE: Dict[str, str] = {}
_ROUTE_CACHE_MAX = 512


def _classify_domain(text: str) -> str:
    """Pick a domain for this request: LLM classifier first, keywords as fallback.

    The classifier is given each registered domain's own `description`, so adding
    a domain needs no router change beyond registering it.
    """
    key = (text or "").strip().lower()
    if not key:
        return default_domain_name()
    if key in _ROUTE_CACHE:
        return _ROUTE_CACHE[key]

    decided = ""
    try:
        import claude_client

        decided = claude_client.classify_domain(
            key,
            {name: d.description for name, d in DOMAINS.items() if d.description},
            default=default_domain_name(),
        )
    except Exception:
        decided = ""

    resolved = decided if decided in DOMAINS else _classify_domain_keywords(key)
    if len(_ROUTE_CACHE) >= _ROUTE_CACHE_MAX:
        _ROUTE_CACHE.clear()
    _ROUTE_CACHE[key] = resolved
    return resolved


def default_domain_name() -> str:
    """The fallback domain: DEFAULT_DOMAIN env var, else whichever registered
    domain declares is_default, else biology. Derived rather than hardcoded so a
    new domain needs no change here."""
    env = os.getenv("DEFAULT_DOMAIN", "").strip()
    if env and env in DOMAINS:
        return env
    for name, d in DOMAINS.items():
        if getattr(d, "is_default", False):
            return name
    return "biology"


def _default_domain() -> Domain:
    return DOMAINS.get(default_domain_name()) or next(iter(DOMAINS.values()))


def get_domain(name: Optional[str] = None) -> Domain:
    return DOMAINS.get((name or default_domain_name()), _default_domain())


def pinnable_domain_names() -> List[str]:
    """Domains a carried profile may pin a conversation to. Everything except the
    default, which is what an unpinned request falls back to anyway."""
    return [n for n in DOMAINS if n != default_domain_name()]


def route(query: str = "", conversation: str = "", override: Optional[str] = None) -> Domain:
    """Pick the domain for a request. `override` (e.g. the carried profile's domain)
    pins the domain across a conversation; otherwise classify from the text."""
    if override and override in DOMAINS:
        return DOMAINS[override]
    return DOMAINS.get(_classify_domain(f"{query} {conversation}"), _default_domain())


def active_domain(query: str = "", conversation: str = "", override: Optional[str] = None) -> Domain:
    return route(query, conversation, override)


# Request-scoped current domain (mirrors claude_client.set_provider): the chat
# endpoint resolves the domain once and sets it here; engine helpers read it via
# current_domain() without threading it through every signature.
_current_domain_name: Optional[str] = None


def set_current_domain(name: Optional[str]) -> None:
    global _current_domain_name
    _current_domain_name = name


def current_domain() -> Domain:
    return get_domain(_current_domain_name)
