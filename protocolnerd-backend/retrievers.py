"""
Pluggable retrieval layer (strategy pattern for search sources).

Each source (protocols.io local index, PubMed, …) is a `Retriever` that returns
a list of **normalized result dicts**. Common keys across sources:

    title        — result title
    description  — body/abstract text (PubMed uses `abstract`; both are read)
    source       — provider tag ("protocols.io", "pubmed", …)
    url / doi     — link
    + source-specific metadata (score, pmid, authors, …)

Retrievers self-register in `RETRIEVERS`, so adding a source is a new class +
`register(...)` — the blend step (`blend_ranking.blend_sources`) already handles
an arbitrary number of sources on one comparable relevance axis.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from query_operators import pubmed_terms

log = logging.getLogger(__name__)

_SKIP = {"", "not specified", "none", "unknown", "not sure", "null"}


@dataclass
class RetrievalContext:
    """Everything a retriever might need for one search turn. Passed to
    `retrieve()` so the orchestrator can drive every source through one loop."""
    queries: List[str] = field(default_factory=list)   # candidate search queries
    structured_query: str = ""                         # profile-derived query string
    profile: Optional[Dict[str, Any]] = None
    k: int = 5                                          # results to pull from this source
    raw_query: str = ""                                # the user's original request
    search_mode: str = "local"                         # "local" | "live"


class Retriever:
    """Common interface every search source implements."""

    name: str = "base"

    def is_enabled(self) -> bool:
        return True

    def search(self, query: str, k: int = 5, *, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Return up to `k` normalized result dicts (each tagged with `source`)."""
        raise NotImplementedError

    def retrieve(self, ctx: RetrievalContext) -> List[Dict[str, Any]]:
        """Context-driven entry point used by the orchestrator's fetch loop.
        Default: search the first candidate query (or the structured query)."""
        query = (ctx.queries[0] if ctx.queries else ctx.structured_query) or ""
        return self.search(query, ctx.k, profile=ctx.profile)


# ---------------------------------------------------------------------------
# PubMed (NCBI E-utilities)
# ---------------------------------------------------------------------------

def pubmed_core_query(profile: Optional[Dict[str, Any]], fallback: str) -> str:
    """Build a trimmed core query for PubMed from the profile.

    PubMed ANDs every term, so the full specific candidate (e.g. "CRISPR Cas
    genome editing tomato whole plant stable transformation") returns ~1 hit.
    The core concepts (technique + organism + a couple of key fields) — e.g.
    "CRISPR genome editing tomato stable transformation" — return ~18. We collect
    the most discriminating fields in priority order, de-dup at the TOKEN level
    (so "CRISPR / genome editing" + "genome editing" collapse cleanly), and cap
    the length so PubMed's AND stays satisfiable.
    """
    if not profile:
        return fallback
    # Priority order: technique first, then target/organism, then delivery.
    # Deliberately EXCLUDES over-constraining qualifiers (tissue/"whole plant",
    # readout, intent_specific extras like "editing tool: CRISPR/Cas").
    # NOTE: `sub_intent` is intentionally NOT here — its value is a raw enum token
    # ("stress_tolerance_assay") that appears in no PubMed article and, since
    # PubMed ANDs every term, single-handedly forces 0 results. The human-readable
    # concept is already carried by `experimental_method`.
    ordered_fields = (
        "modification_type", "experimental_method",
        "target", "gene_or_construct", "organism", "delivery_method",
    )
    # For stress-tolerance / phenotyping assays the CONDITION (e.g. "drought") is
    # the core searchable concept, not an over-constraint — search on it up front.
    sub_intent = str(profile.get("sub_intent") or "").strip().lower()
    intent_family = str(profile.get("intent_family") or "").strip().lower()
    if sub_intent == "stress_tolerance_assay" or "phenotyp" in intent_family:
        ordered_fields = ("condition",) + ordered_fields
    # Each field renders to PubMed fragment(s): OR -> "(rice OR tomato)", AND ->
    # both tokens, LIKE -> bare term, plain -> deduped tokens. Fragments are
    # space-joined (PubMed ANDs them); a parenthesized OR group stays a real
    # disjunction inside that AND.
    terms: List[str] = []
    seen: set = set()
    for field in ordered_fields:
        val = str(profile.get(field) or "").strip()
        if not val or val.lower() in _SKIP:
            continue
        for frag in pubmed_terms(val):
            key = frag.lower()
            if frag.startswith("("):  # OR group — keep verbatim, don't dedup tokens
                terms.append(frag)
                seen.update(frag.strip("()").lower().split())
            elif key not in seen:
                seen.add(key)
                terms.append(frag)
        if len(terms) >= 7:  # cap concepts so PubMed's AND stays satisfiable
            break
    query = " ".join(terms[:7]).strip()
    return query if len(query.split()) >= 2 else fallback


class PubMedRetriever(Retriever):
    name = "pubmed"

    def core_query(self, profile: Optional[Dict[str, Any]], fallback: str) -> str:
        """The trimmed profile-derived query PubMed is searched with."""
        return pubmed_core_query(profile, fallback)

    def is_enabled(self) -> bool:
        # PubMed is best-effort; always attempt it (failures are caught upstream).
        return True

    def search(self, query: str, k: int = 5, *, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        from pubmed_client import search_pubmed
        results = search_pubmed(query, k) or []
        for r in results:
            r.setdefault("source", "pubmed")
        return results


# ---------------------------------------------------------------------------
# protocols.io local TF-IDF index
# ---------------------------------------------------------------------------

class LocalProtocolsRetriever(Retriever):
    """Wraps the prebuilt local TF-IDF index. The in-memory index is injected at
    startup (see `set_index`). NOTE: the orchestrator's richer multi-variant local
    search still runs in main.py; this retriever is the single-query interface and
    the natural home to migrate that logic into next."""

    name = "protocols.io"

    def __init__(self, index: Optional[Dict[str, Any]] = None) -> None:
        self._index = index

    def set_index(self, index: Optional[Dict[str, Any]]) -> None:
        self._index = index

    def is_enabled(self) -> bool:
        return bool(self._index)

    def search(self, query: str, k: int = 5, *, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._index:
            return []
        from protocol_rag import multi_search_protocols
        results = multi_search_protocols(self._index, [query], k) or []
        for r in results:
            r.setdefault("source", "protocols.io")
        return results


# ---------------------------------------------------------------------------
# Europe PMC (methods-section literature search) — per-domain
# ---------------------------------------------------------------------------

class EuropePMCRetriever(Retriever):
    """Europe PMC, the literature lane for domains that declare it.

    Chemistry pairs protocols.io with Europe PMC instead of PubMed because
    Protocols.io holds almost no synthetic chemistry, while Europe PMC's
    METHODS: field query searches inside papers' methods sections, which is
    where chemistry procedures actually live.

    On for any domain whose `paper_sources` names it and for no other, so a
    biology request never sees this source and its result set is untouched.
    ENABLE_EUROPEPMC=0 is the kill switch that forces it off everywhere.
    """

    name = "europepmc"

    def is_enabled(self) -> bool:
        if os.getenv("ENABLE_EUROPEPMC", "1").strip().lower() in ("0", "false", "no"):
            return False
        try:
            # Deferred: domains/biology.py imports THIS module, so a top-level
            # `from domains import ...` here would be a circular import.
            from domains import current_domain
            return self.name in (current_domain().paper_sources or ())
        except Exception:  # noqa: BLE001 -- unknown state means "source absent"
            return False

    def search(self, query: str, k: int = 5, *, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        from europepmc_client import search_with_fallback
        results = search_with_fallback(profile, query, k) or []
        for r in results:
            r.setdefault("source", self.name)
        return results

    def retrieve(self, ctx: "RetrievalContext") -> List[Dict[str, Any]]:
        query = (ctx.queries[0] if ctx.queries else ctx.structured_query) or ctx.raw_query
        return self.search(query, ctx.k, profile=ctx.profile)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RETRIEVERS: Dict[str, Retriever] = {}


def register(retriever: Retriever) -> None:
    RETRIEVERS[retriever.name] = retriever


register(PubMedRetriever())
register(LocalProtocolsRetriever())
register(EuropePMCRetriever())


class _DemoRetriever(Retriever):
    """Example third source. Registered only when ENABLE_DEMO_RETRIEVER is set —
    a template showing that adding a source is a Retriever class + register(),
    with no orchestrator edit (the registry-driven fetch loop picks it up)."""

    name = "demo"

    def search(self, query: str, k: int = 5, *, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [{
            "title": f"[demo] {query}",
            "description": f"Synthetic demo result for: {query}",
            "source": "demo",
            "url": "https://example.org/demo",
        }][:k]


if os.getenv("ENABLE_DEMO_RETRIEVER", "").strip():
    register(_DemoRetriever())


def active_retrievers() -> List[Retriever]:
    return [r for r in RETRIEVERS.values() if r.is_enabled()]
