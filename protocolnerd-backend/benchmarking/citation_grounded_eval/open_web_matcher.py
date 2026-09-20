"""Scoring rule for systems that search the open web, for the Reviewer 1.1
comparison against Consensus, Elicit and Perplexity.

Why this exists. The biology benchmark's 100 target protocols are not all
hosted on Protocols.io: 52 carry a Protocols.io DOI, 26 a Nature Protocol
Exchange DOI, 21 a Springer or Nature Protocols DOI and 1 a Humana DOI. Every
one of them is in the Protocols.io corpus, so a corpus-scoped system can always
reach it by id, but a system that searches the whole web will often return the
same protocol at its publisher instead. Scoring only Protocols.io URLs would
mark those correct answers wrong, which is indefensible in review.

So a candidate counts as the target protocol under three tiers:

  1. protocols.io   the candidate is a Protocols.io id, DOI slug or /view/<slug>
                    URL for a confirmed target, resolving the slug through
                    Protocols.io's own search API when the raw strings differ.
                    This is the existing rule, reused unchanged from
                    run_llm_websearch_baseline.
  2. doi            the target's DOI appears in the candidate URL or in the
                    fetched page text. Covers Springer chapters
                    (link.springer.com/protocol/10.1007/...) and Nature
                    Protocols (nature.com/articles/nprot.2010.53).
  3. title          the candidate's title matches the target's. Needed because
                    Protocol Exchange protocols now live at
                    researchsquare.com/article/nprot-<n>, whose pages carry the
                    title but no DOI, so tiers 1 and 2 both miss them.

Tiers 1 and 2 identify a protocol; tier 3 only suggests one, because several
target titles are generic ("Enzyme-Linked Immunosorbent Assay (ELISA)",
"Introduction to Cell Culture") and some unrelated page may carry the same
words. Tier 3 hits are therefore returned flagged and are meant to be confirmed
by hand, never counted automatically.

The tiers are a superset of the existing rule, and they can only help a system
that returns pages from outside Protocols.io. ProtocolNerd, DP and the
protocols.io-scoped web search baseline return Protocols.io entries only, so
tiers 2 and 3 can never fire for them and their published find rates are
unchanged. `--verify` proves that by re-scoring their stored per-paper results
through this matcher and checking the counts still come out at 22, 34 and 19.

Usage:
    python open_web_matcher.py --verify
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmarking/
import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_DIR  # noqa: E402

import run_llm_websearch_baseline as WS  # tier 1, reused as-is

SCRIPT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH = SCRIPT_DIR / "citation_ground_truth_Biology_100.csv"
RESOLVE_CACHE = Path(_bootstrap.CACHE_DIR) / "open_web_matcher_slug_resolution.json"
TOP_K = 10
TITLE_RATIO = 0.92


# --------------------------------------------------------------------------- targets

class Target:
    """The confirmed protocol(s) for one citing paper, in every form a web
    result might identify them by."""

    def __init__(self) -> None:
        self.ids: Set[str] = set()
        self.lookup: Dict[str, Tuple[str, str]] = {}   # id -> (doi slug, protocols.io uri)
        self.dois: Set[str] = set()                    # normalised, e.g. 10.1007/978-1-0716-2903-1_7
        self.titles: Set[str] = set()


def normalise_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    return d.rstrip(" .")


def normalise_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[‐-―─-╿]", "-", t)     # dashes and box glyphs
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def load_targets(path: Path = GROUND_TRUTH) -> Dict[str, Target]:
    """pmid of the citing paper -> Target.

    Built the same way run_llm_websearch_baseline builds its target_lookup: the
    confirmed ids come from used_in_methods_protocol_ids split on "|" (18 of the
    100 papers name more than one protocol), and each id's Protocols.io DOI and
    uri come from the corpus index, not from this CSV. A second target of a
    multi-target paper has no row of its own here, so reading the CSV alone
    would leave it with no uri and make it unmatchable.

    The CSV still supplies the publisher DOI (Springer, Nature, Protocol
    Exchange) and the title, which tiers 2 and 3 need and the corpus does not
    always carry."""
    index = {str(p.get("id")): p for p in WS.get_index()["protocols"]}
    csv_by_id: Dict[str, Tuple[str, str]] = {}
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        csv_by_id[(r["protocol_id"] or "").strip()] = (
            normalise_doi(r.get("protocol_doi", "")), (r.get("protocol_title") or "").strip())

    by_pmid: Dict[str, Target] = {}
    for r in rows:
        pmid = (r["pmid"] or "").strip()
        t = by_pmid.setdefault(pmid, Target())
        for tid in [x.strip() for x in (r.get("used_in_methods_protocol_ids") or "").split("|") if x.strip()]:
            entry = index.get(tid, {})
            t.ids.add(tid)
            t.lookup[tid] = (WS._doi_slug(entry.get("doi", "")), (entry.get("uri", "") or "").lower())
            for d in (normalise_doi(entry.get("doi", "")), csv_by_id.get(tid, ("", ""))[0]):
                if d:
                    t.dois.add(d)
            for title in ((entry.get("title") or "").strip(), csv_by_id.get(tid, ("", ""))[1]):
                if title:
                    t.titles.add(title)
    return by_pmid


# --------------------------------------------------------------------------- tiers

def _load_resolve_cache() -> Dict[str, Any]:
    if RESOLVE_CACHE.exists():
        try:
            return json.loads(RESOLVE_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_resolve_cache(cache: Dict[str, Any]) -> None:
    RESOLVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RESOLVE_CACHE.write_text(json.dumps(cache, indent=1))


def match_by_id(protocol_id: str, target: Target) -> bool:
    """Tier 1 for systems that return Protocols.io ids rather than URLs."""
    return str(protocol_id).strip() in target.ids


def match_by_doi(url: str, target: Target, page_text: str = "", doi: str = "") -> Optional[str]:
    """Tier 2. Returns the matched DOI, or None.

    `doi` is the candidate's own DOI when the source reports one (Elicit does),
    which is more reliable than finding the string inside a URL."""
    hay = f"{normalise_doi(doi)} {(url or '').lower()} {(page_text or '').lower()}"
    hay = hay.replace("%2f", "/")
    for doi in target.dois:
        if doi and doi in hay:
            return doi
    return None


def match_by_title(title: str, target: Target) -> Optional[str]:
    """Tier 3, advisory only. Returns the matched target title, or None."""
    cand = normalise_title(title)
    if len(cand) < 12:
        return None
    for t in target.titles:
        tgt = normalise_title(t)
        if cand == tgt or SequenceMatcher(None, cand, tgt).ratio() >= TITLE_RATIO:
            return t
    return None


def classify(candidate: Dict[str, str], target: Target,
             resolve_cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Score one candidate. `candidate` may carry url, title, protocol_id and
    page_text. Returns None for a miss, else the tier and what it matched."""
    pid = candidate.get("protocol_id")
    if pid and match_by_id(pid, target):
        return {"tier": "protocols.io", "by": f"id {pid}", "needs_review": False}

    url = candidate.get("url") or ""
    if url:
        rank = WS._determine_hit_rank([url], target.ids, target.lookup, resolve_cache)
        if rank is not None:
            return {"tier": "protocols.io", "by": url, "needs_review": False}

    doi = match_by_doi(url, target, candidate.get("page_text", ""), candidate.get("doi", ""))
    if doi:
        return {"tier": "doi", "by": doi, "needs_review": False}

    title = candidate.get("title") or ""
    if title:
        hit = match_by_title(title, target)
        if hit:
            return {"tier": "title", "by": hit, "needs_review": True}
    return None


def hit_rank(candidates: List[Dict[str, str]], target: Target,
             resolve_cache: Dict[str, Any], top_k: int = TOP_K) -> Optional[Dict[str, Any]]:
    """Best match within the top k, preferring certainty over position.

    Identifier matches (tiers 1 and 2) are searched across the whole window
    before any title match is considered. Taking simply the first match would
    let an ambiguous title match at rank 8 mask a verified DOI match at rank 9,
    which both sends a settled case to human review and, if that title match
    were rejected, scores the paper a miss despite a confirmed hit."""
    window = list(candidates[:top_k])
    scored = [(i, classify(c, target, resolve_cache)) for i, c in enumerate(window, start=1)]
    for i, got in scored:
        if got and not got["needs_review"]:
            return {"rank": i, **got}
    for i, got in scored:
        if got:
            return {"rank": i, **got}
    return None


# --------------------------------------------------------------------------- verification

def _split_urls(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    return [u for u in re.split(r"[\s;|,]+", raw) if u.startswith("http")]


def _score_url_vector(path: Path, targets: Dict[str, Target],
                      resolve_cache: Dict[str, Any]) -> Tuple[int, int]:
    """Scored against whichever resolution cache is passed in. For --verify that
    is the published run's own frozen cache, so the check measures this
    matcher's tiers and not drift in Protocols.io's live search."""
    rows = list(csv.DictReader(open(path)))
    hits = 0
    for r in rows:
        tgt = targets.get((r.get("pmid") or "").strip())
        if not tgt:
            continue
        cands = [{"url": u} for u in _split_urls(r.get("candidate_urls", ""))]
        if hit_rank(cands, tgt, resolve_cache):
            hits += 1
    return hits, len(targets)


def _score_dp(path: Path, targets: Dict[str, Target]) -> Tuple[int, int]:
    """DP stores one row per returned protocol; a paper is a hit when a
    confirmed target appears in its top 10."""
    by_pmid: Dict[str, List[Tuple[int, str]]] = {}
    for r in csv.DictReader(open(path)):
        pmid = (r.get("pmid") or "").strip()
        try:
            rank = int(float(r.get("rank") or 0))
        except ValueError:
            continue
        by_pmid.setdefault(pmid, []).append((rank, (r.get("result_protocol_id") or "").strip()))
    hits = 0
    for pmid, items in by_pmid.items():
        tgt = targets.get(pmid)
        if not tgt:
            continue
        if any(match_by_id(pid, tgt) for rank, pid in sorted(items) if 0 < rank <= TOP_K):
            hits += 1
    return hits, len(targets)


def verify() -> int:
    targets = load_targets()
    resolve_cache = _load_resolve_cache()
    cache_dir = Path(_bootstrap.CACHE_DIR)
    checks = [
        ("DP keyword baseline", RESULTS_DIR / "keyword_baseline_protocolsio_100.csv", 22, _score_dp, None),
        ("LLM + web search (abstract)", RESULTS_DIR / "websearch_llm_baseline_protocolsio_100.csv", 34, None,
         cache_dir / "websearch_llm_baseline_slug_resolution.json"),
        ("LLM + web search (short query)", RESULTS_DIR / "websearch_llm_baseline_shortquery_protocolsio_100.csv", 19, None,
         cache_dir / "websearch_llm_baseline_shortquery_slug_resolution.json"),
    ]
    print(f"  targets loaded: {len(targets)} papers, "
          f"{sum(len(t.ids) for t in targets.values())} confirmed protocol ids, "
          f"{sum(len(t.dois) for t in targets.values())} DOIs\n")
    failures = []
    for label, path, expected, scorer, frozen in checks:
        if not Path(path).exists():
            print(f"  {label:<34} MISSING {path}")
            failures.append(label)
            continue
        # the published run's own cache, copied so this check never writes to it
        pinned = dict(json.loads(Path(frozen).read_text())) if frozen and Path(frozen).exists() else dict(resolve_cache)
        got, n = (scorer(Path(path), targets) if scorer
                  else _score_url_vector(Path(path), targets, pinned))
        ok = got == expected
        print(f"  {label:<34} {got:>3}/{n:<4} published {expected:>3}   {'OK' if ok else 'CHANGED'}")
        if not ok:
            failures.append(f"{label}: matcher gives {got}, published {expected}")
    _save_resolve_cache(resolve_cache)
    print()
    if failures:
        print("  FAILED, the extended criterion moved a published number:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("  The extended criterion leaves every published find rate unchanged.")
    return 0


CAPTURE_COLUMNS = ["tool", "pmid", "rank", "url", "title", "notes"]


def write_template(n: int, out: Path) -> int:
    """The sheet a person fills in while running Consensus or Elicit by hand.

    It carries the pmid and the query only. The target protocol is deliberately
    absent: whoever records the results should not know the answer, so a
    borderline "is this the same protocol" call cannot be swayed by it. Scoring
    happens afterwards, from the ids in the ground truth."""
    rows = list(csv.DictReader(open(GROUND_TRUTH)))
    seen, picked = set(), []
    for r in sorted(rows, key=lambda x: int(x["citing_pmid"])):     # stated rule: lowest pmids first
        pmid = r["citing_pmid"].strip()
        if pmid in seen:
            continue
        seen.add(pmid)
        picked.append((pmid, (r.get("query") or "").replace("Search query: ", "").strip()))
        if len(picked) >= n:
            break
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pmid", "query"])
        w.writerows(picked)
    print(f"  wrote {out}  ({len(picked)} queries, the {len(picked)} lowest pmids)")
    print(f"  record results into a second file with columns: {', '.join(CAPTURE_COLUMNS)}")
    print("  one row per returned result, rank 1 to 10, for each tool and pmid")
    return 0


def score_capture(path: Path, tool: Optional[str]) -> int:
    """Score a hand-recorded results sheet by the same rule as everything else."""
    targets = load_targets()
    resolve_cache = _load_resolve_cache()
    rows = [r for r in csv.DictReader(open(path))
            if not tool or (r.get("tool") or "").strip().lower() == tool.lower()]
    missing = [c for c in ("pmid", "rank", "url") if c not in (rows[0].keys() if rows else [])]
    if missing:
        print(f"  the sheet is missing required columns: {', '.join(missing)}")
        return 1
    by_tool: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for r in rows:
        t = (r.get("tool") or tool or "tool").strip()
        pmid = (r.get("pmid") or "").strip()
        try:
            rank = int(float(r.get("rank") or 0))
        except ValueError:
            continue
        by_tool.setdefault(t, {}).setdefault(pmid, []).append(
            {"rank": rank, "url": (r.get("url") or "").strip(), "title": (r.get("title") or "").strip()})

    exit_code = 0
    for t, papers in sorted(by_tool.items()):
        hits, review = [], []
        for pmid, cands in papers.items():
            tgt = targets.get(pmid)
            if not tgt:
                print(f"  {t}: pmid {pmid} is not in the benchmark, skipped")
                continue
            ordered = [c for c in sorted(cands, key=lambda c: c["rank"])]
            got = hit_rank(ordered, tgt, resolve_cache)
            if got:
                hits.append((pmid, got))
                if got["needs_review"]:
                    review.append((pmid, got))
        n = len(papers)
        auto = [h for h in hits if not h[1]["needs_review"]]
        print(f"\n  {t}: {len(auto)}/{n} confirmed"
              f"{f', plus {len(review)} title matches awaiting confirmation' if review else ''}")
        for pmid, got in sorted(hits):
            flag = "  NEEDS REVIEW" if got["needs_review"] else ""
            print(f"     pmid {pmid}  rank {got['rank']}  via {got['tier']}  {got['by'][:70]}{flag}")
        if review:
            exit_code = 0   # not an error, just work left for a human
    _save_resolve_cache(resolve_cache)
    return exit_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="re-score the stored baseline vectors and check the published numbers are unchanged")
    ap.add_argument("--template", type=int, metavar="N",
                    help="write a capture sheet of N queries for a hand run (Consensus, Elicit)")
    ap.add_argument("--score", metavar="FILE",
                    help="score a hand-recorded results sheet")
    ap.add_argument("--tool", help="score only this tool's rows")
    ap.add_argument("--out", default="manual_queries.csv", help="output path for --template")
    args = ap.parse_args()
    if args.verify:
        return verify()
    if args.template:
        return write_template(args.template, Path(args.out))
    if args.score:
        return score_capture(Path(args.score), args.tool)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
