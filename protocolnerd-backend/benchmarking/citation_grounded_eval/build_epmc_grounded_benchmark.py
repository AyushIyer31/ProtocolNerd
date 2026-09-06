"""Build a Europe PMC-grounded chemistry benchmark (Section 6.4 experiment).

Ground truth transplants the citation-grounded recipe: a protocol-describing
paper X in Europe PMC is used in the Methods section of a citing paper P,
confirmed by an LLM reading that Methods section. The query is written from
P's title and abstract, and the find rate asks whether X (matched by PMID or
DOI, from any retrieval lane) appears in the top 10.

Pipeline per candidate X:
  1. seed X from Europe PMC searches over protocol-shaped chemistry venues
  2. list papers P citing X (Europe PMC citations API)
  3. keep P only if its open-access full text cites X INSIDE a Methods section
     (reference-list entry matched by DOI or title, then xref rid located
     within a methods-titled section)
  4. LLM-confirm with the same chemist judge the frozen chemistry set used

    python build_epmc_grounded_benchmark.py --probe      # small-cap feasibility run
    python build_epmc_grounded_benchmark.py --target 100 # full build
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _bootstrap  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

SEED_QUERIES = [
    '(JOURNAL:"MethodsX") AND (chromatograph* OR "mass spectrometry" OR '
    '"sample preparation" OR titration OR "solid phase extraction" OR '
    'spectrophotometr* OR voltammetr* OR "chemical synthesis")',
    '(JOURNAL:"Nature Protocols" OR JOURNAL:"STAR Protocols") AND (synthesis '
    'OR nanoparticle OR chromatograph* OR electrochemi* OR "chemical")',
    'TITLE:(method OR protocol OR determination) AND ("ICP-MS" OR "GC-MS" OR '
    'HPLC OR QuEChERS OR "X-ray diffraction" OR "FTIR")',
    '(JOURNAL:"Analytical methods" OR JOURNAL:"Analytica chimica acta" OR '
    'JOURNAL:"Talanta") AND TITLE:(method OR determination OR protocol)',
    '(JOURNAL:"Journal of chromatography. A" OR JOURNAL:"Food chemistry" OR '
    'JOURNAL:"Microchemical journal") AND TITLE:(method OR determination OR '
    'quantification OR protocol)',
    '(JOURNAL:"MethodsX") AND (electrochemi* OR polymer OR "atomic absorption" '
    'OR "gas chromatography" OR pesticide OR "heavy metal*" OR adsorption)',
    'TITLE:(protocol OR procedure) AND ("thin layer chromatography" OR '
    '"column chromatography" OR "UV-Vis" OR "Raman" OR electrophoresis OR '
    '"atomic force microscopy" OR "dynamic light scattering")',
    'TITLE:(method OR determination OR protocol) AND ("NMR" OR '
    '"nuclear magnetic resonance" OR "Karl Fischer" OR "elemental analysis" OR '
    'thermogravimetr* OR "differential scanning calorimetry")',
    '(JOURNAL:"Journal of pharmaceutical and biomedical analysis" OR '
    'JOURNAL:"Analytical and bioanalytical chemistry" OR JOURNAL:"Journal of '
    'agricultural and food chemistry") AND TITLE:(method OR determination OR '
    'validation OR quantification)',
    'TITLE:(synthesis OR preparation OR fabrication) AND ("green synthesis" OR '
    '"sol-gel" OR hydrothermal OR "co-precipitation" OR electrospinning OR '
    '"ball milling")',
    '(JOURNAL:"Molecules" OR JOURNAL:"RSC advances" OR JOURNAL:"Heliyon") AND '
    'TITLE:(method OR determination OR quantification OR validation OR assay)',
    'TITLE:(extraction OR derivatization OR digestion OR hydrolysis) AND '
    'TITLE:(optimization OR "response surface" OR optimized OR improved)',
    'TITLE:(electrode OR sensor OR electrochemical) AND TITLE:(fabrication OR '
    'preparation OR determination OR detection)',
    '(JOURNAL:"Food analytical methods" OR JOURNAL:"Journal of analytical '
    'methods in chemistry" OR JOURNAL:"International journal of analytical '
    'chemistry") AND TITLE:(method OR determination OR analysis)',
]

METHODS_TITLES = re.compile(
    r"method|material|experimental|procedure|protocol|sample preparation",
    re.I)


_UA = {"User-Agent": "ProtocolNerd-benchmark/1.0 (mailto:iyer.ayush31@gmail.com)"}

# certifi CA bundle, same pattern as pubmed_client: macOS framework Python has
# no default cafile, so verification fails without this.
import ssl
try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _CTX = None


def _get_json(url: str, tries: int = 2) -> Optional[Any]:
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=20, context=_CTX) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.0)
    return None


def _get_text(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


BROAD_QUERIES = [
    '(TITLE:"method" OR TITLE:"protocol" OR TITLE:"procedure") AND OPEN_ACCESS:y AND SRC:MED',
    '(TITLE:determination OR TITLE:quantification OR TITLE:assay) AND OPEN_ACCESS:y AND SRC:MED',
    '(TITLE:synthesis OR TITLE:preparation OR TITLE:extraction) AND OPEN_ACCESS:y AND SRC:MED',
    '(TITLE:characterization OR TITLE:validation OR TITLE:optimization) AND OPEN_ACCESS:y AND SRC:MED',
]


def _enumerate(query: str, max_pages: int, page_size: int = 1000) -> List[Dict[str, Any]]:
    """Deep paging via cursorMark: the hand-written seed queries top out at about
    a thousand candidates, which is why earlier builds stalled short of target."""
    out, cursor = [], "*"
    for page in range(max_pages):
        # citation-sorted, otherwise deep paging returns overwhelmingly uncited
        # papers (94 of 100 in relevance order) and the cited-by filter drops them
        url = (f"{EPMC}/search?query={urllib.parse.quote(query)}"
               f"&resultType=core&pageSize={page_size}&sort=CITED%20desc"
               f"&cursorMark={urllib.parse.quote(cursor)}&format=json")
        data = _get_json(url) or {}
        res = ((data.get("resultList") or {}).get("result") or [])
        for r in res:
            pmid = r.get("pmid") or ""
            if not pmid or int(r.get("citedByCount") or 0) < 2:
                continue
            out.append({
                "x_pmid": pmid,
                "x_doi": (r.get("doi") or "").lower(),
                "x_pmcid": r.get("pmcid") or "",
                "x_title": r.get("title") or "",
                "x_journal": (r.get("journalInfo") or {}).get("journal", {}).get("title", ""),
                "x_cited_by": int(r.get("citedByCount") or 0),
                "x_abstract": (r.get("abstractText") or "")[:2000],
            })
        nxt = data.get("nextCursorMark")
        print(f"  enumerating: page {page+1}, {len(out)} candidates so far", flush=True)
        if not res or not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(0.4)
    return out


def seed_candidates(per_query: int, oa_only: bool = False) -> List[Dict[str, Any]]:
    """Protocol-shaped chemistry papers X, most-cited first."""
    seen, out = set(), []
    for q in SEED_QUERIES:
        if oa_only:
            q = f"({q}) AND OPEN_ACCESS:y"
        url = (f"{EPMC}/search?query={urllib.parse.quote(q)}"
               f"&resultType=core&sort=CITED%20desc&pageSize={per_query}&format=json")
        data = _get_json(url) or {}
        for r in (data.get("resultList") or {}).get("result", []):
            pmid = r.get("pmid") or ""
            if not pmid or pmid in seen or int(r.get("citedByCount") or 0) < 2:
                continue
            seen.add(pmid)
            out.append({
                "x_pmid": pmid,
                "x_doi": (r.get("doi") or "").lower(),
                "x_pmcid": r.get("pmcid") or "",
                "x_title": r.get("title") or "",
                "x_journal": (r.get("journalInfo") or {}).get("journal", {}).get("title", ""),
                "x_cited_by": int(r.get("citedByCount") or 0),
                "x_abstract": (r.get("abstractText") or "")[:2000],
            })
        time.sleep(0.4)
    return out


# P papers whose title+abstract the query model (claude-sonnet-4-6) refuses to
# paraphrase (API stop_reason "refusal", deterministic). Their X's may still
# pair with a different citing paper.
EXCLUDED_P_PMIDS = {"42069706", "42330110"}


def x_has_tagged_refs(x: Dict[str, Any], cache: Dict[str, bool]) -> bool:
    """v2 gate: X must be open access with machine-readable references, so the
    precedence criterion (Methods-citation lineage) is computable for every
    benchmark pair. Requires a PMCID, fetchable full text, and at least one
    pub-id-tagged entry in the reference list."""
    pmid = x["x_pmid"]
    if pmid in cache:
        return cache[pmid]
    ok = False
    pmcid = x.get("x_pmcid") or ""
    if not pmcid:
        data = _get_json(f"{EPMC}/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json") or {}
        res = (data.get("resultList") or {}).get("result", [])
        pmcid = (res[0].get("pmcid") or "") if res else ""
    xml = _get_text(f"{EPMC}/{pmcid}/fullTextXML") if pmcid else None
    if xml and len(xml) > 2000 and "<pub-id" in xml:
        try:
            root = ET.fromstring(xml)
            ok = any(ref.find(".//pub-id") is not None for ref in root.iter("ref"))
        except ET.ParseError:
            ok = False
    cache[pmid] = ok
    json.dump(cache, open(SCRIPT_DIR / ".epmc_refsgate_cache.json", "w"))
    return ok


def citers(x_pmid: str, cap: int) -> List[Dict[str, str]]:
    """Open-access citing papers, via the CITES search field.

    The search route (unlike the citations endpoint) can filter to the
    open-access subset and returns pmcid and abstract in the same call;
    full text is only fetchable by PMCID.
    """
    q = f"CITES:{x_pmid}_MED AND OPEN_ACCESS:y AND SRC:MED"
    url = (f"{EPMC}/search?query={urllib.parse.quote(q)}"
           f"&resultType=core&pageSize={cap}&format=json")
    data = _get_json(url) or {}
    out = []
    for r in (data.get("resultList") or {}).get("result", []):
        if r.get("pmid") and r.get("pmcid"):
            out.append({"p_pmid": r["pmid"], "p_pmcid": r["pmcid"],
                        "p_title": r.get("title") or "",
                        "p_abstract": (r.get("abstractText") or "")[:2500]})
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def methods_cites(p_pmcid: str, x_doi: str, x_title: str) -> Tuple[bool, str]:
    """Does P's open-access full text cite X inside a Methods section?

    Returns (verdict, methods_text). Reference matched by DOI when X has one,
    else by normalized-title containment; the xref rid must appear within a
    section whose sec-type or title looks methods-like.
    """
    xml = _get_text(f"{EPMC}/{p_pmcid}/fullTextXML")
    if not xml or len(xml) < 2000:
        return False, ""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False, ""

    ref_ids = set()
    want_title = _norm(x_title)[:80]
    for ref in root.iter("ref"):
        blob = " ".join(t for t in ref.itertext())
        hit = False
        if x_doi and x_doi in blob.lower():
            hit = True
        elif want_title and len(want_title) > 25 and want_title in _norm(blob):
            hit = True
        if hit and ref.get("id"):
            ref_ids.add(ref.get("id"))
    if not ref_ids:
        return False, ""

    for sec in root.iter("sec"):
        title_el = sec.find("title")
        sec_name = (sec.get("sec-type") or "") + " " + \
                   ("".join(title_el.itertext()) if title_el is not None else "")
        if not METHODS_TITLES.search(sec_name):
            continue
        for xr in sec.iter("xref"):
            rids = (xr.get("rid") or "").split()
            if any(r in ref_ids for r in rids):
                text = " ".join(t.strip() for t in sec.itertext() if t.strip())
                return True, text[:12000]
    return False, ""


from llm_providers import call_llm  # noqa: E402

JUDGE_PROMPT = (
    "You are an expert chemist. Given the METHODS section and ABSTRACT of a "
    "research paper and the title and description of one specific protocol, judge "
    "whether the methods section describes the paper's authors actually USING that "
    "protocol to run part of their experiment, not just citing it elsewhere in the "
    "paper (introduction, discussion) without using it, and that the abstract does "
    "not directly name the protocol. Answer with exactly one word: YES or NO.")
DOMAIN_PROMPT = (
    "Classify the protocol described by this title and abstract. Answer with "
    "exactly one word: CHEMISTRY if it is an analytical, synthetic, or physical "
    "chemistry procedure; BIOLOGY if molecular/cell biology; OTHER for anything "
    "else, including review methodology or statistics.")


def _llm(system: str, user: str, model: str) -> str:
    try:
        return (call_llm(messages=[{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                         temperature=0.0, provider="claude", model=model) or "").strip().upper()
    except Exception:
        return ""


def x_is_chemistry(x: Dict[str, Any]) -> bool:
    out = _llm(DOMAIN_PROMPT,
               f"TITLE: {x['x_title']}\nABSTRACT: {x['x_abstract'][:1200]}",
               "claude-haiku-4-5")
    return out.startswith("CHEM")


def judge_pair(pr: Dict[str, Any]) -> bool:
    user = (f"PROTOCOL TITLE: {pr['x_title']}\n"
            f"PROTOCOL DESCRIPTION: {pr['x_abstract'][:1200]}\n\n"
            f"PAPER ABSTRACT:\n{pr['p_abstract'][:2000]}\n\n"
            f"PAPER METHODS SECTION:\n{pr['p_methods'][:8000]}")
    return _llm(JUDGE_PROMPT, user, "claude-sonnet-4-6").startswith("YES")


def _load_json(path: Path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def run(per_query: int, citer_cap: int, per_x_checks: int, stop_at: int,
        out_path: Path, max_per_x: int = 99, refs_gate: bool = False,
        oa_only: bool = False, exhaustive: bool = False) -> None:
    domain_cache = _load_json(SCRIPT_DIR / ".epmc_domain_cache.json", {})
    refsgate_cache = _load_json(SCRIPT_DIR / ".epmc_refsgate_cache.json", {})
    judge_cache = _load_json(SCRIPT_DIR / ".epmc_judge_cache.json", {})
    # resume: keep max_per_x pairs per existing X, skip those X's entirely
    prior = _load_json(out_path, [])
    kept: Dict[str, List[Dict[str, Any]]] = {}
    for pr in prior:
        kept.setdefault(pr["x_pmid"], [])
        if len(kept[pr["x_pmid"]]) < max_per_x:
            kept[pr["x_pmid"]].append(pr)
    done_x = set(kept)
    xs = seed_candidates(per_query, oa_only=oa_only)
    if exhaustive:
        seen = {x["x_pmid"] for x in xs}
        for q in BROAD_QUERIES:
            for cand in _enumerate(q, max_pages=6):
                if cand["x_pmid"] not in seen:
                    seen.add(cand["x_pmid"]); xs.append(cand)
        xs.sort(key=lambda x: -x["x_cited_by"])
    print(f"seed X candidates: {len(xs)}  (cited-by median "
          f"{sorted(x['x_cited_by'] for x in xs)[len(xs)//2] if xs else 0})")
    funnel = {"x": len(xs), "x_chemistry": 0, "x_with_citers": 0, "p_checked": 0,
              "p_methods_cited": 0, "judge_confirmed": 0}
    pairs: List[Dict[str, Any]] = [pr for prs in kept.values() for pr in prs]
    print(f"resumed with {len(pairs)} pairs from {len(done_x)} existing X's")
    for x in xs:
        if len(pairs) >= stop_at:
            break
        if x["x_pmid"] in done_x:
            continue
        dkey = x["x_pmid"]
        if dkey not in domain_cache:
            domain_cache[dkey] = x_is_chemistry(x)
            json.dump(domain_cache, open(SCRIPT_DIR / ".epmc_domain_cache.json", "w"))
        if not domain_cache[dkey]:
            continue
        funnel["x_chemistry"] += 1
        if refs_gate and not x_has_tagged_refs(x, refsgate_cache):
            continue
        funnel["x_oa_tagged_refs"] = funnel.get("x_oa_tagged_refs", 0) + 1
        if len(pairs) >= stop_at:
            break
        cs = citers(x["x_pmid"], citer_cap)
        time.sleep(0.3)
        if cs:
            funnel["x_with_citers"] += 1
        checked = 0
        for c in cs:
            if c["p_pmid"] in EXCLUDED_P_PMIDS:
                continue
            if checked >= per_x_checks or len(pairs) >= stop_at:
                break
            checked += 1
            funnel["p_checked"] += 1
            ok, methods = methods_cites(c["p_pmcid"], x["x_doi"], x["x_title"])
            time.sleep(0.3)
            if not ok:
                continue
            funnel["p_methods_cited"] += 1
            pr = {**x, **c, "p_methods": methods}
            jkey = f"{x['x_pmid']}:{c['p_pmid']}"
            if jkey not in judge_cache:
                judge_cache[jkey] = judge_pair(pr)
                json.dump(judge_cache, open(SCRIPT_DIR / ".epmc_judge_cache.json", "w"))
            if not judge_cache[jkey]:
                continue
            funnel["judge_confirmed"] += 1
            del pr["p_methods"]
            pairs.append(pr)
            print(f"  [{len(pairs):>3}/{stop_at}] X: {x['x_title'][:46]:<48} <- P: {c['p_title'][:40]}",
                  flush=True)
            if sum(1 for q in pairs if q["x_pmid"] == x["x_pmid"]) >= max_per_x:
                break
    print("\nfunnel:", json.dumps(funnel))
    print(f"candidate pairs (pre-LLM): {len(pairs)}")
    out_path.write_text(json.dumps(pairs, indent=1))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="small-cap feasibility run")
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--exhaustive", action="store_true",
                    help="deep-page the corpus instead of relying on seed queries")
    ap.add_argument("--v2", action="store_true",
                    help="v2 benchmark: X must be open access with tagged references")
    args = ap.parse_args()
    if args.probe:
        run(per_query=6, citer_cap=15, per_x_checks=4, stop_at=25,
            out_path=SCRIPT_DIR / ".epmc_probe_pairs.json")
    elif args.v2:
        run(per_query=100, citer_cap=40, per_x_checks=10, stop_at=args.target,
            out_path=SCRIPT_DIR / "epmc_ground_truth_chemistry_v2.json", max_per_x=1,
            refs_gate=True, oa_only=True, exhaustive=args.exhaustive)
    else:
        run(per_query=60, citer_cap=40, per_x_checks=10, stop_at=args.target,
            out_path=SCRIPT_DIR / "epmc_ground_truth_chemistry.json", max_per_x=1)
