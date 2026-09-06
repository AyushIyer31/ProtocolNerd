"""Precedence-overlap criterion (Shasha) for the Europe PMC grounded benchmark.

Immediate(P) is the set of papers P cites inside Methods-type sections of its
open-access full text. Depth-1 Precedence(P) = {P} plus Immediate(P). A query
counts as a precedence hit when some paper X' in the arm's top 10 (excluding
the citing paper P itself, the self-return) has Precedence(X') overlapping
Precedence(X), where X is the confirmed ground-truth protocol paper. Overlap
means a shared PMID or DOI, so it captures X' citing X, X citing X', and X and
X' both building on the same underlying protocol paper.

Only paper results participate: Protocols.io items carry no citation graph.
Only open-access papers have fetchable Methods, the same sampling limit the
benchmark itself carries.
"""
import argparse, csv, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

import re
import xml.etree.ElementTree as ET
import run_epmc_grounded_test as R

# Many publishers deposit reference lists with no <pub-id> tags, but the DOI is
# still present inside the citation string. Harvest those too, otherwise the
# ancestry of such a paper looks empty when it is merely untagged.
_DOI_RE = re.compile(r'''10\.\d{4,9}/[^\s"'<>,;]+''', re.I)


def _doi_in_text(el) -> str:
    txt = " ".join(s.strip() for s in el.itertext() if s.strip())
    m = _DOI_RE.search(txt)
    return m.group(0).rstrip(".;,").lower() if m else ""


def _ids_from_ext_links(ref) -> set:
    """Publishers commonly put the identifier in an <ext-link> ATTRIBUTE rather
    than in <pub-id> or the citation text, e.g.
    <ext-link ext-link-type="doi" xlink:href="10.1378/chest.104.5.1498"/>.
    Attribute values never appear in itertext(), so they must be read directly."""
    out = set()
    for el in ref.iter():
        if not el.tag.endswith("ext-link"):
            continue
        kind = (el.get("ext-link-type") or "").lower()
        href = ""
        for k, v in el.attrib.items():
            if k.endswith("href"):
                href = (v or "").strip()
                break
        if not href:
            continue
        if kind == "doi":
            out.add(f"doi:{href.lower().rstrip('.;,')}")
        elif kind == "pmid":
            out.add(f"pmid:{href}")
        elif kind == "pmcid":
            out.add(f"pmcid:{href.upper().replace('PMC', 'PMC')}")
    return out
from build_epmc_grounded_benchmark import _get_json, _get_text, METHODS_TITLES, _CTX, _UA

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
CACHE = SCRIPT_DIR / ".precedence_cache.json"


def pmcid_of(pmid, cache):
    key = f"pmcid:{pmid}"
    if key not in cache:
        data = _get_json(f"{EPMC}/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json") or {}
        res = (data.get("resultList") or {}).get("result", [])
        cache[key] = (res[0].get("pmcid") or "") if res else ""
        json.dump(cache, open(CACHE, "w"))
    return cache[key]


def epmc_reference_list(pmid):
    """Europe PMC's structured reference list for an article.

    This works for CLOSED-ACCESS papers too: Europe PMC holds the reference
    metadata even where it cannot serve full text, which is most of what the
    full-text route was failing on. Returns None when the service is
    unavailable (the endpoint returns 503 during outages) so the caller can
    fall back to parsing full text rather than caching a false empty.
    """
    ids, page = set(), 1
    while page <= 4:
        url = (f"{EPMC}/MED/{pmid}/references?format=json"
               f"&pageSize=100&page={page}")
        data = _get_json(url)
        if data is None:
            return None if page == 1 else sorted(ids)
        refs = ((data.get("referenceList") or {}).get("reference") or [])
        if not refs:
            break
        for r in refs:
            if r.get("id") and (r.get("source") or "MED") == "MED":
                ids.add(f"pmid:{r['id']}")
            if r.get("doi"):
                ids.add(f"doi:{r['doi'].lower()}")
        if len(refs) < 100:
            break
        page += 1
    return sorted(ids)


def crossref_reference_list(doi):
    """Reference list from Crossref, which publishes it for CLOSED-ACCESS papers
    too: the criterion needs only the citation graph, not the article text.
    Used when Europe PMC cannot supply references (its references endpoint has
    been unavailable, and full text is withheld for non-open-access articles).
    Returns None on failure so a transient error is not cached as 'no refs'."""
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "ProtocolNerd-benchmark/1.0 (mailto:iyer.ayush31@gmail.com)"})
    for _ in range(2):
        try:
            msg = json.load(urllib.request.urlopen(req, timeout=30, context=_CTX))["message"]
            break
        except Exception:
            time.sleep(1.0)
    else:
        return None
    out = set()
    for r in msg.get("reference") or []:
        if r.get("DOI"):
            out.add(f"doi:{r['DOI'].lower()}")
        if r.get("PMID"):
            out.add(f"pmid:{r['PMID']}")
    return sorted(out)


def doi_of(pmid, cache):
    """The article's own DOI, needed to ask Crossref for its references."""
    key = f"doi:{pmid}"
    if key not in cache:
        data = _get_json(f"{EPMC}/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&resultType=core&format=json") or {}
        res = (data.get("resultList") or {}).get("result", [])
        cache[key] = (res[0].get("doi") or "").lower() if res else ""
        json.dump(cache, open(CACHE, "w"))
    return cache[key]


def immediate(pmid, cache, fallback_all_refs=False):
    """PMIDs and DOIs cited inside Methods-type sections of pmid's full text.

    With fallback_all_refs (used for the ground-truth protocol paper X only),
    a paper whose Methods sections yield no tagged references falls back to its
    full reference list: X is by construction a protocol-describing paper, so
    its bibliography approximates its method lineage even when its section
    structure does not label a Methods section."""
    key = f"immfull5:{pmid}" if fallback_all_refs else f"imm5:{pmid}"
    if key in cache:
        return set(cache[key])
    out = set()
    all_refs = set()
    # Preferred route: the structured reference list, which exists for
    # closed-access articles as well. Only fall through to full-text parsing
    # when the service is unavailable or returns nothing.
    api_refs = epmc_reference_list(pmid)
    if api_refs:
        cache[key] = api_refs
        json.dump(cache, open(CACHE, "w"))
        return set(api_refs)
    cr_refs = crossref_reference_list(doi_of(pmid, cache))
    if cr_refs:
        cache[key] = cr_refs
        json.dump(cache, open(CACHE, "w"))
        return set(cr_refs)
    pmcid = pmcid_of(pmid, cache)
    xml = _get_text(f"{EPMC}/{pmcid}/fullTextXML") if pmcid else None
    if xml and len(xml) > 2000:
        try:
            root = ET.fromstring(xml)
            refs = {}
            for ref in root.iter("ref"):
                rid = ref.get("id")
                if not rid:
                    continue
                ids = set()
                for pid in ref.iter("pub-id"):
                    v = (pid.text or "").strip().lower()
                    if v:
                        ids.add(f"pmid:{v}" if pid.get("pub-id-type") == "pmid" else f"doi:{v}"
                                if pid.get("pub-id-type") == "doi" else v)
                if not ids:                      # no <pub-id>: try ext-link attributes, then the text
                    ids |= _ids_from_ext_links(ref)
                if not ids:
                    d = _doi_in_text(ref)
                    if d:
                        ids.add(f"doi:{d}")
                if ids:
                    refs[rid] = ids
                    all_refs |= ids
            for sec in root.iter("sec"):
                title_el = sec.find("title")
                sec_name = (sec.get("sec-type") or "") + " " + \
                           ("".join(title_el.itertext()) if title_el is not None else "")
                if not METHODS_TITLES.search(sec_name):
                    continue
                for xr in sec.iter("xref"):
                    for rid in (xr.get("rid") or "").split():
                        out |= refs.get(rid, set())
        except ET.ParseError:
            pass
    if fallback_all_refs and not out:
        out = all_refs
    cache[key] = sorted(out)
    json.dump(cache, open(CACHE, "w"))
    return out


def precedence(pmid, doi, cache, fallback_all_refs=False):
    ids = {f"pmid:{pmid}"}
    if doi:
        ids.add(f"doi:{doi.lower()}")
    return ids | immediate(pmid, cache, fallback_all_refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["epmc", "epmc_pubmed", "epmc_lit", "epmc_pubmed_lit"], required=True)
    ap.add_argument("--paper-candidates", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N pairs: score only newly added ones")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--relaxed", action="store_true",
                    help="X' side also falls back to the full reference list")
    ap.add_argument("--fresh", default="",
                    help="tag suffix that forces new re-rank calls instead of reusing cached top-10s")
    args = ap.parse_args()
    gt_json = "epmc_ground_truth_chemistry_v2.json" if args.v2 else "epmc_ground_truth_chemistry.json"
    gt_csv = "citation_ground_truth_Chemistry_EPMC_100_v2.csv" if args.v2 else "citation_ground_truth_Chemistry_EPMC_100.csv"
    v2tag = "_v2" if args.v2 else ""
    R.PAPER_CANDIDATES = args.paper_candidates
    suffix = "" if args.paper_candidates == 5 else f"_pc{args.paper_candidates}"
    cache_tag = f"rerank_epmcgt_{args.arm}{suffix}{v2tag}{args.fresh}"

    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    rows = list(csv.DictReader(open(SCRIPT_DIR / gt_csv)))
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[:args.limit]

    report, hits = [], []
    for i, r in enumerate(rows, 1):
        x_prec = precedence(r["protocol_id"], r["protocol_doi"], cache, fallback_all_refs=True)
        final_ids, slim = R.run_one(r["query"], args.arm, cache_tag)
        hit_detail = ""
        n_papers = n_traceable = 0
        for pos, j in enumerate(final_ids[:R.K], 1):
            it = slim.get(int(j), {})
            xp = it.get("pmid")
            if not xp or xp == r["citing_pmid"]:      # papers only; exclude P itself
                continue
            n_papers += 1
            xp_prec = precedence(xp, it.get("doi") or "", cache, fallback_all_refs=args.relaxed)
            if len(xp_prec) > (2 if it.get("doi") else 1):   # more than the paper's own ids
                n_traceable += 1
            shared = x_prec & xp_prec
            if shared and not hit_detail:
                hit_detail = f"rank {pos} X'={xp} shared={';'.join(sorted(shared)[:4])}"
        hits.append(1 if hit_detail else 0)
        report.append({"p_pmid": r["citing_pmid"], "x_pmid": r["protocol_id"], "query": r["query"],
                       "precedence_hit": bool(hit_detail), "detail": hit_detail,
                       "papers_returned": n_papers, "papers_traceable": n_traceable,
                       "x_imm_size": len(x_prec) - (2 if r["protocol_doi"] else 1)})
        print(f"  [{i:>3}/{len(rows)}] {'HIT ' if hit_detail else 'miss'}  {hit_detail[:70] or r['query'][:60]}",
              flush=True)

    out = Path(R.RESULTS_DIR) / f"epmc_grounded_{args.arm}{suffix}{v2tag}{args.fresh}_precedence{'_relaxed' if args.relaxed else ''}_{('from%d_' % args.offset) if args.offset else ''}{len(rows)}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader(); w.writerows(report)
    lo, hi = R.bootstrap_ci(hits)
    empty_x = sum(1 for r in report if r["x_imm_size"] == 0)
    testable = sum(1 for r in report if r["papers_traceable"] > 0)
    no_papers = sum(1 for r in report if r["papers_returned"] == 0)
    hits_in_testable = sum(1 for r in report if r["papers_traceable"] > 0 and r["precedence_hit"])
    print(f"\nARM {args.arm}{suffix} (precedence{'-relaxed' if args.relaxed else ''}): {sum(hits)}/{len(hits)}  90% CI [{lo:.0f}%, {hi:.0f}%]")
    print(f"X's with no fetchable Methods refs (closed access or no xrefs): {empty_x}/{len(report)}")
    print(f"DENOMINATOR: queries returning no papers {no_papers}, "
          f"queries with >=1 traceable paper {testable}, hits among those {hits_in_testable}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
