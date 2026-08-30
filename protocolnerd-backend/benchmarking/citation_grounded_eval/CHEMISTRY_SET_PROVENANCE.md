# Chemistry benchmark: how it was built

`citation_ground_truth_Chemistry_100.csv` is the frozen chemistry benchmark, n=100,
matching biology's n. It is the evaluated set; nothing regenerates it.

## Method

Identical to the biology methodology (paper Section 5.3), differing only in the seeds:

1. Semantic Scholar citation graph, chemistry-seeded, filtered to entries with a PMID.
2. Embedding relevance filter between the citing paper and the protocol.
3. **Methods-section confirmation** (`confirm_methods_chemistry.py`): an LLM reads the
   citing paper's own Methods section and confirms the authors actually USED the
   protocol, rather than citing it in the introduction or discussion.
   Judge model: **claude-sonnet-4-6**, the same judge biology's set was built with.
   This matters: judging with Haiku instead moved the confirm rate from 50% to 30%,
   so the two find rates would not have been comparable.

Query per row is written from the citing paper's abstract alone, using the same
Query Paraphrase prompt as biology, so the query never names the protocol.

## Result

| | |
|---|---|
| Find rate | 32.0% (32/100) |
| 90% CI (bootstrap) | 24%–40% |
| 90% CI (bias-corrected) | 23%–39% |

Europe PMC is deliberately excluded from this measurement. Ground truth is always a
protocols.io protocol id, so a paper can never BE the right answer and any slot it took
could only depress the score for reasons unrelated to retrieval quality. Europe PMC's
effect on the live product is measured separately, below.

## What Europe PMC actually costs (`run_epmc_ab.sh`)

Both arms replay the same 100 benchmark queries through the same live `/chat` endpoint,
differing only in `ENABLE_EUROPEPMC`, so the delta is attributable to Europe PMC alone.
We do NOT compare against the offline runner's 32%: that comes from a different code path
(`build_shortlist` + `llm_rerank` rather than `main.py`), and comparing to it would
confound Europe PMC's displacement with path differences.

| | Europe PMC OFF | Europe PMC ON |
|---|---|---|
| Find rate | 31/100 | 31/100 |
| Failures | 0 | 0 |

**No detectable cost.** Note the live path's 31% independently corroborates the offline
runner's 32%, which is why the paper's number describes the shipped product and not just
the benchmark harness.

Where the slots come from, over the 23 queries where Europe PMC surfaced at all:

| source | OFF | ON | change |
|---|---|---|---|
| protocols.io | 7.57 | 4.87 | **-2.70** |
| pubmed | 2.26 | 2.74 | +0.48 |
| europepmc | 0.00 | 2.39 | +2.39 |

Europe PMC displaces **protocols.io**, not PubMed. It nonetheless costs no find rate,
because the protocols it displaces were mostly wrong anyway: of the 12 queries that
changed outcome, only 4 had Europe PMC in the top 10 at all, and those split 3 gained
against 1 lost.

**Read this as "no detectable cost", not "zero cost".** The remaining 8 of those 12
flips had no Europe PMC present, so they are pure run-to-run variance from the LLM
re-ranker and live PubMed. That 8-in-100 noise floor is the experiment's resolution
limit: an effect smaller than roughly 8 points cannot be separated from it.

## Known asymmetry against the biology set

Biology is one protocol per citing paper throughout: 100 rows, 100 distinct protocols,
100 distinct citing papers. Chemistry is 100 rows, **80 distinct protocols**, 97 distinct
citing papers, because reaching n=100 from a thinner pool required allowing a protocol to
be the target for more than one citing paper (`--max-per-protocol`). Some protocols are
therefore tested repeatedly, so trials are somewhat less independent than in biology.
This is worth a sentence in the paper's methodology.
