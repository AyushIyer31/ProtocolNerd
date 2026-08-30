# Evaluation

The paper reports **one experiment**: the citation-grounded benchmark in Section 5.
Everything that produces a number in the paper lives in `citation_grounded_eval/`.
Everything else is archived under `archive/`.

## Layout

```
benchmarking/
  _bootstrap.py              env + shared paths.  Must stay at this level:
  systems.py                 scripts find them via a parent-relative sys.path.
  citation_grounded_eval/    the paper's experiment (see below)
  replay_cache/              committed: replays all four numbers with no API key
  export_replay_cache.py     regenerates replay_cache/ from cache/
  cache/                     local working cache (gitignored, grows unboundedly)
  results/                   run outputs
```

`cache/` is gitignored, so a fresh clone starts with none. `_bootstrap.py` seeds it
from `replay_cache/` — the same entries pruned to the 100 benchmark papers, 0.6 MB —
so every script below runs offline and free on a clean checkout. A local cache file
is never overwritten by the seed, so a fresh run's results always win.

## The paper's experiment

`citation_grounded_eval/` — 100 papers, each confirmed from its own Methods
section to have used a specific Protocols.io protocol.

| Script | Produces | Paper |
|---|---|---|
| `gen_citation_ground_truth.py` | the benchmark itself | Section 5.1 |
| `run_citation_ground_truth_test.py` | ProtocolNerd find rate, **46%** | Section 5.2 |
| `run_llm_keyword_baseline.py` | DP (Wang et al. Detailed Prompt), **22%** | Section 5.2 |
| `run_llm_websearch_baseline.py` | LLM + web search, full abstract, **34%** | Section 5.2 |
| `run_llm_websearch_baseline_shortquery.py` | same, matched short query, **19%** | Section 5.2 |

Statistics: `MeanConf.py` (bootstrap CI) and `pairedtest.py` (paired difference
test), both following the method in Shasha & Wilson, *Statistics is Easy!*

## Reproducing the numbers

One input file feeds all four methods:
**`citation_ground_truth_Biology_100.csv`**. It carries each paper's query, title,
abstract, and the protocol IDs confirmed in its Methods section — everything any
of the four scripts needs. It is the evaluated set and is committed as-is.

One script per experiment, one run each:

```bash
python run_citation_ground_truth_test.py \
  --queries-csv citation_ground_truth_Biology_100.csv \
  --out ../results/citation_ground_truth_report_FINAL_100.csv \
  --cache-name rerank_final100_haiku          # ProtocolNerd, 46%

python run_llm_websearch_baseline.py             # 34%
python run_llm_websearch_baseline_shortquery.py  # 19%
python run_llm_keyword_baseline.py               # 22%
```

All four replay from `replay_cache/` on a clean checkout, so they are free, offline,
and exact. Pass a `--cache-name` that does not exist yet to force ProtocolNerd to
re-rank live against Claude instead.

**Expect the fresh ProtocolNerd run to move by a paper or two.** The re-ranker is an
LLM and PubMed is searched live, so the pipeline is not deterministic: the run
committed here scores 45/100 against the paper's 46/100. The three baselines are
deterministic replays and match exactly.

A hit is credited when **any** Methods-confirmed protocol for that paper lands in
the top 10. Inputs without a `used_in_methods_protocol_ids` column fall back to
the single originally-cited protocol.

## The chemistry extension

The same methodology applied to a second domain, differing only in the seeds. The
evaluated set is `citation_ground_truth_Chemistry_100.csv`, n=100 matching biology,
committed as-is. See `CHEMISTRY_SET_PROVENANCE.md` for how it was built.

```bash
cd citation_grounded_eval
./run_chemistry_findrate.sh          # 32.0%, 90% CI 24-40%
```

Europe PMC is a chemistry-only search source in the product, but is deliberately
excluded from the find-rate measurement: ground truth is always a protocols.io
protocol id, so a paper can never BE the right answer and any slot it takes could
only depress the score for reasons unrelated to retrieval quality. Its effect on
the live product is measured separately:

| Script | Measures |
|---|---|
| `run_epmc_ab.sh` | find rate with Europe PMC on vs off, one identical code path |
| `epmc_surfacing_analysis.py` | where Europe PMC lands in the live top 10 |
| `slot_occupancy_analysis.py` | PubMed slot occupancy, chemistry 3.0 vs biology 2.3 of 10 |

### Checking without re-running

```bash
python verify_paper_numbers.py
```

Checks all four find rates offline against committed artifacts — no API key, no
network. The baselines and the published per-paper record must match exactly; the
fresh run is allowed ±2 papers of drift. Exits non-zero otherwise.

## Archived

Exploratory and superseded evaluations live in the private development repository and are not part of this snapshot.

### Statistics scripts

`MeanConf.py` and `pairedtest.py` read their input (`MeanConf.vals`, `paired_vectors.json`)
from the current directory, so run them from `citation_grounded_eval/`:

```bash
cd citation_grounded_eval
python MeanConf.py      # find-rate CI:            0.38 - 0.54  (bias corrected 0.37 - 0.53)
python pairedtest.py    # ProtocolNerd vs DP diff: 0.16 - 0.33  (bias corrected 0.15 - 0.31)
```
