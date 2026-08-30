#!/bin/bash
# End-to-end chemistry find rate on the FROZEN n=100 benchmark, using the SAME
# runner and the SAME bootstrap CI code as the biology 100-paper experiment.
#
# The evaluated set is citation_ground_truth_Chemistry_100.csv, n=100 matching biology.
# It is frozen and nothing here regenerates it. How it was built is recorded in
# CHEMISTRY_SET_PROVENANCE.md.
#
# Europe PMC is deliberately NOT part of this measurement. Ground truth is always a
# protocols.io protocol id, so a paper can never BE the right answer and any slot it
# takes could only depress the score for reasons unrelated to retrieval quality.
# Europe PMC's effect on the live product is measured separately by run_epmc_ab.sh.
#
# Usage: ./run_chemistry_findrate.sh
set -e
cd "$(dirname "$0")"
PY=/Users/ayushiyer/Documents/ProtocolNerd-Working/venv/bin/python
CSV=citation_ground_truth_Chemistry_100.csv
REPORT=../results/citation_ground_truth_report_chemistry_100.csv

N=$("$PY" -c "import csv;print(len(list(csv.DictReader(open('$CSV')))))")
echo "frozen chemistry benchmark: $N pairs"
echo

echo "=== 1/2  find-rate run (BENCHMARK_DOMAIN=chemistry) ==="
# BENCHMARK_DOMAIN routes systems._nerd_profile through the chemistry plugin, so the
# profile is built with chemistry's own fields and prompt. A chemistry-specific rerank
# cache keeps this from colliding with the biology run's cache.
BENCHMARK_DOMAIN=chemistry "$PY" run_citation_ground_truth_test.py \
  --queries-csv "$CSV" \
  --out "$REPORT" \
  --cache-name rerank_finalchem_sonnet

echo
echo "=== 2/2  bootstrap 90% CI (MeanConf.py, same code as biology) ==="
# MeanConf.py reads a hardcoded ./MeanConf.vals, so run it in a temp dir to avoid
# overwriting the committed biology vector.
TMP=$(mktemp -d)
cp MeanConf.py "$TMP"/
"$PY" - "$TMP" "$REPORT" <<'PYEOF'
import csv, sys, pathlib
rows = list(csv.DictReader(open(sys.argv[2])))
vals = [1 if int(r["exact_match_position"]) > 0 else 0 for r in rows]
out = pathlib.Path(sys.argv[1]) / "MeanConf.vals"
out.write_text(">ProtocolNerd-chemistry-100\n" + " ".join(str(v) for v in vals) + "\n")
print(f"  n={len(vals)}  found={sum(vals)}  find rate={sum(vals)/len(vals):.1%}")
PYEOF
( cd "$TMP" && "$PY" MeanConf.py )
rm -rf "$TMP"
