#!/bin/bash
# Finalize the corpus after a full enumeration: backfill step text for any newly
# added protocols (looping until progress stalls), then rebuild the index pickle.
set -u
cd "$(dirname "$0")/.."
LOG_TS() { date '+%Y-%m-%d %H:%M:%S'; }
COUNT_EMPTY() { python3 -c "import json,glob; print(sum(1 for f in glob.glob('data/protocols/*.json') if not any((s or '').strip() for s in (json.load(open(f)).get('steps') or [])) ))"; }

echo "[$(LOG_TS)] FINALIZE — backfilling step text (loop until stable)…"
prev=-1
for pass in $(seq 1 12); do
    empty=$(COUNT_EMPTY)
    echo "[$(LOG_TS)]   pass $pass: $empty missing steps"
    if [ "$empty" -le 50 ]; then break; fi
    if [ "$empty" -eq "$prev" ]; then echo "[$(LOG_TS)]   no progress — stopping"; break; fi
    prev=$empty
    python3 backfill_steps.py --workers 8
done
echo "[$(LOG_TS)] backfill done — $(COUNT_EMPTY) genuinely step-less remain"

echo "[$(LOG_TS)] rebuilding index pickle…"
python3 scripts/build_index.py
echo "[$(LOG_TS)] FINALIZE DONE — total: $(ls data/protocols/*.json | wc -l | tr -d ' ') protocols"
