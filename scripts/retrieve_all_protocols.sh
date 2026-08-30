#!/bin/bash
# Full-library retrieval pipeline for the protocols.io RAG corpus.
#   Phase 1: enumerate ALL public protocols (metadata) via a universal probe
#   Phase 2: backfill per-step instruction text for every protocol missing it,
#            looping until no further progress (resilient to a flaky network)
#   Phase 3: rebuild the prebuilt TF-IDF index pickle
#
# Designed to run unattended to completion. Every phase is resumable.
set -u
cd "$(dirname "$0")/.."   # repo root
LOG_TS() { date '+%Y-%m-%d %H:%M:%S'; }
COUNT_ALL()   { ls data/protocols/*.json 2>/dev/null | wc -l | tr -d ' '; }
COUNT_EMPTY() { python3 -c "import json,glob; print(sum(1 for f in glob.glob('data/protocols/*.json') if not any((s or '').strip() for s in (json.load(open(f)).get('steps') or [])) ))"; }

echo "===================================================================="
echo "[$(LOG_TS)] RETRIEVE-ALL pipeline starting — protocols before: $(COUNT_ALL)"
echo "===================================================================="

# ---- PHASE 1: enumerate the full public library (metadata only) ----------
echo "[$(LOG_TS)] PHASE 1/3 — enumerating full public library (page_size=50, with retries)…"
python3 fetch_protocols.py --keywords " " --max-per-keyword 30000 --skip-steps --page-size 50
echo "[$(LOG_TS)] PHASE 1 complete — protocols cached now: $(COUNT_ALL)"

# ---- PHASE 2: backfill step text, looping until progress stalls ----------
echo "[$(LOG_TS)] PHASE 2/3 — backfilling per-step text (looping until stable)…"
prev_empty=-1
for pass in $(seq 1 12); do
    empty=$(COUNT_EMPTY)
    echo "[$(LOG_TS)]   pass $pass: $empty protocols still missing steps"
    if [ "$empty" -le 50 ]; then echo "[$(LOG_TS)]   essentially complete ($empty left)"; break; fi
    if [ "$empty" -eq "$prev_empty" ]; then echo "[$(LOG_TS)]   no progress this pass — stopping backfill loop"; break; fi
    prev_empty=$empty
    python3 backfill_steps.py --workers 8
done
echo "[$(LOG_TS)] PHASE 2 complete — $(COUNT_EMPTY) still missing steps (genuinely step-less or unreachable)"

# ---- PHASE 3: rebuild the prebuilt index pickle --------------------------
echo "[$(LOG_TS)] PHASE 3/3 — rebuilding prebuilt index pickle…"
python3 scripts/build_index.py
echo "[$(LOG_TS)] PHASE 3 complete"

echo "===================================================================="
echo "[$(LOG_TS)] PIPELINE DONE — total protocols cached: $(COUNT_ALL)"
echo "===================================================================="
