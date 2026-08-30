#!/bin/bash
# A/B the cost of Europe PMC on the chemistry find rate.
#
# Both arms replay the SAME 100 benchmark queries through the SAME live /chat
# endpoint. The only difference is ENABLE_EUROPEPMC, so the delta is attributable
# to Europe PMC alone. We deliberately do NOT compare against the offline runner's
# 32%: that number comes from a different code path (build_shortlist + llm_rerank
# rather than main.py itself), so comparing to it would confound Europe PMC's
# displacement with path differences.
set -e
cd "$(dirname "$0")"
PY=/Users/ayushiyer/Documents/ProtocolNerd-Working/venv/bin/python
CSV=citation_ground_truth_Chemistry_100.csv
BACKEND=/Users/ayushiyer/Documents/ProtocolNerd-Working/protocolnerd-backend

# Bring up a backend with the given ENABLE_EUROPEPMC value.
#
# The previous version of this script raced: it pkill'd the old backend and
# immediately started the new one, which then died on "address already in use"
# because the port had not been released yet, leaving the readiness loop waiting
# forever on a port nothing was listening to. So: wait for the port to actually
# free, then confirm readiness against /chat rather than /docs. /docs answers
# before the app can serve a real search, which is how an entire arm previously
# ran to completion against a backend that answered nothing.
restart_backend () {
  pkill -f "uvicorn main:app" 2>/dev/null || true
  until [ -z "$(lsof -ti:8001 2>/dev/null)" ]; do sleep 1; done
  ( cd "$BACKEND" && ENABLE_EUROPEPMC="$1" nohup "$PY" -m uvicorn main:app \
      --host 0.0.0.0 --port 8001 > /tmp/ab_backend_$1.log 2>&1 & )
  for _ in $(seq 1 90); do
    if curl -s -m 20 -X POST http://localhost:8001/chat \
         -H "Content-Type: application/json" \
         -d '{"query":"ping","search_mode":"local","top_k":1,"explain":false,"no_log":true,"skip_clarification":true,"search_confirmed":true}' \
         -o /dev/null 2>/dev/null; then
      echo "backend READY (serving /chat) with ENABLE_EUROPEPMC=$1"; return 0
    fi
    sleep 3
  done
  echo "FATAL: backend never became ready with ENABLE_EUROPEPMC=$1"; exit 1
}

# Full arm output goes to its own log. The old script piped this through
# `tail -20`, which threw away the per-query lines and hid the fact that every
# single query in the arm had failed.
run_arm () {   # $1 = env value, $2 = out csv, $3 = arm log
  restart_backend "$1"
  "$PY" epmc_surfacing_analysis.py --queries-csv "$CSV" --out "$2" > "$3" 2>&1
  local n
  n=$("$PY" -c "import csv;print(len(list(csv.DictReader(open('$2')))))")
  echo "arm ENABLE_EUROPEPMC=$1 wrote $n rows -> $2"
  if [ "$n" -lt 50 ]; then
    echo "FATAL: arm produced $n rows, too few to compare. See $3"; exit 1
  fi
  grep -E "find rate|slots / query|Europe PMC in top-10" "$3" || true
}

echo "=== ARM A: Europe PMC OFF ==="
run_arm 0 ../results/epmc_ab_off.csv /tmp/ab_arm_off.log

echo
echo "=== ARM B: Europe PMC ON ==="
run_arm 1 ../results/epmc_ab_on.csv /tmp/ab_arm_on.log

echo
echo "=== DELTA ==="
"$PY" - <<'PYEOF'
import csv
def load(p):
    rows = list(csv.DictReader(open(p)))
    return rows, [1 if int(r["found_rank"]) else 0 for r in rows]
off_rows, off = load("../results/epmc_ab_off.csv")
on_rows,  on  = load("../results/epmc_ab_on.csv")
print(f"  Europe PMC OFF : {sum(off)}/{len(off)} = {sum(off)/len(off):.1%}")
print(f"  Europe PMC ON  : {sum(on)}/{len(on)} = {sum(on)/len(on):.1%}")
print(f"  cost of Europe PMC: {(sum(on)/len(on) - sum(off)/len(off))*100:+.1f} points")
# Pair by query text, NOT by position: a failed request is skipped rather than
# written, so the two arms can drift out of alignment by index.
byq = {r["query"]: r for r in off_rows}
flips = []
for b in on_rows:
    o = byq.get(b["query"])
    if o and bool(int(o["found_rank"])) != bool(int(b["found_rank"])):
        flips.append((b["query"][:60], f"{o['found_rank']}->{b['found_rank']}", b["epmc_ranks"]))
if flips:
    print(f"\n  {len(flips)} queries changed outcome (off -> on):")
    for q, ch, e in flips:
        print(f"    rank {ch:<10} epmc@{e or '-':<10} {q}")
PYEOF
