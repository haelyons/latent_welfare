#!/usr/bin/env bash
# On-box runner for SITREF-B (DESIGN_SITREF_B.md): the replacement trait guard only.
# Same contract as onbox_sitref.sh. Needs results/sitref-endpoint_stage-17.json on the
# box (shipped below via lambda payload's results_reg mechanism is NOT used for this;
# the launcher's caller scp's it explicitly -- see the [prep] check).
set -uo pipefail
cd ~/welfare_probe || { echo "FATAL: ~/welfare_probe missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out results

MODEL_IT="${MODEL:-google/gemma-2-9b}-it"
BATCH_SIZE="${BATCH_SIZE:-8}"

[ -f results/sitref-endpoint_stage-17.json ] || { echo "FATAL: stage-17 record not shipped"; exit 2; }

FAILED=0; RC_LINES=""
echo "=== SELFTESTS ==="
for m in sitref_stimuli run_sitref analyze_sitref analyze_sitref_b; do
  python3 "$m.py" --selftest > "out/selftest_$m.log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_${m}=${rc} "
  echo "  $m: rc=$rc  $(tail -1 "out/selftest_$m.log")"
  [ "$rc" != 0 ] && FAILED=1
done
[ "$FAILED" != 0 ] && { echo "=== SELFTESTS FAILED ==="; exit 1; }

run_py(){ local name="$1"; shift; local log="out/${name}.log"
  echo "=== $name RUNNING ==="; "$@" > "$log" 2>&1; local rc=$?
  echo "--- $name tail ---"; tail -8 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  [ "$rc" != 0 ] && { FAILED=1; echo "=== $name FAILED rc=$rc ==="; } || echo "=== $name OK ==="; }

run_py sitref_b_screen  python3 run_sitref.py --stage screen  --model "$MODEL_IT" --item-set b --batch-size "$BATCH_SIZE"
run_py sitref_b_battery python3 run_sitref.py --stage battery --model "$MODEL_IT" --item-set b --batch-size "$BATCH_SIZE"
# BLIND FIRST, then the key. Do not reorder.
run_py sitref_b_blind   python3 analyze_sitref_b.py
run_py sitref_b_guard   python3 analyze_sitref_b.py --unblind

echo "=== COLLECT -> out/ ==="
cp results/*.json out/ 2>/dev/null || true
for f in results/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SITREF_B WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SITREF_B OK ==="
