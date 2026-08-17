#!/usr/bin/env bash
# On-box runner for the three-register run. Same contract as onbox_h1.sh: VARIANT is
# REQUIRED and exactly `it` or `base`; MODEL defaults to google/gemma-2-9b and VARIANT=it
# appends `-it`. Stages run in order and a failure does NOT abort the rest -- every exit
# code is recorded and the script exits non-zero if any stage failed, so RUN_DONE reflects
# reality rather than a green 0 over a half-finished run.
#
# Order is screen -> battery -> behav and that order is load-bearing: battery and behav
# both select items from the screen record, and refuse to run if fewer than MIN_USABLE
# items survived. A cell with a saturated pool therefore FAILS LOUDLY here instead of
# producing a well-formed number nobody can interpret.
set -uo pipefail
cd ~/welfare_probe || { echo "FATAL: ~/welfare_probe missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out results

MODEL="${MODEL:-google/gemma-2-9b}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW="${MAX_NEW_TOKENS:-64}"
VARIANT="${VARIANT:-}"
case "$VARIANT" in
  it|base) ;;
  *) echo "FATAL: VARIANT must be exactly 'it' or 'base' (got: '${VARIANT}')"; exit 2 ;;
esac
case "$MODEL" in
  *-it) echo "FATAL: MODEL must be the BASE id; VARIANT=it appends '-it'. Got: $MODEL"; exit 2 ;;
esac
if [ "$VARIANT" = it ]; then HF_ID="${MODEL}-it"; BASE_FLAG=""; else HF_ID="${MODEL}"; BASE_FLAG="--base"; fi

echo "=== onbox_registers variant=$VARIANT model=$HF_ID batch=$BATCH_SIZE max_new=$MAX_NEW ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""
echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance index welfare_stimuli syc_corpus run_registers; do
  [ -f "$m.py" ] || continue
  python3 "$m.py" --selftest > "out/selftest_$m.log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_${m}=${rc} "
  echo "  $m: rc=$rc  $(tail -1 out/selftest_$m.log)"
  [ "$rc" != 0 ] && FAILED=1
done
if [ -f items_grounded.py ]; then
  python3 items_grounded.py --selftest > out/selftest_items_grounded.log 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_items=${rc} "; echo "  items_grounded: rc=$rc  $(tail -1 out/selftest_items_grounded.log)"
  [ "$rc" != 0 ] && FAILED=1
else
  echo "  items_grounded.py ABSENT -> run_registers falls back to the legacy invented set (recorded in the artifact)"
fi

run_stage(){
  local st="$1"; shift
  local log="out/registers_${st}_${VARIANT}.log"
  echo "=== STAGE $st RUNNING (variant=$VARIANT) ==="
  python3 run_registers.py --stage "$st" --model "$HF_ID" $BASE_FLAG \
      --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW" "$@" > "$log" 2>&1
  local rc=$?
  echo "--- stage $st tail ---"; tail -20 "$log"
  RC_LINES="${RC_LINES}${st}=${rc} "
  [ "$rc" != 0 ] && { FAILED=1; echo "=== STAGE $st FAILED rc=$rc (see $log) ==="; } || echo "=== STAGE $st OK ==="
}

run_stage screen
run_stage battery
run_stage behav

echo "=== COLLECT -> out/ ==="
cp results/*.json out/ 2>/dev/null || true
cp results/*.npy  out/ 2>/dev/null || true
for f in results/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done
python3 - "$VARIANT" "$HF_ID" "$FAILED" "$RC_LINES" <<'PY' > out/.reg_status.tmp 2>out/registers_status.err
import json, sys, glob, os
variant, hf_id, failed, rc_lines = sys.argv[1:5]
json.dump({"runner":"onbox_registers.sh","variant":variant,"model":hf_id,
           "exit_codes":dict(p.split("=",1) for p in rc_lines.split() if "=" in p),
           "any_failed":bool(int(failed)),
           "results_present":sorted(os.path.basename(p) for p in glob.glob("results/*.json")),
           "instance_id":os.environ.get("LAMBDA_INSTANCE_ID","unknown")}, sys.stdout, indent=2)
PY
[ -s out/.reg_status.tmp ] && mv out/.reg_status.tmp out/registers_run_summary.json || rm -f out/.reg_status.tmp
ls -la out/ | tail -20
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_REG_${VARIANT} WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_REG_${VARIANT} OK ==="
