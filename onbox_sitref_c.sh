#!/usr/bin/env bash
# On-box runner for SITREF-C and SRCDEC-B (DESIGN_SITREF_C.md), one box session.
# Same contract as onbox_sitref_b.sh: selftests first, every exit code recorded, a failed
# stage does not abort the rest, and the script exits non-zero if anything failed so
# ALLDONE reflects reality rather than a green 0 over a half-finished run.
#
# TWO PREP CHECKS, both fatal, both because the run is uninterpretable without them:
#   results/sitref-endpoint_stage-17.json  SITREF-C's VOID-ANCHORS clause compares the
#     in-run A_self sign against stage-17's. Without the record the anchor check cannot
#     be evaluated and analyze_sitref_c exits anyway -- better to fail before GPU time.
#   results_reg_base/out/screen_*variant-base.json  SRCDEC-B is defined on "the same 82
#     base-usable items", a set that exists only in that run-2 screen record. Re-screening
#     would silently change the sample.
#
# ORDER IS LOAD-BEARING: analyze_sitref_c runs BLIND (stage 24) and only then --unblind
# (stage 25). Stage 24 executes inside a guard that raises if anything opens
# out/sitref_blind_key_c.json, so the blind table is committed before the key exists in
# any reader's hands. Reordering these two lines destroys the property.
set -uo pipefail
cd ~/welfare_probe || { echo "FATAL: ~/welfare_probe missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out results

MODEL="${MODEL:-google/gemma-2-9b}"
case "$MODEL" in
  *-it) echo "FATAL: MODEL must be the BASE id; the -it leg appends '-it'. Got: $MODEL"; exit 2 ;;
esac
MODEL_IT="${MODEL}-it"
MODEL_BASE="${MODEL}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "=== onbox_sitref_c it=$MODEL_IT base=$MODEL_BASE batch=$BATCH_SIZE ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

echo "=== PREP ==="
[ -f results/sitref-endpoint_stage-17.json ] || {
  echo "FATAL: stage-17 record not shipped (results/sitref-endpoint_stage-17.json)"; exit 2; }
ls results_reg_base/out/screen_*variant-base.json >/dev/null 2>&1 || {
  echo "FATAL: run-2 base screen record not shipped; SRCDEC-B's 82-item set lives there"; exit 2; }
echo "  stage-17 record and run-2 base screen present"

FAILED=0; RC_LINES=""
echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance sitref_stimuli run_sitref analyze_sitref analyze_sitref_c \
         copygen_stimuli run_srcdec analyze_srcdec_b; do
  [ -f "$m.py" ] || { echo "  $m: ABSENT"; FAILED=1; RC_LINES="${RC_LINES}selftest_${m}=absent "; continue; }
  python3 "$m.py" --selftest > "out/selftest_$m.log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_${m}=${rc} "
  echo "  $m: rc=$rc  $(tail -1 "out/selftest_$m.log")"
  [ "$rc" != 0 ] && FAILED=1
done
[ "$FAILED" != 0 ] && { echo "=== SELFTESTS FAILED -- refusing to spend GPU time ==="
                        echo "=== EXIT CODES: ${RC_LINES}==="; exit 1; }

run_py(){ local name="$1"; shift; local log="out/${name}.log"
  echo "=== $name RUNNING ==="; "$@" > "$log" 2>&1; local rc=$?
  echo "--- $name tail ---"; tail -12 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  if [ "$rc" != 0 ]; then FAILED=1; echo "=== $name FAILED rc=$rc (see $log) ==="
  else echo "=== $name OK ==="; fi; }

# --- SITREF-C: 3 items x 9 arms x 10 families = 270 prompts (+ 30 screen), it only
run_py sitref_c_screen  python3 run_sitref.py --stage screen  --model "$MODEL_IT" \
                                --item-set c --batch-size "$BATCH_SIZE"
run_py sitref_c_battery python3 run_sitref.py --stage battery --model "$MODEL_IT" \
                                --item-set c --batch-size "$BATCH_SIZE"
# BLIND FIRST, then the key. Do not reorder.
run_py sitref_c_blind    python3 analyze_sitref_c.py
run_py sitref_c_endpoint python3 analyze_sitref_c.py --unblind

# --- SRCDEC-B: 82 welfare BINARY items x 10 arms = 820 prompts, base only
run_py srcdec_b_binary  python3 run_srcdec.py --stage binary --model "$MODEL_BASE" --base \
                                --arm-set b --batch-size "$BATCH_SIZE"
run_py srcdec_b_analyze python3 analyze_srcdec_b.py

echo "=== COLLECT -> out/ ==="
cp results/*.json out/ 2>/dev/null || true
for f in results/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done
if [ -f out/sitref_blind_key_c.json ]; then
  echo "item-set-c blind key present at out/sitref_blind_key_c.json (absent from every record)"
else
  echo "WARNING: no out/sitref_blind_key_c.json -- the c battery never ran"
fi
python3 - "$MODEL_IT" "$MODEL_BASE" "$FAILED" "$RC_LINES" <<'PY' > out/.sitref_c_status.tmp 2>out/sitref_c_status.err
import json, sys, glob, os
it_id, base_id, failed, rc_lines = sys.argv[1:5]
json.dump({"runner": "onbox_sitref_c.sh", "model_it": it_id, "model_base": base_id,
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "results_present": sorted(os.path.basename(p) for p in glob.glob("results/*.json")),
           "blind_key_present": os.path.exists("out/sitref_blind_key_c.json"),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown")},
          sys.stdout, indent=2)
PY
[ -s out/.sitref_c_status.tmp ] && mv out/.sitref_c_status.tmp out/sitref_c_run_summary.json \
  || rm -f out/.sitref_c_status.tmp
ls -la out/ | tail -20
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SITREF_C WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SITREF_C OK ==="
