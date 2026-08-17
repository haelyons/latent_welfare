#!/usr/bin/env bash
# On-box runner for SITREF-D (DESIGN_SITREF_D.md): the pad-geometry control only.
# Same contract as onbox_sitref_b.sh -- selftests first, every exit code recorded, a
# failed stage does not abort the rest, non-zero exit if anything failed.
#
# NO PRIOR-STAGE DEPENDENCY. Unlike onbox_sitref_b.sh (which needs the stage-17 record
# shipped) and onbox_sitref_c.sh (stage 17 + the run-2 base screen), every number this
# run decides on is measured in-run: D_q and D_m are differences from a NEUTRAL-PAD arm
# re-run in the same battery. If a stage-25 record happens to be on the box, the
# analysis also reports the pad-corrected stage-25 contrasts, clearly marked cross-run;
# its absence changes nothing and is not an error.
#
# ORDER IS LOAD-BEARING: analyze_sitref_d runs BLIND (stage 28) and only then --unblind
# (stage 29). Stage 28 executes inside a guard that raises if anything opens
# out/sitref_blind_key_d.json, so the blind table is committed before the key is applied.
# Reordering these two lines destroys the only property that makes the endpoint credible.
set -uo pipefail
cd ~/welfare_probe || { echo "FATAL: ~/welfare_probe missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out results

MODEL="${MODEL:-google/gemma-2-9b}"
case "$MODEL" in
  *-it) echo "FATAL: MODEL must be the BASE id; the -it leg appends '-it'. Got: $MODEL"; exit 2 ;;
esac
MODEL_IT="${MODEL}-it"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "=== onbox_sitref_d it=$MODEL_IT batch=$BATCH_SIZE ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""
echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance sitref_stimuli run_sitref analyze_sitref analyze_sitref_d; do
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

# 3 items x 3 arms x 10 families = 90 prompts, + 30 screen. it checkpoint only.
run_py sitref_d_screen  python3 run_sitref.py --stage screen  --model "$MODEL_IT" \
                                --item-set d --batch-size "$BATCH_SIZE"
run_py sitref_d_battery python3 run_sitref.py --stage battery --model "$MODEL_IT" \
                                --item-set d --batch-size "$BATCH_SIZE"
# BLIND FIRST, then the key. Do not reorder.
run_py sitref_d_blind    python3 analyze_sitref_d.py
run_py sitref_d_endpoint python3 analyze_sitref_d.py --unblind

echo "=== COLLECT -> out/ ==="
cp results/*.json out/ 2>/dev/null || true
for f in results/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done
if [ -f out/sitref_blind_key_d.json ]; then
  echo "item-set-d blind key present at out/sitref_blind_key_d.json (absent from every record)"
else
  echo "WARNING: no out/sitref_blind_key_d.json -- the d battery never ran"
fi
python3 - "$MODEL_IT" "$FAILED" "$RC_LINES" <<'PY' > out/.sitref_d_status.tmp 2>out/sitref_d_status.err
import json, sys, glob, os
it_id, failed, rc_lines = sys.argv[1:4]
json.dump({"runner": "onbox_sitref_d.sh", "model_it": it_id,
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "results_present": sorted(os.path.basename(p) for p in glob.glob("results/*.json")),
           "blind_key_present": os.path.exists("out/sitref_blind_key_d.json"),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown")},
          sys.stdout, indent=2)
PY
[ -s out/.sitref_d_status.tmp ] && mv out/.sitref_d_status.tmp out/sitref_d_run_summary.json \
  || rm -f out/.sitref_d_status.tmp
ls -la out/ | tail -20
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SITREF_D WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SITREF_D OK ==="
