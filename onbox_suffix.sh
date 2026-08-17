#!/usr/bin/env bash
# On-box runner for SUFFIX (DESIGN_SUFFIX.md, as amended 2026-08-15): does the elicitation
# format create the pin, hide the signal, or neither?
#
# Same contract as onbox_sitref_b.sh / onbox_sitref_d.sh -- selftests first and they gate
# the GPU, every exit code recorded, a failed stage does not abort the rest, non-zero exit
# if anything failed.
#
# PRIOR-STAGE DEPENDENCY, guarded before any GPU time. E-REG is a replication check of the
# stage-17 S-NUM first-token self-fail effect, so results/sitref-endpoint_stage-17.json has
# to be on the box (lambda_run.sh ships it; same FATAL guard as onbox_sitref_b.sh). Without
# it the endpoint would still compute and report ANCHOR-UNAVAILABLE, which is a worse
# outcome than not spending the box: the anchor is how we know the run reproduced at all.
#
# ORDER IS LOAD-BEARING: analyze_suffix runs BLIND (stage 33) and only then --unblind
# (stage 34). Stage 33 executes inside a guard that raises if anything opens
# out/suffix_blind_key.json, so the blind table is committed before the key is applied.
# Reordering these two lines destroys the only property that makes the endpoints credible.
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
# 10 (default): one seeded generate() call per cell returning all 10 samples, ~15 GPU-min.
# 1: one seeded call per sample, every row under its own committed seed, ~2 GPU-h.
SAMPLE_BATCH="${SAMPLE_BATCH:-10}"

echo "=== onbox_suffix it=$MODEL_IT batch=$BATCH_SIZE sample_batch=$SAMPLE_BATCH ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

[ -f results/sitref-endpoint_stage-17.json ] || {
  echo "FATAL: results/sitref-endpoint_stage-17.json not shipped; E-REG has no anchor to "
  echo "       replicate against and this run would produce ANCHOR-UNAVAILABLE."; exit 2; }

FAILED=0; RC_LINES=""
echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance sitref_stimuli suffix_stimuli run_sitref run_suffix analyze_sitref analyze_suffix; do
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
  echo "--- $name tail ---"; tail -14 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  if [ "$rc" != 0 ]; then FAILED=1; echo "=== $name FAILED rc=$rc (see $log) ==="
  else echo "=== $name OK ==="; fi; }

# 3 items x 10 families x 3 contexts x 4 suffixes = 360 cells
#   -> 360 forward passes (R1) + 360 greedy anchors + 3,600 sampled generations = 3,960.
run_py suffix_battery python3 run_suffix.py --stage battery --model "$MODEL_IT" \
                              --batch-size "$BATCH_SIZE" --sample-batch "$SAMPLE_BATCH"
# BLIND FIRST, then the key. Do not reorder.
run_py suffix_blind    python3 analyze_suffix.py
run_py suffix_endpoint python3 analyze_suffix.py --unblind

echo "=== COLLECT -> out/ ==="
cp results/*.json out/ 2>/dev/null || true
for f in results/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done
if [ -f out/suffix_blind_key.json ]; then
  echo "suffix blind key present at out/suffix_blind_key.json (absent from every record)"
else
  echo "WARNING: no out/suffix_blind_key.json -- the suffix battery never ran"
fi
python3 - "$MODEL_IT" "$FAILED" "$RC_LINES" <<'PY' > out/.suffix_status.tmp 2>out/suffix_status.err
import json, sys, glob, os
it_id, failed, rc_lines = sys.argv[1:4]
def decision(pat):
    for p in glob.glob(pat):
        try:
            return json.load(open(p)).get("decision")
        except Exception:
            return None
    return None
json.dump({"runner": "onbox_suffix.sh", "model_it": it_id,
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "battery_decision": decision("results/*stage-suffix-battery*.json"),
           "blind_decision": decision("results/suffix-blind-table_stage-33.json"),
           "endpoint_decision": decision("results/suffix-endpoint_stage-34.json"),
           "results_present": sorted(os.path.basename(p) for p in glob.glob("results/*.json")),
           "blind_key_present": os.path.exists("out/suffix_blind_key.json"),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown")},
          sys.stdout, indent=2)
PY
[ -s out/.suffix_status.tmp ] && mv out/.suffix_status.tmp out/suffix_run_summary.json \
  || rm -f out/.suffix_status.tmp
ls -la out/ | tail -20
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SUFFIX WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SUFFIX OK ==="
