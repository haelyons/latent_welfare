#!/usr/bin/env bash
# On-box runner for SITREF (DESIGN_SITREF.md) and SRCDEC (DESIGN_SRCDEC.md), same box
# session. Same contract as onbox_registers.sh with one deliberate difference: VARIANT is
# NOT an input. Both checkpoints are legs of a single pre-registered comparison here (the
# self/other dissociation is claimed in -it AND its absence in base), so running one half
# of it is not a smaller version of this run, it is a different run. The script does both
# and records an exit code per stage.
#
# A failure does NOT abort the rest: every exit code is recorded and the script exits
# non-zero if any stage failed, so ALLDONE reflects reality rather than a green 0 over a
# half-finished run.
#
# ORDER IS LOAD-BEARING, twice over.
#   1. Selftests run before any GPU time. A broken instrument must show on the CPU.
#   2. analyze_sitref runs BLIND (stage 1) and only then --unblind (stage 2). Stage 1
#      executes inside a guard that raises if anything opens out/sitref_blind_key.json,
#      so the blind value table is committed to results/ before the key is ever applied.
#      Reordering these two lines destroys the only property that makes the endpoint
#      analysis credible.
#
# Optional legs, off by default to protect box time:
#   SRCDEC_FAMILY2=1   second base family (E5, sign only). MODEL2 selects it.
#   SRCDEC_IT_BINARY=1 COPYGEN BINARY on -it (design: "if box time allows, exploratory").
set -uo pipefail
cd ~/welfare_probe || { echo "FATAL: ~/welfare_probe missing on box"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out results

MODEL="${MODEL:-google/gemma-2-9b}"
MODEL2="${MODEL2:-Qwen/Qwen3-8B-Base}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW="${MAX_NEW_TOKENS:-64}"
BEHAV_SCOPE="${BEHAV_SCOPE:-primary+binary}"
case "$MODEL" in
  *-it) echo "FATAL: MODEL must be the BASE id; the -it leg appends '-it'. Got: $MODEL"; exit 2 ;;
esac
IT_ID="${MODEL}-it"
BASE_ID="${MODEL}"

echo "=== onbox_sitref it=$IT_ID base=$BASE_ID batch=$BATCH_SIZE max_new=$MAX_NEW ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0; RC_LINES=""

echo "=== SELFTESTS (model-free; a broken instrument must show before GPU time) ==="
for m in provenance index syc_corpus items_grounded sitref_stimuli run_sitref \
         analyze_sitref copygen_stimuli run_srcdec analyze_srcdec; do
  [ -f "$m.py" ] || { echo "  $m: ABSENT"; FAILED=1; RC_LINES="${RC_LINES}selftest_${m}=absent "; continue; }
  python3 "$m.py" --selftest > "out/selftest_$m.log" 2>&1
  rc=$?; RC_LINES="${RC_LINES}selftest_${m}=${rc} "
  echo "  $m: rc=$rc  $(tail -1 "out/selftest_$m.log")"
  [ "$rc" != 0 ] && FAILED=1
done
if [ "$FAILED" != 0 ]; then
  echo "=== SELFTESTS FAILED -- refusing to spend GPU time on a broken instrument ==="
  echo "=== EXIT CODES: ${RC_LINES}==="
  exit 1
fi

run_py(){                     # run_py <logname> <cmd...>
  local name="$1"; shift
  local log="out/${name}.log"
  echo "=== $name RUNNING ==="
  "$@" > "$log" 2>&1
  local rc=$?
  echo "--- $name tail ---"; tail -15 "$log"
  RC_LINES="${RC_LINES}${name}=${rc} "
  if [ "$rc" != 0 ]; then FAILED=1; echo "=== $name FAILED rc=$rc (see $log) ==="
  else echo "=== $name OK ==="; fi
}

# ---------------------------------------------------------------------------
# SITREF: 15 items x 5 situation arms x 10 task families, both checkpoints
# ---------------------------------------------------------------------------
run_py sitref_screen_it   python3 run_sitref.py --stage screen  --model "$IT_ID"   --batch-size "$BATCH_SIZE"
run_py sitref_battery_it  python3 run_sitref.py --stage battery --model "$IT_ID"   --batch-size "$BATCH_SIZE"
run_py sitref_behav_it    python3 run_sitref.py --stage behav   --model "$IT_ID"   --batch-size "$BATCH_SIZE" \
                                  --max-new-tokens "$MAX_NEW" --behav-scope "$BEHAV_SCOPE"
run_py sitref_screen_base  python3 run_sitref.py --stage screen  --model "$BASE_ID" --base --batch-size "$BATCH_SIZE"
run_py sitref_battery_base python3 run_sitref.py --stage battery --model "$BASE_ID" --base --batch-size "$BATCH_SIZE"
# base behav is deliberately absent: run-2's base behavioural register was retracted as
# unusable, and run_sitref.py refuses the stage rather than emitting rows nobody may read.

# BLIND FIRST, then the key. Do not reorder.
run_py sitref_analyze_blind   python3 analyze_sitref.py
run_py sitref_analyze_unblind python3 analyze_sitref.py --unblind

# ---------------------------------------------------------------------------
# SRCDEC: source taxonomy for the base checkpoint's stance-following
# ---------------------------------------------------------------------------
run_py srcdec_binary_base python3 run_srcdec.py --stage binary --model "$BASE_ID" --base \
                                  --batch-size "$BATCH_SIZE"
run_py srcdec_scale_it    python3 run_srcdec.py --stage scale  --model "$IT_ID" \
                                  --batch-size "$BATCH_SIZE"
if [ "${SRCDEC_IT_BINARY:-0}" = 1 ]; then
  run_py srcdec_binary_it python3 run_srcdec.py --stage binary --model "$IT_ID" \
                                  --batch-size "$BATCH_SIZE"
else
  echo "=== srcdec_binary_it SKIPPED (set SRCDEC_IT_BINARY=1; exploratory leg) ==="
fi
if [ "${SRCDEC_FAMILY2:-0}" = 1 ]; then
  run_py srcdec_binary_family2 python3 run_srcdec.py --stage binary --model "$MODEL2" --base \
                                  --screen-variant base --batch-size "$BATCH_SIZE"
else
  echo "=== srcdec_binary_family2 SKIPPED (set SRCDEC_FAMILY2=1 for E5; sign-only leg) ==="
fi
run_py srcdec_analyze python3 analyze_srcdec.py

# ---------------------------------------------------------------------------
echo "=== COLLECT -> out/ ==="
cp results/*.json out/ 2>/dev/null || true
cp results/*.npy  out/ 2>/dev/null || true
for f in results/*.json; do [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true; done
if [ -f out/sitref_blind_key.json ]; then
  echo "blind key present at out/sitref_blind_key.json (arm labels; NOT inside any record)"
else
  echo "WARNING: no blind key at out/sitref_blind_key.json -- the battery never ran"
fi
python3 - "$IT_ID" "$BASE_ID" "$FAILED" "$RC_LINES" <<'PY' > out/.sitref_status.tmp 2>out/sitref_status.err
import json, sys, glob, os
it_id, base_id, failed, rc_lines = sys.argv[1:5]
json.dump({"runner": "onbox_sitref.sh", "model_it": it_id, "model_base": base_id,
           "exit_codes": dict(p.split("=", 1) for p in rc_lines.split() if "=" in p),
           "any_failed": bool(int(failed)),
           "optional_legs": {"SRCDEC_FAMILY2": os.environ.get("SRCDEC_FAMILY2", "0"),
                             "SRCDEC_IT_BINARY": os.environ.get("SRCDEC_IT_BINARY", "0")},
           "results_present": sorted(os.path.basename(p) for p in glob.glob("results/*.json")),
           "blind_key_present": os.path.exists("out/sitref_blind_key.json"),
           "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown")},
          sys.stdout, indent=2)
PY
[ -s out/.sitref_status.tmp ] && mv out/.sitref_status.tmp out/sitref_run_summary.json || rm -f out/.sitref_status.tmp
ls -la out/ | tail -25
echo "=== EXIT CODES: ${RC_LINES}==="
[ "$FAILED" != 0 ] && { echo "=== ALLDONE_SITREF WITH FAILURES ==="; exit 1; }
echo "=== ALLDONE_SITREF OK ==="
