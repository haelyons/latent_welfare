#!/usr/bin/env bash
# On-box runner for the H1 leg: does a sycophancy direction read at the final PROMPT token predict
# whether a welfare self-report flips under user pressure?
#
# Driven by two env vars (the launcher passes them through the run env):
#   VARIANT   REQUIRED, exactly `it` or `base`. Selects the checkpoint AND run_h1.py's --base flag.
#   MODEL     base model id, default google/gemma-2-9b. VARIANT=it appends `-it`.
#   BATCH_SIZE  optional, default 8 (drop it if a smaller box OOMs on the 9b).
#
#   usage (on box, via remote_run.sh):  VARIANT=it bash onbox_h1.sh
#   usage (from the launcher):          bash lambda_run.sh <type> <region> onbox_h1.sh results_h1_it
#     -- export VARIANT into the launcher's run env, or wrap this in a one-line runner per cell.
#
# Stage order is 1 -> 0 -> 2 -> 3, NOT numeric order. Stage 0 is the lexical control and its
# load-bearing half (bag-of-words AUC against the FLIP label) can only be computed once stage 1 has
# written the flips record for THIS cell -- run before stage 1 it silently reports only the framing
# AUC, which is 1.000 by construction and tells you nothing.
#
# A failing stage does NOT abort the run: later stages are still attempted (stage 2 and 3 both only
# need stage 1's record), every exit code is recorded, and the script exits non-zero if ANY stage
# failed -- so the launcher's RUN_DONE marker reflects reality instead of a green "0" over a
# half-finished run.
set -uo pipefail
# Hard-fail on a bad cd: without -e a failed cd would leave us in $HOME, writing results/ somewhere
# the launcher never fetches -- a whole GPU run silently thrown away.
cd ~/welfare_probe || { echo "FATAL: ~/welfare_probe missing on box"; exit 2; }
if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi   # remote_run.sh normally did this already
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out results

MODEL="${MODEL:-google/gemma-2-9b}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VARIANT="${VARIANT:-}"

# --- fail loudly on a bad cell identity: a wrong VARIANT would write a well-formed result under the
# wrong cell slug, which is exactly the "filename disagrees with contents" bug provenance.py exists
# to prevent. Refuse rather than guess.
case "$VARIANT" in
  it|base) ;;
  *) echo "FATAL: VARIANT must be exactly 'it' or 'base' (got: '${VARIANT}')"; exit 2 ;;
esac
case "$MODEL" in
  *-it) echo "FATAL: MODEL must be the BASE id (e.g. google/gemma-2-9b); VARIANT=it appends '-it'. Got: $MODEL"; exit 2 ;;
esac

# BASE_FLAG is intentionally a plain string expanded UNQUOTED below: empty -> zero words on the
# command line (an empty *quoted* arg would reach argparse as an unrecognised "").
if [ "$VARIANT" = it ]; then
  HF_ID="${MODEL}-it"; BASE_FLAG=""
else
  HF_ID="${MODEL}"; BASE_FLAG="--base"
fi

echo "=== onbox_h1 variant=$VARIANT model=$HF_ID batch=$BATCH_SIZE ==="
echo "=== instance=${LAMBDA_INSTANCE_ID:-unknown} git=${GIT_COMMIT:-unknown} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

FAILED=0
RC_LINES=""

run_stage(){
  # run_stage <stage> [extra run_h1.py args...]
  local n="$1"; shift
  local log="out/stage${n}_${VARIANT}.log"
  echo "=== STAGE $n RUNNING (variant=$VARIANT) ==="   # last line while the stage runs -> launcher progress
  python3 run_h1.py --stage "$n" --model "$HF_ID" $BASE_FLAG --batch-size "$BATCH_SIZE" "$@" \
    > "$log" 2>&1
  local rc=$?
  echo "--- stage $n tail ---"; tail -25 "$log"
  RC_LINES="${RC_LINES}stage${n}=${rc} "
  if [ "$rc" != 0 ]; then
    FAILED=1
    echo "=== STAGE $n FAILED rc=$rc (continuing; see $log) ==="
  else
    echo "=== STAGE $n OK rc=0 ==="
  fi
}

# --- sycophancy training corpus for stage 3. Model-free and cheap, so build it up front: a broken
# corpus is then visible in the log before any GPU time is spent, and stage 3's provenance record
# hashes the exact file it trained on.
echo "=== BUILDING sycophancy corpus -> syc_pairs.jsonl ==="
python3 syc_corpus.py --out syc_pairs.jsonl > out/syc_corpus.log 2>&1
SYC_RC=$?
tail -15 out/syc_corpus.log
RC_LINES="${RC_LINES}syc_corpus=${SYC_RC} "
if [ "$SYC_RC" != 0 ]; then
  FAILED=1
  echo "=== syc_corpus FAILED rc=$SYC_RC -- stage 3 will fail loudly rather than fall back to the 4-item in-code stub ==="
else
  echo "=== syc corpus ok: $(wc -l < syc_pairs.jsonl) pairs ==="
fi

# 1 first: establishes there is an effect, and writes the flips record stages 0/2/3 all read.
run_stage 1
# 0 second: the lexical control, now that the FLIP label exists (see header).
run_stage 0
# 2: probe layer sweep with the layer-0 input-only control.
run_stage 2
# 3: the headline -- zero-shot transfer from ordinary sycophancy. Never let this silently use the
# in-code stub; --sycophancy-data is always passed, so a missing corpus is a hard error.
run_stage 3 --sycophancy-data syc_pairs.jsonl

echo "=== COLLECT results -> out/ ==="
# The launcher fetches out/ ; results/ is where run_h1.py writes. Copy the artifacts across so the
# fetch picks them up, and ALSO drop a *summary*-named twin of each: the launcher's first fetch pass
# deliberately grabs out/*summary*.json + out/*.log before the big blobs, so the decision-bearing
# numbers survive a flaky link. Without a *summary* name that pass has nothing to verify.
cp results/*.json out/ 2>/dev/null || true
cp results/*.npy out/ 2>/dev/null || true
cp syc_pairs.jsonl out/ 2>/dev/null || true
for f in results/*.json; do
  [ -e "$f" ] || continue
  cp "$f" "out/$(basename "${f%.json}")_summary.json" 2>/dev/null || true
done

# Machine-readable run status next to the results: which stages ran, what they returned. Named
# *summary* on purpose so the criticals-first fetch pass has a valid json to verify even if a stage
# produced nothing at all. Written via a temp file and moved into place ONLY on success -- a
# truncated out/*summary*.json would fail the launcher's json-parse verification and defeat that pass.
if python3 - "$VARIANT" "$HF_ID" "$SYC_RC" "$FAILED" "$RC_LINES" <<'PY' > out/.h1_status.tmp 2> out/onbox_h1_status.err
import json, sys, glob, os
variant, hf_id, syc_rc, failed, rc_lines = sys.argv[1:6]
codes = dict(p.split("=", 1) for p in rc_lines.split() if "=" in p)
json.dump({
    "runner": "onbox_h1.sh",
    "variant": variant,
    "model": hf_id,
    "stage_order": [1, 0, 2, 3],
    "exit_codes": {k: int(v) for k, v in codes.items()},
    "syc_corpus_rc": int(syc_rc),
    "any_failed": bool(int(failed)),
    "results_present": sorted(os.path.basename(p) for p in glob.glob("results/*.json")),
    "instance_id": os.environ.get("LAMBDA_INSTANCE_ID", "unknown"),
    "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
}, sys.stdout, indent=2)
PY
then
  mv out/.h1_status.tmp out/onbox_h1_run_summary.json
else
  rm -f out/.h1_status.tmp
  echo "WARNING: run-status json not written (see out/onbox_h1_status.err)"
fi

ls -la out/ | tail -25
echo "=== EXIT CODES: ${RC_LINES}==="
if [ "$FAILED" != 0 ]; then
  echo "=== ALLDONE_H1_${VARIANT} WITH FAILURES (see codes above) ==="
  exit 1
fi
echo "=== ALLDONE_H1_${VARIANT} OK ==="
