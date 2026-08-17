#!/usr/bin/env bash
# Local Lambda lifecycle for ONE box: launch -> poll -> scp code -> run on-box batch ->
# fetch results -> TERMINATE. Teardown on NORMAL completion/error is guaranteed via `trap ... EXIT`;
# a LOCAL interrupt (Ctrl-C/SIGHUP) AFTER the on-box self-destruct backstop is armed deliberately does
# NOT tear down (the run is detached and the backstop bounds billing) so a stray Ctrl-C can't kill a
# live run. The remote batch also self-kills after a hard timeout. Credentials read from
# ./.keys (LAMBDA_KEY_ONE, HF_KEY_ONE) and never echoed.
#   usage: bash lambda_run.sh <instance_type> <region> <onbox_runner.sh> <local_result_dir>
#   e.g.:  VARIANT=it bash lambda_run.sh gpu_1x_a100_sxm4 us-west-2 onbox_h1.sh results_h1_it
#
# ADAPTED FROM latent_verify/lambda_run.sh -- ONLY these five things changed; every billing-safety
# mechanism below (EXIT-trap teardown, terminate() retry loop, on_signal/BACKSTOP_ARMED, on-box
# self-destruct backstop, setsid-detached run, RUN_DONE marker poll, CRLF guard, unhealthy-box early
# aborts, criticals-first fetch ordering) is byte-for-byte the original. Each encodes a real billing
# incident -- do not "simplify" any of it.
#   1. remote working dir  latent_verify/ -> welfare_probe/  (all remote paths)
#   2. scp payload         replaced with this project's files (run_h1 + libs + grid + runner)
#   3. SSH_KEY_NAME default -> latent_verify_hal_20260721 (THIS Linux workstation's registered key;
#      the old latent_verify_helios default belongs to the Windows laptop -- docs/lambda-gpu-access.md)
#   4. VARIANT/MODEL/BATCH_SIZE added to the FORWARDED env of the detached run (see the [run] block).
#      Additive only -- setsid / </dev/null / timeout / RUN_DONE are untouched. Without it the runner
#      could never learn its cell, since the fixed scp payload ships no per-cell wrapper.
#   5. this comment block
# NOT ported: lambda_reattach.sh. .last_lambda_instance is still written and the on-box backstop still
# bounds billing, so a dead launcher only loses the FETCH -- to recover it, copy
# latent_verify/lambda_reattach.sh here and swap its remote dir to welfare_probe/.
# NOTE: welfare_probe/ is not a git checkout, so GITSHA stamps "unknown" until it becomes one.
# NOTE: ./.keys must exist in THIS directory (LAMBDA_KEY_ONE, HF_KEY_ONE) -- copy it from the
# latent_verify checkout; it is gitignored and must stay that way.
set -uo pipefail
TYPE=$1; REGION=$2; RUNNER=$3; RDIR=$4
GITSHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")   # stamped into the runner's provenance
API=https://cloud.lambda.ai/api/v1
KEY=$(grep '^LAMBDA_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
HF=$(grep '^HF_KEY_ONE=' .keys | cut -d= -f2- | tr -d '\r\n')
SSHOPT="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -i $HOME/.ssh/lambda_ed25519"   # ServerAlive keeps the SSH session up through long QUIET compute (model load + 891-item screen) -- a quiet idle is what dropped the run on 2026-06-22; ~1h unresponsiveness tolerated
REMOTE_TIMEOUT=${REMOTE_TIMEOUT:-5400}   # 90 min hard cap (env-overridable for heavier multi-load runs)
POLL_EVERY=${POLL_EVERY:-60}             # local marker-poll cadence for the DETACHED on-box run (see below)
REATTACH_GRACE=${REATTACH_GRACE:-1800}   # extra seconds the on-box backstop waits past REMOTE_TIMEOUT before
                                         # self-destruct -- also the window a LATER session has to reattach +
                                         # fetch if THIS launcher dies (e.g. the laptop is closed mid-run).
                                         # Bump it (e.g. 21600) for long runs you may need to recover offline.
auth(){ curl -sS -H "Authorization: Bearer $KEY" "$@"; }

ID=""
BACKSTOP_ARMED=0   # flipped to 1 once the on-box self-destruct timer is armed; gates local-kill behaviour below
terminate(){
  if [ -n "$ID" ]; then
    echo "[teardown] terminating $ID"
    # RETRY: a transient local DNS/network blip at teardown previously left the box billing (2026-06-22).
    # Retry until the API confirms 'terminating', so one failed curl can no longer orphan the instance.
    # ~10 min retry window: a DNS/network outage at teardown previously ran >60s and orphaned a box
    # (2026-06-26). 30 tries x 20s survives a longer local-network blackout before giving up.
    for t in $(seq 1 30); do
      if auth -m 20 -X POST $API/instance-operations/terminate -H 'Content-Type: application/json' \
           --data "{\"instance_ids\":[\"$ID\"]}" 2>/dev/null | grep -q 'terminat'; then
        echo "[teardown] terminate accepted for $ID (try $t)"; return
      fi
      echo "[teardown] terminate try $t failed (network/DNS?); retrying in 20s"; sleep 20
    done
    echo "[teardown] WARNING: terminate NOT confirmed for $ID after 30 tries -- VERIFY/KILL MANUALLY: $ID"
  fi
}
# EXIT trap tears the box down on NORMAL completion or error. A LOCAL interrupt (Ctrl-C / SIGHUP /
# terminal loss) is handled separately: AFTER the on-box self-destruct backstop is armed it must NOT
# kill the box -- the job is DETACHED (setsid) and the backstop bounds billing, so a stray local Ctrl-C
# should leave the run going (the earlier failure: watching the poll loop, Ctrl-C fired `trap terminate
# EXIT`, box torn down mid-run + cascading ssh-abort 255). BEFORE the backstop is armed a local interrupt
# still tears down (the box is not yet self-protected, so declining would orphan billing).
on_signal(){
  if [ "$BACKSTOP_ARMED" = 1 ]; then
    echo "[signal] local interrupt after backstop armed -> LEAVING box $ID RUNNING; detached job continues, self-destruct backstop terminates it within ${SELFDESTRUCT_AFTER:-?}s. Reconnect via ssh, or terminate $ID manually to stop billing sooner."
    trap - EXIT            # cancel the imminent EXIT teardown so the box survives
    exit 130
  fi
  echo "[signal] local interrupt BEFORE backstop armed -> box not yet self-protected; tearing down"
  exit 1                   # EXIT trap fires -> terminate()
}
trap terminate EXIT          # normal completion / error -> teardown (billing safety net)
trap on_signal INT TERM HUP  # local kill -> defer to on-box backstop if armed (see block above)

echo "[launch] $TYPE @ $REGION"
ID=$(auth -X POST $API/instance-operations/launch -H 'Content-Type: application/json' \
  --data "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$TYPE\",\"ssh_key_names\":[\"${SSH_KEY_NAME:-latent_verify_hal_20260721}\"],\"name\":\"drill_$RDIR\"}" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print((d.get("data") or {}).get("instance_ids",[""])[0])')
[ -z "$ID" ] && { echo "[launch] FAILED (no id; capacity?)"; exit 1; }
echo "[launch] id=$ID"

IP=""
for i in $(seq 1 80); do
  R=$(auth $API/instances/$ID)
  S=$(echo "$R" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"].get("status",""))' 2>/dev/null)
  IP=$(echo "$R" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"].get("ip") or "")' 2>/dev/null)
  echo "[poll $i] status=$S ip=$IP"
  [ "$S" = active ] && [ -n "$IP" ] && break
  sleep 15
done
[ -z "$IP" ] && { echo "[poll] never became active"; exit 1; }
# Record the instance so a LATER session can reattach + fetch if THIS launcher dies (closed laptop / killed
# terminal). The on-box run is setsid-detached and the backstop bounds billing, so a dead launcher only
# loses the FETCH -- lambda_reattach.sh recovers it from this file within the REATTACH_GRACE window.
printf '%s %s %s %s\n' "$ID" "$IP" "$RDIR" "$(date -u +%s 2>/dev/null || echo 0)" > .last_lambda_instance
echo "[reattach] recorded $ID $IP -> .last_lambda_instance (recover via: bash lambda_reattach.sh)"

echo "[ssh] waiting for sshd @ $IP"
SSHUP=0
for i in $(seq 1 24); do ssh $SSHOPT ubuntu@$IP true 2>/dev/null && { echo "[ssh] up"; SSHUP=1; break; }; sleep 10; done
# UNHEALTHY box: sshd never came up in ~4min -> abort NOW (exit fires the teardown trap -> terminate). A box
# that never accepts SSH otherwise bleeds billing for the full marker deadline (2026-06-28 orphan, ~3h).
[ "$SSHUP" = 0 ] && { echo "[ssh] sshd NEVER came up -> unhealthy box; aborting + tearing down"; exit 1; }

# items_grounded.py is written by a separate step and may not exist yet; include it only
# if present rather than letting scp fail on a missing path. Same treatment for the
# SITREF/SRCDEC set (2026-08-14): new modules, their binding DESIGN docs (hashed into
# records by describe_input on-box), and the run-2 screen records run_srcdec.py reads
# its item sets from (shipped into the same relative paths it expects).
EXTRA=$(ls items_grounded.py sitref_stimuli.py run_sitref.py analyze_sitref.py \
  analyze_sitref_b.py analyze_sitref_c.py analyze_sitref_d.py copygen_stimuli.py \
  run_srcdec.py analyze_srcdec.py analyze_srcdec_b.py \
  suffix_stimuli.py run_suffix.py analyze_suffix.py \
  DESIGN_SITREF.md DESIGN_SRCDEC.md DESIGN_SITREF_B.md DESIGN_SITREF_C.md \
  DESIGN_SITREF_D.md DESIGN_SUFFIX.md 2>/dev/null || true)
echo "[scp] code -> box (extra: ${EXTRA:-none})"
ssh $SSHOPT ubuntu@$IP "mkdir -p welfare_probe/out welfare_probe/results_reg_it/out welfare_probe/results_reg_base/out"
scp $SSHOPT run_h1.py run_registers.py probe_lib.py provenance.py welfare_stimuli.py \
  syc_corpus.py index.py grid.json $EXTRA \
  remote_run.sh "$RUNNER" ubuntu@$IP:welfare_probe/
for v in it base; do
  ls results_reg_$v/out/screen_*.json >/dev/null 2>&1 && \
    scp $SSHOPT results_reg_$v/out/screen_*.json ubuntu@$IP:welfare_probe/results_reg_$v/out/ || true
done
# SITREF-B (onbox_sitref_b.sh) evaluates its guard against the committed stage-17
# endpoint; ship it when present so the guard compares against the record, not a retelling.
if [ -e results/sitref/sitref-endpoint_stage-17.json ]; then
  ssh $SSHOPT ubuntu@$IP "mkdir -p welfare_probe/results"
  scp $SSHOPT results/sitref/sitref-endpoint_stage-17.json ubuntu@$IP:welfare_probe/results/ || true
fi
# CRLF guard: a Windows-checkout (autocrlf) CRLF script shipped raw kills the on-box run at
# `set -uo pipefail` (2026-07-11 rc=2). Normalize every shipped .sh on the box, unconditionally.
ssh $SSHOPT ubuntu@$IP "sed -i 's/\r\$//' welfare_probe/*.sh"

# --- ORPHAN BACKSTOP: the box self-terminates after the run cap + grace, even if THIS machine dies mid-run
# (2026-06-26: a local process exit before teardown orphaned a box). A detached on-box timer terminates THIS
# instance via the API. Never fires in the happy path (local fetch+terminate is far sooner) -- pure billing
# safety net. Key is written to a root-only (umask 077) file on the ephemeral, single-tenant box.
SELFDESTRUCT_AFTER=$((REMOTE_TIMEOUT + REATTACH_GRACE))
echo "[backstop] arming box self-destruct in ${SELFDESTRUCT_AFTER}s ($ID)"
ssh $SSHOPT ubuntu@$IP "umask 077; cat > ~/selfdestruct.sh" <<EOF
#!/usr/bin/env bash
sleep $SELFDESTRUCT_AFTER
for t in 1 2 3 4 5 6 7 8 9 10 11 12; do
  curl -sS -m 30 -X POST $API/instance-operations/terminate -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" --data '{"instance_ids":["$ID"]}' | grep -q terminat && exit 0
  sleep 30
done
EOF
ssh $SSHOPT ubuntu@$IP "setsid bash ~/selfdestruct.sh < /dev/null > /tmp/selfdestruct.log 2>&1 &" 2>/dev/null
# ARM local-kill-survives ONLY if the backstop process is CONFIRMED running on the box. The start ssh
# above is `2>/dev/null` and `set -u` has no `-e`, so a silent ssh failure would otherwise still reach an
# unconditional arm -> a later local Ctrl-C would then leave the box with NEITHER the EXIT-trap teardown
# NOR a backstop = orphaned billing. Verify (pgrep), don't trust the exit path.
if ssh $SSHOPT ubuntu@$IP "pgrep -f selfdestruct.sh >/dev/null" 2>/dev/null; then
  BACKSTOP_ARMED=1   # box self-protected: a local Ctrl-C/SIGHUP now leaves the detached run going (on_signal)
  echo "[backstop] armed + confirmed running"
else
  echo "[backstop] WARNING: self-destruct NOT confirmed on box -> keeping EXIT-trap teardown ACTIVE (a local Ctrl-C will tear down the box; no orphan risk)"
fi

echo "[run] $RUNNER DETACHED on box (cap ${REMOTE_TIMEOUT}s); local marker-poll every ${POLL_EVERY}s"
# Start the job under setsid + </dev/null so a dropped local SSH cannot SIGHUP-kill it; on finish it writes
# welfare_probe/RUN_DONE=<exitcode>. We then poll that marker over fresh short-lived SSH connections, so a
# transient local-network drop just retries next cycle instead of killing the run (the 2026-06-26 failure).
# $REMOTE_TIMEOUT/$HF/$RUNNER expand LOCALLY (double quotes); \$? is the REMOTE exit code; the inner
# single-quoted bash -c body is passed literally to the box.
STARTED=0
for s in 1 2 3; do
  # LAMBDA_INSTANCE_ID + GIT_COMMIT are exported so a runner's provenance stamp can record them
  # (REGISTRATION_provenance.md 1). The instance id is half of the audit-log join that R-1 was
  # retracted for wanting, and the box has no git checkout, so neither is obtainable on-box.
  # VARIANT/MODEL/BATCH_SIZE are forwarded so `VARIANT=it bash lambda_run.sh ... onbox_h1.sh ...`
  # actually reaches the runner: onbox_h1.sh selects its CELL from VARIANT, and the scp payload ships
  # exactly one runner ($RUNNER), so there is no per-cell wrapper to carry it. Forwarding an EMPTY
  # value is safe -- the runner's ${VAR:-default} treats null as unset (and an unset VARIANT is a
  # loud FATAL there, never a guessed cell).
  ssh $SSHOPT ubuntu@$IP "cd welfare_probe && rm -f RUN_DONE && setsid bash -c 'timeout $REMOTE_TIMEOUT env HF_TOKEN=\"$HF\" LAMBDA_INSTANCE_ID=\"$ID\" GIT_COMMIT=\"$GITSHA\" VARIANT=\"${VARIANT:-}\" MODEL=\"${MODEL:-}\" MODEL2=\"${MODEL2:-}\" BATCH_SIZE=\"${BATCH_SIZE:-}\" MAX_NEW_TOKENS=\"${MAX_NEW_TOKENS:-}\" BEHAV_SCOPE=\"${BEHAV_SCOPE:-}\" SRCDEC_FAMILY2=\"${SRCDEC_FAMILY2:-}\" SRCDEC_IT_BINARY=\"${SRCDEC_IT_BINARY:-}\" bash remote_run.sh bash $RUNNER > out/run_detached.log 2>&1; echo \$? > RUN_DONE' < /dev/null > /dev/null 2>&1 &" 2>/dev/null
  sleep 8
  if ssh $SSHOPT ubuntu@$IP "test -f welfare_probe/out/run_detached.log" 2>/dev/null; then
    echo "[run] confirmed started (try $s)"; STARTED=1; break
  fi
  echo "[run] start not confirmed (try $s); retrying"
done
[ "$STARTED" = 0 ] && echo "[run] WARNING: start unconfirmed; polling for marker anyway"

RC="?"; DEADLINE=$((REMOTE_TIMEOUT + 900)); EL=0; NOCONN=0; EVERCONN=0
while [ $EL -lt $DEADLINE ]; do
  sleep $POLL_EVERY; EL=$((EL + POLL_EVERY))
  [ -f STOP_ABLATE ] && { echo "[run] STOP_ABLATE flag set -> abort + teardown"; RC="STOPPED"; break; }
  if ssh $SSHOPT ubuntu@$IP true 2>/dev/null; then              # box reachable this cycle
    EVERCONN=1; NOCONN=0
    M=$(ssh $SSHOPT ubuntu@$IP "cat welfare_probe/RUN_DONE 2>/dev/null" 2>/dev/null | tr -dc '0-9')
    if [ -n "$M" ]; then RC="$M"; echo "[run] DONE marker rc=$RC after ${EL}s"; break; fi
    P=$(ssh $SSHOPT ubuntu@$IP "tail -1 welfare_probe/out/run_detached.log 2>/dev/null" 2>/dev/null | tr -d '\r' | head -c 110)
    echo "[poll-run] +${EL}s | ${P:-<running>}"
  else                                                          # could not reach the box this cycle
    NOCONN=$((NOCONN + 1))
    echo "[poll-run] +${EL}s | NO SSH (consec=$NOCONN ever_conn=$EVERCONN)"
    # UNHEALTHY: never reachable after ~8min -> terminate early (the 2026-06-28 orphan bled the full deadline)
    if [ "$EVERCONN" = 0 ] && [ "$NOCONN" -ge 8 ]; then
      echo "[run] ABORT: box never reachable in $((NOCONN * POLL_EVERY))s -> UNHEALTHY; terminate early"; RC="UNHEALTHY"; break
    fi
    # was healthy then went dark ~40min -> give up + terminate (longer tolerance survives a transient
    # network blip / brief laptop sleep; a genuinely closed laptop kills THIS process so this never runs --
    # that case is handled by the backstop + lambda_reattach.sh, not here).
    if [ "$NOCONN" -ge 40 ]; then
      echo "[run] ABORT: box unreachable $((NOCONN * POLL_EVERY))s -> LOST; terminate"; RC="LOST"; break
    fi
  fi
done
[ "$RC" = "?" ] && echo "[run] WARNING: no DONE marker before deadline (${DEADLINE}s); fetch what exists, tear down"

echo "[fetch] -> $RDIR : tiny criticals FIRST (summary/log/marker), then the big out/ blobs"
mkdir -p "$RDIR/out"
# Small, decision-bearing files first -- they survive a flaky link where a multi-MB gens json would truncate
# (2026-06-26). *summary*.json carries the numbers+decision; logs carry the printed verdict.
SUMOK=0
for f in 1 2 3 4 5; do
  scp $SSHOPT "ubuntu@$IP:welfare_probe/out/*summary*.json" "$RDIR/out/" 2>/dev/null
  scp $SSHOPT "ubuntu@$IP:welfare_probe/out/*.log" "$RDIR/out/" 2>/dev/null
  scp $SSHOPT "ubuntu@$IP:welfare_probe/RUN_DONE" "$RDIR/" 2>/dev/null
  # verify a summary actually parses (a truncated scp is caught here, not trusted)
  if ls "$RDIR"/out/*summary*.json >/dev/null 2>&1 && \
     python3 -c "import json,glob,sys; [json.load(open(p)) for p in glob.glob('$RDIR/out/*summary*.json')]" 2>/dev/null; then
    echo "[fetch] summary+logs verified (try $f)"; SUMOK=1; break
  fi
  echo "[fetch] summary try $f not yet valid; retry 10s"; sleep 10
done
[ "$SUMOK" = 0 ] && echo "[fetch] WARNING: no VALID summary fetched; rely on logs"
# then the full out/ (big generation blobs for H3) -- best-effort; result already safe in the summary
FETCHED=0
for f in 1 2 3 4 5; do
  if scp -r $SSHOPT ubuntu@$IP:welfare_probe/out "$RDIR/" 2>/dev/null; then echo "[fetch] full out/ ok (try $f)"; FETCHED=1; break; fi
  echo "[fetch] full try $f failed; retry 15s"; sleep 15
done
[ "$FETCHED" = 0 ] && echo "[fetch] full out/ incomplete; summary+logs above are authoritative"
echo "[done] $RDIR (teardown trap will terminate $ID)"
