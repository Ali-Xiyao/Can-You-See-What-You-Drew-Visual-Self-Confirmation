#!/usr/bin/env bash
# The L3 preview, start to terminal state, per the pre-registration in STATUS 34.
#
# Runs itself to one of two ends and stops:
#   A  divergence.json reports a d_star above the curve's own noise floor
#   B  failure, by one of the four conditions section 34 names
# and writes which one to runs/v4/l3-preview/TERMINAL.txt either way.
#
# Section 33 predicts B.1 ("no training happened"): thirty LoRA updates cannot
# move the backbone, so two flat curves are a statement about the corpus and
# not about the hypothesis. That prediction is on the record before the run,
# which is the only thing that keeps a null from being reinterpreted later.
set -u
cd "$(dirname "$0")/.."
source scripts/set_h_env.sh >/dev/null 2>&1 || true
export PYTHONPATH="$(cygpath -w "$(pwd)/src")"

OUT=runs/v4/l3-preview
CFG=configs/v4_l3_preview.yaml
OBS=envs/observer/python.exe
CORE=envs/core/python.exe
SHOWO=envs/showo2/python.exe
RUNS=(runs/v4/main-2plus1 runs/v4/main-1plus1plus1)
ROUNDS=10
ARMS=(naive rfo_gold)
STARTED=$(date +%s)
BUDGET=$((40 * 3600))            # section 34, condition B.4

log  () { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
stop () { echo "$*" > "$OUT/TERMINAL.txt"; log "TERMINAL: $*"; exit "${2:-0}"; }

over_budget () { [ $(( $(date +%s) - STARTED )) -gt "$BUDGET" ]; }

# --- which cards ----------------------------------------------------------
# This needs two cards and the earlier version of it did not know that. It
# demanded GPU1 be all but empty, and when it was not it put both stages on
# GPU0 and called that "the slower plan". It is not a slower plan, it is not a
# plan: the training process holds the backbone resident while it shells out to
# the ladder, so the adjudicator has to load beside 13.5 GB and cannot. That
# fallback ran twice and died twice in round 0, both times at 0xC0000005 during
# shard loading (STATUS 37).
#
# So the test is now "does the generator fit on GPU1 beside whatever is already
# there", not "is GPU1 empty" -- other people's jobs are not to be preempted,
# but a card with room is a card with room. The generator is the smaller of the
# two consumers, so it takes the shared card and the adjudicator gets GPU0 to
# itself; that is also what the frozen config asks for.
#
# When no assignment fits, this stops. Refusing beats degrading into something
# that cannot work, which is the whole lesson of the fallback it replaces.
GENERATOR_MIB=15000     # 13.5 GB resident plus room to breathe
pick_devices () {
  local used free
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 2p)
  free=$(( 24576 - ${used:-24576} ))
  if [ "$free" -ge "$GENERATOR_MIB" ]; then
    DEV=cuda:1; LADDER=cuda:0
    log "GPU1 has ${free} MiB free (${used} used, not ours): backbone on $DEV, ladder on $LADDER"
  else
    stop "B.3 no two-card assignment fits: GPU1 has only ${free} MiB free and the \
generator needs ${GENERATOR_MIB}. One card cannot host both stages -- the ladder \
loads while the backbone is resident. Wait for GPU1 or lower the scale." 1
  fi
}

# --- one attempt, one retry, then it is condition B.3 ---------------------
attempt () {
  local name="$1"; shift
  if "$@" >> "$OUT/$name.log" 2>&1; then return 0; fi
  log "$name failed once, retrying after 300s"
  sleep 300
  if "$@" >> "$OUT/$name.log" 2>&1; then return 0; fi
  stop "B.3 stage '$name' failed twice; see $OUT/$name.log" 1
}

adjudicate () {                   # $1 = directory holding manifest.jsonl
  local dir="$1"
  [ -f "$dir/verified.jsonl" ] && return 0
  local total; total=$(wc -l < "$dir/manifest.jsonl")
  for det in qwen3vl internvl; do
    local have=0
    [ -f "$dir/detections.$det.jsonl" ] && have=$(wc -l < "$dir/detections.$det.jsonl")
    [ "$have" -ge "$total" ] && continue
    attempt "detect.$det" "$OBS" scripts/v4_run_pipeline.py detect \
      --manifest "$dir/manifest.jsonl" --detector "$det" --device "$LADDER"
  done
  attempt crop   "$OBS"  scripts/v4_run_pipeline.py crop   --run "$dir" --device "$LADDER"
  attempt verify "$CORE" scripts/v4_run_pipeline.py verify --run "$dir"
}

# --- wait for the batch-3 chain to clear the cards ------------------------
log "waiting for the batch-3 pipeline to finish"
while [ ! -f runs/v4/tierb-conf2/conf_fork.txt ]; do
  over_budget && stop "B.4 over the 40h budget still waiting on batch-3" 1
  sleep 120
done
log "batch-3 done"

# The Gate B report walks three 17 GB gradient memmaps. Running it beside the
# training loop is the mistake that cost both cards fifty idle minutes on
# 2026-09-03, so wait it out rather than race it.
log "waiting for the queued Gate B report"
while [ ! -f runs/v4/gate-b/REPORT_DONE ]; do
  over_budget && stop "B.4 over the 40h budget still waiting on the Gate B report" 1
  sleep 60
done
log "Gate B report done ($(cat runs/v4/gate-b/REPORT_DONE))"

mkdir -p "$OUT"
pick_devices

# --- train ----------------------------------------------------------------
if [ ! -f "$OUT/checkpoints/rfo_gold/round-$(printf %03d $((ROUNDS-1)))/DONE.json" ]; then
  log "training $ROUNDS rounds, arms ${ARMS[*]}"
  attempt train "$SHOWO" scripts/v4_train.py train --outdir "$OUT" --config "$CFG" \
    --runs "${RUNS[@]}" --device "$DEV" --ladder-device "$LADDER" --max-epochs 1
fi

# --- per-checkpoint evaluation -------------------------------------------
STEPS_PER_ROUND=$("$CORE" -c "import yaml;print(yaml.safe_load(open('$CFG'))['training']['optimizer_steps_per_round'])")
for arm in "${ARMS[@]}"; do
  for r in $(seq 0 $((ROUNDS - 1))); do
    over_budget && stop "B.4 over the 40h budget during evaluation" 1
    step=$(printf "%05d" $(( (r + 1) * STEPS_PER_ROUND )))
    dir="$OUT/evaluations/$arm/step-$step"
    if [ ! -f "$dir/cycle.json" ]; then
      log "draw $arm round $r (step $step)"
      attempt "generate.$arm.$r" "$SHOWO" scripts/v4_train.py generate \
        --outdir "$OUT" --config "$CFG" --runs "${RUNS[@]}" \
        --arm "$arm" --round "$r" --device "$DEV"
    fi
    log "adjudicate $arm round $r"
    adjudicate "$dir"
  done
done

# --- score and decide -----------------------------------------------------
attempt score  "$CORE" scripts/v4_train.py score  --outdir "$OUT"
attempt report "$CORE" scripts/v4_train.py report --outdir "$OUT"

"$CORE" scripts/v4_l3_verdict.py --outdir "$OUT" --config "$CFG" | tee "$OUT/verdict.txt"
verdict=$(sed -n 1p "$OUT/verdict.txt")
stop "$verdict"
