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
# GPU1 is fair game only when it is idle: there are jobs on this machine that
# are not this project's and they are not to be preempted. When it is busy the
# run still works, it just serialises drawing behind adjudication on GPU0 and
# the wall clock becomes their sum rather than the larger (STATUS 32).
pick_devices () {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 2p)
  if [ "${used:-9999}" -lt 500 ]; then
    DEV=cuda:1; LADDER=cuda:0
    log "GPU1 idle (${used} MiB): backbone on $DEV, ladder on $LADDER"
  else
    DEV=cuda:0; LADDER=cuda:0
    log "GPU1 busy (${used} MiB, not ours): everything on $DEV, expect the slower plan"
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
