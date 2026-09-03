#!/usr/bin/env bash
# The third batch's pipeline, from finished detections to the section 27 test.
#
# Section 27 was written before any of these images existed and says the design
# is byte-identical to section 25's; only n and the seed offset move. So this
# script is section 26's sequence with the run directories swapped, and nothing
# in it may be tuned to what the numbers turn out to be. In particular:
#
#   * no --sham. Section 23 already ruled the filler's fingerprint out and the
#     confirmatory run does not pay for it twice.
#   * --edits delete --per-image 3, the section 25 setting.
#   * the two check passes append to one checks.jsonl and must run in sequence,
#     never in parallel on two cards.
#   * labels for the fork come from runs/v4/main-* only. --old is left at its
#     default here so that stays true by not being said.
#
# Everything detector-facing is on GPU0: STATUS line 44 measured GPU1 at 7.5x
# slower per image on PCIe gen3 x4, since a single forward pass per image waits
# on data rather than computing.
#
# Resumable. Every stage either appends or is cheap to redo, and no stage is
# given --overwrite -- one of those on the wrong line erases a finished
# detector pass.
set -u

cd "$(dirname "$0")/.." || exit 1
source scripts/set_h_env.sh >/dev/null 2>&1
PYTHONPATH="$(cygpath -w "$(pwd)/src")"
export PYTHONPATH

OBS=envs/observer/python.exe
CORE=envs/core/python.exe
OUT=runs/v4/tierb-conf2
RUNS=(runs/v4/conf2-2plus1 runs/v4/conf2-1plus1plus1)
DEV=cuda:0

step () { echo "=== $(date +%H:%M:%S) $* ==="; }
die  () { echo "!!! $(date +%H:%M:%S) $* -- stopping"; exit 1; }

# --- wait for the detect chain -------------------------------------------
# Watching the files, not a process name: the Gate B probe runs from the same
# interpreter, and a stalled job should stop this rather than spin forever.
step "waiting for four detection files"
stalled=0
last=""
while :; do
  now=""
  ready=1
  for dir in "${RUNS[@]}"; do
    total=$(wc -l < "$dir/manifest.jsonl")
    for det in qwen3vl internvl; do
      have=$(wc -l < "$dir/detections.$det.jsonl" 2>/dev/null || echo 0)
      now="$now $have"
      [ "$have" -ge "$total" ] || ready=0
    done
  done
  [ "$ready" -eq 1 ] && break
  if [ "$now" = "$last" ]; then
    stalled=$((stalled + 1))
    # 45 minutes: InternVL has not started writing yet, and priming 16 GB of
    # safetensors plus the load can sit silent for well over twenty on a busy
    # disk. A premature exit here would look exactly like a crashed detector.
    [ "$stalled" -ge 45 ] && die "detections stalled at$now"
  else
    stalled=0
  fi
  last="$now"
  sleep 60
done
step "detections complete:$last"

# --- ladder levels 2 and 3 ------------------------------------------------
for dir in "${RUNS[@]}"; do
  step "crop $dir"
  "$OBS" scripts/v4_run_pipeline.py crop --run "$dir" --device "$DEV" \
    >> "$dir/crop.log" 2>&1 || die "crop failed on $dir"
done

for dir in "${RUNS[@]}"; do
  step "verify $dir"
  "$CORE" scripts/v4_run_pipeline.py verify --run "$dir" || die "verify failed on $dir"
done

# --- Tier B, deletion arm -------------------------------------------------
step "plan"
"$CORE" scripts/v4_tierb_build.py plan \
  --run "${RUNS[0]}" --run "${RUNS[1]}" \
  --outdir "$OUT" --edits delete --per-image 3 || die "plan failed"

for det in qwen3vl internvl; do
  step "check $det"
  "$OBS" scripts/v4_tierb_build.py check --outdir "$OUT" --detector "$det" \
    --device "$DEV" >> "$OUT/check.$det.log" 2>&1 || die "check $det failed"
done

step "accept"
"$CORE" scripts/v4_tierb_build.py accept --outdir "$OUT" || die "accept failed"

step "questions"
"$CORE" scripts/v4_tierb_build.py questions --outdir "$OUT" || die "questions failed"

for condition in image_only prompted; do
  step "observe $condition"
  "$OBS" scripts/v4_tierb_build.py observe --outdir "$OUT" \
    --condition "$condition" --device "$DEV" \
    >> "$OUT/observe.$condition.log" 2>&1 || die "observe $condition failed"
done

# --- the pre-registered test ---------------------------------------------
step "section 27 confirmatory fork"
"$CORE" scripts/v4_conf_fork.py --outdir "$OUT" --new "${RUNS[@]}" \
  | tee "$OUT/conf_fork.txt"

step "done"
