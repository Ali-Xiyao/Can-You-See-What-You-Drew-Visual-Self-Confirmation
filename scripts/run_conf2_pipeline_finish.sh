#!/usr/bin/env bash
# Finish batch 3 from the observe stage on.
#
# Everything up to `questions` is already on disk and cost about two and a half
# hours of GPU time: crop, verify, plan, both checks, accept, questions. The
# full pipeline has no per-stage resume, so re-running it would redo all of
# that to reach the one stage that failed -- observe was launched with the
# detector environment, which has no diffusers, and died on the import.
#
# The name matters: the queued Gate B report checks liveness by matching
# '*run_conf2_pipeline*' in the process table, and this file is what is running
# in its place, so it has to match that pattern too.
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/set_h_env.sh >/dev/null 2>&1
PYTHONPATH="$(cygpath -w "$(pwd)/src")"
export PYTHONPATH

OUT=runs/v4/tierb-conf2
RUNS=(runs/v4/conf2-2plus1 runs/v4/conf2-1plus1plus1)
CORE=envs/core/python.exe
SHOWO=envs/showo2/python.exe
DEV=cuda:0

step () { echo "=== $(date +%H:%M:%S) $* ==="; }
die  () { echo "!!! $(date +%H:%M:%S) $* -- stopping"; exit 1; }

for f in plan.jsonl accepted.jsonl questions.jsonl; do
  [ -s "$OUT/$f" ] || die "$OUT/$f is missing; run the full pipeline instead"
done

# image_only writes answers.jsonl, prompted writes answers.prompted.jsonl
for condition in image_only prompted; do
  case "$condition" in
    image_only) target="$OUT/answers.jsonl" ;;
    prompted)   target="$OUT/answers.prompted.jsonl" ;;
  esac
  if [ -s "$target" ]; then
    step "observe $condition already done ($(wc -l < "$target") rows)"
    continue
  fi
  step "observe $condition"
  "$SHOWO" scripts/v4_tierb_build.py observe --outdir "$OUT" \
    --condition "$condition" --device "$DEV" \
    >> "$OUT/observe.$condition.log" 2>&1 || die "observe $condition failed"
done

step "section 27 confirmatory fork"
"$CORE" scripts/v4_conf_fork.py --outdir "$OUT" --new "${RUNS[@]}" \
  | tee "$OUT/conf_fork.txt"

step "done"
