#!/usr/bin/env bash
# Finish the Gate B probe on GPU1 once the naive arm is done.
#
# GPU1 is the gen3 x4 card, 7.5x slower on per-image inference, so detect stays
# on GPU0. The gradient stage is forward+backward on a 1.5B backbone -- compute
# bound, the regime where generation measured GPU1 at only 7% behind -- so this
# is the work that belongs here.
#
# select and report are CPU-only and run from envs/core, which is the env that
# has numpy/scipy; the GPU stages run from envs/showo2.
set -u

cd "$(dirname "$0")/.." || exit 1
source scripts/set_h_env.sh >/dev/null 2>&1
PYTHONPATH="$(cygpath -w "$(pwd)/src")"
export PYTHONPATH

OUT=runs/v4/gate-b
PROBE=scripts/v4_gate_b_probe.py

# Wait on the output file rather than a process name: both arms run python from
# the project's envs and a name match would catch the wrong one. Give up if the
# count stops moving instead of spinning on a job that died.
wait_for () {
  local file="$1" target="$2" label="$3"
  local last=-1 stalled=0 now
  while :; do
    now=$(wc -l < "$file" 2>/dev/null || echo 0)
    [ "$now" -ge "$target" ] && break
    if [ "$now" -eq "$last" ]; then
      stalled=$((stalled + 1))
      if [ "$stalled" -ge 10 ]; then
        echo "=== $(date +%H:%M:%S) $label stalled at $now/$target, stopping ==="
        exit 1
      fi
    else
      stalled=0
    fi
    last=$now
    sleep 60
  done
  echo "=== $(date +%H:%M:%S) $label finished at $now rows ==="
}

wait_for "$OUT/observations.naive.jsonl" 552 "naive observations"

echo "=== $(date +%H:%M:%S) select ==="
envs/core/python.exe "$PROBE" select --outdir "$OUT" 2>&1 || exit 1

echo "=== $(date +%H:%M:%S) gradients ==="
envs/showo2/python.exe "$PROBE" gradients --outdir "$OUT" --device cuda:1 \
  >> "$OUT/gradients.log" 2>&1
echo "=== $(date +%H:%M:%S) gradients exited $? ==="

echo "=== $(date +%H:%M:%S) report ==="
envs/core/python.exe "$PROBE" report --outdir "$OUT" 2>&1

echo "=== $(date +%H:%M:%S) Gate B probe done ==="
