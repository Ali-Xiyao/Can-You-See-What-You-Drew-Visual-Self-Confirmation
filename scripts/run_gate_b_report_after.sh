#!/usr/bin/env bash
# Re-run the Gate B report once the GPU work is out of the way.
#
# The report walks three 17 GB gradient memmaps -- 51 GB of near-random reads.
# Run concurrently with a detect job it seek-starves both down to under
# 2 MB/s (measured 2026-09-03: 0.8 MB/s, GPU idle for fifty minutes). So it
# waits for the batch-3 pipeline's terminal artefact instead of racing it.
# No data is at risk either way: the gradients are already on disk and the
# report is a pure read.
set -u
cd "$(dirname "$0")/.."

SENTINEL="runs/v4/tierb-conf2/conf_fork.txt"
DEADLINE=$(( $(date +%s) + 12 * 3600 ))

while [ ! -f "$SENTINEL" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "$(date +%H:%M:%S) gave up waiting for $SENTINEL after 12h" >&2
    exit 1
  fi
  sleep 120
done

echo "$(date +%H:%M:%S) batch-3 finished, running the Gate B report at n=232"
exec envs/core/python.exe scripts/v4_gate_b_probe.py report --outdir runs/v4/gate-b
