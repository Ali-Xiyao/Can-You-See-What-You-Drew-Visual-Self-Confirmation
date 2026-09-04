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

# Sourcing the env was missing here. Every other stage of this chain gets it
# from its launcher; this waiter is launched on its own, so without these two
# lines the report dies on an import before it reads a single byte.
source scripts/set_h_env.sh
export PYTHONPATH="$(cygpath -w "$(pwd)/src")"

SENTINEL="runs/v4/tierb-conf2/conf_fork.txt"
DONE="runs/v4/gate-b/REPORT_DONE"

# 36h, not the 12h this started with. That budget was sized against an estimate
# that batch-3 would finish by 04:15; the pipeline died of a parse error at
# 04:13 and sat dead for seven hours, which ate the whole margin and expired
# this waiter at 12:46 while the restarted pipeline was running fine. A wall
# clock is the wrong instrument for "is what I depend on still alive", so the
# liveness check below is what should normally end this early.
DEADLINE=$(( $(date +%s) + 36 * 3600 ))

pipeline_alive () {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*run_conf2_pipeline*' }).Count" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | head -1
}

# Every exit from this script must write the sentinel the L3 preview waits on.
# The first version wrote it only after the report ran, so when the liveness
# check below fired early -- it did, at 14:48:11, nine minutes before
# conf_fork.txt appeared -- the waiter exited and the L3 orchestrator sat
# waiting on a file nobody would ever write. Blocking a downstream stage
# forever is a worse failure than starting it with a stale Gate B number.
give_up () {
  echo "$(date +%H:%M:%S) $1" >&2
  echo "1 $(date +%FT%T) gave-up: $1" > "$DONE"
  exit 1
}

# The liveness check has a real blind spot: the pipeline's last act is a python
# analysis step that does not match *run_conf2_pipeline*, so between the shell
# exiting and conf_fork.txt landing there is a window where nothing is alive by
# this definition and the sentinel does not exist yet. Requiring the condition
# to hold twice, two minutes apart, covers a window of that shape. It does not
# make the check correct in general -- a longer trailing step would still fool
# it -- but the give_up path above now degrades safely instead of deadlocking.
misses=0
while [ ! -f "$SENTINEL" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    give_up "gave up waiting for $SENTINEL after 36h"
  fi
  alive=$(pipeline_alive); alive=${alive:-0}
  if [ "$alive" -eq 0 ]; then
    misses=$(( misses + 1 ))
    if [ "$misses" -ge 2 ]; then
      give_up "the batch-3 pipeline is gone and $SENTINEL was never written"
    fi
  else
    misses=0
  fi
  sleep 120
done

echo "$(date +%H:%M:%S) batch-3 finished, running the Gate B report at n=232"
envs/core/python.exe scripts/v4_gate_b_probe.py report --outdir runs/v4/gate-b
status=$?
# The sentinel is written whether the report succeeded or not, and carries the
# exit code. The L3 preview waits on this file, and a failed report should let
# it start rather than block it forever -- the report is a read over gradients
# that are already on disk, so it can be re-run at any time.
echo "$status $(date +%FT%T)" > "$DONE"
exit "$status"
