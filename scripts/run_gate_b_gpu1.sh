#!/usr/bin/env bash
# Run the Gate B probe end to end on GPU1, for whatever probe set is frozen in
# $OUT/pools.jsonl.
#
# GPU1 is the gen3 x4 card: 7.5x slower than GPU0 on per-image inference, only
# 7% slower on generation. Detect therefore stays on GPU0 and this stays here --
# the gradient stage is forward+backward on a 1.5B backbone, which is the
# compute-bound regime GPU1 handles fine, and the observation stages use small
# models on a probe set two orders of magnitude smaller than the corpus.
#
# Both observe stages resume on (prompt_id, candidate_id), so enlarging the
# probe set re-observes only the candidates that are new.
set -u

cd "$(dirname "$0")/.." || exit 1
source scripts/set_h_env.sh >/dev/null 2>&1
PYTHONPATH="$(cygpath -w "$(pwd)/src")"
export PYTHONPATH

OUT=runs/v4/gate-b
PROBE=scripts/v4_gate_b_probe.py

step () {
  echo "=== $(date +%H:%M:%S) $1 ==="
}

step "rfo observations"
envs/observer/python.exe "$PROBE" observe --outdir "$OUT" --arm rfo --device cuda:1 \
  >> "$OUT/observe.rfo.log" 2>&1 || { echo "rfo observe failed"; exit 1; }

step "naive observations"
envs/showo2/python.exe "$PROBE" observe --outdir "$OUT" --arm naive --device cuda:1 \
  >> "$OUT/observe.naive.log" 2>&1 || { echo "naive observe failed"; exit 1; }

step "select"
envs/core/python.exe "$PROBE" select --outdir "$OUT" || exit 1

step "gradients"
envs/showo2/python.exe "$PROBE" gradients --outdir "$OUT" --device cuda:1 \
  >> "$OUT/gradients.log" 2>&1 || { echo "gradients failed"; exit 1; }

step "report"
envs/core/python.exe "$PROBE" report --outdir "$OUT"

step "Gate B probe done"
