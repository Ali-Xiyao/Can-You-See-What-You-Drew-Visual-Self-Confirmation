#!/usr/bin/env bash
# Launch one bank-probe shard per GPU, refusing to double-launch.
#
# Two processes writing the same output dir race on os.replace and die with
# WinError 5, so this checks for a live process on each output dir first.
# Packets are cached, so re-running after an interruption resumes.
#
# usage: bash scripts/launch_bank_shards.sh <records.jsonl> <output-root> <total-prompts>
set -euo pipefail

RECORDS="$1"; OUT_ROOT="$2"; TOTAL="$3"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/set_h_env.sh

HALF=$(( (TOTAL + 1) / 2 ))
mkdir -p "${OUT_ROOT}"

running_on() {
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | Where-Object { \$_.CommandLine -like '*$1*' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r\n '
}

for shard in 0 1; do
  out="${OUT_ROOT}/gpu${shard}"
  if [ "$(running_on "gpu${shard}")" != "0" ]; then
    echo "gpu${shard}: already running, skipping"
    continue
  fi
  if [ -f "${out}/gate_a.json" ]; then
    echo "gpu${shard}: already complete, skipping"
    continue
  fi
  offset=$(( shard * HALF ))
  nohup ./envs/showo2/python.exe scripts/run_v3_bank_probe.py \
    --records "${RECORDS}" --output "${out}" --device "cuda:${shard}" \
    --offset "${offset}" --limit "${HALF}" --rate-only \
    >> "${OUT_ROOT}/gpu${shard}.log" 2>&1 &
  disown
  echo "gpu${shard}: launched offset=${offset} limit=${HALF}"
done
