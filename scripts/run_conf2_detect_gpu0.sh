#!/usr/bin/env bash
# Run the third batch's remaining detect jobs back-to-back on GPU0.
#
# Everything here is on GPU0 on purpose. STATUS.md records the measurement:
# GPU1 is on PCIe gen3 x4 and per-image detection there is 7.5x slower because
# it waits on data rather than computing. Generation did not care (many steps
# per image, compute bound); a single forward pass per image cares a great deal.
#
# No --overwrite anywhere. The detect stage resumes on image_path and appends,
# and one --overwrite on the wrong line would erase a detector's finished work.
set -u

cd "$(dirname "$0")/.." || exit 1
source scripts/set_h_env.sh >/dev/null 2>&1
PYTHONPATH="$(cygpath -w "$(pwd)/src")"
export PYTHONPATH
PY=envs/observer/python.exe

# Wait for the qwen3vl job already on GPU0 to finish its 1638 images. Waiting on
# a process name would be wrong: the Gate B probe runs from the same interpreter
# on GPU1, so this watches the output file instead, and gives up if it stalls
# rather than spinning forever on a job that died.
target=1638
out=runs/v4/conf2-2plus1/detections.qwen3vl.jsonl
stalled=0
last=-1
while :; do
  now=$(wc -l < "$out" 2>/dev/null || echo 0)
  [ "$now" -ge "$target" ] && break
  if [ "$now" -eq "$last" ]; then
    stalled=$((stalled + 1))
    if [ "$stalled" -ge 10 ]; then
      echo "=== $(date +%H:%M:%S) conf2-2plus1 qwen3vl stalled at $now/$target, stopping ==="
      exit 1
    fi
  else
    stalled=0
  fi
  last=$now
  sleep 60
done
echo "=== $(date +%H:%M:%S) conf2-2plus1 qwen3vl finished at $(wc -l < "$out") rows ==="

run () {
  local dir="$1" detector="$2"
  echo "=== $(date +%H:%M:%S) $detector on $dir ==="
  "$PY" scripts/v4_run_pipeline.py detect \
    --manifest "runs/v4/$dir/manifest.jsonl" \
    --detector "$detector" --device cuda:0 \
    >> "runs/v4/$dir/detect.$detector.log" 2>&1
  echo "=== $(date +%H:%M:%S) $detector on $dir exited $? ==="
}

run conf2-1plus1plus1 qwen3vl

# InternVL3_5-8B is the 16 GB one whose cold read pattern crawls at ~5 MB/s.
# Priming costs 40 s and saves ~35 min of a load that looks exactly like a hang.
echo "=== $(date +%H:%M:%S) priming internvl page cache ==="
cat "H:/Xiyao_Wang/001_models/InternVL3_5-8B"/model-*.safetensors > /dev/null

run conf2-2plus1 internvl
run conf2-1plus1plus1 internvl

echo "=== $(date +%H:%M:%S) all detect jobs done ==="
wc -l runs/v4/conf2-2plus1/detections.*.jsonl runs/v4/conf2-1plus1plus1/detections.*.jsonl
