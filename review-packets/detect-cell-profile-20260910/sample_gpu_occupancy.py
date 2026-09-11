"""Record who is on the two GPUs, once a minute, so the next episode is attributable.

On 2026-09-11 the gold arm's step-72 qwen3vl cell ran 5x its own baseline for
forty minutes (04:30-05:10).  By the time the spike was noticed and nvidia-smi
was run, the only foreign jobs on the cards had started at 06:51 -- two hours
too late to be the cause.  Whatever was there at 04:50 had exited, and nothing
on this machine keeps a history of it.  That is not a fact about the episode,
it is a missing instrument; see EXECUTION.md 0.37.

So: one nvidia-smi pair per minute, appended to a CSV.  Command lines are
resolved once per new PID and cached, so the per-tick cost is two nvidia-smi
calls.  Read-only with respect to everything -- it never touches the run
directory and never signals a process, including the foreign ones.

    python sample_gpu_occupancy.py --out gpu-occupancy.csv --deadline-hours 26
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def smi(query: str, kind: str) -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{kind}={query}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def cmdline(pid: str, cache: dict[str, str]) -> str:
    if pid in cache:
        return cache[pid]
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\"; "
             "if ($p) { $p.CommandLine }"],
            capture_output=True, text=True, timeout=30)
        c = " ".join(out.stdout.split())[:400]
    except Exception:
        c = ""
    cache[pid] = c or "(gone before it could be resolved)"
    return cache[pid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--deadline-hours", type=float, default=26.0)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.exists():
        out.write_text("sampled_at,kind,a,b,c,d\n", encoding="utf-8", newline="\n")
    cache: dict[str, str] = {}
    stop = time.time() + args.deadline_hours * 3600

    while time.time() < stop:
        now = time.time()
        lines = []
        for row in smi("index,utilization.gpu,memory.used,temperature.gpu", "gpu"):
            idx, util, mem, temp = [x.strip() for x in row.split(",")]
            lines.append(f"{now:.3f},gpu,{idx},{util},{mem},{temp}")
        for row in smi("gpu_uuid,pid,used_gpu_memory", "compute-apps"):
            parts = [x.strip() for x in row.split(",")]
            uuid, pid = parts[0], parts[1]
            mem = parts[2] if len(parts) > 2 else ""
            c = cmdline(pid, cache).replace(",", ";")
            lines.append(f"{now:.3f},proc,{uuid[-8:]},{pid},{mem},{c}")
        with out.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        time.sleep(max(1.0, args.interval - (time.time() - now)))


if __name__ == "__main__":
    main()
