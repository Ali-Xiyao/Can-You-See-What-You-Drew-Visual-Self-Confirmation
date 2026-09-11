"""Sample (mtime, row count) while rfo_gold.step-00064.detect.internvl runs.

Detect rows carry no timestamp, so once the stage ends only the total span is
recoverable and a load episode inside the window is invisible (0.32 section 4).
The file's mtime advances as rows are appended, so sampling it live recovers
the within-cell profile that the finished file cannot give.

Read-only with respect to the run; writes one CSV under the scratchpad.
Costs one stat + one line count on a <=256-row file per minute.
"""
import csv, os, sys, time
from pathlib import Path

SRC = Path(r"H:\Xiyao_Wang\062_Can You See What You Drew Visual Self-Confirmation"
           r"\runs\v4\decoupling-main-20260908\evaluations\rfo_gold\step-00064"
           r"\detections.internvl.jsonl")
OUT = Path(sys.argv[1])
DEADLINE = time.time() + 110 * 60

with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["sampled_at", "mtime", "rows", "bytes"])
    last = -1
    while time.time() < DEADLINE:
        if SRC.exists():
            st = SRC.stat()
            try:
                rows = sum(1 for _ in SRC.open("rb"))
            except OSError:
                rows = last
            w.writerow([f"{time.time():.3f}", f"{st.st_mtime:.3f}", rows, st.st_size])
            fh.flush()
            if rows >= 256 and rows == last:
                print(f"complete at {rows} rows")
                break
            last = rows
        time.sleep(60)
    else:
        print("deadline reached before 256 rows")
