"""Record the within-cell trajectory of every detect stage at one checkpoint.

Detect rows carry no timestamp, so once a stage ends the only recoverable wall
clock is the file's ctime -> mtime span and anything inside the cell is gone
(EXECUTION.md 0.32 section 4).  The file's mtime advances as rows are appended,
so sampling (mtime, row count) live recovers the trajectory.

0.33 section 4 is why this now runs on both arms: one cell's first-half against
second-half ratio was the size of the whole nine-cell column's range, so cell
totals are comparable only if every cell has the same internal trajectory --
an assumption that can only be checked while the cells run.

Read-only with respect to the run: one stat and one line count per file per
interval, writing only to the CSV named on the command line.  Appends, so a
relaunch does not lose what was already recorded.

    python sample_detect_cells.py --step 00072 --out profile-72.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

RUN = Path(r"H:\Xiyao_Wang\062_Can You See What You Drew Visual Self-Confirmation"
           r"\runs\v4\decoupling-main-20260908")
CELLS = [(arm, det) for arm in ("naive", "rfo_gold")
         for det in ("qwen3vl", "internvl")]
ROWS_PER_CELL = 256


def path_for(step: str, arm: str, det: str) -> Path:
    return RUN / "evaluations" / arm / f"step-{step}" / f"detections.{det}.jsonl"


def count(p: Path) -> int:
    try:
        with p.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--deadline-hours", type=float, default=9.0)
    args = ap.parse_args()

    out = Path(args.out)
    fresh = not out.exists()
    deadline = time.time() + args.deadline_hours * 3600
    done: set[tuple[str, str]] = set()
    last: dict[tuple[str, str], int] = {}

    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(["sampled_at", "step", "arm", "detector", "mtime", "rows", "bytes"])
        while time.time() < deadline and len(done) < len(CELLS):
            for arm, det in CELLS:
                if (arm, det) in done:
                    continue
                p = path_for(args.step, arm, det)
                if not p.exists():
                    continue
                st = p.stat()
                rows = count(p)
                w.writerow([f"{time.time():.3f}", args.step, arm, det,
                            f"{st.st_mtime:.3f}", rows, st.st_size])
                # a cell is finished only after a full row count is seen twice,
                # so a sample that lands mid-write does not close it early
                if rows >= ROWS_PER_CELL and last.get((arm, det)) == rows:
                    done.add((arm, det))
                    print(f"complete {arm}.{det} at {rows} rows", flush=True)
                last[(arm, det)] = rows
            fh.flush()
            time.sleep(args.interval)
    print(f"finished: {len(done)}/{len(CELLS)} cells complete", flush=True)


if __name__ == "__main__":
    main()
