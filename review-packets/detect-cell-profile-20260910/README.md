# The within-cell profile of a detect stage, recovered live

`detections.*.jsonl` rows carry only `detections`, `image_path` and `reply` --
no timestamp.  Once a detect stage ends, the only recoverable wall clock is the
file's `ctime -> mtime` span, so anything that happens *inside* the cell is
invisible.  The file's mtime advances as rows are appended, so sampling
`(mtime, row count)` while the stage runs recovers the trajectory that the
finished file cannot give.

`internvl64_profile.csv` is one such recording:
`rfo_gold.step-00064.detect.internvl`, 2026-09-10 22:52 -> 23:48, sampled every
60 s.  57 usable segments, row increments 3-7 per minute (smooth, so the
profile is not an artifact of batched flushing).

What it shows, and why it matters, is EXECUTION.md section 0.33:

- the cell is strongly non-stationary -- second half vs first is -17.5% in
  s/img and -36.2% in s/100chars, while reply length *rises* 52%;
- one cell's first-half-vs-second-half ratio (1.568x in s/100chars) is the
  same size as the whole nine-cell column's range (1.592x).

So the comparability of cell totals across checkpoints rests on every cell
having the same internal trajectory -- an assumption never checked, and
checkable only while a cell runs.

`sample_internvl.py` is the recorder.  Read-only with respect to the run: one
stat and one line count on a <=256-row file per minute, writing only to a path
given on the command line.
