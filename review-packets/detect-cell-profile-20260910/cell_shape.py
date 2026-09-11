"""Do the four detect cells of one step share the same internal trajectory shape?

That is the question 0.33 SS4 left open: a cell total is comparable to another
cell's total only if the two have the same internal shape, and a finished
detect file cannot answer it (rows carry no timestamp).  Live row-count
sampling can.

Method.  Each pair of consecutive samples is one segment: dt from the file's
own mtime (not the poll time, which carries polling jitter), drows from the
row count, and the output volume from summing `reply` lengths over exactly
the rows that landed in that interval.  Within a cell the rate is reported in
s per 100 output chars, per the 0.27 unit rule; s/img is printed alongside but
is the between-cell unit, not the within-cell one.

The judge.  For each cell, first-half rate vs second-half rate, split at the
midpoint of the OBSERVED row range.  The null is not "no difference" -- it is
the same segments with their time ordering destroyed: odd-indexed segments vs
even-indexed ones.  That control is the same data, same cell, same segment
lengths, and differs from the treatment in exactly one thing, the ordering
that a trajectory claim is about.  If the odd/even spread is as large as the
half/half spread, there is no trajectory here, only segment noise.

Provenance.  Regression-tested against the one published trajectory before
being allowed to produce new ones: EXECUTION.md 0.33 SS3.  The whole cell
reproduced (13.24 s/img, the same value 0.33 settled on), the first half did
not.  0.33's own three first-half numbers turned out to be mutually
inconsistent -- 14.65 s/img over 202 chars/img is 7.25 s/100chars, not the
8.337 it registered, a 15.0% error; the whole-cell row checks out to 0.0%.
The remeasurement and the error bar are EXECUTION.md 0.35.  0.33 is left
byte-identical, as is every file that was already in this packet.

Usage:  python cell_shape.py <profile.csv> [--step 00072]
        python cell_shape.py <profile.csv> --step 00064 --arm rfo_gold --det internvl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUN = Path(r"H:\Xiyao_Wang\062_Can You See What You Drew Visual Self-Confirmation"
           r"\runs\v4\decoupling-main-20260908")


def reply_lengths(step: str, arm: str, det: str) -> list[int]:
    p = RUN / "evaluations" / arm / f"step-{step}" / f"detections.{det}.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(len(json.loads(line).get("reply", "")))
    return out


def segments(samples):
    """[(row_lo, row_hi, dt)] for consecutive samples that advanced."""
    segs = []
    for (m0, r0), (m1, r1) in zip(samples, samples[1:]):
        if r1 > r0 and m1 > m0:
            segs.append((r0, r1, m1 - m0))
    return segs


def rate(segs, lens):
    """(s/img, s/100chars, n_img, n_seg) aggregated over a set of segments."""
    t = sum(s[2] for s in segs)
    n = sum(s[1] - s[0] for s in segs)
    c = sum(sum(lens[s[0]:s[1]]) for s in segs)
    if not n or not c:
        return None
    return t / n, t / (c / 100.0), n, len(segs)


def fmt(r):
    return "   (none)" if r is None else f"{r[0]:6.2f} s/img  {r[1]:6.3f} s/100ch  n={r[2]:3d} seg={r[3]:2d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--step", default=None, help="override; else read from the csv")
    ap.add_argument("--arm", default=None, help="for single-cell csvs with no arm column")
    ap.add_argument("--det", default=None)
    args = ap.parse_args()

    rows = Path(args.profile).read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    cells: dict[tuple[str, str, str], list] = {}
    for line in rows[1:]:
        f = line.split(",")
        if len(f) != len(header):
            continue
        if "arm" in header:
            step, arm, det = f[1], f[2], f[3]
            mtime, nrows = float(f[4]), int(f[5])
        else:  # the single-cell recorder: sampled_at,mtime,rows,bytes
            step, arm, det = args.step, args.arm, args.det
            mtime, nrows = float(f[1]), int(f[2])
        cells.setdefault((step, arm, det), []).append((mtime, nrows))

    print(f"{'cell':<22}{'coverage':<22}{'whole':<38}")
    results = {}
    for key in sorted(cells):
        step, arm, det = key
        samples = sorted(cells[key])
        lens = reply_lengths(step, arm, det)
        if not lens:
            print(f"{arm}.{det:<12}  detect file not readable yet -- skipped")
            continue
        segs = [s for s in segments(samples) if s[1] <= len(lens)]
        if len(segs) < 6:
            print(f"{arm}.{det:<12}  only {len(segs)} usable segments -- skipped")
            continue
        lo, hi = segs[0][0], segs[-1][1]
        mid = (lo + hi) // 2
        first = [s for s in segs if s[1] <= mid]
        second = [s for s in segs if s[0] >= mid]
        odd = segs[1::2]
        even = segs[0::2]
        whole = rate(segs, lens)
        results[key] = dict(first=rate(first, lens), second=rate(second, lens),
                            odd=rate(odd, lens), even=rate(even, lens), whole=whole,
                            lo=lo, hi=hi, total=len(lens))
        print(f"{arm+'.'+det:<22}rows {lo:3d}-{hi:3d} of {len(lens):3d}   {fmt(whole)}")

    print("\nwithin-cell trajectory, in s/100chars (0.27 unit rule)")
    print(f"{'cell':<22}{'1st half':>10}{'2nd half':>10}{'change':>9}   "
          f"{'odd':>9}{'even':>9}{'control':>9}")
    for key, r in results.items():
        _, arm, det = key
        f1, f2 = r["first"], r["second"]
        o, e = r["odd"], r["even"]
        if not (f1 and f2 and o and e):
            continue
        traj = (f2[1] - f1[1]) / f1[1] * 100
        ctrl = (e[1] - o[1]) / o[1] * 100
        print(f"{arm+'.'+det:<22}{f1[1]:10.3f}{f2[1]:10.3f}{traj:+8.1f}%   "
              f"{o[1]:9.3f}{e[1]:9.3f}{ctrl:+8.1f}%")

    if len(results) >= 2:
        print("\nbetween-cell, in s/img (the between-cell unit)")
        vals = {f"{k[1]}.{k[2]}": r["whole"][0] for k, r in results.items()}
        lo_k = min(vals, key=vals.get)
        hi_k = max(vals, key=vals.get)
        print(f"  slowest {hi_k} {vals[hi_k]:.2f}  /  fastest {lo_k} {vals[lo_k]:.2f}"
              f"  = {vals[hi_k]/vals[lo_k]:.3f}x")
        trajs = {f"{k[1]}.{k[2]}": (r['second'][1] - r['first'][1]) / r['first'][1] * 100
                 for k, r in results.items() if r["first"] and r["second"]}
        if len(trajs) >= 2:
            lo_t, hi_t = min(trajs.values()), max(trajs.values())
            print(f"  trajectory change spans {lo_t:+.1f}% to {hi_t:+.1f}%"
                  f"  (spread {hi_t - lo_t:.1f} points)")


if __name__ == "__main__":
    main()
