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
import random
import statistics
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


# ---- the 0.36 pre-registration, written 2026-09-11 04:1x, before the gold
# arm's two step-72 cells existed.  Do not edit these five constants to fit a
# result; that is the 0.32 failure, where an if/elif order decided the verdict.
MIN_SEGMENTS = 10        # a cell with fewer is excluded, not down-weighted
EXPECTED_ROWS = 256      # a cell counts only once it is FULL and sampled to the end;
                         # a partial cell passes MIN_SEGMENTS easily and would be the
                         # head-read-as-cell error 0.33 was written about
MIN_CELLS = 4            # the test refuses to run on a partial step
SIM_DRAWS = 20000
ZONE_DIFFERENT = 0.01    # p <  0.01          -> shapes differ
ZONE_AMBIGUOUS = 0.10    # 0.01 <= p <  0.10  -> ambiguous
                         # 0.10 <= p          -> indistinguishable
# The three zones are [0, 0.01), [0.01, 0.10), [0.10, 1].  Mutually exclusive
# and exhaustive: no value belongs to two of them, which is the defect 0.33
# registered against 0.32's own boundaries.


def compare_shapes(results, seed: int = 20260911):
    """Do the cells of one step share an internal trajectory shape?

    Statistic: R_i = (first-half s/100chars) / (second-half s/100chars) per
    cell, rows-balanced split -- the same estimator and the same unit that 0.35
    registered for step-64 (R = 1.331).  Each cell's standard error is the sd of
    its own shuffled-ordering null.

    That last step carries an assumption worth naming: the shuffle null is
    centred on 1.0 because it destroys the trajectory, so using its sd as the
    error on R assumes the noise scale does not grow with R.  If a future cell
    shows a much larger R than step-64's, that assumption should be rechecked
    rather than relied on.

    Null: every cell has the same true R0 (inverse-variance weighted mean of the
    observed R_i).  Draw R_i* ~ N(R0, sd_i), take the largest pairwise
    z = |R_i* - R_j*| / sqrt(sd_i^2 + sd_j^2), and ask how often it reaches the
    observed one.  Taking the max over all six pairs is why the null has to be
    simulated rather than read off a table.
    """
    import math

    cells = {}
    for key, r in results.items():
        if not r["null"] or not r["first"] or not r["second"]:
            continue
        if r["first"][3] + r["second"][3] < MIN_SEGMENTS:
            continue
        if r["total"] < EXPECTED_ROWS or r["hi"] < EXPECTED_ROWS:
            continue
        med, sd, _, _, _, obs, _, _ = r["null"]
        cells[f"{key[1]}.{key[2]}"] = (obs, sd)
    if len(cells) < MIN_CELLS:
        print()
        print(f"0.36 shape test REFUSED: {len(cells)} qualifying cell(s), needs {MIN_CELLS}.")
        print(f"  (a cell qualifies at {EXPECTED_ROWS} rows sampled to the end, "
              f"with >= {MIN_SEGMENTS} segments)")
        print("  The test is pre-registered over a complete step.  Running it on a")
        print("  partial one is the head-read-as-cell error 0.33 was written about.")
        return

    names = sorted(cells)
    R = [cells[n][0] for n in names]
    SD = [cells[n][1] for n in names]
    w = [1.0 / s ** 2 for s in SD]
    R0 = sum(x * y for x, y in zip(R, w)) / sum(w)

    def max_z(vals):
        return max(abs(vals[i] - vals[j]) / math.sqrt(SD[i] ** 2 + SD[j] ** 2)
                   for i in range(len(vals)) for j in range(i + 1, len(vals)))

    obs_z = max_z(R)
    rng = random.Random(seed)
    hits = 0
    for _ in range(SIM_DRAWS):
        if max_z([rng.gauss(R0, s) for s in SD]) >= obs_z:
            hits += 1
    pv = hits / SIM_DRAWS

    if pv < ZONE_DIFFERENT:
        verdict = "SHAPES DIFFER"
        meaning = ("cell totals in this column are not comparable without a shape "
                   "correction; 0.27's nine-cell column has to be re-read")
    elif pv < ZONE_AMBIGUOUS:
        verdict = "AMBIGUOUS"
        meaning = "undetermined -- neither side, and it does not become one by waiting"
    else:
        verdict = "INDISTINGUISHABLE"
        meaning = ("no shape difference at this power; cell totals stay comparable "
                   "on this evidence, which is not the same as shown equal")

    print()
    print(f"0.36 four-cell shape test ({SIM_DRAWS} draws, seed {seed})")
    for n, r_, s_ in zip(names, R, SD):
        print(f"  {n:<22} R = {r_:.3f} +/- {s_:.3f}")
    print(f"  weighted common R0     {R0:.3f}")
    print(f"  largest pairwise z     {obs_z:.2f}")
    print(f"  p(null >= observed)    {pv:.4f}")
    print(f"  zones  p<{ZONE_DIFFERENT}: differ | "
          f"{ZONE_DIFFERENT}<=p<{ZONE_AMBIGUOUS}: ambiguous | p>={ZONE_AMBIGUOUS}: indistinguishable")
    print(f"  VERDICT  {verdict}")
    print(f"           {meaning}")


def null_ratio(segs, lens, draws: int, seed: int = 20260911):
    """The half/half ratio with the time ordering destroyed and nothing else.

    Each segment keeps its own (dt, rows, chars); only their order is shuffled,
    then the same rows-balanced split is applied.  This is the control 0.35 SS4
    registered: treatment and control are the same data, and differ in exactly
    the one variable a trajectory claim is about.  Returns
    (median, sd, lo95, hi95, max, observed, sd_above_median, p).
    """
    triples = [(dt, hi - lo, sum(lens[lo:hi])) for lo, hi, dt in segs]

    def ratio(order):
        half = sum(x[1] for x in order) / 2
        acc, cut = 0.0, 0
        for i, x in enumerate(order):
            if acc >= half:
                break
            acc += x[1]
            cut = i + 1
        a, b = order[:cut], order[cut:]
        if not a or not b or not sum(x[2] for x in a) or not sum(x[2] for x in b):
            return None
        ra = sum(x[0] for x in a) / (sum(x[2] for x in a) / 100.0)
        rb = sum(x[0] for x in b) / (sum(x[2] for x in b) / 100.0)
        return ra / rb

    obs = ratio(triples)
    if obs is None:
        return None
    rng = random.Random(seed)
    draw = []
    for _ in range(draws):
        o = triples[:]
        rng.shuffle(o)
        r = ratio(o)
        if r is not None:
            draw.append(r)
    if not draw:
        return None
    draw.sort()
    n = len(draw)
    med, sd = statistics.median(draw), statistics.pstdev(draw)
    # two-sided: how often does a shuffle depart from 1.0 by at least as much?
    pv = sum(1 for x in draw if abs(x - 1.0) >= abs(obs - 1.0)) / n
    return (med, sd, draw[int(0.025 * n)], draw[int(0.975 * n)], draw[-1],
            obs, (obs - med) / sd if sd else float("nan"), pv)


def fmt(r):
    return "   (none)" if r is None else f"{r[0]:6.2f} s/img  {r[1]:6.3f} s/100ch  n={r[2]:3d} seg={r[3]:2d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--step", default=None, help="override; else read from the csv")
    ap.add_argument("--arm", default=None, help="for single-cell csvs with no arm column")
    ap.add_argument("--det", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="the 0.36 pre-registered four-cell shape test; refuses to run "
                         "on fewer than MIN_CELLS qualifying cells, so it cannot be "
                         "peeked at while the run is still filling them in")
    ap.add_argument("--perm", type=int, default=0,
                    help="shuffled-ordering null for the half/half ratio, N draws "
                         "(0.35 SS4 used 5000); off by default so the plain output is stable")
    args = ap.parse_args()
    if args.compare and not args.perm:
        args.perm = 5000  # --compare needs each cell's own bar; do not make me remember

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
                            lo=lo, hi=hi, total=len(lens),
                            null=null_ratio(segs, lens, args.perm) if args.perm else None)
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

    if any(r["null"] for r in results.values()):
        print()
        print(f"shuffled-ordering null for the half/half ratio ({args.perm} draws)")
        print(f"{'cell':<22}{'observed':>10}{'null med':>10}{'sd':>8}"
              f"{'95% of null':>18}{'null max':>10}{'sd above':>10}{'p':>9}")
        for key, r in results.items():
            if not r["null"]:
                continue
            med, sd, lo95, hi95, mx, obs, z, pv = r["null"]
            print(f"{key[1] + '.' + key[2]:<22}{obs:10.3f}{med:10.3f}{sd:8.4f}"
                  f"   [{lo95:6.3f}, {hi95:6.3f}]{mx:10.3f}{z:10.1f}{pv:9.4f}")

    if args.compare:
        compare_shapes(results)

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
