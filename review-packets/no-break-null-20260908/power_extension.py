"""How many checkpoints would a smaller decoupling need?

calibrate.py stopped at 24 checkpoints, where a 0.20 external rise reaches 78%
power and a 0.10 rise reaches 29%. The design question that leaves open is
whether the smaller effect is reachable at all by running more checkpoints, or
whether the external curve simply has to move further. Extrapolating from the
existing table is not licensed, so this measures it.

Same estimator, same null, same thresholds as calibrate.py; only the checkpoint
counts and the rises are different. Gated only -- the ungated arm is already
known to be uninformative.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfsight.analysis.breakpoints import estimate_d_star

OUTCOME_N = 120
BASE_RATE = 0.37
INTERNAL_RISE = 0.004
INTERNAL_SEM = 0.002
SPAN = 175.0
TRIALS = 300
RESAMPLES = 300
P_THRESHOLD = 0.05
CHECKPOINTS = (32, 48)
RISES = (0.0, 0.10, 0.15)
SEED = 20260908


def external_floor(external):
    sems = sorted(math.sqrt(max(0.0, p * (1.0 - p)) / OUTCOME_N) for p in external)
    return sems[len(sems) // 2] / SPAN


def truth(steps, rise):
    knot = SPAN / 2.0
    return np.array([BASE_RATE + rise * min(step, knot) / knot for step in steps])


def one_trial(generator, steps, rise):
    expected = truth(steps, rise)
    external = generator.binomial(OUTCOME_N, np.clip(expected, 0.0, 1.0)) / OUTCOME_N
    internal = -2.0 + INTERNAL_RISE * steps + generator.normal(0.0, INTERNAL_SEM, steps.size)
    estimate = estimate_d_star(
        steps,
        internal,
        external,
        min_coupled_external_slope=external_floor(external),
        max_break_p_value=P_THRESHOLD,
        break_resamples=RESAMPLES,
        break_seed=int(generator.integers(1, 2**31 - 1)),
    )
    return estimate.d_star is not None


def main():
    started = time.time()
    rows = []
    for rise in RISES:
        for n_checkpoints in CHECKPOINTS:
            steps = np.linspace(0.0, SPAN, n_checkpoints)
            generator = np.random.default_rng(SEED + n_checkpoints * 100 + int(rise * 1000))
            hits = sum(one_trial(generator, steps, rise) for _ in range(TRIALS))
            rows.append({"rise": rise, "checkpoints": n_checkpoints, "gated": hits / TRIALS})
            print(f"  rise {rise:.2f}  n={n_checkpoints:>2}  gated {hits / TRIALS:.1%}",
                  flush=True)

    report = {
        "settings": {
            "trials_per_cell": TRIALS,
            "break_resamples": RESAMPLES,
            "p_threshold": P_THRESHOLD,
            "monte_carlo_se_at_50pct": round(math.sqrt(0.25 / TRIALS), 4),
            "extends": "calibration.json, which covered 8 to 24 checkpoints",
        },
        "cells": rows,
        "seconds": round(time.time() - started, 1),
    }
    (HERE / "power-extension.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["cells"], indent=2))
    print(f"{report['seconds']}s")


if __name__ == "__main__":
    main()
