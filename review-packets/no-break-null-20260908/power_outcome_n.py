"""The other lever: the effect does not have to grow if the noise can shrink.

calibrate.py fixed the outcome set at 120 prompts per checkpoint, which puts one
binomial SEM at 4.4 correctness points and makes ~20 points of external rise the
price of 78% power. That framed the whole thing as a training-budget problem.

But the requirement is a ratio, and the denominator is ours to choose. Evaluating
more outcome prompts per checkpoint costs generation and adjudication, not
optimizer steps. This measures the trade directly instead of assuming SEM scaling
carries through the estimator's four conditions unchanged.
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

BASE_RATE = 0.37
INTERNAL_RISE = 0.004
INTERNAL_SEM = 0.002
SPAN = 175.0
TRIALS = 300
RESAMPLES = 300
P_THRESHOLD = 0.05
CHECKPOINTS = 24
OUTCOME_SIZES = (120, 240, 480, 960)
RISES = (0.0, 0.05, 0.10)
SEED = 20260908


def external_floor(external, outcome_n):
    sems = sorted(math.sqrt(max(0.0, p * (1.0 - p)) / outcome_n) for p in external)
    return sems[len(sems) // 2] / SPAN


def truth(steps, rise):
    knot = SPAN / 2.0
    return np.array([BASE_RATE + rise * min(step, knot) / knot for step in steps])


def one_trial(generator, steps, rise, outcome_n):
    expected = truth(steps, rise)
    external = generator.binomial(outcome_n, np.clip(expected, 0.0, 1.0)) / outcome_n
    internal = -2.0 + INTERNAL_RISE * steps + generator.normal(0.0, INTERNAL_SEM, steps.size)
    estimate = estimate_d_star(
        steps,
        internal,
        external,
        min_coupled_external_slope=external_floor(external, outcome_n),
        max_break_p_value=P_THRESHOLD,
        break_resamples=RESAMPLES,
        break_seed=int(generator.integers(1, 2**31 - 1)),
    )
    return estimate.d_star is not None


def main():
    started = time.time()
    steps = np.linspace(0.0, SPAN, CHECKPOINTS)
    rows = []
    for rise in RISES:
        for outcome_n in OUTCOME_SIZES:
            generator = np.random.default_rng(SEED + outcome_n + int(rise * 1000))
            hits = sum(one_trial(generator, steps, rise, outcome_n) for _ in range(TRIALS))
            sem = math.sqrt(BASE_RATE * (1 - BASE_RATE) / outcome_n)
            rows.append({
                "rise": rise,
                "outcome_n": outcome_n,
                "one_sem_points": round(100 * sem, 2),
                "rise_in_sem": round(rise / sem, 2) if sem else None,
                "power": hits / TRIALS,
            })
            print(f"  rise {rise:.2f}  outcome_n {outcome_n:>3}  "
                  f"(1 SEM = {100 * sem:.2f} pt, rise = {rise / sem:.1f} SEM)  "
                  f"power {hits / TRIALS:.1%}", flush=True)

    (HERE / "power-outcome-n.json").write_text(
        json.dumps({
            "settings": {
                "checkpoints": CHECKPOINTS,
                "trials_per_cell": TRIALS,
                "break_resamples": RESAMPLES,
                "p_threshold": P_THRESHOLD,
                "monte_carlo_se_at_50pct": round(math.sqrt(0.25 / TRIALS), 4),
                "question": (
                    "does buying outcome prompts substitute for buying optimizer steps"
                ),
            },
            "cells": rows,
            "seconds": round(time.time() - started, 1),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{round(time.time() - started, 1)}s")


if __name__ == "__main__":
    main()
