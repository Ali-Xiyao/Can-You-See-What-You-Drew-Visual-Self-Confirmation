"""How often does estimate_d_star report a D* that is not there, and does the
no-break null fix it without destroying the signal it exists to find?

Two questions, both answered by simulation because the analytic answer does not
exist: the knot is not identified under the no-break null, so the usual F-test
against chi-square quantiles is invalid (Davies' problem).

    1. FALSE POSITIVE. External correctness is genuinely flat -- no coupling, no
       break, only binomial noise from adjudicating n=120 prompts. Internal
       genuinely rises. How often is a D* returned?

    2. POWER. External genuinely rises and then stalls. How often is the D*
       still found once the no-break null has to be cleared?

Both are run against the estimator with and without `max_break_p_value`, at
several checkpoint counts, so the answer is a design table rather than a verdict.
Nothing here touches a model, a GPU, or runs/.
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

OUTCOME_N = 120           # prompts adjudicated per checkpoint
BASE_RATE = 0.37          # external correctness under the flat null
INTERNAL_RISE = 0.004     # per step, comfortably above its own noise floor
INTERNAL_SEM = 0.002
SPAN = 175.0
TRIALS = 300
RESAMPLES = 300
P_THRESHOLD = 0.05
CHECKPOINTS = (8, 12, 16, 24)
RISES = (0.0, 0.05, 0.10, 0.15, 0.20)   # total external gain before the knot
SEED = 20260908


def external_floor(external):
    """What v4.evaluate.external_noise_slope would return for this curve."""
    sems = sorted(math.sqrt(max(0.0, p * (1.0 - p)) / OUTCOME_N) for p in external)
    return sems[len(sems) // 2] / SPAN


def truth(steps, rise):
    """External rises by `rise` over the first half, then holds. rise=0 is the null."""
    knot = SPAN / 2.0
    return np.array([
        BASE_RATE + rise * min(step, knot) / knot for step in steps
    ])


def one_trial(generator, steps, rise, gate):
    expected = truth(steps, rise)
    external = generator.binomial(OUTCOME_N, np.clip(expected, 0.0, 1.0)) / OUTCOME_N
    internal = -2.0 + INTERNAL_RISE * steps + generator.normal(0.0, INTERNAL_SEM, steps.size)
    estimate = estimate_d_star(
        steps,
        internal,
        external,
        min_coupled_external_slope=external_floor(external),
        max_break_p_value=P_THRESHOLD if gate else None,
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
            cell = {"rise": rise, "checkpoints": n_checkpoints}
            for gate in (False, True):
                generator = np.random.default_rng(SEED + n_checkpoints * 100 + int(rise * 1000))
                hits = sum(one_trial(generator, steps, rise, gate) for _ in range(TRIALS))
                cell["gated" if gate else "ungated"] = hits / TRIALS
            rows.append(cell)
            print(f"  rise {rise:.2f}  n={n_checkpoints:>2}  "
                  f"ungated {cell['ungated']:.1%}  gated {cell['gated']:.1%}", flush=True)

    null_rows = [r for r in rows if r["rise"] == 0.0]
    report = {
        "settings": {
            "outcome_n": OUTCOME_N,
            "base_rate": BASE_RATE,
            "trials_per_cell": TRIALS,
            "break_resamples": RESAMPLES,
            "p_threshold": P_THRESHOLD,
            "monte_carlo_se_at_5pct": round(math.sqrt(0.05 * 0.95 / TRIALS), 4),
            "seed": SEED,
            "note": (
                "rise is the total external gain before the knot, in correctness points; "
                "one binomial SEM at n=120 and p=0.37 is 0.044, so rise=0.10 is 2.3 SEM"
            ),
        },
        "cells": rows,
        "summary": {
            "false_positive_ungated": {
                str(r["checkpoints"]): r["ungated"] for r in null_rows
            },
            "false_positive_gated": {
                str(r["checkpoints"]): r["gated"] for r in null_rows
            },
            "ungated_rate_rises_with_checkpoints": (
                null_rows[-1]["ungated"] > null_rows[0]["ungated"]
            ),
            "gated_rate_within_threshold_everywhere": all(
                r["gated"] <= P_THRESHOLD + 3 * math.sqrt(0.05 * 0.95 / TRIALS)
                for r in null_rows
            ),
        },
        "seconds": round(time.time() - started, 1),
    }
    (HERE / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"{report['seconds']}s")


if __name__ == "__main__":
    main()
