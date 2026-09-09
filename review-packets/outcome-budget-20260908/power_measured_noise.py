"""Redo the power table against the noise the pilot actually has, not binomial.

calibrate.py and power_outcome_n.py both drew the external curve as
binomial(n, p)/n. The pilot's own checkpoint_metrics.csv says that is wrong by
a factor of about 2.6: residuals about a fitted line are 2.33 points where the
binomial prediction is 6.13. The reason is in the config -- evaluation.fixed
_latents is true, so the same (prompt, latent) pairs are re-rendered at every
checkpoint and the between-spec difficulty spread, which dominates binomial
variance, is a fixed effect that the pairing cancels.

That deflation decides the run configuration, so it gets its own measurement
rather than an assumption. Noise is a direct SD parameter here, swept across
the point estimate and both ends of its (wide, 6 df) confidence interval.

One thing deliberately NOT deflated: external_floor still computes the binomial
SEM, because src/selfsight/v4/evaluate.py:external_noise_slope does. Modelling
the deployed instrument means keeping its too-conservative floor.
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

BASE_RATE = 0.28          # the pilot's own step-0 external rate, not 0.37
OUTCOME_N = 64            # what the binomial floor is computed from
INTERNAL_RISE = 0.004
INTERNAL_SEM = 0.002
SPAN = 175.0
TRIALS = 300
RESAMPLES = 300
P_THRESHOLD = 0.05
SEED = 20260908

# 2.33 pt measured at n=57; 95% CI [1.50, 5.14]. Scaled to each image count by
# 1/sqrt(n), which is the one part of the binomial model the pairing preserves.
SD_AT_57 = (0.0150, 0.0233, 0.0514)
IMAGE_COUNTS = (64, 128, 256)
CHECKPOINTS = (12, 16, 24)
RISES = (0.0, 0.05, 0.10)


def external_floor(external, outcome_n):
    sems = sorted(math.sqrt(max(0.0, p * (1.0 - p)) / outcome_n) for p in external)
    return sems[len(sems) // 2] / SPAN


def truth(steps, rise):
    knot = SPAN / 2.0
    return np.array([BASE_RATE + rise * min(step, knot) / knot for step in steps])


def one_trial(generator, steps, rise, sd, images):
    expected = truth(steps, rise)
    external = expected + generator.normal(0.0, sd, steps.size)
    internal = -2.3 + INTERNAL_RISE * steps + generator.normal(0.0, INTERNAL_SEM, steps.size)
    estimate = estimate_d_star(
        steps, internal, external,
        min_coupled_external_slope=external_floor(external, images),
        max_break_p_value=P_THRESHOLD,
        break_resamples=RESAMPLES,
        break_seed=int(generator.integers(1, 2**31 - 1)),
    )
    return estimate.d_star is not None


def main():
    started = time.time()
    rows = []
    for sd57 in SD_AT_57:
        for images in IMAGE_COUNTS:
            sd = sd57 * math.sqrt(57.0 / images)
            for n_ck in CHECKPOINTS:
                steps = np.linspace(0.0, SPAN, n_ck)
                for rise in RISES:
                    g = np.random.default_rng(
                        SEED + int(sd57 * 1e5) + images * 7 + n_ck * 13 + int(rise * 1000)
                    )
                    hits = sum(one_trial(g, steps, rise, sd, images) for _ in range(TRIALS))
                    rows.append({
                        "sd_at_57_pt": round(100 * sd57, 2),
                        "images": images,
                        "sd_pt": round(100 * sd, 2),
                        "checkpoints": n_ck,
                        "rise": rise,
                        "rise_in_sd": round(rise / sd, 2) if sd else None,
                        "power": hits / TRIALS,
                    })
                    print(f"  sd57={100*sd57:.2f}pt img={images:>3} sd={100*sd:.2f}pt "
                          f"ck={n_ck:>2} rise={rise:.2f} ({rise/sd:5.1f} SD)  "
                          f"power {hits/TRIALS:.1%}", flush=True)
    (HERE / "power-measured-noise.json").write_text(json.dumps({
        "settings": {
            "base_rate": BASE_RATE, "trials_per_cell": TRIALS,
            "break_resamples": RESAMPLES, "p_threshold": P_THRESHOLD,
            "monte_carlo_se_at_50pct": round(math.sqrt(0.25 / TRIALS), 4),
            "noise_source": "runs/v4/decoupling-pilot-20260906/checkpoint_metrics.csv residuals",
            "floor_kept_binomial": "external_noise_slope in evaluate.py computes it that way",
        },
        "cells": rows, "seconds": round(time.time() - started, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{round(time.time()-started,1)}s")


if __name__ == "__main__":
    main()
