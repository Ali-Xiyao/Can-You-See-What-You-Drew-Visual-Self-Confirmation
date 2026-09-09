"""D_g's power once it uses the per-prompt statistic and D*'s own knot machinery.

Two changes from the shipped estimator, both of which are re-analysis of probes
that already ran -- no new GPU work:

  1. statistic: mean over prompts of the per-prompt cosine, paired against base,
     instead of cosine(mean_L, mean_R).  Measured SEM 0.052 against 0.211.
  2. rule: `fit_segmented` + `break_support` (the residual bootstrap that
     calibrates the knot search) + a negative post-break slope, instead of
     "two consecutive checkpoints below an absolute floor".

`break_support`'s statistic `1 - sse_break/sse_null` is scale-free, so its null
distribution depends on the design (12 evenly spaced points, 3 aside) and not on
the noise level.  It is therefore simulated once here and reused, which is what
makes 4000 trials per cell affordable; the critical value is checked against the
real `break_support` at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from selfsight.analysis.breakpoints import break_support, fit_segmented

CHECKPOINTS = 12
SPAN = 88.0
STEPS = np.linspace(0.0, SPAN, CHECKPOINTS)
D_G_TRUE = SPAN * 0.40
ALPHA = 0.05
NULL_TRIALS = 20000
TRIALS = 4000

# --- measured on the pilot's saved grams, per_prompt_cosine.py ---
N_PILOT = 14
SEM_TYPICAL = 0.0521        # median paired per-prompt SEM across 8 checkpoints
SEM_PESSIMISTIC = 0.0834    # the worst single checkpoint (rfo_gold, step 24)
PROBE_SIZES = (14, 32, 64)
DROPS = (0.0, 0.05, 0.10, 0.15, 0.20)

_LINE = np.column_stack((np.ones_like(STEPS), STEPS))


def reduction_and_slope(y: np.ndarray) -> tuple[float, float]:
    coefficients, *_ = np.linalg.lstsq(_LINE, y, rcond=None)
    residual = y - _LINE @ coefficients
    sse_null = float(residual @ residual)
    fit = fit_segmented(STEPS, y)
    if sse_null <= 0.0:
        return 0.0, fit.slope_after
    return 1.0 - fit.sse / sse_null, fit.slope_after


def critical_value(rng: np.random.Generator) -> float:
    """95th percentile of the scale-free SSE reduction under a no-break null."""
    draws = np.empty(NULL_TRIALS)
    for index in range(NULL_TRIALS):
        draws[index], _ = reduction_and_slope(rng.normal(0.0, 1.0, CHECKPOINTS))
    return float(np.quantile(draws, 1.0 - ALPHA))


def truth(drop: float) -> np.ndarray:
    after = np.clip((STEPS - D_G_TRUE) / (SPAN - D_G_TRUE), 0.0, 1.0)
    return -drop * after


def power(sem: float, drop: float, crit: float, rng: np.random.Generator) -> float:
    signal = truth(drop)
    fired = 0
    for _ in range(TRIALS):
        observed = signal + rng.normal(0.0, sem, CHECKPOINTS)
        value, slope = reduction_and_slope(observed)
        fired += value >= crit and slope < 0.0
    return fired / TRIALS


def table(label: str, sem_at_pilot: float, crit: float) -> None:
    print(f"\n{label}  (SEM at n={N_PILOT} is {sem_at_pilot:.4f})")
    print(f"{'n':>5} {'SEM':>7}   " + "".join(f"{d:>9.2f}" for d in DROPS))
    for n in PROBE_SIZES:
        sem = sem_at_pilot * np.sqrt(N_PILOT / n)
        rng = np.random.default_rng(20260908 + n)
        cells = "".join(f"{power(sem, drop, crit, rng):>9.1%}" for drop in DROPS)
        print(f"{n:>5} {sem:>7.4f}   {cells}")


def main() -> None:
    rng = np.random.default_rng(20260908)
    crit = critical_value(rng)
    print(f"{CHECKPOINTS} checkpoints over {SPAN:.0f} steps, true D_g at step "
          f"{D_G_TRUE:.0f}, {TRIALS} trials")
    print(f"no-break null: 95th pct of 1 - sse_break/sse_null = {crit:.4f} "
          f"({NULL_TRIALS} draws)")

    # The shortcut is only legitimate if it agrees with the shipped calibration.
    check = rng.normal(0.0, SEM_TYPICAL, CHECKPOINTS)
    shipped = break_support(STEPS, check, resamples=2000)
    mine, _ = reduction_and_slope(check)
    print(f"cross-check on one flat draw: reduction={mine:.4f}, "
          f"break_support p={shipped.p_value:.3f}, "
          f"shortcut fires={mine >= crit}, shipped fires={shipped.p_value < ALPHA}")

    table("drop in paired per-prompt cosine ->", SEM_TYPICAL, crit)
    table("pessimistic (worst checkpoint's SEM) ->", SEM_PESSIMISTIC, crit)
    print("\ndrop=0.00 is the false-alarm rate; it should land near 5% by construction.")
    print("pilot moved -0.014 by step 8 and +0.089 by step 32: no drop to detect yet.")


if __name__ == "__main__":
    main()
