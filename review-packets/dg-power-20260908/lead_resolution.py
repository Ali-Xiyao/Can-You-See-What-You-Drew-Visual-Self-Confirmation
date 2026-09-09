"""What `Lead = D* - D_g` can resolve, given that both knots are searched.

The endpoint is a difference of two breakpoints, each estimated by minimising
SSE over the same 12-point grid.  A searched knot is a coarse estimate even when
the break is real: it can only land on one of six admissible candidates, and it
lands on the wrong one whenever noise happens to favour it.  Lead inherits both
errors, so its standard error is roughly sqrt(2) times a single knot's -- before
any of the biology.  This measures the single-knot error at the SNRs the two
curves actually run at, and reads Lead's resolution off it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from selfsight.analysis.breakpoints import fit_segmented

CHECKPOINTS = 12
SPAN = 88.0
STEPS = np.linspace(0.0, SPAN, CHECKPOINTS)
GRID = STEPS[1] - STEPS[0]
TRUE_KNOT = SPAN * 0.40
TRIALS = 4000
CANDIDATES = fit_segmented(STEPS, np.arange(CHECKPOINTS, dtype=float)).candidate_breakpoints


def knot_error(move: float, sem: float, rng: np.random.Generator) -> tuple[float, float]:
    """SD of the estimated knot, and how often it lands on the right candidate."""
    after = np.clip((STEPS - TRUE_KNOT) / (SPAN - TRUE_KNOT), 0.0, 1.0)
    signal = move * after
    nearest = min(CANDIDATES, key=lambda c: abs(c - TRUE_KNOT))
    found = np.empty(TRIALS)
    for index in range(TRIALS):
        found[index] = fit_segmented(STEPS, signal + rng.normal(0.0, sem, CHECKPOINTS)).breakpoint
    return float(found.std(ddof=1)), float(np.mean(np.isclose(found, nearest)))


def main() -> None:
    print(f"{CHECKPOINTS} checkpoints, grid spacing {GRID:.1f} steps, true knot at "
          f"{TRUE_KNOT:.0f}")
    print(f"admissible knots: {[round(c) for c in CANDIDATES]}  ({len(CANDIDATES)} of them)\n")
    print(f"{'curve':26s}{'move':>7}{'SEM':>8}{'move/SEM':>10}"
          f"{'knot SD':>10}{'on target':>11}{'Lead SE':>10}")
    cases = [
        # external correctness, from review-packets/outcome-budget-20260908
        ("external, rise 0.10 R=4", 0.10, 0.0150),
        ("external, rise 0.05 R=4", 0.05, 0.0150),
        ("external, pessimistic SD", 0.10, 0.0514),
        # gradient, per-prompt paired, from per_prompt_cosine.py
        ("gradient, drop 0.20 n=32", -0.20, 0.0345),
        ("gradient, drop 0.10 n=32", -0.10, 0.0345),
        ("gradient, drop 0.20 n=14", -0.20, 0.0521),
    ]
    single = {}
    for label, move, sem in cases:
        rng = np.random.default_rng(20260908)
        sd, hit = knot_error(move, sem, rng)
        single[label] = sd
        print(f"{label:26s}{move:>+7.2f}{sem:>8.4f}{abs(move) / sem:>10.1f}"
              f"{sd:>10.1f}{hit:>11.1%}{sd * np.sqrt(2):>10.1f}")

    print("\nknot SD and Lead SE are in training steps, on an 88-step span.")
    best = single["external, rise 0.10 R=4"]
    worst = single["gradient, drop 0.10 n=32"]
    print(f"best case pairing  : Lead SE ~= {np.hypot(best, single['gradient, drop 0.20 n=32']):.1f} steps")
    print(f"likely case pairing: Lead SE ~= {np.hypot(best, worst):.1f} steps")
    print("a Lead has to exceed ~2x that to be distinguishable from zero.")


if __name__ == "__main__":
    main()
