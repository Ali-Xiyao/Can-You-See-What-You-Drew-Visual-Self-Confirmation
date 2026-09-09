"""D_g's power, measured the same way D*'s was.

The false-alarm rate for D_g was measured (0/2000) and its power never was.
Those are the same fact seen from two sides if the noise floor is loose enough,
and the pilot's own bootstrap says it is: the base checkpoint's gda_free is
0.715 with a 95% interval of [0.156, 0.982] over 14 scene clusters. A floor at
0.156 can only be crossed by a cosine that has collapsed to near-orthogonality.

Everything here is driven off that measured interval, not off an assumption:
  - per-checkpoint SD  = width / 3.92, scaled by sqrt(14 / n)
  - noise floor        = g0 - (g0 - lower), the same gap, scaled by sqrt(14 / n)
The estimator is the real `estimate_d_g`, run with the config's ema and
persistence, so what is measured is the shipped decision rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from selfsight.analysis.breakpoints import estimate_d_g

# --- measured on runs/v4/decoupling-pilot-20260906/gradient_sensitivity.json ---
N_PILOT = 14
G0 = 0.715                      # base checkpoint, gda_free
CI_PILOT = (0.156, 0.982)
SD_PILOT = (CI_PILOT[1] - CI_PILOT[0]) / 3.92        # 0.211
GAP_PILOT = G0 - CI_PILOT[0]                          # 0.559, floor distance

CHECKPOINTS = 12
SPAN = 88.0
STEPS = np.linspace(0.0, SPAN, CHECKPOINTS)
D_G_TRUE = SPAN * 0.40          # alignment starts eroding 40% of the way in
EMA_ALPHA = 0.35                # breakpoints.estimate_d_g default
PERSISTENCE = 2                 # gates.gradient_confirmation_checkpoints
TRIALS = 4000
PROBE_SIZES = (14, 32, 64, 128, 256)
DROPS = (0.0, 0.10, 0.20, 0.40, 0.60)   # total fall in gda_free by the last ck


def truth(drop: float) -> np.ndarray:
    """Flat, then a linear erosion of `drop` over the rest of the span."""
    after = np.clip((STEPS - D_G_TRUE) / (SPAN - D_G_TRUE), 0.0, 1.0)
    return G0 - drop * after


def run(n: int, drop: float, rng: np.random.Generator) -> tuple[float, float]:
    scale = np.sqrt(N_PILOT / n)
    sd, floor = SD_PILOT * scale, G0 - GAP_PILOT * scale
    high = min(1.0, G0 + GAP_PILOT * scale)
    signal = truth(drop)
    fired = 0
    for _ in range(TRIALS):
        observed = np.clip(signal + rng.normal(0.0, sd, CHECKPOINTS), -1.0, 1.0)
        estimate = estimate_d_g(STEPS, observed, noise_low=floor, noise_high=high,
                                ema_alpha=EMA_ALPHA, persistence=PERSISTENCE)
        fired += estimate.d_g is not None
    return fired / TRIALS, floor


def main() -> None:
    print(f"pilot: n={N_PILOT}  gda_free={G0:.3f}  CI={CI_PILOT}  "
          f"SD={SD_PILOT:.3f}  floor gap={GAP_PILOT:.3f}")
    print(f"{CHECKPOINTS} checkpoints over {SPAN:.0f} steps, true D_g at step "
          f"{D_G_TRUE:.0f}, {TRIALS} trials\n")
    header = "  drop ->" + "".join(f"{d:>9.2f}" for d in DROPS)
    print(f"{'n':>5} {'floor':>7} {'SD':>6}   " + header[9:])
    for n in PROBE_SIZES:
        rng = np.random.default_rng(20260908 + n)
        rates = [run(n, drop, rng) for drop in DROPS]
        floor = rates[0][1]
        sd = SD_PILOT * np.sqrt(N_PILOT / n)
        cells = "".join(f"{rate:>9.1%}" for rate, _ in rates)
        print(f"{n:>5} {floor:>7.3f} {sd:>6.3f}   {cells}")
    print("\nleft column (drop=0.00) is the false-alarm rate; the rest is power.")
    print("a drop must clear G0 - floor before the rule can ever fire:")
    for n in PROBE_SIZES:
        need = GAP_PILOT * np.sqrt(N_PILOT / n)
        print(f"  n={n:>4}  needs gda_free to fall by more than {need:.3f}")


if __name__ == "__main__":
    main()
