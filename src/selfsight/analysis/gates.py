"""Machine-readable registered Gate 2 and 2b decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    gate: str
    passed: bool
    checks: dict[str, bool]
    narrative: str


def decide_gate_2(
    *,
    d_star_by_seed: Mapping[int, float | None],
    internal_post_positive_by_seed: Mapping[int, bool],
    external_post_nonpositive_by_seed: Mapping[int, bool],
    delta_scfr: float,
    delta_scfr_ci_low: float,
) -> GateDecision:
    found = sum(value is not None for value in d_star_by_seed.values()) >= 2
    slopes = sum(
        bool(internal_post_positive_by_seed.get(seed)) and bool(external_post_nonpositive_by_seed.get(seed))
        for seed, value in d_star_by_seed.items()
        if value is not None
    ) >= 2
    scfr = delta_scfr >= 0.03 and delta_scfr_ci_low > 0.0
    checks = {"d_star_in_at_least_2_of_3_seeds": found, "registered_post_slopes": slopes, "delta_scfr": scfr}
    passed = all(checks.values())
    narrative = "divergence anchor established" if passed else "registered safe-self-training fallback"
    return GateDecision("gate_2", passed, checks, narrative)


def decide_gate_2b(
    *,
    lead_ci_low: float,
    median_lead: float,
    total_steps: int,
    d_g_below_noise: bool,
    early_safe: bool,
    median_spearman: float,
    beats_entropy_baseline: bool,
) -> GateDecision:
    checks = {
        "paired_bootstrap_ci_above_zero": lead_ci_low > 0.0,
        "median_lead_at_least_10_percent": median_lead >= total_steps * 0.10,
        "d_g_below_noise_floor": d_g_below_noise,
        "early_10_percent_inside_noise": early_safe,
        "gda_free_gold_spearman": median_spearman >= 0.70,
        "beats_entropy_baseline": beats_entropy_baseline,
    }
    passed = all(checks.values())
    narrative = "unlabeled early-warning claim enabled" if passed else "registered rigorous negative-signal fallback"
    return GateDecision("gate_2b", passed, checks, narrative)
