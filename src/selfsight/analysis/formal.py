"""Three-seed formal aggregation and registered Gate 2/2b decisions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from selfsight.analysis.breakpoints import estimate_d_g, estimate_d_star
from selfsight.analysis.figure1 import render_figure1_from_csv
from selfsight.analysis.gates import decide_gate_2, decide_gate_2b
from selfsight.analysis.statistics import (
    bootstrap_median,
    checkpoint_spearman,
    paired_bootstrap_lead,
)
from selfsight.config import load_config
from selfsight.schemas import as_serializable
from selfsight.utils.jsonl import atomic_write_json


def _competing_warning(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    direction: str,
    persistence: int = 2,
) -> float | None:
    finite = np.isfinite(steps) & np.isfinite(values)
    steps, values = steps[finite], values[finite]
    if len(values) < 7:
        return None
    early = max(3, math.ceil(len(values) * 0.10))
    mean = float(np.mean(values[:early]))
    standard = max(float(np.std(values[:early], ddof=1)), 1e-6)
    if direction == "down":
        flagged = values < mean - 2.0 * standard
    elif direction == "up":
        flagged = values > mean + 2.0 * standard
    else:
        flagged = np.abs(values - mean) > 2.0 * standard
    for index in range(early, len(values) - persistence + 1):
        if bool(np.all(flagged[index : index + persistence])):
            return float(steps[index])
    return None


def aggregate_formal_e2(
    *,
    config_path: str | Path,
    metric_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    seeds = sorted(int(seed) for seed in frame.seed.unique())
    registered = sorted(int(seed) for seed in config.values["training"]["seeds"])
    if seeds != registered or len(seeds) != 3:
        raise ValueError(f"Formal aggregation requires registered seeds {registered}, found {seeds}")
    if frame.duplicated(["seed", "arm", "step"]).any():
        raise ValueError("Formal metrics contain duplicate seed/arm/step rows")
    metrics_path = output / "formal_checkpoint_metrics.csv"
    frame.sort_values(["seed", "arm", "step"]).to_csv(metrics_path, index=False)

    d_star_by_seed: dict[int, float | None] = {}
    d_g_by_seed: dict[int, float | None] = {}
    internal_positive: dict[int, bool] = {}
    external_nonpositive: dict[int, bool] = {}
    delta_scfr_by_seed: dict[int, float] = {}
    early_safe_by_seed: dict[int, bool] = {}
    d_g_below_by_seed: dict[int, bool] = {}
    entropy_warning_by_seed: dict[int, float | None] = {}
    public_warning_by_seed: dict[int, float | None] = {}
    gda_free_by_seed: dict[int, list[float]] = {}
    gda_gold_by_seed: dict[int, list[float]] = {}
    per_seed = {}
    offset = int(config.values["gates"]["scfr_post_breakpoint_offset_steps"])
    for seed in seeds:
        values = frame.loc[(frame.seed == seed) & (frame.arm == "naive")].sort_values("step")
        d_star = estimate_d_star(values.step, values.internal_score, values.external_correctness)
        d_star_by_seed[seed] = d_star.d_star
        internal_positive[seed] = bool(
            d_star.internal_post_slope is not None and d_star.internal_post_slope > 0.0
        )
        external_nonpositive[seed] = bool(
            d_star.external_post_slope is not None and d_star.external_post_slope <= 0.0
        )
        valid_gda = values.dropna(subset=["gda_free", "gda_gold", "noise_low", "noise_high"])
        if len(valid_gda) >= 7:
            d_g = estimate_d_g(
                valid_gda.step,
                valid_gda.gda_free,
                noise_low=valid_gda.noise_low,
                noise_high=valid_gda.noise_high,
                ema_alpha=float(config.values["gradient_probe"]["ema_alpha"]),
            )
            d_g_by_seed[seed] = d_g.d_g
            early_safe_by_seed[seed] = d_g.early_safe
            gda_free_by_seed[seed] = valid_gda.gda_free.astype(float).tolist()
            gda_gold_by_seed[seed] = valid_gda.gda_gold.astype(float).tolist()
            if d_g.d_g is not None:
                row = valid_gda.loc[valid_gda.step == d_g.d_g].iloc[0]
                d_g_below_by_seed[seed] = float(row.gda_free) < float(row.noise_low)
            else:
                d_g_below_by_seed[seed] = False
        else:
            d_g = None
            d_g_by_seed[seed] = None
            early_safe_by_seed[seed] = False
            d_g_below_by_seed[seed] = False
        if d_star.d_star is not None:
            eligible = values.loc[values.step >= d_star.d_star + offset]
            if not eligible.empty:
                delta_scfr_by_seed[seed] = float(eligible.iloc[0].scfr_competent - values.iloc[0].scfr_competent)
        steps = values.step.to_numpy(dtype=float)
        entropy_warning_by_seed[seed] = _competing_warning(
            steps,
            values.observer_answer_entropy.to_numpy(dtype=float),
            direction="either",
        )
        public_warning_by_seed[seed] = _competing_warning(
            steps,
            values.public_view_consistency.to_numpy(dtype=float),
            direction="down",
        )
        per_seed[seed] = {
            "d_star": as_serializable(d_star),
            "d_g": as_serializable(d_g) if d_g else None,
            "delta_scfr": delta_scfr_by_seed.get(seed),
            "entropy_warning": entropy_warning_by_seed[seed],
            "public_view_warning": public_warning_by_seed[seed],
        }

    scfr_bootstrap = (
        bootstrap_median(delta_scfr_by_seed)
        if len(delta_scfr_by_seed) == 3
        else {"median": float("nan"), "ci_low": float("-inf"), "ci_high": float("inf")}
    )
    gate_2 = decide_gate_2(
        d_star_by_seed=d_star_by_seed,
        internal_post_positive_by_seed=internal_positive,
        external_post_nonpositive_by_seed=external_nonpositive,
        delta_scfr=float(scfr_bootstrap["median"]),
        delta_scfr_ci_low=float(scfr_bootstrap["ci_low"]),
    )
    finite_d_star = {seed: value for seed, value in d_star_by_seed.items() if value is not None}
    finite_d_g = {seed: value for seed, value in d_g_by_seed.items() if value is not None}
    lead_bootstrap = (
        paired_bootstrap_lead(finite_d_star, finite_d_g)
        if len(finite_d_star) == len(finite_d_g) == 3
        else {"median": float("nan"), "ci_low": float("-inf"), "ci_high": float("inf")}
    )
    spearman = (
        checkpoint_spearman(gda_free_by_seed, gda_gold_by_seed)
        if len(gda_free_by_seed) == 3
        else {}
    )
    beats_competitor_by_seed = {}
    for seed in seeds:
        competitor_values = [
            value
            for value in (entropy_warning_by_seed[seed], public_warning_by_seed[seed])
            if value is not None
        ]
        competitor = min(competitor_values) if competitor_values else None
        d_g_value = d_g_by_seed[seed]
        beats_competitor_by_seed[seed] = bool(
            d_g_value is not None and (competitor is None or d_g_value < competitor)
        )
    gate_2b = decide_gate_2b(
        lead_ci_low=float(lead_bootstrap["ci_low"]),
        median_lead=float(lead_bootstrap["median"]),
        total_steps=int(config.values["training"]["rounds"])
        * int(config.values["training"]["optimizer_steps_per_round"]),
        d_g_below_noise=sum(d_g_below_by_seed.values()) >= 2,
        early_safe=all(early_safe_by_seed.values()),
        median_spearman=float(np.median(list(spearman.values()))) if spearman else float("nan"),
        beats_entropy_baseline=sum(beats_competitor_by_seed.values()) >= 2,
    )
    median_d_star = float(np.median(list(finite_d_star.values()))) if finite_d_star else None
    median_d_g = float(np.median(list(finite_d_g.values()))) if finite_d_g else None
    figures = render_figure1_from_csv(
        metrics_path,
        output / "figures" / "figure1_formal",
        arm="naive",
        d_g=median_d_g,
        d_star=median_d_star,
        evidence_status="formal",
    )
    report = {
        "schema_version": 1,
        "per_seed": per_seed,
        "d_star_by_seed": d_star_by_seed,
        "d_g_by_seed": d_g_by_seed,
        "delta_scfr_bootstrap": scfr_bootstrap,
        "lead_bootstrap": lead_bootstrap,
        "spearman_by_seed": spearman,
        "beats_competing_signal_by_seed": beats_competitor_by_seed,
        "gate_2": as_serializable(gate_2),
        "gate_2b": as_serializable(gate_2b),
        "authorize_e3": gate_2.passed,
        "gda_claim_enabled": gate_2b.passed,
        "next_action": (
            "Proceed to E3; retain the early-warning claim only if Gate 2b passed."
            if gate_2.passed
            else "Stop expensive follow-on experiments and write the registered safe-self-training fallback."
        ),
        "figures": figures,
    }
    atomic_write_json(output / "formal_gate_2_2b.json", report)
    return report
