"""Describe a dynamic pilot without forcing a breakpoint or confirming a lead.

Reads evaluations/{arm}/step-*/{manifest,verified,s_select}.jsonl and
gradient-probes/**/report.json (also accepts probes/). The windows and thresholds are
fixed before this pilot; every look remains exploratory and unadjusted for
multiple looks. Missing image verdicts are unknown, never negative labels.

    envs/core/python.exe scripts/v4_decoupling_report.py --outdir RUN
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


RULES = {
    "window_checkpoints": 3,
    "internal_delta_min": 0.02,
    "external_delta_max": 0.02,
    "gradient_drop_min": 0.01,
    "gradient_consecutive_checkpoints": 2,
    "gradient_paired_ci_min_common": 4,
    "confidence_level": 0.95,
}
LIMITATIONS = [
    "exploratory: this report does not confirm D* or predictive utility",
    "multiple looks and arms are not multiplicity-adjusted",
    "observed lead is descriptive; forecasting improvement requires separate validation",
    "complete paired specs may differ from the full outcome population; inspect coverage",
    "bootstrap_rule_supported is the frozen descriptive rule, not reliable evidence of a plateau",
    "supplementary robustness bounds do not correct repeated looks or establish predictive utility",
    "spec-level binomial bounds require independent pairs; duplicate canonical scenes block that support flag",
    "scene-disjoint sensitivity and scene-cluster bootstrap supplement the original frozen outcome population",
    "the supplementary screen retains the internal-score percentile bootstrap; it is not an exact joint confidence test",
]
NUMERIC_TOLERANCE = 1e-12
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v4_decoupling_pilot.yaml"
DEFAULT_PROTOCOL = ROOT / "docs/prereg/2026-09-06-decoupling-pilot.md"


def configured_rules(config: dict[str, Any]) -> dict[str, Any]:
    rules = dict(RULES)
    pilot = config.get("pilot", {})
    mapping = {"window_checkpoints": "effect_window_checkpoints",
               "internal_delta_min": "internal_change_margin",
               "external_delta_max": "external_improvement_margin",
               "gradient_drop_min": "gradient_drop_margin",
               "gradient_consecutive_checkpoints": "gradient_confirmation_checkpoints"}
    for key, source in mapping.items():
        if source in pilot:
            rules[key] = pilot[source]
    rules["gradient_paired_ci_min_common"] = config.get("gradient_probe", {}).get("minimum_size", 4)
    for key in ("window_checkpoints", "gradient_consecutive_checkpoints",
                "gradient_paired_ci_min_common"):
        rules[key] = int(rules[key])
        if rules[key] < 2:
            raise ValueError(f"{key} must be at least two")
    for key in ("internal_delta_min", "external_delta_max", "gradient_drop_min"):
        rules[key] = float(rules[key])
        if not 0 <= rules[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    return rules


def file_identity(path: Path | None) -> dict[str, str] | None:
    return ({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if path is not None else None)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _image_index(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["image_path"])
        if key in result:
            raise ValueError(f"Duplicate image_path in {label}: {key}")
        result[key] = row
    return result


def read_outcome(directory: Path, expected_spec_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Keep a spec only when all its outcome images have both measurements."""
    step = int(directory.name.removeprefix("step-"))
    manifest_path = directory / "manifest.jsonl"
    verified_path = directory / "verified.jsonl"
    selection_path = directory / "s_select.jsonl"
    manifest = _jsonl(manifest_path)
    _image_index(manifest, str(manifest_path))
    verified = _image_index(_jsonl(verified_path), str(verified_path))
    selection = _image_index(_jsonl(selection_path), str(selection_path))
    grouped: dict[str, list[tuple[float | None, float | None]]] = defaultdict(list)
    counts = {"images": len(manifest), "known_score_images": 0,
              "known_verdict_images": 0, "joint_images": 0,
              "answer_available": 0, "answer_total": 0}
    for row in manifest:
        image_path = str(row["image_path"])
        spec_id = str(row["spec_id"])
        score_row = selection.get(image_path, {})
        recorded_id = score_row.get("prompt_id", score_row.get("spec_id"))
        if recorded_id is not None and str(recorded_id) != spec_id:
            raise ValueError(f"s_select spec mismatch for {image_path}")
        score = _finite(score_row.get("s_select"))
        if score is not None and not 0 <= score <= 1:
            raise ValueError(f"s_select outside [0, 1] for {image_path}")
        verdict_row = verified.get(image_path, {})
        verdict: float | None = None
        # Human unnameable images have a known image-level failure verdict.
        # Their per-question visual facts are unknown, a different quantity.
        if (verdict_row.get("resolution") != "pending_human"
                and isinstance(verdict_row.get("image_correct"), bool)):
            verdict = float(verdict_row["image_correct"])
        counts["known_score_images"] += score is not None
        counts["known_verdict_images"] += verdict is not None
        counts["joint_images"] += score is not None and verdict is not None
        counts["answer_available"] += int(score_row.get("available", 0))
        counts["answer_total"] += int(score_row.get("total", 0))
        grouped[spec_id].append((score, verdict))
    expected = set(map(str, expected_spec_ids)) if expected_spec_ids is not None else set(grouped)
    if set(grouped) - expected:
        raise ValueError("Outcome manifest contains specs outside the frozen outcome split")
    complete = {}
    measurements = {}
    for spec_id in expected - set(grouped):
        measurements[spec_id] = {"s_select": None, "external": None, "n_images": None,
                                 "s_select_low": 0.0, "s_select_high": 1.0,
                                 "external_low": 0.0, "external_high": 1.0}
    for spec_id, values in grouped.items():
        size = len(values)
        scores = [score for score, _ in values if score is not None]
        verdicts = [verdict for _, verdict in values if verdict is not None]
        measurements[spec_id] = {
            "s_select": sum(scores) / size if len(scores) == size else None,
            "external": sum(verdicts) / size if len(verdicts) == size else None,
            "n_images": size,
            "s_select_low": sum(scores) / size,
            "s_select_high": (sum(scores) + size - len(scores)) / size,
            "external_low": sum(verdicts) / size,
            "external_high": (sum(verdicts) + size - len(verdicts)) / size,
        }
        if all(score is not None and verdict is not None for score, verdict in values):
            complete[spec_id] = {
                "s_select": sum(score for score, _ in values) / len(values),
                "external": sum(verdict for _, verdict in values) / len(values),
                "n_images": len(values),
            }
    counts["specs"] = len(grouped)
    counts["expected_specs"] = len(expected)
    counts["missing_manifest_specs"] = len(expected - set(grouped))
    counts["complete_specs"] = len(complete)
    counts["incomplete_specs"] = len(expected) - len(complete)
    counts["paired_measurement_coverage"] = len(complete) / len(expected) if expected else None
    counts["answer_coverage"] = (counts["answer_available"] / counts["answer_total"]
                                if counts["answer_total"] else None)
    return {"step": step, "arm": directory.parent.name, "directory": str(directory),
            "coverage": counts, "complete_specs": complete, "spec_measurements": measurements,
            "missing_files": [p.name for p in (manifest_path, verified_path, selection_path)
                              if not p.exists()]}


def paired_changes(start: dict[str, Any], end: dict[str, Any], *,
                   bootstrap_samples: int = 2000, seed: int = 20260905) -> dict[str, Any]:
    """Bootstrap paired spec-level changes using the same resamples for both metrics."""
    ids = sorted(set(start["complete_specs"]) & set(end["complete_specs"]))
    n = len(ids)
    out: dict[str, Any] = {"n_paired_specs": n, "paired_spec_ids": ids,
                           "n_start_complete": len(start["complete_specs"]),
                           "n_end_complete": len(end["complete_specs"]),
                           "ci_method": "paired_spec_percentile_bootstrap",
                           "bootstrap_samples": bootstrap_samples}
    delta = np.asarray([[end["complete_specs"][sid][key]
                         - start["complete_specs"][sid][key]
                         for key in ("s_select", "external")] for sid in ids], dtype=float)
    intervals: np.ndarray | None = None
    if n >= 2:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, n, size=(bootstrap_samples, n))
        intervals = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975], axis=0)
    for index, key in enumerate(("s_select", "external")):
        out[key] = {"delta": float(delta[:, index].mean()) if n else None,
                    "ci_low": float(intervals[0, index]) if intervals is not None else None,
                    "ci_high": float(intervals[1, index]) if intervals is not None else None}
    out["ci_status"] = "available" if n >= 2 else "insufficient_paired_specs"
    return out


def _binomial_probability(n: int, k: int, probability: float, *, upper: bool) -> float:
    """Small pilot counts: evaluate the required tail directly without scipy."""
    if probability == 0:
        return float(k <= 0) if upper else 1.0
    if probability == 1:
        return 1.0 if upper else float(k >= n)
    indices = range(k, n + 1) if upper else range(k + 1)
    return math.fsum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                              - math.lgamma(n - i + 1)
                              + i * math.log(probability)
                              + (n - i) * math.log1p(-probability)) for i in indices)


def _clopper_pearson(k: int, n: int, tail_probability: float) -> tuple[float, float]:
    """Exact binomial inversion; each excluded tail has the supplied probability."""
    if not 0 <= k <= n or n < 1:
        raise ValueError("Clopper-Pearson needs 0 <= k <= n and n >= 1")
    if not 0 < tail_probability < 0.5:
        raise ValueError("Invalid Clopper-Pearson tail probability")
    lower = 0.0
    if k:
        lo, hi = 0.0, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2
            if _binomial_probability(n, k, mid, upper=True) < tail_probability:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    upper = 1.0
    if k < n:
        lo, hi = 0.0, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2
            if _binomial_probability(n, k, mid, upper=False) > tail_probability:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def paired_binary_bounds(improvements: int, deteriorations: int, n: int,
                         confidence: float = 0.95) -> dict[str, Any]:
    """Joint bounds for p(+1)-p(-1), avoiding all-zero bootstrap collapse.

    Each of the two Clopper-Pearson intervals has total error alpha/2.
    Bonferroni therefore bounds their joint error by alpha; independence of
    improvement and deterioration counts is neither assumed nor required.
    This is an additional fixed-sample check, not a replacement registered test.
    """
    if n < 1 or improvements < 0 or deteriorations < 0 or improvements + deteriorations > n:
        raise ValueError("Invalid paired binary counts")
    tail = (1 - confidence) / 4
    plus_low, plus_high = _clopper_pearson(improvements, n, tail)
    minus_low, minus_high = _clopper_pearson(deteriorations, n, tail)
    return {"method": "paired_discordance_clopper_pearson_bonferroni",
            "confidence": confidence, "category_tail_probability": tail,
            "n": n, "improvements": improvements, "deteriorations": deteriorations,
            "unchanged": n - improvements - deteriorations,
            "delta": (improvements - deteriorations) / n,
            "ci_low": plus_low - minus_high, "ci_high": plus_high - minus_low,
            "improvement_probability_ci": [plus_low, plus_high],
            "deterioration_probability_ci": [minus_low, minus_high]}


def _measurements(outcome: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "spec_measurements" in outcome:
        return outcome["spec_measurements"]
    return {sid: {**row, "s_select_low": row["s_select"], "s_select_high": row["s_select"],
                  "external_low": row["external"], "external_high": row["external"]}
            for sid, row in outcome["complete_specs"].items()}


def supplementary_robustness(start: dict[str, Any], end: dict[str, Any], *,
                             bootstrap_samples: int, seed: int,
                             rules: dict[str, Any]) -> dict[str, Any]:
    """Keep the frozen rule, while checking full-population missingness and sparse pairs."""
    left, right = _measurements(start), _measurements(end)
    ids = sorted(set(left) | set(right))
    n = len(ids)
    unknown = {"s_select": None, "external": None,
               "s_select_low": 0.0, "s_select_high": 1.0,
               "external_low": 0.0, "external_high": 1.0}
    raw_lower = raw_upper = missing_lower = missing_upper = 0.0
    completion_lower: list[float] = []
    completion_upper: list[float] = []
    paired_external: list[tuple[float, float]] = []
    internal: list[float] = []
    for sid in ids:
        a, b = left.get(sid, unknown), right.get(sid, unknown)
        lower = b["external_low"] - a["external_high"]
        upper = b["external_high"] - a["external_low"]
        raw_lower += lower
        raw_upper += upper
        completion_lower.append(lower)
        completion_upper.append(upper)
        if a["external"] is not None and b["external"] is not None:
            paired_external.append((a["external"], b["external"]))
        else:
            missing_lower += lower
            missing_upper += upper
        if a["s_select"] is not None and b["s_select"] is not None:
            internal.append(b["s_select"] - a["s_select"])
    external_n = len(paired_external)
    internal_n = len(internal)
    internal_ci = None
    if internal_n >= 2:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, internal_n, size=(bootstrap_samples, internal_n))
        means = np.asarray(internal)[indices].mean(axis=1)
        internal_ci = [float(value) for value in np.quantile(means, [0.025, 0.975])]
    binary = all(a in (0.0, 1.0) and b in (0.0, 1.0) for a, b in paired_external)
    cp = None
    confidence_envelope = None
    if external_n and binary:
        cp = paired_binary_bounds(sum(b > a for a, b in paired_external),
                                  sum(b < a for a, b in paired_external), external_n,
                                  confidence=rules["confidence_level"])
        # Retain the conditional observed-pair uncertainty and allow all
        # unknown pairs their permitted endpoint completions. Missing pairs
        # are never presumed exchangeable with the observed subset.
        confidence_envelope = [
            max(-1.0, (external_n * cp["ci_low"] + missing_lower) / n),
            min(1.0, (external_n * cp["ci_high"] + missing_upper) / n),
        ]
    internal_mean = float(np.mean(internal)) if internal else None
    internal_supported = bool(n and internal_n == n and internal_ci is not None
                              and internal_mean >= rules["internal_delta_min"] - NUMERIC_TOLERANCE
                              and internal_ci[0] > NUMERIC_TOLERANCE)
    fixed_cohort_screen = bool(confidence_envelope is not None
                               and confidence_envelope[1] <= rules["external_delta_max"] + NUMERIC_TOLERANCE)
    # For every spec, L <= actual change <= U. Treat the complete N-element
    # L/U arrays as the sampled variables; do not hold observability weights
    # fixed. Each selected one-sided bound uses two CP tails of alpha/4, so
    # the lower(L)/upper(U) interval has joint error at most alpha.
    completion_interval = lower_cp = upper_cp = None
    if n and binary and all(value in (-1.0, 0.0, 1.0)
                            for value in completion_lower + completion_upper):
        lower_cp = paired_binary_bounds(completion_lower.count(1.0), completion_lower.count(-1.0), n,
                                        confidence=rules["confidence_level"])
        upper_cp = paired_binary_bounds(completion_upper.count(1.0), completion_upper.count(-1.0), n,
                                        confidence=rules["confidence_level"])
        completion_interval = [lower_cp["ci_low"], upper_cp["ci_high"]]
    external_supported = bool(completion_interval is not None
                              and completion_interval[1] <= rules["external_delta_max"] + NUMERIC_TOLERANCE)
    return {
        "role": "supplementary_robustness_check_not_registered_confirmation",
        "n_population_specs": n, "n_external_paired_specs": external_n,
        "n_external_unknown_pairs": n - external_n, "n_internal_paired_specs": internal_n,
        "external_pair_coverage": external_n / n if n else None,
        "internal_pair_coverage": internal_n / n if n else None,
        "external_unknown_completion_delta_bounds": [raw_lower / n, raw_upper / n] if n else None,
        "external_paired_binary_bounds": cp,
        "external_confidence_and_unknown_envelope": confidence_envelope,
        "external_fixed_cohort_screen": fixed_cohort_screen,
        "external_complete_population": bool(n and external_n == n),
        "external_envelope_scope": "fixed observed cohort and missingness pattern; not a population confidence interval",
        "external_full_cohort_completion_confidence_bounds": completion_interval,
        "external_lower_completion_binary_bounds": lower_cp,
        "external_upper_completion_binary_bounds": upper_cp,
        "external_completion_bound_method": "full_N_endpoint_completion_CP_four_tail_union_bound",
        "external_completion_bound_scope": "L <= actual delta <= U; simultaneous lower E[L] and upper E[U] assuming independent spec pairs",
        "external_ci_status": ("available" if cp else "nonbinary_spec_rates"
                               if external_n and not binary else "no_known_external_pairs"),
        "internal_all_available_delta": internal_mean,
        "internal_all_available_bootstrap_ci": internal_ci,
        "internal_complete_population": bool(n and internal_n == n),
        "internal_supported": internal_supported, "external_supported": external_supported,
        "supported": internal_supported and external_supported,
        "independence_assumption": "independent spec pairs; canonical-scene duplicates invalidate this assumption",
        "support_interpretation": "conditional supplementary screen, not an exact joint test or confirmation of D*",
        "internal_inference_limitation": "internal-score percentile bootstrap can under-cover rare unobserved changes",
        "multiple_looks_adjusted": False,
        "scope": "complete-case CP is diagnostic; supplementary support uses full-N worst-completion bounds and independent-pair assumptions",
    }


def scene_cluster_changes(start: dict[str, Any], end: dict[str, Any], scene_of: dict[str, str], *,
                          bootstrap_samples: int, seed: int) -> dict[str, Any]:
    """Resample whole canonical scenes, retaining the original prompt-weighted mean."""
    population = sorted(set(_measurements(start)) | set(_measurements(end)))
    if set(population) - set(scene_of):
        raise ValueError("Scene audit does not cover the outcome population")
    scenes = sorted({scene_of[sid] for sid in population})
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    paired = sorted(set(start["complete_specs"]) & set(end["complete_specs"]))
    counts = np.zeros(len(scenes), dtype=int)
    totals = np.zeros((len(scenes), 2), dtype=float)
    for sid in paired:
        index = scene_index[scene_of[sid]]
        counts[index] += 1
        totals[index] += [end["complete_specs"][sid][key] - start["complete_specs"][sid][key]
                          for key in ("s_select", "external")]
    paired_scenes = int((counts > 0).sum())
    intervals = None
    valid_draws = 0
    if paired_scenes >= 2:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(scenes), size=(bootstrap_samples, len(scenes)))
        denominators = counts[indices].sum(axis=1)
        valid = denominators > 0
        valid_draws = int(valid.sum())
        estimates = totals[indices].sum(axis=1)[valid] / denominators[valid, None]
        if valid_draws:
            intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
    result = {"method": "canonical_scene_cluster_percentile_bootstrap",
              "estimand": "original prompt-weighted paired change, not scene-equal mean",
              "n_population_specs": len(population), "n_paired_specs": len(paired),
              "n_population_scene_clusters": len(scenes),
              "n_paired_scene_clusters": paired_scenes, "valid_bootstrap_draws": valid_draws,
              "bootstrap_samples": bootstrap_samples,
              "scope": "supplementary complete-case analysis; sparse binary bootstrap can still degenerate"}
    for index, key in enumerate(("s_select", "external")):
        result[key] = {"delta": float(totals[:, index].sum() / len(paired)) if paired else None,
                       "ci_low": float(intervals[0, index]) if intervals is not None else None,
                       "ci_high": float(intervals[1, index]) if intervals is not None else None}
    return result


def add_scene_sensitivity(arm_report: dict[str, Any], outcomes: Sequence[dict[str, Any]],
                          audit: dict[str, Any], *, bootstrap_samples: int, seed: int,
                          rules: dict[str, Any], representative_audit: dict[str, Any] | None = None) -> None:
    """Read the prospectively frozen audit; never choose exclusions from outcomes."""
    scene_of = {}
    for group in audit["within_outcome_clusters"]:
        for sid in group["spec_ids"]:
            if sid in scene_of:
                raise ValueError("Scene audit repeats an outcome spec")
            scene_of[sid] = group["scene_sha256"]
    sensitivity_ids = set(audit["scene_disjoint_outcome_sensitivity"]["spec_ids"])
    if not sensitivity_ids <= set(scene_of):
        raise ValueError("Sensitivity IDs fall outside the audited outcome population")
    representative_ids = set()
    if representative_audit is not None:
        representative_ids = set(representative_audit["spec_ids"])
        expected_representatives = {min(group["spec_ids"]) for group in audit["within_outcome_clusters"]
                                    if set(group["spec_ids"]) <= sensitivity_ids}
        if representative_ids != expected_representatives:
            raise ValueError("Representative IDs do not match the frozen lexicographic scene rule")
    points = {point["step"]: point for point in outcomes}

    def attach_independence(window: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> None:
        cluster = scene_cluster_changes(a, b, scene_of, bootstrap_samples=bootstrap_samples,
                                        seed=seed + int(window["available_at_step"]))
        window["scene_cluster_bootstrap"] = cluster
        iid = window["robustness_supported"]
        window["robustness_supported_assuming_independent_specs"] = iid
        duplicates = cluster["n_population_scene_clusters"] < cluster["n_population_specs"]
        window["robustness_support_status"] = (
            "blocked_pending_scene_adjusted_sparse_binary_inference" if duplicates
            else "conditional_on_independent_canonical_scenes")
        if duplicates:
            # Cluster percentile intervals alone do not fix all-zero binary
            # degeneracy. Do not quietly count duplicate prompts as independent
            # CP trials or promote a conditionally valid flag to robust evidence.
            window["robustness_supported"] = False

    for window in arm_report["windows"]:
        first, last = points[window["window_steps"][0]], points[window["window_steps"][-1]]
        attach_independence(window, first, last)
        subsets = [("scene_disjoint_sensitivity", sensitivity_ids)]
        if representative_audit is not None:
            subsets.append(("independent_scene_representative_sensitivity", representative_ids))
        for label, selected_ids in subsets:
            filtered = []
            for step in window["window_steps"]:
                point = points[step]
                filtered.append({**point,
                                 "complete_specs": {sid: row for sid, row in point["complete_specs"].items()
                                                    if sid in selected_ids},
                                 "spec_measurements": {sid: row for sid, row in _measurements(point).items()
                                                       if sid in selected_ids}})
            sensitivity = behavior_windows(filtered, bootstrap_samples=bootstrap_samples,
                                           seed=seed, rules=rules)[0]
            attach_independence(sensitivity, filtered[0], filtered[-1])
            sensitivity["role"] = "frozen_sensitivity_not_replacement_primary_analysis"
            sensitivity["requested_sensitivity_n"] = len(selected_ids)
            window[label] = sensitivity
    robust = next((window["available_at_step"] for window in arm_report["windows"]
                   if window["robustness_supported"]), None)
    arm_report["first_robustness_supported_step"] = robust
    arm_report["lead"] = classify_lead(robust, arm_report["gradient"]["interval_supported_alarm_step"])
    arm_report["scene_sensitivity_status"] = "supplementary; gradient-bank exclusions are not applied here"
    sensitivity_windows = [window["scene_disjoint_sensitivity"] for window in arm_report["windows"]]
    arm_report["scene_disjoint_sensitivity_summary"] = {
        "requested_specs": len(sensitivity_ids),
        "canonical_scene_clusters": len({scene_of[sid] for sid in sensitivity_ids}),
        "first_candidate_step": next((window["available_at_step"] for window in sensitivity_windows
                                      if window["candidate"]), None),
        "first_bootstrap_rule_supported_step": next((window["available_at_step"] for window in sensitivity_windows
                                                     if window["bootstrap_rule_supported"]), None),
        "first_robustness_supported_step": next((window["available_at_step"] for window in sensitivity_windows
                                                 if window["robustness_supported"]), None),
        "role": "prospectively_frozen_sensitivity_not_replacement_primary_analysis",
    }
    if representative_audit is not None:
        representative_windows = [window["independent_scene_representative_sensitivity"]
                                  for window in arm_report["windows"]]
        arm_report["independent_scene_representative_summary"] = {
            "requested_specs_and_scenes": len(representative_ids),
            "first_candidate_step": next((window["available_at_step"] for window in representative_windows
                                          if window["candidate"]), None),
            "first_bootstrap_rule_supported_step": next((window["available_at_step"] for window in representative_windows
                                                         if window["bootstrap_rule_supported"]), None),
            "first_robustness_supported_step": next((window["available_at_step"] for window in representative_windows
                                                     if window["robustness_supported"]), None),
            "role": "conditional independent-scene supplement; gradient-bank exclusions require a separate matched analysis",
        }


def behavior_windows(outcomes: Sequence[dict[str, Any]], *, bootstrap_samples: int = 2000,
                     seed: int = 20260905, rules: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rules = rules or RULES
    ordered = sorted(outcomes, key=lambda row: row["step"])
    result = []
    width = int(rules["window_checkpoints"])
    for index in range(width - 1, len(ordered)):
        window = ordered[index - width + 1:index + 1]
        changes = paired_changes(window[0], window[-1], bootstrap_samples=bootstrap_samples,
                                 seed=seed + int(window[-1]["step"]))
        internal, external = changes["s_select"], changes["external"]
        candidate = (internal["delta"] is not None and external["delta"] is not None
                     and internal["delta"] >= rules["internal_delta_min"] - NUMERIC_TOLERANCE
                     and external["delta"] <= rules["external_delta_max"] + NUMERIC_TOLERANCE)
        supported = bool(candidate and internal["ci_low"] is not None
                         and internal["ci_low"] > NUMERIC_TOLERANCE
                         and external["ci_high"] <= rules["external_delta_max"] + NUMERIC_TOLERANCE)
        robustness = supplementary_robustness(window[0], window[-1],
                                              bootstrap_samples=bootstrap_samples,
                                              seed=seed + int(window[-1]["step"]), rules=rules)
        result.append({"window_steps": [row["step"] for row in window],
                       "available_at_step": window[-1]["step"], **changes,
                       "candidate": bool(candidate), "interval_supported": supported,
                       "interval_support_scope": "legacy_alias_for_descriptive_bootstrap_rule_only",
                       "bootstrap_rule_supported": supported,
                       "bootstrap_degenerate_external_interval": bool(
                           external["ci_low"] is not None and
                           abs(external["ci_high"] - external["ci_low"]) <= NUMERIC_TOLERANCE),
                       "robustness": robustness,
                       "robustness_supported": bool(candidate and robustness["supported"]),
                       "robustness_support_status": "conditional_on_independent_spec_pairs_not_scene_audited",
                       "exploratory": True, "multiple_looks_adjusted": False})
    return result


def gradient_trajectory(reports: Sequence[dict[str, Any]], steps: Sequence[int], *,
                        rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or RULES
    by_step: dict[int, dict[str, Any]] = {}
    for row in reports:
        step = int(row["checkpoint"]["step"])
        if step in by_step:
            raise ValueError(f"Duplicate gradient checkpoint {step}")
        by_step[step] = row
    baseline = _finite(by_step.get(0, {}).get("gda_free", {}).get("cosine"))
    all_steps = sorted(set(steps) | set(by_step))
    points = []
    candidate_streak = supported_streak = 0
    candidate_alarm = supported_alarm = None
    for step in all_steps:
        row = by_step.get(step, {})
        cosine = _finite(row.get("gda_free", {}).get("cosine"))
        point_delta = cosine - baseline if cosine is not None and baseline is not None else None
        marginal_delta = point_delta
        delta = row.get("delta_from_reference", {}).get("gda_free", {})
        low, high = _finite(delta.get("ci_low")), _finite(delta.get("ci_high"))
        paired_point = _finite(delta.get("point_difference"))
        paired_ci = (delta.get("reference_step") == 0
                     and "paired" in str(delta.get("method", ""))
                     and int(delta.get("n_common", 0)) >= rules["gradient_paired_ci_min_common"]
                     and paired_point is not None and low is not None and high is not None
                     and low <= high)
        if paired_ci:
            point_delta = paired_point
        point_flag = bool(step > 0 and point_delta is not None
                          and point_delta <= -rules["gradient_drop_min"] + NUMERIC_TOLERANCE)
        supported = bool(point_flag and paired_ci and high < -NUMERIC_TOLERANCE)
        candidate_streak = candidate_streak + 1 if point_flag else 0
        supported_streak = supported_streak + 1 if supported else 0
        # Timestamp is the second confirming checkpoint, never the earlier one.
        if candidate_alarm is None and candidate_streak >= rules["gradient_consecutive_checkpoints"]:
            candidate_alarm = step
        if supported_alarm is None and supported_streak >= rules["gradient_consecutive_checkpoints"]:
            supported_alarm = step
        points.append({"step": step, "cosine": cosine, "delta_from_step0": point_delta,
                       "marginal_point_delta": marginal_delta,
                       "paired_ci_low": low if paired_ci else None,
                       "paired_ci_high": high if paired_ci else None,
                       "n_common": int(delta.get("n_common", 0)),
                       "point_flag": point_flag, "interval_supported_flag": supported,
                       "support_status": ("paired_difference_ci" if paired_ci
                                          else "unsupported_no_paired_difference_ci"),
                       "missing_report": step not in by_step})
    return {"baseline_cosine": baseline, "points": points,
            "candidate_alarm_step": candidate_alarm,
            "interval_supported_alarm_step": supported_alarm,
            "alarm_timestamp": "final_checkpoint_of_required_consecutive_flags",
            "warning": "marginal cosine intervals are never subtracted to obtain a difference CI"}


def classify_lead(event_step: int | None, alarm_step: int | None) -> dict[str, Any]:
    if event_step is None:
        status = "no_event"
    elif alarm_step is None:
        status = "missed"
    elif alarm_step < event_step:
        status = "earlier"
    elif alarm_step == event_step:
        status = "simultaneous"
    else:
        status = "late"
    return {"status": status, "event_step": event_step, "alarm_step": alarm_step,
            "lead_steps": event_step - alarm_step
            if event_step is not None and alarm_step is not None else None,
            "interpretation": "descriptive as of the currently available checkpoints"}


def analyze_arm(outcomes: Sequence[dict[str, Any]], gradients: Sequence[dict[str, Any]], *,
                bootstrap_samples: int = 2000, seed: int = 20260905,
                rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or RULES
    windows = behavior_windows(outcomes, bootstrap_samples=bootstrap_samples, seed=seed, rules=rules)
    candidate = next((w["available_at_step"] for w in windows if w["candidate"]), None)
    supported = next((w["available_at_step"] for w in windows if w["interval_supported"]), None)
    robust = next((w["available_at_step"] for w in windows if w["robustness_supported"]), None)
    trajectory = gradient_trajectory(gradients, [row["step"] for row in outcomes], rules=rules)
    return {"status": "insufficient_checkpoints"
            if len(outcomes) < rules["window_checkpoints"] else "descriptive",
            "checkpoints": [{k: v for k, v in row.items()
                             if k not in ("complete_specs", "spec_measurements")}
                            for row in sorted(outcomes, key=lambda row: row["step"])],
            "windows": windows, "first_candidate_step": candidate,
            "first_interval_supported_step": supported,
            "interval_support_scope": "legacy_alias_for_descriptive_bootstrap_rule_only",
            "first_bootstrap_rule_supported_step": supported,
            "first_robustness_supported_step": robust, "gradient": trajectory,
            "bootstrap_rule_lead": classify_lead(supported, trajectory["interval_supported_alarm_step"]),
            "lead": classify_lead(robust, trajectory["interval_supported_alarm_step"]),
            "lead_reference": "supplementary_robustness_screened_behavior_and_paired_gradient_candidates",
            "point_only_lead": classify_lead(candidate, trajectory["candidate_alarm_step"]),
            "point_only_lead_is_supported": False}


def build_report(outdir: Path, *, bootstrap_samples: int = 2000, seed: int = 20260905,
                 config_path: Path | None = None, protocol_path: Path | None = None) -> dict[str, Any]:
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gradients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path else {}
    rules = configured_rules(config)
    audit_path = outdir / "audit-splits" / "scene_overlap.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    representative_path = outdir / "audit-splits" / "scene_representatives.json"
    representative_audit = (json.loads(representative_path.read_text(encoding="utf-8"))
                            if representative_path.exists() else None)
    provenance = {"config": file_identity(config_path), "protocol": file_identity(protocol_path),
                  "scene_overlap": file_identity(audit_path) if audit is not None else None,
                  "scene_representatives": file_identity(representative_path) if representative_audit is not None else None}
    prior_path = outdir / "decoupling_report.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        for key in ("config", "protocol", "scene_overlap", "scene_representatives"):
            old = prior.get("provenance", {}).get(key)
            new = provenance[key]
            if old and (not new or old["sha256"] != new["sha256"]):
                raise ValueError(f"Frozen {key} changed; use a new run version")
        if prior.get("rules", rules) != rules:
            raise ValueError("Frozen reporting rules changed; use a new run version")
    split_path = outdir / "split.json"
    expected_outcomes = (json.loads(split_path.read_text(encoding="utf-8"))["outcome"]
                         if split_path.exists() else None)
    if audit is not None:
        if not audit.get("created_before_any_outcome_evaluation_artifact"):
            raise ValueError("Scene sensitivity audit is not marked as prospectively frozen")
        audit_provenance = audit.get("provenance", {})
        for key, identity in (("config_sha256", provenance["config"]),
                              ("split_sha256", file_identity(split_path) if split_path.exists() else None)):
            if audit_provenance.get(key) and (not identity or audit_provenance[key] != identity["sha256"]):
                raise ValueError(f"Scene audit {key} does not match this run")
        # An audit is a claim about the run it was built in, and copying one
        # between runs is how a post hoc subset becomes a prospective freeze:
        # two runs off the same config share a config digest and a split
        # digest, so nothing checked above tells them apart. Compared by
        # directory name, not by full path, because moving a run must not make
        # its audit unreadable -- once outcomes exist it cannot be rebuilt.
        recorded_run = audit_provenance.get("run")
        if recorded_run and Path(recorded_run).name != outdir.resolve().name:
            raise ValueError(
                f"Scene audit was built in {recorded_run}, not {outdir}; an audit copied "
                f"from another run of the same config clears every other check here and "
                f"would freeze this run's outcome subsets on that run's authority")
        audited_ids = [sid for group in audit["within_outcome_clusters"] for sid in group["spec_ids"]]
        if len(audited_ids) != len(set(audited_ids)):
            raise ValueError("Scene audit repeats an outcome spec")
        if expected_outcomes is not None and set(audited_ids) != set(expected_outcomes):
            raise ValueError("Scene audit outcome population does not match frozen split")
    if representative_audit is not None:
        if (audit is None or not representative_audit.get("created_before_any_outcome_evaluation_artifact")
                or representative_audit.get("parent_scene_audit_sha256") != provenance["scene_overlap"]["sha256"]):
            raise ValueError("Representative audit does not match the prospectively frozen parent audit")
        ids = representative_audit["spec_ids"]
        digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        if len(ids) != len(set(ids)) or digest != representative_audit.get("spec_ids_sha256"):
            raise ValueError("Representative spec IDs do not match their frozen hash")
    for directory in sorted((outdir / "evaluations").glob("*/step-*")):
        if directory.is_dir():
            outcome = read_outcome(directory, expected_spec_ids=expected_outcomes)
            outcomes[outcome["arm"]].append(outcome)
    probe_paths = sorted({path for name in ("gradient-probes", "probes")
                          for path in (outdir / name).glob("**/report.json")})
    for path in probe_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if "checkpoint" not in row or "gda_free" not in row:
            continue
        arm = row["checkpoint"].get("arm")
        if arm is None:
            raise ValueError(f"Gradient report has no checkpoint.arm: {path}")
        gradients[str(arm)].append(row)
    shared_base = gradients.pop("base", [])
    if len(shared_base) > 1 or any(int(row["checkpoint"]["step"]) != 0 for row in shared_base):
        raise ValueError("The shared base must be one step-0 gradient report")
    arms = sorted(set(outcomes) | set(gradients))
    for arm in arms:
        if shared_base and not any(int(row["checkpoint"]["step"]) == 0 for row in gradients[arm]):
            gradients[arm] = [*shared_base, *gradients[arm]]
    arm_reports = {arm: analyze_arm(outcomes[arm], gradients[arm],
                                   bootstrap_samples=bootstrap_samples, seed=seed, rules=rules)
                   for arm in arms}
    if audit is not None:
        for arm in arms:
            add_scene_sensitivity(arm_reports[arm], outcomes[arm], audit,
                                  bootstrap_samples=bootstrap_samples, seed=seed, rules=rules,
                                  representative_audit=representative_audit)
    return {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
            "outdir": str(outdir), "exploratory": True, "multiple_looks_adjusted": False,
            "rules": rules, "provenance": provenance,
            "reporting_repair": "registered bootstrap rules retained; supplemental sparse-pair and missingness checks",
            "numeric_comparison_tolerance": NUMERIC_TOLERANCE,
            "bootstrap_seed": seed, "limitations": LIMITATIONS,
            "scene_audit_status": "prospectively_frozen_supplement_loaded" if audit is not None
                                  else "absent; spec independence is an unchecked assumption",
            "scene_audit_summary": audit.get("summary") if audit is not None else None,
            "arms": arm_reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if not args.outdir.is_dir():
        parser.error("--outdir must be an existing pilot directory")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_report(args.outdir,
                          bootstrap_samples=int(config.get("gradient_probe", {}).get("resamples", 2000)),
                          seed=int(config["seed"]), config_path=args.config, protocol_path=args.protocol)
    output = args.outdir / "decoupling_report.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(f"wrote {output}; exploratory, multiple looks not adjusted")
    for arm, row in report["arms"].items():
        print(f"{arm}: behavioral candidate={row['first_candidate_step']}, "
              f"descriptive bootstrap rule={row['first_bootstrap_rule_supported_step']}, "
              f"supplementary robustness={row['first_robustness_supported_step']}, "
              f"gradient alarm={row['gradient']['interval_supported_alarm_step']}, "
              f"descriptive lead={row['lead']['status']}")


if __name__ == "__main__":
    main()
