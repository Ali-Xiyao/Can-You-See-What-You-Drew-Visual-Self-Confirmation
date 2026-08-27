"""Pre-specified continuous segmented regressions for exploratory D* and D_g."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from selfsight.training.gradients import exponential_moving_average


@dataclass(frozen=True)
class SegmentedFit:
    breakpoint: float
    intercept: float
    slope_before: float
    slope_after: float
    sse: float
    candidate_breakpoints: tuple[float, ...]
    profile_sse: tuple[float, ...]


@dataclass(frozen=True)
class DivergenceEstimate:
    d_star: float | None
    fit_internal: SegmentedFit | None
    fit_external: SegmentedFit | None
    internal_post_slope: float | None
    external_post_slope: float | None
    reason: str


@dataclass(frozen=True)
class GradientWarningEstimate:
    d_g: float | None
    fit: SegmentedFit | None
    noise_low: float
    early_safe: bool
    reason: str


def _design(x: np.ndarray, breakpoint: float) -> np.ndarray:
    return np.column_stack((np.ones_like(x), x, np.maximum(0.0, x - breakpoint)))


def _fit_at(x: np.ndarray, y: np.ndarray, breakpoint: float) -> tuple[np.ndarray, float]:
    design = _design(x, breakpoint)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return coefficients, float(np.dot(residual, residual))


def fit_segmented(
    x: Sequence[float],
    y: Sequence[float],
    *,
    min_points_each_side: int = 3,
) -> SegmentedFit:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    order = np.argsort(x_array)
    x_array, y_array = x_array[order], y_array[order]
    if x_array.size < min_points_each_side * 2 + 1:
        raise ValueError("Not enough checkpoints for segmented regression")
    candidates = np.unique(x_array)[min_points_each_side:-min_points_each_side]
    if candidates.size == 0:
        raise ValueError("No admissible breakpoint candidates")
    profile = []
    coefficients_by_break = []
    for candidate in candidates:
        coefficients, sse = _fit_at(x_array, y_array, float(candidate))
        coefficients_by_break.append(coefficients)
        profile.append(sse)
    best_index = int(np.argmin(profile))
    coefficients = coefficients_by_break[best_index]
    slope_before = float(coefficients[1])
    slope_after = float(coefficients[1] + coefficients[2])
    return SegmentedFit(
        breakpoint=float(candidates[best_index]),
        intercept=float(coefficients[0]),
        slope_before=slope_before,
        slope_after=slope_after,
        sse=float(profile[best_index]),
        candidate_breakpoints=tuple(float(item) for item in candidates),
        profile_sse=tuple(float(item) for item in profile),
    )


def estimate_d_star(
    steps: Sequence[float],
    internal: Sequence[float],
    external: Sequence[float],
    *,
    min_positive_internal_slope: float = 0.0,
) -> DivergenceEstimate:
    try:
        internal_fit = fit_segmented(steps, internal)
        external_fit = fit_segmented(steps, external)
    except ValueError as exc:
        return DivergenceEstimate(None, None, None, None, None, str(exc))
    candidates = sorted(set(internal_fit.candidate_breakpoints).intersection(external_fit.candidate_breakpoints))
    best: tuple[float, SegmentedFit, SegmentedFit] | None = None
    for candidate in candidates:
        x = np.asarray(steps, dtype=float)
        internal_coefficients, internal_sse = _fit_at(x, np.asarray(internal, dtype=float), candidate)
        external_coefficients, external_sse = _fit_at(x, np.asarray(external, dtype=float), candidate)
        internal_candidate = SegmentedFit(
            candidate,
            float(internal_coefficients[0]),
            float(internal_coefficients[1]),
            float(internal_coefficients[1] + internal_coefficients[2]),
            internal_sse,
            (),
            (),
        )
        external_candidate = SegmentedFit(
            candidate,
            float(external_coefficients[0]),
            float(external_coefficients[1]),
            float(external_coefficients[1] + external_coefficients[2]),
            external_sse,
            (),
            (),
        )
        if internal_candidate.slope_after <= min_positive_internal_slope:
            continue
        if external_candidate.slope_after > 0.0:
            continue
        objective = internal_sse + external_sse
        if best is None or objective < best[0]:
            best = (objective, internal_candidate, external_candidate)
    if best is None:
        return DivergenceEstimate(
            None,
            internal_fit,
            external_fit,
            internal_fit.slope_after,
            external_fit.slope_after,
            "No shared breakpoint satisfies internal post-slope > 0 and external post-slope <= 0",
        )
    _, internal_best, external_best = best
    return DivergenceEstimate(
        d_star=internal_best.breakpoint,
        fit_internal=internal_best,
        fit_external=external_best,
        internal_post_slope=internal_best.slope_after,
        external_post_slope=external_best.slope_after,
        reason="exploratory local estimate; formal inference requires seed-level bootstrap",
    )


def estimate_d_g(
    steps: Sequence[float],
    gda_free: Sequence[float],
    *,
    noise_low: float | Sequence[float],
    noise_high: float | Sequence[float],
    ema_alpha: float = 0.35,
    persistence: int = 2,
) -> GradientWarningEstimate:
    smoothed = np.asarray(exponential_moving_average(gda_free, ema_alpha), dtype=float)
    steps_array = np.asarray(steps, dtype=float)
    low_array = np.broadcast_to(np.asarray(noise_low, dtype=float), smoothed.shape)
    high_array = np.broadcast_to(np.asarray(noise_high, dtype=float), smoothed.shape)
    early_count = max(1, int(np.ceil(len(smoothed) * 0.10)))
    early_safe = bool(
        np.all(
            (smoothed[:early_count] >= low_array[:early_count])
            & (smoothed[:early_count] <= high_array[:early_count])
        )
    )
    try:
        fit = fit_segmented(steps_array, smoothed)
    except ValueError as exc:
        return GradientWarningEstimate(None, None, float(low_array[0]), early_safe, str(exc))
    below = smoothed < low_array
    persistent_candidates = [
        index
        for index in range(len(smoothed) - persistence + 1)
        if bool(np.all(below[index : index + persistence]))
    ]
    eligible = [index for index in persistent_candidates if steps_array[index] >= fit.breakpoint]
    if fit.slope_after >= 0.0 or not eligible:
        return GradientWarningEstimate(
            None,
            fit,
            float(low_array[0]),
            early_safe,
            "GDA-free does not persist below the noise floor with a negative post-break slope",
        )
    index = eligible[0]
    return GradientWarningEstimate(
        float(steps_array[index]),
        fit,
        float(low_array[0]),
        early_safe,
        "exploratory local estimate; formal inference requires seed-level bootstrap",
    )


def estimate_lead(d_star: float | None, d_g: float | None) -> float | None:
    return None if d_star is None or d_g is None else float(d_star - d_g)
