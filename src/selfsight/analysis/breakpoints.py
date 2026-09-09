"""Pre-specified continuous segmented regressions for exploratory D* and D_g."""

from __future__ import annotations

import math
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
class BreakSupport:
    reduction: float
    p_value: float
    sse_null: float
    sse_break: float
    breakpoint: float
    resamples: int


@dataclass(frozen=True)
class DivergenceEstimate:
    d_star: float | None
    fit_internal: SegmentedFit | None
    fit_external: SegmentedFit | None
    internal_post_slope: float | None
    external_post_slope: float | None
    reason: str
    external_pre_slope: float | None = None
    coupled_candidates: int = 0
    admissible_candidates: int = 0
    break_support: BreakSupport | None = None


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


def break_support(
    x: Sequence[float],
    y: Sequence[float],
    *,
    min_points_each_side: int = 3,
    resamples: int = 2000,
    seed: int = 20260908,
) -> BreakSupport:
    """Would a straight line with this much noise have produced a knot this good?

    `fit_segmented` minimises SSE over candidate knots and returns the winner
    unconditionally, with nothing compared against a no-break alternative. The
    knot search is the problem, not the fit: searching more candidates finds a
    better apparent bend in pure noise, so on a flat curve the false-positive
    rate *rises* with the number of checkpoints.

    The null is refit here rather than assumed. Residuals from the straight-line
    fit are resampled (rescaled by sqrt(n/(n-2)), since residuals are shrunk
    relative to the errors they estimate), added back to the fitted line, and
    the **same knot search** is rerun on each surrogate. Calibrating the search
    by running the search is the whole point; an F-test against chi-square
    quantiles would not, because the knot is not identified under the null.

    The statistic is the scale-free SSE reduction `1 - sse_break/sse_null`, and
    the p-value carries the usual +1 so it is never exactly zero.

    This is an instrument check, not a hypothesis test of the science: it asks
    whether the estimator found structure or found noise. Passing it does not
    make a single-seed descriptive trajectory a significance result.
    """

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    order = np.argsort(x_array)
    x_array, y_array = x_array[order], y_array[order]

    fit = fit_segmented(x_array, y_array, min_points_each_side=min_points_each_side)
    line = np.column_stack((np.ones_like(x_array), x_array))
    coefficients, *_ = np.linalg.lstsq(line, y_array, rcond=None)
    residuals = y_array - line @ coefficients
    sse_null = float(np.dot(residuals, residuals))

    def reduction(sse_break: float, sse_line: float) -> float:
        return 0.0 if sse_line <= 0.0 else 1.0 - sse_break / sse_line

    observed = reduction(fit.sse, sse_null)
    size = x_array.size
    scaled = residuals * math.sqrt(size / (size - 2)) if size > 2 else residuals
    fitted = line @ coefficients
    generator = np.random.default_rng(seed)
    at_least = 0
    for _ in range(resamples):
        surrogate = fitted + generator.choice(scaled, size=size, replace=True)
        surrogate_line, *_ = np.linalg.lstsq(line, surrogate, rcond=None)
        surrogate_residual = surrogate - line @ surrogate_line
        surrogate_null = float(np.dot(surrogate_residual, surrogate_residual))
        surrogate_fit = fit_segmented(
            x_array, surrogate, min_points_each_side=min_points_each_side
        )
        if reduction(surrogate_fit.sse, surrogate_null) >= observed:
            at_least += 1
    return BreakSupport(
        reduction=observed,
        p_value=(1 + at_least) / (1 + resamples),
        sse_null=sse_null,
        sse_break=fit.sse,
        breakpoint=fit.breakpoint,
        resamples=resamples,
    )


def estimate_d_star(
    steps: Sequence[float],
    internal: Sequence[float],
    external: Sequence[float],
    *,
    min_positive_internal_slope: float = 0.0,
    min_coupled_external_slope: float = 0.0,
    max_break_p_value: float | None = None,
    break_resamples: int = 2000,
    break_seed: int = 20260908,
) -> DivergenceEstimate:
    """The breakpoint where internal keeps rising and external stops improving.

    Four conditions, all necessary. The first is the precondition: "training
    caused decoupling" is only meaningful if something was coupled beforehand,
    so the external curve has to have been rising into the knot. Without it a
    curve that was flat from step 0 -- one the internal score never predicted --
    is scored as a decoupling event the moment the internal curve bends upward,
    and so is a curve that only ever declined (measured: both return a confident
    D* of 32 on noiseless synthetic data; see
    review-packets/bon-coupling-20260908/dstar-precondition-defect.json).

    `min_coupled_external_slope` is that precondition's floor and zero is not a
    floor, for the same reason `min_positive_internal_slope` documents: a flat
    curve fits at a slope of order 1e-05 and clears it. The defensible floor is
    built from how precisely the external curve was measured -- see
    `selfsight.v4.evaluate.external_noise_slope`, which `divergence_report`
    supplies. The estimator's own default reproduces the sign test only.

    The fourth condition is `max_break_p_value`, which sends the external curve
    through `break_support`. The three slope conditions are marginal checks
    evaluated at a knot the search chose, so between them they do not ask
    whether the bend beat noise: measured on a flat external curve with binomial
    noise at n=120 prompts, this function returns a false D* 32% of the time at
    8 checkpoints and 54% at 40 -- rising with the number of checkpoints,
    because a wider knot search finds a better apparent bend in noise. More data
    does not rescue it. Leaving it None reproduces that behaviour;
    `selfsight.v4.evaluate.divergence_report` supplies the registered value.

    The test runs on the external curve's own SSE-optimal knot, not on the
    shared knot the joint search settles on. That is deliberate and conservative
    in the right direction: the external-optimal knot is the best case, so a
    curve that fails there has no supported break at any knot. The external
    curve is also the right one to test, because D* is a claim that external
    improvement *stopped* -- with no bend in it there is no such moment,
    whatever the internal curve does.

    Adding a necessary condition can only withdraw D* estimates, never create
    one, so no previously reported estimate can be raised by this.
    """

    try:
        internal_fit = fit_segmented(steps, internal)
        external_fit = fit_segmented(steps, external)
    except ValueError as exc:
        return DivergenceEstimate(None, None, None, None, None, str(exc))
    support = None
    if max_break_p_value is not None:
        support = break_support(steps, external, resamples=break_resamples, seed=break_seed)
        if support.p_value > max_break_p_value:
            return DivergenceEstimate(
                None,
                internal_fit,
                external_fit,
                internal_fit.slope_after,
                external_fit.slope_after,
                (
                    f"No supported break: the external curve's knot cuts SSE by "
                    f"{support.reduction:.1%}, which a straight line with the same noise "
                    f"reaches in {support.p_value:.1%} of {support.resamples} surrogates "
                    f"(threshold {max_break_p_value:g}). The knot search finds a bend in "
                    "noise, so there is no moment at which improvement can be said to stop."
                ),
                external_pre_slope=external_fit.slope_before,
                admissible_candidates=len(external_fit.candidate_breakpoints),
                break_support=support,
            )
    candidates = sorted(set(internal_fit.candidate_breakpoints).intersection(external_fit.candidate_breakpoints))
    best: tuple[float, SegmentedFit, SegmentedFit] | None = None
    coupled = 0
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
        if external_candidate.slope_before <= min_coupled_external_slope:
            continue
        coupled += 1
        if internal_candidate.slope_after <= min_positive_internal_slope:
            continue
        if external_candidate.slope_after > 0.0:
            continue
        objective = internal_sse + external_sse
        if best is None or objective < best[0]:
            best = (objective, internal_candidate, external_candidate)
    if best is None:
        if coupled == 0:
            reason = (
                "No coupled phase: the external curve does not rise faster than "
                f"{min_coupled_external_slope:g} before any of the {len(candidates)} admissible "
                "breakpoints, so there is no coupling for training to break. D* is undefined "
                "here, which is not the same as a decoupling that was looked for and not found."
            )
        else:
            reason = (
                f"{coupled} of {len(candidates)} admissible breakpoints have a coupled phase, but "
                f"none of those also has internal post-slope > {min_positive_internal_slope:g} "
                "with external post-slope <= 0"
            )
        return DivergenceEstimate(
            None,
            internal_fit,
            external_fit,
            internal_fit.slope_after,
            external_fit.slope_after,
            reason,
            external_pre_slope=external_fit.slope_before,
            coupled_candidates=coupled,
            admissible_candidates=len(candidates),
            break_support=support,
        )
    _, internal_best, external_best = best
    return DivergenceEstimate(
        d_star=internal_best.breakpoint,
        fit_internal=internal_best,
        fit_external=external_best,
        internal_post_slope=internal_best.slope_after,
        external_post_slope=external_best.slope_after,
        reason="exploratory local estimate; formal inference requires seed-level bootstrap",
        external_pre_slope=external_best.slope_before,
        coupled_candidates=coupled,
        admissible_candidates=len(candidates),
        break_support=support,
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
