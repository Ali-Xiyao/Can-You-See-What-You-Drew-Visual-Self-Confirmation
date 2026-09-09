"""Tests for E3 endpoint 2.

Like the endpoint 1 tests, most of these hold a sentence of the
pre-registration in place rather than a property of the code, so each one
names the clause it is pinning. Deviation 13.1 is seven numbered rules and
every one of them has a test here; the ones with no test would be the ones
free to drift.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from selfsight.analysis.endpoint2 import (
    A_SLOPE_MAJORITY,
    BLIND_CONDITION,
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    MIN_CHECKPOINTS_PER_FIT,
    REGISTERED_SEED_COUNT,
    AcrossSeeds,
    BlindPassMissing,
    BlindSeries,
    SeedVerdict,
    across_seeds,
    available_steps,
    gap_series,
    load_series,
    ols_slope,
    paired_bootstrap,
    seed_verdict,
)

STEPS = (0, 8, 16, 24)


def _series(arm: str, gaps: list[float], *, steps: tuple[int, ...] = STEPS,
            prompts: int = 8) -> BlindSeries:
    """A series whose gap at checkpoint j is exactly `gaps[j]`.

    Half the prompts are labelled right and half wrong; the right half scores
    `gap` and the wrong half scores 0, so the mean difference is the gap and
    the fitted slope is whatever the caller asked for.
    """

    names = tuple(f"p{index:02d}" for index in range(prompts))
    scores = np.zeros((prompts, len(steps)))
    labels = np.zeros((prompts, len(steps)), dtype=int)
    for index in range(prompts):
        right = index < prompts // 2
        labels[index, :] = 1 if right else 0
        if right:
            scores[index, :] = np.array(gaps)
    return BlindSeries(arm=arm, steps=steps, prompts=names, scores=scores, labels=labels)


def _varied_series(arm: str, gaps: list[float], weights: list[float],
                   *, steps: tuple[int, ...] = STEPS) -> BlindSeries:
    """Right prompt i scales the whole gap trajectory by `weights[i]`.

    `_series` gives every right prompt the same score, so its bootstrap
    distribution is a point mass and every percentile of it is the same
    number. That hides the difference between reading the 97.5th percentile,
    the median and the point estimate -- which is precisely what deviation
    13.1 point 2 pins down -- so the percentile tests need a series whose
    slope actually moves when the prompts are resampled. A negative weight is
    a prompt whose gap widens while the arm's average gap narrows.
    """

    names = tuple(f"p{index:02d}" for index in range(2 * len(weights)))
    scores = np.zeros((len(names), len(steps)))
    labels = np.zeros((len(names), len(steps)), dtype=int)
    for index, weight in enumerate(weights):
        labels[index, :] = 1
        scores[index, :] = np.array(gaps) * weight
    return BlindSeries(arm=arm, steps=steps, prompts=names, scores=scores, labels=labels)


def _write_blind(run: Path, arm: str, step: int, rows: list[dict]) -> None:
    directory = run / "analysis" / "blind_observe" / arm
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"step-{step:05d}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _blind_row(prompt: str, score: float, **overrides: object) -> dict:
    row = {"prompt_id": prompt, "candidate_index": 0,
           "condition": BLIND_CONDITION, "s_select": score}
    row.update(overrides)
    return row


def _write_verified(run: Path, arm: str, step: int,
                    verdicts: list[tuple[str, int, bool | None]]) -> None:
    directory = run / "evaluations" / arm / f"step-{step:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec, draw, correct in verdicts:
        if correct is None:
            rows.append({"spec_id": spec, "candidate_index": draw,
                         "image_correct": None, "resolution": "pending_human"})
        else:
            rows.append({"spec_id": spec, "candidate_index": draw,
                         "image_correct": correct, "resolution": "agreed"})
    (directory / "verified.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# --- the constants deviation 13.1 point 3 registers -------------------------


def test_the_final_analysis_constants_are_the_registered_ones():
    # Deviation 13.1 point 3: the confirmatory slope fit takes deviation 6.4's
    # final-analysis branch, not the running report's 2000 / 20260906.
    assert FINAL_RESAMPLES == 20000
    assert FINAL_BOOTSTRAP_SEED == 20260908


def test_the_majority_rule_is_four_of_five():
    # Deviation 9.4, fixed before any seed ran.
    assert A_SLOPE_MAJORITY == 4
    assert REGISTERED_SEED_COUNT == 5


# --- the gap and the fit ----------------------------------------------------


def test_the_gap_is_right_minus_wrong():
    series = _series("naive", [0.3, 0.2, 0.1, 0.0])
    gaps = gap_series(series, np.arange(len(series.prompts)))
    assert gaps == pytest.approx([0.3, 0.2, 0.1, 0.0])


def test_the_abscissa_is_the_step_not_the_checkpoint_index():
    # Deviation 13.1 point 5. Equally spaced steps make the two readings
    # proportional, so the test uses unequal spacing where they differ.
    steps = np.array([0, 8, 40])
    gaps = np.array([0.3, 0.2, 0.1])
    by_step = ols_slope(steps, gaps)
    by_index = ols_slope(np.array([0, 1, 2]), gaps)
    # -4.0 / 896, worked by hand off centred x = [-16, -8, 24].
    assert by_step == pytest.approx(-4.0 / 896)
    assert by_index == pytest.approx(-0.1)
    assert by_step != pytest.approx(by_index)


def test_a_checkpoint_with_one_side_empty_has_no_gap():
    """Deviation 13.1 point 4: undefined, not zero, and guarded explicitly.

    Warnings are errors here on purpose. `wrong.mean()` on an empty array
    already returns NaN, so a version that computed the difference anyway
    would give the same number while emitting "Mean of empty slice" 20000
    times per seed -- and would break outright under an errstate that raises.
    The guard has to be the thing producing the NaN.
    """

    series = _series("naive", [0.3, 0.2, 0.1, 0.0])
    only_right = np.arange(len(series.prompts) // 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaps = gap_series(series, only_right)
    assert np.all(np.isnan(gaps))


def test_a_resampled_prompt_counts_as_many_times_as_it_was_drawn():
    # The bootstrap resamples with replacement, so gap_series has to weight by
    # multiplicity. Taking the set of drawn prompts instead would make every
    # resample identical to the point estimate.
    series = _series("naive", [0.4, 0.4, 0.4, 0.4], prompts=4)
    # Two right prompts scoring 0.4, one wrong prompt scoring 0 -> 0.4.
    balanced = gap_series(series, np.array([0, 1, 2, 3]))
    # One right, three wrong -> still 0.4 because the sides are averaged, but
    # a score that varied by prompt would differ. Vary one.
    series.scores[0, :] = 1.0
    lopsided = gap_series(series, np.array([0, 0, 0, 2]))
    assert balanced == pytest.approx([0.4] * 4)
    assert lopsided == pytest.approx([1.0] * 4)


def test_a_fit_on_fewer_than_three_checkpoints_is_refused():
    # Deviation 13.1 point 4's second clause.
    steps = np.array([0, 8, 16, 24])
    gaps = np.array([0.3, np.nan, np.nan, 0.0])
    assert math.isnan(ols_slope(steps, gaps))
    assert MIN_CHECKPOINTS_PER_FIT == 3


def test_the_fit_drops_undefined_checkpoints_rather_than_the_whole_series():
    steps = np.array([0, 8, 16, 24])
    with_hole = ols_slope(steps, np.array([0.3, np.nan, 0.1, 0.0]))
    without = ols_slope(np.array([0, 16, 24]), np.array([0.3, 0.1, 0.0]))
    assert with_hole == pytest.approx(without)
    assert not math.isnan(with_hole)


# --- the bootstrap ----------------------------------------------------------


def test_both_arms_see_the_same_resampled_prompts():
    """Deviation 13.1 point 1, the whole reason the difference is paired.

    Two arms with identical data must give a difference distribution that is
    exactly zero. Under independent resampling per arm it would be a spread
    around zero instead, and the 2.5th percentile would sit below it.
    """

    series = _series("naive", [0.3, 0.2, 0.1, 0.0])
    twin = _series("blind_self", [0.3, 0.2, 0.1, 0.0])
    draws = paired_bootstrap(series, twin, resamples=200, seed=1)
    assert np.all(draws.difference == 0.0)


def test_the_bootstrap_is_reproducible_from_its_seed():
    series = _series("naive", [0.3, 0.2, 0.1, 0.0])
    twin = _series("blind_self", [0.30, 0.25, 0.24, 0.23])
    first = paired_bootstrap(series, twin, resamples=100, seed=7)
    second = paired_bootstrap(series, twin, resamples=100, seed=7)
    other = paired_bootstrap(series, twin, resamples=100, seed=8)
    assert np.array_equal(first.difference, second.difference)
    assert not np.array_equal(first.difference, other.difference)


def test_arms_scored_on_different_prompts_are_refused():
    left = _series("naive", [0.3, 0.2, 0.1, 0.0], prompts=8)
    right = _series("blind_self", [0.3, 0.2, 0.1, 0.0], prompts=6)
    with pytest.raises(ValueError, match="same prompts"):
        paired_bootstrap(left, right, resamples=10, seed=1)


def test_arms_scored_at_different_steps_are_refused():
    left = _series("naive", [0.3, 0.2, 0.1, 0.0])
    right = _series("blind_self", [0.3, 0.2, 0.1], steps=(0, 8, 16))
    with pytest.raises(ValueError, match="same steps"):
        paired_bootstrap(left, right, resamples=10, seed=1)


def test_a_degenerate_resample_is_discarded_and_counted():
    """Deviation 13.1 point 4: both counts are reported, not swallowed.

    Two prompts, one right and one wrong, makes an all-right or all-wrong
    resample common enough to hit in a hundred draws.
    """

    left = _series("naive", [0.3, 0.2, 0.1, 0.0], prompts=2)
    right = _series("blind_self", [0.3, 0.2, 0.1, 0.0], prompts=2)
    draws = paired_bootstrap(left, right, resamples=100, seed=3)
    assert draws.discarded_resamples > 0
    assert draws.dropped_checkpoints > 0
    assert len(draws.arm_a) == 100 - draws.discarded_resamples


def test_a_bootstrap_that_discards_everything_raises():
    # One prompt: every resample is all-right, so no gap is ever defined.
    names = ("p00",)
    scores = np.ones((1, len(STEPS)))
    labels = np.ones((1, len(STEPS)), dtype=int)
    left = BlindSeries("naive", STEPS, names, scores, labels)
    right = BlindSeries("blind_self", STEPS, names, scores.copy(), labels.copy())
    with pytest.raises(ValueError, match="discarded"):
        paired_bootstrap(left, right, resamples=10, seed=1)


# --- the per-seed verdict ---------------------------------------------------


def test_a_collapsing_a_and_a_flat_b_confirms_the_seed():
    left = _series("naive", [0.30, 0.20, 0.10, 0.00])
    right = _series("blind_self", [0.30, 0.30, 0.30, 0.30])
    draws = paired_bootstrap(left, right, resamples=400, seed=11)
    result = seed_verdict(left, right, draws, seed=20260906)
    assert result.slope_a == pytest.approx(-0.0125)
    assert result.slope_b == pytest.approx(0.0)
    assert result.a_collapses
    assert result.b_exceeds_a
    assert result.confirmed


def test_an_a_that_does_not_collapse_fails_the_first_clause():
    left = _series("naive", [0.30, 0.30, 0.30, 0.30])
    right = _series("blind_self", [0.30, 0.35, 0.40, 0.45])
    draws = paired_bootstrap(left, right, resamples=400, seed=11)
    result = seed_verdict(left, right, draws, seed=20260906)
    assert not result.a_collapses
    assert result.b_exceeds_a
    assert not result.confirmed


def test_a_b_no_better_than_a_fails_the_second_clause():
    left = _series("naive", [0.30, 0.20, 0.10, 0.00])
    right = _series("blind_self", [0.30, 0.20, 0.10, 0.00])
    draws = paired_bootstrap(left, right, resamples=400, seed=11)
    result = seed_verdict(left, right, draws, seed=20260906)
    assert result.a_collapses
    assert not result.b_exceeds_a
    assert not result.confirmed


def test_significance_is_read_off_the_registered_percentiles():
    """Deviation 13.1 point 2, read against the draws themselves.

    The verdict has to equal the percentile condition and not some other
    summary of the same draws -- the median, or the point estimate, both of
    which are different and much weaker claims. The series is `_varied_` so
    that those three numbers are actually distinct.
    """

    weights = [0.4, 0.8, 1.2, 1.6]
    left = _varied_series("naive", [0.30, 0.20, 0.10, 0.00], weights)
    right = _varied_series("blind_self", [0.30, 0.30, 0.30, 0.30], weights)
    draws = paired_bootstrap(left, right, resamples=600, seed=11)
    result = seed_verdict(left, right, draws, seed=20260906)
    assert result.a_interval[1] == pytest.approx(float(np.percentile(draws.arm_a, 97.5)))
    assert result.a_interval[1] != pytest.approx(float(np.percentile(draws.arm_a, 50.0)))
    assert result.difference_interval[0] == pytest.approx(
        float(np.percentile(draws.difference, 2.5)))
    assert result.difference_interval[0] != pytest.approx(result.difference_interval[1])
    assert result.a_collapses == (result.a_interval[1] < 0.0)
    assert result.b_exceeds_a == (result.difference_interval[0] > 0.0)


def test_a_slope_negative_on_average_but_uncertain_does_not_collapse():
    """Deviation 13.1 point 2 again, on the case where the two readings part.

    One prompt's gap widens hard while the arm's average gap narrows. The
    point estimate is negative; the bootstrap says the sign is not settled.
    The registered clause is the interval one, so this seed does not count
    towards deviation 9.4's 4 of 5.
    """

    weights = [-2.0, 0.5, 1.0, 1.5]
    left = _varied_series("naive", [0.30, 0.20, 0.10, 0.00], weights)
    right = _varied_series("blind_self", [0.30, 0.30, 0.30, 0.30], weights)
    draws = paired_bootstrap(left, right, resamples=600, seed=11)
    result = seed_verdict(left, right, draws, seed=20260906)
    assert result.slope_a < 0.0
    assert result.a_interval[1] > 0.0
    assert not result.a_collapses
    assert not result.confirmed


def test_a_difference_that_straddles_zero_is_not_significant():
    """Deviation 13.1 point 2's lower end, which is the whole one-sidedness.

    B's average slope is above A's, and the interval still contains 0. Reading
    the upper end of the paired difference instead of the lower would call
    that confirmed, which is the registered criterion read backwards.
    """

    left = _varied_series("naive", [0.30, 0.20, 0.10, 0.00], [0.4, 0.8, 1.2, 1.6])
    right = _varied_series("blind_self", [0.30, 0.22, 0.11, 0.02], [1.6, 1.2, 0.8, 0.4])
    draws = paired_bootstrap(left, right, resamples=600, seed=5)
    result = seed_verdict(left, right, draws, seed=20260906)
    assert result.slope_b > result.slope_a
    assert result.difference_interval[0] < 0.0 < result.difference_interval[1]
    assert not result.b_exceeds_a
    assert not result.confirmed


def test_the_verdict_carries_the_checkpoints_it_could_not_get():
    # Deviation 13.1 point 6: report the gap in the series, do not hide it.
    left = _series("naive", [0.30, 0.20, 0.10], steps=(0, 8, 16))
    right = _series("blind_self", [0.30, 0.30, 0.30], steps=(0, 8, 16))
    draws = paired_bootstrap(left, right, resamples=100, seed=11)
    result = seed_verdict(left, right, draws, seed=20260906, missing_steps=(24,))
    assert result.steps == (0, 8, 16)
    assert result.missing_steps == (24,)


# --- across seeds -----------------------------------------------------------


def _verdict(*, a_collapses: bool, difference: float) -> SeedVerdict:
    return SeedVerdict(seed=0, steps=STEPS, missing_steps=(),
                       slope_a=0.0, slope_b=difference,
                       a_interval=(-1.0, -0.1 if a_collapses else 0.1),
                       difference_interval=(-1.0, 1.0),
                       a_collapses=a_collapses, b_exceeds_a=difference > 0,
                       dropped_checkpoints=0, discarded_resamples=0)


def test_five_supporting_seeds_confirm():
    result = across_seeds([_verdict(a_collapses=True, difference=0.01)] * 5)
    assert result.collapsing == 5
    assert result.sign_supporting == 5
    assert result.sign_p == pytest.approx(0.03125)
    assert result.a_half and result.b_half
    assert result.confirmed


def test_four_of_five_collapsing_still_passes_the_first_half():
    verdicts = [_verdict(a_collapses=True, difference=0.01)] * 4
    verdicts.append(_verdict(a_collapses=False, difference=0.01))
    result = across_seeds(verdicts)
    assert result.collapsing == 4
    assert result.a_half
    assert result.confirmed


def test_three_of_five_collapsing_fails_the_first_half():
    verdicts = [_verdict(a_collapses=True, difference=0.01)] * 3
    verdicts += [_verdict(a_collapses=False, difference=0.01)] * 2
    result = across_seeds(verdicts)
    assert result.collapsing == 3
    assert not result.a_half
    assert not result.confirmed


def test_one_seed_in_the_wrong_direction_fails_the_sign_test():
    verdicts = [_verdict(a_collapses=True, difference=0.01)] * 4
    verdicts.append(_verdict(a_collapses=True, difference=-0.01))
    result = across_seeds(verdicts)
    assert result.sign_supporting == 4
    assert result.sign_p == pytest.approx(0.1875)
    assert result.a_half
    assert not result.b_half
    assert not result.confirmed


def test_an_exactly_zero_difference_counts_against():
    # Matching endpoint1.sign_test: a tie stays in the denominator.
    verdicts = [_verdict(a_collapses=True, difference=0.01)] * 4
    verdicts.append(_verdict(a_collapses=True, difference=0.0))
    result = across_seeds(verdicts)
    assert result.sign_supporting == 4


def test_the_sign_test_is_one_sided():
    verdicts = [_verdict(a_collapses=True, difference=-0.01)] * 5
    result = across_seeds(verdicts)
    assert result.sign_supporting == 0
    assert result.sign_p == pytest.approx(1.0)
    assert not result.b_half


def test_fewer_than_five_seeds_is_not_the_registered_test():
    # Deviation 9.4 registers a five-seed test; three seeds can reach p=0.125
    # at best, and calling that confirmed would be a different test.
    result = across_seeds([_verdict(a_collapses=True, difference=0.01)] * 3)
    assert not result.confirmatory
    assert not result.confirmed


def test_more_than_five_seeds_is_not_the_registered_test_either():
    """Deviation 9.4 registers five seeds, not "at least five".

    Six unanimous seeds clear both halves on their own -- p = 0.015625 and 6
    of 6 collapsing -- so this is the case where the seed count is the only
    thing standing between the numbers and the word confirmed. Adding a sixth
    replicate after seeing five is the manoeuvre the count exists to block.
    """

    result = across_seeds([_verdict(a_collapses=True, difference=0.01)] * 6)
    assert result.a_half and result.b_half
    assert result.sign_p == pytest.approx(0.015625)
    assert not result.confirmatory
    assert not result.confirmed


def test_across_seeds_needs_at_least_one_seed():
    with pytest.raises(ValueError, match="at least one seed"):
        across_seeds([])


def test_across_seeds_is_a_conjunction():
    assert AcrossSeeds(5, 5, 5, 0.03, True, False, True).confirmed is False
    assert AcrossSeeds(5, 5, 5, 0.03, False, True, True).confirmed is False
    assert AcrossSeeds(5, 5, 5, 0.03, True, True, False).confirmed is False
    assert AcrossSeeds(5, 5, 5, 0.03, True, True, True).confirmed is True


# --- the loader -------------------------------------------------------------


def test_load_series_reads_scores_and_labels(tmp_path: Path):
    run = tmp_path / "run"
    for step in (0, 8):
        _write_blind(run, "naive", step, [_blind_row("p1", 0.75), _blind_row("p2", 0.25)])
        _write_verified(run, "naive", step,
                        [("p1", 0, True), ("p1", 1, False), ("p2", 0, False)])
    series = load_series(run, "naive", [0, 8])
    assert series.prompts == ("p1", "p2")
    assert series.steps == (0, 8)
    assert series.scores[:, 0] == pytest.approx([0.75, 0.25])
    assert list(series.labels[:, 0]) == [1, 0]


def test_the_label_comes_from_candidate_zero(tmp_path: Path):
    """Deviation 11.1: the draw the self-report was computed on.

    Candidate 0 is wrong here and candidate 1 is right. Reading the label off
    any other draw -- the first row in the file, say -- would flip it.
    """

    run = tmp_path / "run"
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5)])
    _write_verified(run, "naive", 0, [("p1", 1, True), ("p1", 0, False)])
    series = load_series(run, "naive", [0])
    assert list(series.labels[:, 0]) == [0]


def test_a_prompted_row_is_refused_rather_than_substituted(tmp_path: Path):
    # Deviation 11.3's whole point: no fallback to the score already on disk.
    run = tmp_path / "run"
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5, condition="observe_naive")])
    _write_verified(run, "naive", 0, [("p1", 0, True)])
    with pytest.raises(ValueError, match="deviation 11.3"):
        load_series(run, "naive", [0])


def test_a_row_for_another_candidate_is_refused(tmp_path: Path):
    run = tmp_path / "run"
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5, candidate_index=2)])
    _write_verified(run, "naive", 0, [("p1", 0, True)])
    with pytest.raises(ValueError, match="deviation 11.1"):
        load_series(run, "naive", [0])


def test_a_duplicate_prompt_is_refused(tmp_path: Path):
    run = tmp_path / "run"
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5), _blind_row("p1", 0.9)])
    _write_verified(run, "naive", 0, [("p1", 0, True)])
    with pytest.raises(ValueError, match="duplicate"):
        load_series(run, "naive", [0])


def test_an_absent_blind_pass_raises_its_own_type(tmp_path: Path):
    """Deviation 11.3 gives this case the verdict "not done".

    A distinct exception type so the driver can tell it from a missing
    manifest, which is a hole in the instrument rather than a pass that has
    not been run yet.
    """

    run = tmp_path / "run"
    _write_verified(run, "naive", 0, [("p1", 0, True)])
    with pytest.raises(BlindPassMissing):
        load_series(run, "naive", [0])
    # Narrower than FileNotFoundError, not parallel to it: a caller that
    # catches the ordinary type must still catch this one, or a missing blind
    # pass would escape the driver's own error handling entirely.
    assert issubclass(BlindPassMissing, FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load_series(run, "naive", [0])


def test_a_blind_pass_with_no_verdicts_is_a_different_failure(tmp_path: Path):
    run = tmp_path / "run"
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5)])
    with pytest.raises(FileNotFoundError) as caught:
        load_series(run, "naive", [0])
    assert not isinstance(caught.value, BlindPassMissing)


def test_an_undecided_verdict_becomes_absent_not_wrong(tmp_path: Path):
    # Deviation 10 through 12: undecided is not a zero. Scoring it as wrong
    # would push the gap down by however many rows the adjudicator hesitated
    # over, which is 14 to 23 per 256 in the main run.
    run = tmp_path / "run"
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5), _blind_row("p2", 0.9)])
    _write_verified(run, "naive", 0, [("p1", 0, True), ("p2", 0, None)])
    series = load_series(run, "naive", [0])
    assert list(series.labels[:, 0]) == [1, -1]


def test_a_prompt_missing_at_one_checkpoint_keeps_its_row(tmp_path: Path):
    run = tmp_path / "run"
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5), _blind_row("p2", 0.9)])
    _write_verified(run, "naive", 0, [("p1", 0, True), ("p2", 0, False)])
    _write_blind(run, "naive", 8, [_blind_row("p1", 0.4)])
    _write_verified(run, "naive", 8, [("p1", 0, True), ("p2", 0, False)])
    series = load_series(run, "naive", [0, 8])
    assert series.prompts == ("p1", "p2")
    assert list(series.labels[:, 1]) == [1, -1]


def test_load_series_needs_at_least_one_step(tmp_path: Path):
    with pytest.raises(ValueError, match="no steps"):
        load_series(tmp_path, "naive", [])


def test_available_steps_lists_what_the_blind_pass_finished(tmp_path: Path):
    run = tmp_path / "run"
    assert available_steps(run, "naive") == []
    _write_blind(run, "naive", 16, [_blind_row("p1", 0.5)])
    _write_blind(run, "naive", 0, [_blind_row("p1", 0.5)])
    assert available_steps(run, "naive") == [0, 16]


# --- the series invariants --------------------------------------------------


def test_a_series_whose_arrays_do_not_match_its_axes_is_refused():
    with pytest.raises(ValueError, match="expected"):
        BlindSeries("naive", (0, 8), ("p1",), np.zeros((1, 3)), np.zeros((1, 3), dtype=int))


def test_steps_out_of_order_are_refused():
    with pytest.raises(ValueError, match="out of order"):
        BlindSeries("naive", (8, 0), ("p1",), np.zeros((1, 2)), np.zeros((1, 2), dtype=int))


def test_a_repeated_step_is_refused():
    with pytest.raises(ValueError, match="twice"):
        BlindSeries("naive", (8, 8), ("p1",), np.zeros((1, 2)), np.zeros((1, 2), dtype=int))


def test_a_repeated_prompt_is_refused():
    with pytest.raises(ValueError, match="prompt appears twice"):
        BlindSeries("naive", (0, 8), ("p1", "p1"),
                    np.zeros((2, 2)), np.zeros((2, 2), dtype=int))
