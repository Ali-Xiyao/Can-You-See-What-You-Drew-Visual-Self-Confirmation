"""The registered analysis for the co-evolution endpoints, on data it cannot cheat on.

Written before the run produced anything, so every case here is synthetic and
chosen to make one property fail if it is dropped. The check that it computes
the *same* thing as the frozen cross-sectional packet is in
`review-packets/bsv-selection-20260908/` and is run against real files; these
tests are about the properties that make the number mean what it says.
"""
from __future__ import annotations

import math

import pytest

from selfsight.analysis.context import (
    accuracy,
    context_effect,
    discrimination_gap,
    dose_response,
    kept,
    ols,
    pools,
    selection_gain,
    spec_agreement,
)


def row(spec="s0", index=0, *, correct, gold_source="spec_matches_image",
        image_correct=True, abstain=False):
    return {"spec_id": spec, "candidate_index": index, "correct": correct,
            "gold_source": gold_source, "image_correct": image_correct,
            "abstain": abstain}


# --------------------------------------------------------- what agreement is


def test_on_an_agreeing_trial_the_description_and_the_pixels_want_the_same_answer():
    assert spec_agreement(row(correct=True)) is True
    assert spec_agreement(row(correct=False)) is False


def test_on_a_conflicting_trial_agreeing_with_the_description_means_being_wrong():
    """The flip is the whole instrument. Without it a self-selection score would
    be a score against the detections, which the loop cannot see."""

    conflict = row(correct=True, gold_source="image_differs_from_spec")
    assert spec_agreement(conflict) is False
    assert spec_agreement(row(correct=False, gold_source="image_differs_from_spec")) is True


def test_a_trial_the_description_never_constrained_is_refused():
    """gold_source "image" asks about something nobody requested; scoring it as
    agreement would quietly count a coin flip as evidence."""

    with pytest.raises(ValueError, match="no description-implied answer"):
        spec_agreement(row(correct=True, gold_source="image"))


# ------------------------------------------------------------ context effect


def test_the_context_effect_ignores_the_trials_that_cannot_separate_the_two():
    """Agreement trials run near ceiling in both conditions and the description
    actually helps there (0.979 -> 0.990 cross-sectionally). Letting them in
    would dilute the effect toward zero with trials that carry no information
    about it."""

    blind = [row(correct=True, gold_source="image_differs_from_spec"),
             row(correct=False)]
    prompted = [row(correct=False, gold_source="image_differs_from_spec"),
                row(correct=True)]
    assert context_effect(blind, prompted) == pytest.approx(1.0)


def test_abstentions_are_not_counted_as_wrong():
    assert accuracy([row(correct=True), row(correct=None, abstain=True)]) == pytest.approx(1.0)


# -------------------------------------------------------------------- pools


def test_a_candidate_only_one_condition_answered_is_dropped():
    """Scoring it on half the evidence would put a condition difference into the
    pool that is really a coverage difference."""

    blind = [row(index=0, correct=True), row(index=1, correct=True),
             row(index=2, correct=True)]
    prompted = [row(index=0, correct=True), row(index=1, correct=True)]
    assert [item["index"] for item in pools(blind, prompted)["s0"]] == [0, 1]


def test_a_description_with_one_usable_candidate_is_not_a_choice():
    blind = [row(index=0, correct=True)]
    assert pools(blind, list(blind)) == {}


def test_a_candidate_cannot_be_both_right_and_wrong():
    """Two rows disagreeing about image_correct means the verdicts were joined
    on the wrong key, and a silent majority vote would hide it."""

    blind = [row(index=0, correct=True, image_correct=True),
             row(index=0, correct=True, image_correct=False)]
    with pytest.raises(ValueError, match="both right and wrong"):
        pools(blind, list(blind))


# ---------------------------------------------------------------- selection


def test_a_tie_is_worth_what_a_shuffled_pool_would_get():
    """Both candidates score 1.0; one is right. A rule that reported 1.0 here
    would be reporting the oracle."""

    pool = [{"index": 0, "blind": 1.0, "prompted": 1.0, "correct": True},
            {"index": 1, "blind": 1.0, "prompted": 1.0, "correct": False}]
    assert kept(pool, "blind") == pytest.approx(0.5)


def test_the_gain_is_positive_when_blind_prefers_the_image_that_is_right():
    pool = {"s0": [{"index": 0, "blind": 1.0, "prompted": 0.0, "correct": True},
                   {"index": 1, "blind": 0.0, "prompted": 1.0, "correct": False}]}
    assert selection_gain(pool) == pytest.approx(1.0)
    assert selection_gain({}) != selection_gain({})  # nan


# ----------------------------------------------------------- discrimination


def test_the_gap_is_between_candidates_not_between_pools():
    pool = {"s0": [{"index": 0, "blind": 0.9, "prompted": 0.5, "correct": True},
                   {"index": 1, "blind": 0.3, "prompted": 0.5, "correct": False}]}
    assert discrimination_gap(pool) == pytest.approx(0.6)
    assert discrimination_gap(pool, "prompted") == pytest.approx(0.0)


def test_a_checkpoint_whose_images_are_all_right_has_no_gap_rather_than_a_zero():
    """Reporting 0.0 there would put "the score stopped discriminating" and
    "there was nothing left to discriminate" on the same curve."""

    pool = {"s0": [{"index": 0, "blind": 0.9, "prompted": 0.5, "correct": True},
                   {"index": 1, "blind": 0.3, "prompted": 0.5, "correct": True}]}
    assert math.isnan(discrimination_gap(pool))


# ------------------------------------------------------------ dose response


def test_the_slope_is_the_slope():
    slope, intercept = ols([(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_a_checkpoint_missing_one_of_the_two_numbers_drops_out():
    """NaN arrives whenever a checkpoint's pools are all-correct, which is not a
    reason to lose the other 25 points."""

    slope, _ = ols([(0.0, 1.0), (float("nan"), 3.0), (2.0, 5.0)])
    assert slope == pytest.approx(2.0)


def test_a_flat_relationship_gives_an_interval_that_contains_zero():
    points = [(x / 10, (-1) ** x * 0.1) for x in range(20)]
    result = dose_response(points, draws=2000)
    assert result["low"] < 0 < result["high"]


def test_a_real_relationship_gives_an_interval_that_does_not():
    points = [(x / 10, x / 10) for x in range(20)]
    result = dose_response(points, draws=2000)
    assert result["slope"] == pytest.approx(1.0)
    assert result["low"] > 0
    assert result["n"] == 20
