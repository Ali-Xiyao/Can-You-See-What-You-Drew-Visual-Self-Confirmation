"""Tests for E3 endpoint 3.

Deviation 13.2 pins the 口径 to `review-packets/bsv-selection-20260908/`
`bsv_selection.py`. That packet is frozen, so this module restates its rules,
and a restatement that can drift is worse than no restatement at all. The
first block below is therefore not a test of endpoint 3: it is a test that
endpoint 3 and the frozen script compute the same three functions.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from selfsight.analysis.endpoint3 import (
    CONFLICT_GOLD,
    DOWNGRADE,
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    MIN_CANDIDATES,
    USABLE_GOLD,
    CheckpointPoint,
    build_pools,
    checkpoint_point,
    context_effect,
    fit_slope,
    load_point,
    mean_agreement,
    nested_bootstrap,
    pick,
    spec_agreement,
    verdict,
)

ROOT = Path(__file__).resolve().parent.parent
PACKET = ROOT / "review-packets" / "bsv-selection-20260908" / "bsv_selection.py"


def _frozen():
    spec = importlib.util.spec_from_file_location("bsv_selection_frozen", PACKET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = _frozen()


# --- the 口径 is the frozen one ---------------------------------------------


def test_the_usable_gold_sources_are_the_frozen_ones():
    assert USABLE_GOLD == frozen.USABLE_GOLD
    assert CONFLICT_GOLD in USABLE_GOLD


def test_spec_agreement_matches_the_frozen_script_on_every_case():
    """Four cases, and the interesting one is the conflict.

    On a conflict trial the description implies the opposite of the pixels, so
    agreeing with the description means being wrong about the picture. Getting
    this backwards would flip the selection score for exactly the trials
    endpoint 3 is about.
    """

    for gold in USABLE_GOLD:
        for correct in (True, False):
            row = {"gold_source": gold, "correct": correct}
            assert spec_agreement(row) == frozen.spec_agreement(row), row
    assert spec_agreement({"gold_source": CONFLICT_GOLD, "correct": True}) is False


def test_mean_agreement_matches_the_frozen_script():
    answers = [(True, False), (False, True), (True, True)]
    assert mean_agreement(answers) == pytest.approx(frozen.mean_agreement(answers))
    assert math.isnan(mean_agreement([]))
    assert math.isnan(frozen.mean_agreement([]))


def test_pick_matches_the_frozen_script_including_the_tie_break():
    pool = [{"index": 0, "blind": 0.5, "correct": False},
            {"index": 1, "blind": 0.5, "correct": True},
            {"index": 2, "blind": 0.25, "correct": True}]
    assert pick(pool, "blind") == pytest.approx(frozen.pick(pool, "blind"))
    assert pick(pool, "blind") == pytest.approx(0.5)
    assert pick(pool, "blind", first_index=True) == frozen.pick(pool, "blind", True)
    assert pick(pool, "blind", first_index=True) == 0.0


def test_the_bootstrap_constants_are_the_frozen_ones():
    # bsv_selection.paired_bootstrap defaults; deviation 13.2 point 5 keeps them
    # and changes only the cluster unit.
    assert FINAL_RESAMPLES == 20000 == frozen.paired_bootstrap.__defaults__[0]
    assert FINAL_BOOTSTRAP_SEED == 20260908 == frozen.paired_bootstrap.__defaults__[1]


# --- building pools ---------------------------------------------------------


def _answer(spec: str, index: int, condition: str, *, gold: str, correct: bool,
            image_correct: bool, abstain: bool = False) -> dict:
    return {"spec_id": spec, "candidate_index": index, "condition": condition,
            "gold_source": gold, "correct": correct, "abstain": abstain,
            "image_correct": image_correct, "question_id": f"{spec}:q{index}"}


def _pool_rows(spec: str, table: dict[int, tuple[bool, list[tuple[str, bool, bool]]]]
               ) -> list[dict]:
    """`table[index] = (image_correct, [(gold, blind_correct, prompted_correct)])`."""

    rows = []
    for index, (image_correct, trials) in table.items():
        for gold, blind_correct, prompted_correct in trials:
            rows.append(_answer(spec, index, "blind", gold=gold, correct=blind_correct,
                                image_correct=image_correct))
            rows.append(_answer(spec, index, "prompted", gold=gold, correct=prompted_correct,
                                image_correct=image_correct))
    return rows


def test_an_image_only_question_is_dropped():
    # Deviation 13.2 point 1: nothing the description implies, nothing a
    # selection loop could have scored.
    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True),
                                        ("image", True, True)]),
                             1: (False, [("spec_matches_image", False, False)])})
    pools = build_pools(rows)
    assert len(pools) == 1
    assert [item["asked"] for item in pools[0]] == [1, 1]


def test_an_abstention_is_dropped():
    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True)]),
                             1: (False, [("spec_matches_image", False, False)])})
    rows.append(_answer("p1", 0, "blind", gold="spec_matches_image", correct=True,
                        image_correct=True, abstain=True))
    pools = build_pools(rows)
    assert [item["asked"] for item in pools[0]] == [1, 1]


def test_a_prompt_with_one_scorable_candidate_is_not_a_choice():
    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True)])})
    assert build_pools(rows) == []
    assert MIN_CANDIDATES == 2


def test_a_candidate_scored_in_only_one_condition_drops_out():
    """It would enter one side of the selection difference and not the other."""

    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True)]),
                             1: (False, [("spec_matches_image", False, False)])})
    rows.append(_answer("p1", 2, "blind", gold="spec_matches_image", correct=True,
                        image_correct=True))
    pools = build_pools(rows)
    assert [item["index"] for item in pools[0]] == [0, 1]


def test_a_candidate_that_is_both_correct_and_incorrect_is_refused():
    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True)]),
                             1: (False, [("spec_matches_image", False, False)])})
    rows.append(_answer("p1", 0, "blind", gold="spec_matches_image", correct=True,
                        image_correct=False))
    with pytest.raises(ValueError, match="both externally correct"):
        build_pools(rows)


def test_an_unknown_condition_is_refused():
    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True)]),
                             1: (False, [("spec_matches_image", False, False)])})
    rows.append(_answer("p1", 0, "observe_rfo", gold="spec_matches_image", correct=True,
                        image_correct=True))
    with pytest.raises(ValueError, match="Unknown condition"):
        build_pools(rows)


# --- the two axes -----------------------------------------------------------


def test_the_dose_is_pixel_agreement_on_conflict_trials_only():
    """Deviation 13.2 point 3: `correct`, not spec agreement, and conflicts only.

    Blind gets both conflict trials right about the picture and prompted gets
    neither, so the dose is +1. The non-conflict trial disagrees in the other
    direction and must not touch it.
    """

    rows = _pool_rows("p1", {
        0: (True, [(CONFLICT_GOLD, True, False), ("spec_matches_image", False, True)]),
        1: (False, [(CONFLICT_GOLD, True, False)])})
    dose, counts = context_effect(rows)
    assert dose == pytest.approx(1.0)
    assert counts["blind_trials"] == 2
    assert counts["prompted_correct"] == 0


def test_the_dose_is_nan_when_nothing_conflicted():
    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True)])})
    dose, counts = context_effect(rows)
    assert math.isnan(dose)
    assert counts["blind_trials"] == 0


def test_the_selection_gain_is_the_frozen_rule_averaged_over_prompts():
    """y is `pick(blind) - pick(prompted)` per pool, meaned over pools.

    Candidate 0 is wrong and candidate 1 is right. Blind scores them apart in
    the right direction; prompted scores them apart the wrong way. So blind
    keeps a correct image, prompted keeps an incorrect one, and the gain is 1.
    """

    rows = _pool_rows("p1", {
        0: (False, [(CONFLICT_GOLD, True, False)]),
        1: (True, [(CONFLICT_GOLD, False, True)])})
    point = checkpoint_point(rows, seed=20260906, arm="naive", step=8)
    assert point.selection_gain == pytest.approx(1.0)
    assert point.pools == 1
    assert point.candidates == 2
    assert point.cluster == (20260906, 8)


def test_the_first_index_tie_break_is_reported_beside_the_expectation():
    # Both candidates tie under both conditions, so the uniform expectation is
    # the pool mean (0.5) in each condition and the gain is 0. First index
    # takes candidate 0 in both conditions, also 0. They agree here; the point
    # is that both are computed and carried.
    rows = _pool_rows("p1", {0: (False, [("spec_matches_image", True, True)]),
                             1: (True, [("spec_matches_image", True, True)])})
    point = checkpoint_point(rows, seed=1, arm="naive", step=0)
    assert point.selection_gain == pytest.approx(0.0)
    assert point.selection_gain_first_index == pytest.approx(0.0)


def test_a_checkpoint_with_no_scorable_prompt_is_refused():
    rows = _pool_rows("p1", {0: (True, [("spec_matches_image", True, True)])})
    with pytest.raises(ValueError, match="scorable candidates"):
        checkpoint_point(rows, seed=1, arm="naive", step=0)


# --- the fit ----------------------------------------------------------------


def _point(seed: int, arm: str, step: int, x: float, y: float,
           strict: float | None = None) -> CheckpointPoint:
    return CheckpointPoint(seed=seed, arm=arm, step=step, context_effect=x,
                           selection_gain=y,
                           selection_gain_first_index=y if strict is None else strict,
                           pools=8, candidates=32)


# --- section 3's two tie-breaks, read by deviation 15.3 ----------------------


def _parting(seed: int = 1) -> list[CheckpointPoint]:
    """Gain rises with the dose under the primary caliber and falls under the
    strict one, which is the case where reporting only one of them would be a
    choice about the conclusion."""

    return [_point(seed, "naive", step, x, 2 * x, strict=-2 * x)
            for step, x in enumerate([0.1, 0.2, 0.3, 0.4])]


def test_the_strict_tie_break_is_a_different_fit():
    points = _parting()
    assert fit_slope(points) == pytest.approx(2.0)
    assert fit_slope(points, first_index=True) == pytest.approx(-2.0)


def test_both_tie_breaks_come_off_the_same_resamples():
    # A contrast between calibers, not between resamples: the two arrays are
    # the same length and drawn in step with each other.
    points = [_point(1, "naive", step, x, 2 * x, strict=1.5 * x)
              for step, x in enumerate([0.1, 0.2, 0.3, 0.4])]
    draws = nested_bootstrap(points, resamples=64, seed=5)
    assert len(draws.slopes) == len(draws.slopes_first_index)
    assert len(draws.slopes) + draws.discarded_resamples == 64


def test_the_verdict_is_the_primary_fit_even_when_the_strict_one_disagrees():
    points = _parting()
    draws = nested_bootstrap(points, resamples=400, seed=5)
    result = verdict(points, draws)
    assert result.dose_response is True
    assert result.dose_response_first_index is False
    assert result.tie_breaks_agree is False
    assert result.wording != DOWNGRADE
    assert result.slope_first_index == pytest.approx(-2.0)


def test_a_resample_the_strict_fit_cannot_take_is_discarded_from_both():
    """Two of the four checkpoints have no first-index gain to fit.

    A resample that lands on fewer than two of the usable ones leaves the
    strict fit undefined while the primary one is fine. Keeping the draw would
    leave a NaN in the robustness distribution, and every percentile of it
    would come back NaN -- a downgrade produced by a missing number rather
    than by a measurement.
    """

    points = [_point(1, "naive", step, x, 2 * x,
                     strict=float("nan") if step < 2 else 1.5 * x)
              for step, x in enumerate([0.1, 0.2, 0.3, 0.4])]
    draws = nested_bootstrap(points, resamples=200, seed=5)
    assert draws.discarded_resamples > 0
    assert len(draws.slopes) == len(draws.slopes_first_index)
    assert not np.isnan(draws.slopes_first_index).any()
    assert not np.isnan(draws.slopes).any()


def test_a_strict_interval_that_spans_zero_is_not_a_dose_response():
    """The strict fit is read at the same end as the primary one.

    Its point estimate is positive here and its interval contains zero, which
    is the shape the primary fit's own test pins for the downgrade. Reading
    the upper end would report a robustness contrast that always agrees.
    """

    points = [_point(1, "naive", step, x, 2 * x, strict=strict)
              for step, (x, strict) in enumerate(
                  [(0.1, 0.30), (0.2, 0.05), (0.3, 0.35), (0.4, 0.22)])]
    draws = nested_bootstrap(points, resamples=600, seed=5)
    result = verdict(points, draws)
    assert result.slope_first_index > 0.0
    assert result.interval_first_index[0] < 0.0 < result.interval_first_index[1]
    assert result.dose_response_first_index is False


def test_agreeing_tie_breaks_say_so():
    points = [_point(1, "naive", step, x, 2 * x, strict=1.9 * x)
              for step, x in enumerate([0.1, 0.2, 0.3, 0.4])]
    result = verdict(points, nested_bootstrap(points, resamples=400, seed=5))
    assert result.tie_breaks_agree is True
    assert result.dose_response is True


def test_the_slope_is_gain_on_dose():
    points = [_point(1, "naive", step, x, 2 * x)
              for step, x in enumerate([0.1, 0.2, 0.3, 0.4])]
    assert fit_slope(points) == pytest.approx(2.0)


def test_a_single_dose_has_no_slope():
    """Not testable by a zero sum of squares, which is how this first went wrong.

    Three copies of 0.2 have mean 0.20000000000000004, so they centre to about
    -2.8e-17 each rather than to nothing. The sum of squares is then 2e-33, the
    slope divides one rounding error by another, and the answer was a confident
    -0.333 from three points carrying no information about the dose. The guard
    has to compare the raw doses.
    """

    points = [_point(1, "naive", step, 0.2, y) for step, y in enumerate([0.1, 0.3, 0.5])]
    assert math.isnan(fit_slope(points))


def test_a_nan_dose_drops_that_point_rather_than_the_fit():
    points = [_point(1, "naive", 0, 0.1, 0.2), _point(1, "naive", 8, 0.2, 0.4),
              _point(1, "naive", 16, float("nan"), 0.6), _point(1, "naive", 24, 0.3, 0.6)]
    assert fit_slope(points) == pytest.approx(2.0)


# --- the nested bootstrap ---------------------------------------------------


def _grid(gains: dict[int, float]) -> list[CheckpointPoint]:
    """Five seeds x four checkpoints x two arms, dose driving gain by `gains`."""

    points = []
    for seed in (1, 2, 3, 4, 5):
        for step, dose in enumerate([0.1, 0.2, 0.3, 0.4]):
            for arm in ("naive", "blind_self"):
                points.append(_point(seed, arm, step * 8, dose, gains[step]))
    return points


def test_the_cluster_is_the_seed_checkpoint_pair_with_both_arms_inside():
    draws = nested_bootstrap(_grid({0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}),
                             resamples=50, seed=3)
    assert draws.seeds == 5
    assert draws.clusters == 20
    # Every resample is a perfect line through the same four doses, so the
    # slope is 2.0 however the clusters fall.
    assert np.allclose(draws.slopes, 2.0)


def test_the_seeds_themselves_are_resampled():
    """The outer level of deviation 13.2 point 5, which is the whole change.

    Each seed sits at a single dose here, so resampling checkpoints *within* a
    seed cannot move anything -- all four of a seed's points are identical. Any
    variation in the slope must therefore come from drawing seeds with
    replacement. A bootstrap that visited each seed exactly once would return
    the same slope 200 times, which is a confidence interval of width zero
    around whatever these five seeds happened to give.
    """

    points = [_point(seed, arm, step * 8, seed / 10.0, seed / 10.0 + (seed % 2) * 0.3)
              for seed in (1, 2, 3, 4, 5)
              for step in range(4)
              for arm in ("naive", "blind_self")]
    draws = nested_bootstrap(points, resamples=200, seed=4)
    assert len(set(draws.slopes.tolist())) > 1
    assert float(draws.slopes.std()) > 0.0


def test_the_bootstrap_is_reproducible_from_its_seed():
    points = _grid({0: 0.1, 1: 0.5, 2: 0.2, 3: 0.9})
    first = nested_bootstrap(points, resamples=80, seed=7)
    second = nested_bootstrap(points, resamples=80, seed=7)
    other = nested_bootstrap(points, resamples=80, seed=8)
    assert np.array_equal(first.slopes, second.slopes)
    assert not np.array_equal(first.slopes, other.slopes)


def test_a_resample_that_lands_on_one_dose_is_discarded_and_counted():
    # One seed with two checkpoints: the resample draws the same checkpoint
    # twice often enough to make a degenerate fit within 200 draws.
    points = [_point(1, "naive", 0, 0.1, 0.2), _point(1, "naive", 8, 0.3, 0.6)]
    draws = nested_bootstrap(points, resamples=200, seed=2)
    assert draws.discarded_resamples > 0
    assert len(draws.slopes) == 200 - draws.discarded_resamples


def test_a_bootstrap_with_no_dose_spread_at_all_raises():
    points = [_point(1, "naive", step, 0.2, 0.5) for step in (0, 8, 16)]
    with pytest.raises(ValueError, match="discarded"):
        nested_bootstrap(points, resamples=10, seed=1)


def test_the_bootstrap_needs_a_seed():
    with pytest.raises(ValueError, match="at least one seed"):
        nested_bootstrap([], resamples=10, seed=1)


# --- the verdict ------------------------------------------------------------


def test_a_positive_slope_whose_interval_clears_zero_is_the_dose_response():
    points = _grid({0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8})
    draws = nested_bootstrap(points, resamples=200, seed=3)
    result = verdict(points, draws)
    assert result.slope == pytest.approx(2.0)
    assert result.interval[0] > 0.0
    assert result.dose_response
    assert "dose-response" in result.wording
    assert result.points == 40
    assert result.seeds == 5
    assert result.clusters == 20


def test_a_slope_whose_interval_spans_zero_takes_the_registered_downgrade():
    """The registered downgrade is as binding as the registered prediction.

    Gains here are unrelated to the dose, so the interval straddles zero even
    though the point estimate happens to be positive. "The point estimate is
    encouraging" is not a clause the pre-registration contains.
    """

    points = _grid({0: 0.5, 1: 0.1, 2: 0.9, 3: 0.4})
    draws = nested_bootstrap(points, resamples=400, seed=3)
    result = verdict(points, draws)
    assert result.interval[0] <= 0.0
    assert not result.dose_response
    assert result.wording == DOWNGRADE
    assert "not a dose-response" in result.wording


def test_a_negative_slope_takes_the_downgrade_too():
    points = _grid({0: 0.8, 1: 0.6, 2: 0.4, 3: 0.2})
    draws = nested_bootstrap(points, resamples=200, seed=3)
    result = verdict(points, draws)
    assert result.slope == pytest.approx(-2.0)
    assert not result.dose_response
    assert result.wording == DOWNGRADE


# --- reading from disk ------------------------------------------------------


def test_load_point_reads_a_checkpoint(tmp_path: Path):
    run = tmp_path / "run"
    directory = run / "analysis" / "selection" / "naive"
    directory.mkdir(parents=True)
    rows = _pool_rows("p1", {0: (False, [(CONFLICT_GOLD, True, False)]),
                             1: (True, [(CONFLICT_GOLD, False, True)])})
    (directory / "step-00008.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    point = load_point(run, "naive", 8, seed=20260906)
    assert point.selection_gain == pytest.approx(1.0)
    assert point.step == 8


def test_a_missing_selection_pass_is_refused(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No selection pass"):
        load_point(tmp_path, "naive", 8, seed=1)
