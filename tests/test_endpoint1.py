"""Tests for E3 endpoint 1.

Most of these pin a sentence in the pre-registration rather than a property
of the code, so each one names the clause it is holding in place. A test that
fails here means the analysis has drifted from what was registered, which is
the only failure mode that matters for this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from selfsight.analysis.endpoint1 import (
    DETECTABLE_EFFECT,
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    LOW_COVERAGE_PAIRS,
    UNADJUDICATED,
    Coverage,
    PairedCheckpoint,
    between_seed_spread,
    bootstrap_interval,
    exact_mcnemar,
    load_checkpoint,
    paired_difference,
    prompt_differences,
    sign_test,
    verdict,
)


def _write_arm(run: Path, arm: str, step: int, rows: list[dict],
               *, manifest: list[tuple[str, int]] | None = None) -> None:
    """One arm's checkpoint on disk.

    `manifest` defaults to exactly the keys in `rows`, which is the case where
    nothing was skipped. Passing it explicitly is how a test says "the run
    asked for this image and no verdict came back" -- deviation 12's case.
    """

    directory = run / "evaluations" / arm / f"step-{step:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "verified.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    keys = manifest if manifest is not None else [
        (row["spec_id"], row["candidate_index"]) for row in rows]
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps({"spec_id": spec, "candidate_index": draw}) + "\n"
                for spec, draw in keys), encoding="utf-8")


def _row(spec: str, draw: int, correct: bool | None) -> dict:
    """One verdict row. `correct=None` means pending_human, deviation 10's case."""

    if correct is None:
        return {"spec_id": spec, "candidate_index": draw,
                "resolution": "pending_human", "image_correct": None}
    return {"spec_id": spec, "candidate_index": draw,
            "resolution": "agreed", "image_correct": correct}


def _checkpoint(outcomes: dict[str, list[tuple[int, int]]], *, step: int = 8,
                coverage: Coverage | None = None) -> PairedCheckpoint:
    arrays = {prompt: np.asarray(rows, dtype=np.int8) for prompt, rows in outcomes.items()}
    kept = sum(len(rows) for rows in outcomes.values())
    return PairedCheckpoint(
        step=step, outcomes=arrays,
        coverage=coverage or Coverage(kept, kept, len(arrays), len(arrays), 0, 0))


# --------------------------------------------------------------------------
# Deviation 10: paired deletion


def test_a_pair_unadjudicated_in_one_arm_is_dropped_from_the_other_too(tmp_path):
    """Deviation 10.2 point 1. The failure this prevents is the quiet one:
    keeping the adjudicated side would leave the two arms' denominators on
    different images while every count still looked plausible."""

    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True), _row("p1", 1, True)])
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, None), _row("p1", 1, False)])

    checkpoint = load_checkpoint(tmp_path, 8, "naive", "blind_self")

    assert checkpoint.coverage.pairs_kept == 1
    assert checkpoint.outcomes["p1"].tolist() == [[1, 0]], "draw 0 survived on one side"


def test_unadjudicated_is_not_the_same_as_wrong(tmp_path):
    """If a dropped pair were counted as a failure for B, this arrangement
    would read as B losing. It has to read as no information at all."""

    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True), _row("p1", 1, False)])
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, None), _row("p1", 1, True)])

    checkpoint = load_checkpoint(tmp_path, 8, "naive", "blind_self")

    assert paired_difference(checkpoint) == pytest.approx(1.0)


def test_a_prompt_that_loses_every_draw_leaves_the_pool(tmp_path):
    """Deviation 10.2 point 3: no backfill, no imputation, the pool shrinks."""

    _write_arm(tmp_path, "naive", 8,
               [_row("p1", 0, True), _row("p1", 1, True), _row("p2", 0, False)])
    _write_arm(tmp_path, "blind_self", 8,
               [_row("p1", 0, None), _row("p1", 1, None), _row("p2", 0, True)])

    checkpoint = load_checkpoint(tmp_path, 8, "naive", "blind_self")

    assert checkpoint.prompts == ["p2"]
    assert checkpoint.coverage.prompts_kept == 1
    assert checkpoint.coverage.prompts_total == 2


def test_coverage_reports_each_arm_separately(tmp_path):
    """Deviation 10.2 point 5: an asymmetry in who fails adjudication is a
    finding about that arm's images, so it cannot be summed away."""

    _write_arm(tmp_path, "naive", 8,
               [_row("p1", 0, None), _row("p1", 1, None), _row("p2", 0, True)])
    _write_arm(tmp_path, "blind_self", 8,
               [_row("p1", 0, True), _row("p1", 1, True), _row("p2", 0, None)])

    coverage = load_checkpoint(tmp_path, 8, "naive", "blind_self").coverage

    assert (coverage.unadjudicated_a, coverage.unadjudicated_b) == (2, 1)
    assert (coverage.pairs_kept, coverage.pairs_total) == (0, 3)


def test_low_coverage_flags_but_does_not_exclude():
    """Deviation 10.2 point 4 calls the threshold a disclosure trigger, and
    fixes it at 192, which is 75% of the 64 prompts times 4 draws. Asserting
    the literal rather than `LOW_COVERAGE_PAIRS - 1` is the point: a test
    written relative to the constant moves wherever the constant moves."""

    assert LOW_COVERAGE_PAIRS == 192 == int(0.75 * 64 * 4)
    assert Coverage(191, 256, 60, 64, 1, 0).low is True
    assert Coverage(192, 256, 64, 64, 0, 0).low is False


def test_unnameable_counts_as_unadjudicated_exactly_as_the_instrument_says():
    """The set is copied from evaluate.py rather than imported, so it can
    drift. This is the pin that stops it."""

    from selfsight.v4.evaluate import UNADJUDICATED as INSTRUMENT

    assert UNADJUDICATED == INSTRUMENT


def test_arms_asked_for_different_keys_is_refused(tmp_path):
    """A paired estimand over two different populations is not a smaller
    sample, it is a different quantity. It has to stop, not shrink.

    Deviation 12.2 point 4 moves this check onto the manifests. The verdict
    files are exactly what a detector skip is allowed to shorten, so a
    difference there is no longer evidence that the arms drew different
    things -- but a difference in what they were *asked* for still is.
    """

    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True), _row("p1", 1, True)])
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, True)])

    with pytest.raises(ValueError, match="asked for different .prompt, draw. keys"):
        load_checkpoint(tmp_path, 8, "naive", "blind_self")


def test_a_duplicate_verdict_row_is_refused(tmp_path):
    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True), _row("p1", 0, False)],
               manifest=[("p1", 0)])
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, True)])

    with pytest.raises(ValueError, match="duplicate verdict"):
        load_checkpoint(tmp_path, 8, "naive", "blind_self")


def test_a_missing_arm_directory_is_refused(tmp_path):
    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True)])

    with pytest.raises(FileNotFoundError, match="blind_self"):
        load_checkpoint(tmp_path, 8, "naive", "blind_self")


# --------------------------------------------------------------------------
# Deviation 12: an absent row, as opposed to an undecided one


def test_a_skipped_row_drops_the_pair_instead_of_stopping_the_analysis(tmp_path):
    """Deviation 12.2 point 1.

    `skipped_no_detection` makes the row vanish rather than arrive undecided,
    and this used to raise. It has already happened once in the main run
    (rfo_gold step 16, key (v4a-0001, 1)); over 13 checkpoints x 2 arms x 5
    seeds it will happen again, and raising there means the final analysis
    fails at the last possible moment.
    """

    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True), _row("p1", 1, True)])
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, False)],
               manifest=[("p1", 0), ("p1", 1)])

    checkpoint = load_checkpoint(tmp_path, 8, "naive", "blind_self")

    assert checkpoint.coverage.pairs_kept == 1
    assert checkpoint.outcomes["p1"].tolist() == [[1, 0]]


def test_a_skip_is_counted_apart_from_an_undecided_row(tmp_path):
    """Deviation 12.2 point 2. Summing them would hide which instrument gave
    up: the adjudicator hesitating and the detector seeing nothing in the
    picture are different facts, and only the second is about the image."""

    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True), _row("p1", 1, None)])
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, True)],
               manifest=[("p1", 0), ("p1", 1)])

    coverage = load_checkpoint(tmp_path, 8, "naive", "blind_self").coverage

    assert (coverage.unadjudicated_a, coverage.unadjudicated_b) == (1, 0)
    assert (coverage.skipped_a, coverage.skipped_b) == (0, 1)


def test_the_denominator_comes_from_the_manifest(tmp_path):
    """Deviation 12.2 point 3.

    Counting pairs_total off verified.jsonl lets a skipped image delete
    itself from the denominator as well as the numerator, so however many
    rows go missing the deletion rate still reads zero -- and the low
    coverage disclosure never fires.
    """

    keys = [("p1", 0), ("p1", 1), ("p2", 0), ("p2", 1)]
    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True)], manifest=keys)
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, True)], manifest=keys)

    coverage = load_checkpoint(tmp_path, 8, "naive", "blind_self").coverage

    assert coverage.pairs_total == 4, "three vanished rows took the denominator with them"
    assert coverage.prompts_total == 2
    assert (coverage.pairs_kept, coverage.prompts_kept) == (1, 1)
    assert (coverage.skipped_a, coverage.skipped_b) == (3, 3)


def test_a_verdict_no_manifest_asked_for_is_refused(tmp_path):
    """The inverse of a skip, and it is not benign: a verdict for an image
    the run never requested means the directory holds output from something
    else."""

    _write_arm(tmp_path, "naive", 8, [_row("p1", 0, True), _row("p9", 0, True)],
               manifest=[("p1", 0)])
    _write_arm(tmp_path, "blind_self", 8, [_row("p1", 0, True)])

    with pytest.raises(ValueError, match="no manifest asked for"):
        load_checkpoint(tmp_path, 8, "naive", "blind_self")


def test_a_missing_manifest_is_refused(tmp_path):
    directory = tmp_path / "evaluations" / "naive" / "step-00008"
    directory.mkdir(parents=True)
    (directory / "verified.jsonl").write_text(
        json.dumps(_row("p1", 0, True)) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No manifest"):
        load_checkpoint(tmp_path, 8, "naive", "blind_self")


def test_losing_every_pair_raises_rather_than_returning_zero():
    """Zero would flow into the sign test as a tie and count against B, which
    is a real verdict reached from no data."""

    with pytest.raises(ValueError, match="every pair was dropped"):
        paired_difference(_checkpoint({}))


# --------------------------------------------------------------------------
# The estimator


def test_the_estimator_is_the_mean_of_prompt_means_not_the_overall_rate():
    """Deviation 10.2 point 2. The two coincide until deletion makes the draw
    counts unequal, so this fixture is built to make them differ: p1 keeps
    four draws where B never wins, p2 keeps one draw where B does."""

    checkpoint = _checkpoint({
        "p1": [(1, 1), (1, 1), (1, 1), (1, 1)],
        "p2": [(0, 1)],
    })
    overall = (sum(row[1] for rows in checkpoint.outcomes.values() for row in rows.tolist())
               - sum(row[0] for rows in checkpoint.outcomes.values() for row in rows.tolist())) / 5

    assert overall == pytest.approx(0.2)
    assert paired_difference(checkpoint) == pytest.approx(0.5)


def test_prompt_differences_follow_the_prompt_order_the_bootstrap_uses():
    checkpoint = _checkpoint({"b": [(0, 1)], "a": [(1, 0)], "c": [(0, 0)]})

    assert checkpoint.prompts == ["a", "b", "c"]
    assert prompt_differences(checkpoint).tolist() == [-1.0, 1.0, 0.0]


# --------------------------------------------------------------------------
# The bootstrap


def test_the_bootstrap_resamples_prompts_and_not_images():
    """E3 endpoint 1 gives the reason: four draws of one prompt share a
    latent and a spec. Resampling images would treat 40 correlated draws as
    40 independent ones and return a visibly tighter interval. Ten prompts,
    four draws each, all agreeing within a prompt -- so an image-level
    bootstrap has 40 atoms of information here and a prompt-level one has
    10."""

    outcomes = {f"p{i}": [(0, 1)] * 4 if i < 5 else [(1, 0)] * 4 for i in range(10)}
    checkpoint = _checkpoint(outcomes)

    low, high = bootstrap_interval(checkpoint, resamples=4000, seed=1)

    flat = np.concatenate([rows for rows in checkpoint.outcomes.values()])
    per_image = (flat[:, 1] - flat[:, 0]).astype(float)
    rng = np.random.default_rng(1)
    image_draws = per_image[rng.integers(0, len(per_image), size=(4000, len(per_image)))]
    image_low, image_high = np.percentile(image_draws.mean(axis=1), [2.5, 97.5])

    assert (high - low) > 1.8 * (image_high - image_low)


def test_the_bootstrap_is_deterministic_and_the_seed_moves_it():
    checkpoint = _checkpoint({f"p{i}": [(i % 2, (i + 1) % 2)] for i in range(20)})

    first = bootstrap_interval(checkpoint, resamples=500, seed=7)

    assert first == bootstrap_interval(checkpoint, resamples=500, seed=7)
    assert first != bootstrap_interval(checkpoint, resamples=500, seed=8)


def test_the_final_analysis_defaults_are_the_ones_deviation_6_4_registered():
    """Not the report's 20260906 / 2000. Deviation 6.4 split those into two
    branches precisely so neither would inherit the other's numbers."""

    assert (FINAL_RESAMPLES, FINAL_BOOTSTRAP_SEED) == (20000, 20260908)

    from selfsight.analysis import endpoint1

    checkpoint = _checkpoint({f"p{i}": [(0, 1)] for i in range(8)})
    explicit = endpoint1.bootstrap_interval(
        checkpoint, resamples=FINAL_RESAMPLES, seed=FINAL_BOOTSTRAP_SEED)

    assert endpoint1.bootstrap_interval(checkpoint) == explicit


# --------------------------------------------------------------------------
# McNemar


def test_mcnemar_is_one_sided_in_the_registered_direction():
    """B > A was registered before any data existed. The same table read the
    other way round has to come out non-significant."""

    for_b = _checkpoint({f"p{i}": [(0, 1)] for i in range(10)})
    for_a = _checkpoint({f"p{i}": [(1, 0)] for i in range(10)})

    b_only, a_only, p_for_b = exact_mcnemar(for_b)
    _, _, p_for_a = exact_mcnemar(for_a)

    assert (b_only, a_only) == (10, 0)
    assert p_for_b == pytest.approx(0.5 ** 10)
    assert p_for_a == pytest.approx(1.0)


def test_mcnemar_ignores_concordant_pairs():
    discordant_only = _checkpoint({"p1": [(0, 1), (1, 0)]})
    padded = _checkpoint({"p1": [(0, 1), (1, 0), (1, 1), (0, 0)]})

    assert exact_mcnemar(discordant_only) == exact_mcnemar(padded)


def test_mcnemar_with_no_discordant_pairs_is_p_one():
    assert exact_mcnemar(_checkpoint({"p1": [(1, 1), (0, 0)]})) == (0, 0, 1.0)


# --------------------------------------------------------------------------
# Deviation 9.3: the seed-level sign test


def test_five_of_five_is_the_registered_confirmatory_p():
    assert sign_test([0.01] * 5) == (5, 5, pytest.approx(0.03125))


def test_four_of_five_does_not_confirm_and_there_is_no_rescue():
    supporting, total, p = sign_test([0.05, 0.05, 0.05, 0.05, -0.01])

    assert (supporting, total) == (4, 5)
    assert p == pytest.approx(0.1875)
    assert p > 0.05


def test_a_seed_that_lands_exactly_on_zero_counts_against():
    """Deviation 9.3 spells this out. Dropping it would turn 4 supporting
    seeds out of 5 into 4 out of 4, and p from 0.1875 into 0.0625 -- a
    rescue, performed by a convention nobody wrote down."""

    supporting, total, p = sign_test([0.05, 0.05, 0.05, 0.05, 0.0])

    assert (supporting, total) == (4, 5)
    assert p == pytest.approx(0.1875)


def test_three_seeds_could_never_have_cleared_the_threshold():
    """Deviation 9 in one assertion: at n=3 the best attainable p is 0.125,
    which is why the seed count moved to five."""

    assert sign_test([0.05] * 3)[2] == pytest.approx(0.125)
    assert sign_test([0.05] * 3)[2] > 0.05


def test_the_sign_test_needs_at_least_one_seed():
    with pytest.raises(ValueError, match="at least one seed"):
        sign_test([])


# --------------------------------------------------------------------------
# The registered failure condition


def test_not_detected_needs_both_clauses():
    """The registered sentence joins them with and. Either clause alone
    changes the failure condition, which deviation 10.4 forbids."""

    assert verdict(0.01, -0.02, 0.04).detected is False
    assert verdict(0.01, 0.005, 0.04).detected is True, "small but clears zero"
    assert verdict(0.20, -0.05, 0.45).detected is True, "large with a wide interval"
    assert verdict(0.20, 0.10, 0.30).detected is True


def test_the_threshold_is_the_one_written_before_the_data():
    assert DETECTABLE_EFFECT == 0.042

    assert verdict(0.0419, -0.01, 0.09).detected is False
    assert verdict(0.0420, -0.01, 0.09).detected is True


def test_an_interval_touching_zero_contains_it():
    """A boundary that reads either way would let the verdict turn on a
    floating-point tie."""

    assert verdict(0.01, 0.0, 0.05).detected is False
    assert verdict(0.01, -0.05, 0.0).detected is False


def test_the_verdict_says_which_clause_decided_it():
    assert "contains 0" in verdict(0.01, -0.02, 0.04).reason
    assert "excludes 0" in verdict(0.01, 0.005, 0.04).reason


# --------------------------------------------------------------------------
# Reported, never tested against


def test_between_seed_spread_uses_the_sample_sd():
    """df = n-1, which deviation 9.3 names as df=4 for five seeds."""

    spread, ratio = between_seed_spread(_results([0.0, 0.1]))

    assert spread == pytest.approx(0.1 / math.sqrt(2))
    assert ratio == pytest.approx(spread / 0.05)


def test_a_zero_mean_effect_has_no_ratio_rather_than_an_infinite_one():
    spread, ratio = between_seed_spread(_results([-0.1, 0.1]))

    assert spread > 0
    assert ratio is None


def _results(points: list[float]):
    from selfsight.analysis.endpoint1 import SeedResult

    return [SeedResult(seed=20260906 + index, run=Path("."), step=88, point=point,
                       ci_low=point - 0.01, ci_high=point + 0.01,
                       mcnemar_b_only=0, mcnemar_a_only=0, mcnemar_p=1.0,
                       coverage=Coverage(256, 256, 64, 64, 0, 0))
            for index, point in enumerate(points)]
