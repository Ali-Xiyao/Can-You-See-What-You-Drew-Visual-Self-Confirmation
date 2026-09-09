"""The five ways a D* could be produced from numbers that do not mean it.

None of these raise on their own. A prompt appearing in both the training and
outcome sets makes the external curve measure memorisation; an unadjudicated
image counted as wrong bends that curve down for a reason unrelated to the
hypothesis; a checkpoint missing one of its two curves silently pairs one step's
internal score with another's external; a lead reported against a D_g that was
never estimated turns "we could not measure this" into a number; and a
trajectory whose external curve was never coupled to the internal one has no
coupling for training to have broken.
"""

from __future__ import annotations

import json
import math

import pytest

from selfsight.v4.evaluate import (
    CheckpointMetrics,
    curves,
    cycle_scores,
    divergence_report,
    evaluation_seed,
    internal_noise_slope,
    external_correctness,
    external_noise_slope,
    read_metrics_csv,
    split_prompts,
    summarize_cycle,
    write_evaluation_manifest,
    write_metrics_csv,
)
from selfsight.analysis.breakpoints import break_support
from selfsight.v4.spec import SceneSpec, SpecObject


def _spec(spec_id: str) -> SceneSpec:
    return SceneSpec(
        spec_id=spec_id,
        prompt=f"a red cube ({spec_id})",
        objects=(SpecObject(object="cube", color="red", count=1),),
    )


def _row(step: int, internal: float | None, external: float | None,
         arm: str = "naive", sem: float | None = 0.002) -> CheckpointMetrics:
    return CheckpointMetrics(
        arm=arm, round_index=step // 25, step=step,
        internal_cycle=internal, internal_sem=sem, internal_n=120,
        external_correct=external, external_n=120, external_unadjudicated=0,
    )


# --------------------------------------------------------------------------- splits


def test_nothing_the_model_trained_on_can_reach_the_outcome_set():
    split = split_prompts([f"s{index}" for index in range(228)],
                          outcome=120, probe=32, seed=20260901)
    split.assert_disjoint()
    assert len(split.outcome) == 120
    assert len(split.probe) == 32
    assert len(split.train) == 76
    assert set(split.train).isdisjoint(split.outcome)


def test_the_split_is_a_function_of_the_seed_and_not_of_the_bank_order():
    ids = [f"s{index}" for index in range(50)]
    left = split_prompts(ids, outcome=10, probe=5, seed=7)
    right = split_prompts(list(reversed(ids)), outcome=10, probe=5, seed=7)
    assert left == right
    assert split_prompts(ids, outcome=10, probe=5, seed=8) != left


def test_an_evaluation_that_would_leave_nothing_to_train_on_says_so():
    with pytest.raises(ValueError, match="leaving 0 to train on"):
        split_prompts([f"s{index}" for index in range(20)], outcome=15, probe=5, seed=1)


def test_a_hand_built_split_that_overlaps_is_caught():
    from selfsight.v4.evaluate import PromptSplit

    with pytest.raises(ValueError, match="share 1 prompts"):
        PromptSplit(train=("a", "b"), outcome=("b",), probe=("c",)).assert_disjoint()


# ---------------------------------------------------------------- shared latents


def test_both_arms_draw_the_outcome_set_from_the_same_latent():
    """A difference between the arms' external curves must come from the weights."""

    left = evaluation_seed(seed=1, arm="naive", step=50, prompt_id="s3")
    right = evaluation_seed(seed=1, arm="rfo_self", step=50, prompt_id="s3")
    assert left == right
    assert left != evaluation_seed(seed=1, arm="naive", step=75, prompt_id="s3")
    assert left != evaluation_seed(seed=1, arm="naive", step=50, prompt_id="s4")


# ------------------------------------------------------------------- manifest


def test_the_manifest_is_in_the_shape_the_corpus_verifier_already_reads(tmp_path):
    """Reusing the schema is what keeps the external curve comparable to 27."""

    specs = {"a": _spec("a"), "b": _spec("b")}
    path = write_evaluation_manifest(
        tmp_path, specs=specs, prompt_ids=["a", "b"],
        images={("a", 0): "/img/a.png", ("b", 0): "/img/b.png"},
        seeds={("a", 0): 11, ("b", 0): 22},
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["spec_id"] for row in rows] == ["a", "b"]
    assert all(row["candidate_index"] == 0 for row in rows)
    assert all({"spec", "image_path", "seed"} <= set(row) for row in rows)
    assert SceneSpec.from_dict(rows[0]["spec"]).prompt == specs["a"].prompt


def test_a_missing_image_stops_the_manifest_rather_than_shortening_it(tmp_path):
    with pytest.raises(KeyError, match="No generated image"):
        write_evaluation_manifest(tmp_path, specs={"a": _spec("a")}, prompt_ids=["a"],
                                  images={}, seeds={("a", 0): 1})


def test_extra_draws_get_their_own_latents_and_leave_the_first_one_alone():
    """R>1 must not move the first draw, or old curves stop being comparable."""

    first = evaluation_seed(seed=1, arm="naive", step=50, prompt_id="s3")
    assert evaluation_seed(seed=1, arm="naive", step=50, prompt_id="s3",
                           candidate_index=0) == first
    second = evaluation_seed(seed=1, arm="naive", step=50, prompt_id="s3",
                             candidate_index=1)
    third = evaluation_seed(seed=1, arm="naive", step=50, prompt_id="s3",
                            candidate_index=2)
    assert len({first, second, third}) == 3
    # Still shared by the arms, the property the draw count must not break.
    assert second == evaluation_seed(seed=1, arm="rfo_self", step=50, prompt_id="s3",
                                     candidate_index=1)


def test_the_manifest_carries_every_draw_under_its_own_candidate_index(tmp_path):
    """R draws ride the pools' existing field, so detect needs no change."""

    specs = {"a": _spec("a"), "b": _spec("b")}
    images = {(p, i): f"/img/{p}{i}.png" for p in ("a", "b") for i in range(3)}
    seeds = {key: 100 + index for index, key in enumerate(sorted(images))}
    path = write_evaluation_manifest(tmp_path, specs=specs, prompt_ids=["a", "b"],
                                     images=images, seeds=seeds)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert [(row["spec_id"], row["candidate_index"]) for row in rows] == [
        ("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)]
    assert len({row["image_path"] for row in rows}) == 6
    assert len({row["seed"] for row in rows}) == 6


def test_a_prompt_missing_its_later_draws_is_not_silently_shortened(tmp_path):
    """A partial draw set means a checkpoint with fewer images, i.e. more noise."""

    with pytest.raises(KeyError, match="No generated image"):
        write_evaluation_manifest(tmp_path, specs={"a": _spec("a"), "b": _spec("b")},
                                  prompt_ids=["a", "b"],
                                  images={("a", 0): "/img/a.png"},
                                  seeds={("a", 0): 1})

# ------------------------------------------------------------ external curve


def _write_verified(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_an_image_the_ladder_could_not_settle_is_excluded_not_counted_wrong(tmp_path):
    """Counting it wrong would bend the external curve down as pictures get messier.

    That is a real effect and not this one. It would push the breakpoint early,
    in exactly the direction the hypothesis predicts, which is the worst possible
    direction for an artefact to point.
    """

    path = tmp_path / "verified.jsonl"
    _write_verified(path, [
        {"image_correct": True, "resolution": "agreed"},
        {"image_correct": False, "resolution": "agreed"},
        {"image_correct": False, "resolution": "pending_human"},
        {"image_correct": False, "resolution": "unnameable"},
    ])
    rate, n, unadjudicated = external_correctness(path)
    assert rate == 0.5
    assert (n, unadjudicated) == (2, 2)


def test_a_checkpoint_with_no_adjudicated_image_yields_no_rate(tmp_path):
    path = tmp_path / "verified.jsonl"
    _write_verified(path, [{"image_correct": True, "resolution": "pending_human"}])
    assert external_correctness(path) == (None, 0, 1)


def test_a_checkpoint_that_was_never_verified_is_absent_not_zero(tmp_path):
    assert external_correctness(tmp_path / "nothing.jsonl") == (None, 0, 0)


# -------------------------------------------------------------- internal curve


def test_the_internal_score_is_the_registered_cycle_criterion():
    """`prompt -> image -> recover prompt`, never the atomic question form.

    Asking the atomic question here is the v2.3 bug: it is the RFO question, and
    it made the two arms the same function at step 0.
    """

    seen = []

    class Backbone:
        def cycle_consistency_score(self, image_path, prompt):
            seen.append((image_path, prompt))
            return -1.5

        def observe_atoms(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("the internal curve must not use the atomic form")

    scores = cycle_scores(Backbone(), specs={"a": _spec("a")}, images={"a": "/img/a.png"})
    assert scores == {"a": -1.5}
    assert seen == [("/img/a.png", "a red cube (a)")]


# ----------------------------------------------------------------- curve pairing


def test_a_checkpoint_missing_either_curve_is_dropped_from_both():
    rows = [_row(0, -2.0, 0.30), _row(25, -1.8, None), _row(50, None, 0.31),
            _row(75, -1.5, 0.29)]
    steps, internal, external = curves(rows, "naive")
    assert steps == [0.0, 75.0]
    assert internal == [-2.0, -1.5]
    assert external == [0.30, 0.29]


def test_curves_are_sorted_by_step_not_by_file_order():
    steps, _, _ = curves([_row(50, -1.0, 0.2), _row(0, -2.0, 0.3)], "naive")
    assert steps == [0.0, 50.0]


def test_one_arm_never_picks_up_the_other_arm_rows():
    rows = [_row(0, -2.0, 0.30, arm="naive"), _row(0, -9.0, 0.90, arm="rfo_self")]
    _, internal, _ = curves(rows, "rfo_self")
    assert internal == [-9.0]


# ------------------------------------------------------------------ round trip


def test_metrics_survive_a_write_and_read_including_the_missing_ones(tmp_path):
    rows = [_row(0, -2.0, 0.30), _row(25, -1.8, None)]
    path = write_metrics_csv(tmp_path / "checkpoint_metrics.csv", rows)
    restored = read_metrics_csv(path)
    assert restored == sorted(rows, key=lambda item: (item.arm, item.step))
    assert restored[1].external_correct is None
    assert restored[0].internal_sem == pytest.approx(0.002)


# --------------------------------------------------------------------- the report


def test_a_lead_is_not_reported_when_d_g_was_never_estimated():
    rows = [_row(step, -2.0 + step * 0.01, min(0.30 + step * 0.002, 0.40))
            for step in range(0, 200, 25)]
    report = divergence_report(rows, "naive")
    assert report.warning is None
    assert report.lead is None
    assert report.to_dict()["d_g"] is None


def test_a_gradient_curve_without_its_noise_floor_is_refused():
    rows = [_row(step, -2.0, 0.3) for step in range(0, 100, 25)]
    with pytest.raises(ValueError, match="measured noise floor"):
        divergence_report(rows, "naive", gda_free=[0.9, 0.8, 0.7, 0.6])


def test_a_gradient_curve_of_the_wrong_length_is_refused_rather_than_zipped():
    """Silent truncation would pair gradient values with the wrong checkpoints."""

    rows = [_row(step, -2.0, 0.3) for step in range(0, 100, 25)]
    with pytest.raises(ValueError, match="2 points, 4 usable checkpoints"):
        divergence_report(rows, "naive", gda_free=[0.9, 0.8],
                          noise_low=-0.1, noise_high=0.3)


def test_a_flat_internal_curve_yields_no_d_star_and_says_why():
    """The estimator's own default would report D* = 75 here, off a slope of 2e-18.

    D* is *defined* by internal still rising while external has stopped, so a
    zero floor converts a null result into the paper's headline finding. The
    floor derived from the curve's own standard error rejects it.

    The external curve here rises and then stalls, so the coupling precondition
    is satisfied and what is being tested is the internal floor alone. This
    fixture used to hold an external curve that declined from step 0, which
    reaches the same verdict for the wrong reason -- that case is now
    `test_an_external_curve_that_only_declines_never_had_a_coupling_to_break`.
    """

    rows = [_row(step, -2.0, min(0.30 + step * 0.001, 0.36)) for step in range(0, 200, 25)]
    assert divergence_report(rows, "naive", min_internal_slope=0.0).divergence.d_star == 75.0
    report = divergence_report(rows, "naive")
    assert report.divergence.d_star is None
    assert "internal post-slope" in report.divergence.reason
    assert report.divergence.coupled_candidates == report.divergence.admissible_candidates


def test_the_noise_floor_is_one_standard_error_across_the_whole_run():
    rows = [_row(step, -2.0, 0.3, sem=0.004) for step in range(0, 200, 25)]
    assert internal_noise_slope(rows, "naive") == pytest.approx(0.004 / 175.0)


def test_a_rise_clearly_larger_than_the_noise_still_produces_a_d_star():
    """The floor must reject noise without rejecting the effect being looked for."""

    rows = [_row(step, -2.0 + step * 0.004, min(0.30 + step * 0.001, 0.36))
            for step in range(0, 200, 25)]
    report = divergence_report(rows, "naive")
    assert report.divergence.d_star is not None
    assert report.divergence.internal_post_slope > report.min_internal_slope
    assert report.divergence.external_pre_slope > report.min_external_slope


# ------------------------------------------------- the coupling that D* presupposes


def test_an_external_curve_that_only_declines_never_had_a_coupling_to_break():
    """D* means training broke a coupling. This curve never had one.

    External correctness falls from step 0, so the internal score never
    predicted it and there is nothing for training to have decoupled. Before the
    precondition existed this returned a confident D*, because the acceptance
    test read `slope_after` only.

    The no-break null would also reject this curve, so it is switched off here:
    the point is that the precondition rejects it on its own.
    """

    rows = [_row(step, -2.0 + step * 0.004, 0.30 - step * 0.0005) for step in range(0, 200, 25)]
    report = divergence_report(rows, "naive", max_break_p_value=None)
    assert report.divergence.d_star is None
    assert report.divergence.coupled_candidates == 0
    assert "No coupled phase" in report.divergence.reason
    assert report.divergence.internal_post_slope > report.min_internal_slope


def test_an_external_curve_flat_from_the_start_never_had_a_coupling_to_break():
    """And a bare sign test is not enough to catch it.

    Flat external correctness with a little jitter fits at a pre-slope of
    1.3e-05, which is positive, so `min_external_slope=0.0` still calls this a
    decoupling at step 100. Only the floor built from the binomial precision of
    the curve rejects it -- the same argument
    `test_a_flat_internal_curve_yields_no_d_star_and_says_why` makes for the
    internal side. The no-break null is off for the same reason as in the
    declining case -- one condition at a time.
    """

    rows = [_row(step, -2.0 + step * 0.004, 0.37 + 0.002 * ((index % 3) - 1))
            for index, step in enumerate(range(0, 200, 25))]
    assert divergence_report(rows, "naive", min_external_slope=0.0,
                             max_break_p_value=None).divergence.d_star == 100.0
    report = divergence_report(rows, "naive", max_break_p_value=None)
    assert report.divergence.d_star is None
    assert report.divergence.coupled_candidates == 0
    assert 0.0 < report.divergence.external_pre_slope < report.min_external_slope


# ------------------------------------------- the break the knot search invented


# Eight checkpoints of external correctness that is genuinely flat: 43, 42, 41,
# 62, 36, 57, 50 and 43 correct out of 120. No coupling, no bend, only binomial
# noise. The knot search finds step 75 anyway.
_NOISE_ONLY = [count / 120 for count in (43, 42, 41, 62, 36, 57, 50, 43)]


def test_a_knot_a_straight_line_would_also_have_found_is_not_a_d_star():
    """The slope conditions are marginal checks at a knot the search chose.

    They ask whether the fitted curve rose and then stopped. They cannot ask
    whether the bend beat noise, because the knot was picked by minimising SSE
    over candidates -- searching harder finds a better bend in noise, which is
    why the ungated false-positive rate *rises* with the number of checkpoints
    (32% at 8, 54% at 40; review-packets/no-break-null-20260908).
    """

    rows = [_row(step, -2.0 + step * 0.004, value)
            for step, value in zip(range(0, 200, 25), _NOISE_ONLY)]
    assert divergence_report(rows, "naive", max_break_p_value=None).divergence.d_star == 75.0
    report = divergence_report(rows, "naive")
    assert report.divergence.d_star is None
    assert "No supported break" in report.divergence.reason
    assert report.divergence.break_support.p_value > 0.4


def test_the_no_break_null_is_on_unless_it_is_switched_off():
    rows = [_row(step, -2.0 + step * 0.004, value)
            for step, value in zip(range(0, 200, 25), _NOISE_ONLY)]
    assert divergence_report(rows, "naive").max_break_p_value == 0.05
    assert divergence_report(rows, "naive").to_dict()["break_p_value"] > 0.4
    assert divergence_report(rows, "naive", max_break_p_value=None).to_dict()[
        "break_p_value"] is None


def test_a_bend_far_larger_than_the_noise_still_clears_the_no_break_null():
    """The null must reject noise without rejecting the effect being looked for."""

    rows = [_row(step, -2.0 + step * 0.004, min(0.30 + step * 0.001, 0.36))
            for step in range(0, 200, 25)]
    report = divergence_report(rows, "naive")
    assert report.divergence.d_star is not None
    assert report.divergence.break_support.p_value <= 0.05
    assert report.divergence.break_support.reduction > 0.9


def test_break_support_is_deterministic():
    """A p-value that moved between runs would not be evidence of anything."""

    steps = [float(step) for step in range(0, 200, 25)]
    first = break_support(steps, _NOISE_ONLY, resamples=200)
    assert first == break_support(steps, _NOISE_ONLY, resamples=200)
    assert first != break_support(steps, _NOISE_ONLY, resamples=200, seed=1)


def test_the_external_floor_is_the_median_binomial_error_over_the_span():
    """External correctness is a proportion, so its precision is binomial."""

    rows = [_row(step, -2.0, 0.30) for step in range(0, 200, 25)]
    expected = math.sqrt(0.30 * 0.70 / 120) / 175.0
    assert external_noise_slope(rows, "naive") == pytest.approx(expected)


def test_an_external_curve_with_no_adjudicated_prompts_cannot_be_given_a_floor():
    rows = [_row(step, -2.0, None) for step in range(0, 100, 25)]
    with pytest.raises(ValueError, match="carrying external_correct"):
        divergence_report(rows, "naive")


def test_a_curve_carrying_no_precision_cannot_be_given_a_floor():
    rows = [_row(step, -2.0, 0.3, sem=None) for step in range(0, 100, 25)]
    with pytest.raises(ValueError, match="carrying internal_sem"):
        divergence_report(rows, "naive")


def test_the_standard_error_travels_with_the_mean():
    mean, sem, count = summarize_cycle({"a": -2.0, "b": -1.0, "c": -3.0})
    assert mean == pytest.approx(-2.0)
    assert sem == pytest.approx(1.0 / (3 ** 0.5))
    assert count == 3


def test_a_single_scored_image_has_a_mean_but_no_precision():
    assert summarize_cycle({"a": -2.0}) == (-2.0, None, 1)


def test_an_empty_score_set_is_absent_rather_than_zero():
    assert summarize_cycle({}) == (None, None, 0)
