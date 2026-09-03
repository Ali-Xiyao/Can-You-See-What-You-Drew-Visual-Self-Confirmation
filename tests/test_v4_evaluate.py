"""The four ways a D* could be produced from numbers that do not mean it.

None of these raise on their own. A prompt appearing in both the training and
outcome sets makes the external curve measure memorisation; an unadjudicated
image counted as wrong bends that curve down for a reason unrelated to the
hypothesis; a checkpoint missing one of its two curves silently pairs one step's
internal score with another's external; and a lead reported against a D_g that
was never estimated turns "we could not measure this" into a number.
"""

from __future__ import annotations

import json

import pytest

from selfsight.v4.evaluate import (
    CheckpointMetrics,
    curves,
    cycle_scores,
    divergence_report,
    evaluation_seed,
    internal_noise_slope,
    external_correctness,
    read_metrics_csv,
    split_prompts,
    summarize_cycle,
    write_evaluation_manifest,
    write_metrics_csv,
)
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
        images={"a": "/img/a.png", "b": "/img/b.png"},
        seeds={"a": 11, "b": 22},
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["spec_id"] for row in rows] == ["a", "b"]
    assert all(row["candidate_index"] == 0 for row in rows)
    assert all({"spec", "image_path", "seed"} <= set(row) for row in rows)
    assert SceneSpec.from_dict(rows[0]["spec"]).prompt == specs["a"].prompt


def test_a_missing_image_stops_the_manifest_rather_than_shortening_it(tmp_path):
    with pytest.raises(KeyError, match="No generated image"):
        write_evaluation_manifest(tmp_path, specs={"a": _spec("a")}, prompt_ids=["a"],
                                  images={}, seeds={"a": 1})


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
    """

    rows = [_row(step, -2.0, 0.30 - step * 0.0005) for step in range(0, 200, 25)]
    assert divergence_report(rows, "naive", min_internal_slope=0.0).divergence.d_star == 75.0
    report = divergence_report(rows, "naive")
    assert report.divergence.d_star is None
    assert "internal post-slope" in report.divergence.reason


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
