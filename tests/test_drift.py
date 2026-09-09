"""Tests for deviation 7.2's drift bound and deviation 7.3's round bootstrap.

Deviation 7.2 registered an inequality and left four things to whoever wrote
the script: which keys the three rates are computed on, how they are
aggregated, which step, and what happens with five replicates instead of one.
Deviation 14 pinned all four. These tests pin the code to deviation 14, and
one of them pins it to endpoint 1 by running both and comparing, because "the
same caliber" is a claim that can be true in the prose and false in the
arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfsight.analysis.drift import (
    BASE_STEP,
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    NOT_SEPARABLE,
    final_common_step,
    load_columns,
    prompt_mean,
    round_bootstrap,
    split_digest,
)
from selfsight.analysis.endpoint1 import load_checkpoint, paired_difference

# The shape a real split.json has: a wall clock stamp, the recipe hash, and
# the three prompt lists. The stamp used to be in the identity hash.
SPLIT = {"created": "2026-09-08T14:33:14Z", "digest": "s" * 64,
         "train": ["p1", "p2"], "outcome": ["p3", "p4"]}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _column(run: Path, arm: str, step: int, verdicts: dict[tuple[str, int], bool | None],
            *, manifest: bool = True) -> None:
    directory = run / "evaluations" / arm / f"step-{step:05d}"
    _write(directory / "verified.jsonl",
           [{"spec_id": spec_id, "candidate_index": index,
             "image_correct": value if value is not None else None,
             "resolution": "agreed" if value is not None else "pending_human"}
            for (spec_id, index), value in sorted(verdicts.items())])
    if manifest:
        _write(directory / "manifest.jsonl",
               [{"spec_id": spec_id, "candidate_index": index}
                for spec_id, index in sorted(verdicts)])


def _run(tmp_path: Path, name: str, *, split: dict | None = None) -> Path:
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "split.json").write_text(json.dumps(split if split is not None else SPLIT),
                                    encoding="utf-8")
    return run


def _pair(values: list[bool]) -> dict[tuple[str, int], bool | None]:
    """Four keys over two prompts, in a fixed order."""

    keys = [("p1", 0), ("p1", 1), ("p2", 0), ("p2", 1)]
    return dict(zip(keys, values))


def _setup(tmp_path: Path, *, a, a_prime, b, step: int = 8):
    main = _run(tmp_path, "main")
    replicate = _run(tmp_path, "e3-s20260906")
    _column(main, "naive", step, a)
    _column(main, "rfo_gold", step, a)
    _column(replicate, "naive", step, a_prime)
    _column(replicate, "blind_self", step, b)
    return main, replicate


# --- the key set ------------------------------------------------------------


def test_the_three_thetas_share_a_key_set(tmp_path: Path):
    """One key undecided anywhere removes it from all three rates.

    Deviation 14.2 exists because 7.2 compares |theta_A - theta_A'| against
    theta_B - theta_A' directly. Two rates over different denominators can be
    ordered but the ordering does not mean what the rule says it means.
    """

    main, replicate = _setup(
        tmp_path,
        a=_pair([True, True, True, True]),
        a_prime=_pair([True, None, True, True]),
        b=_pair([True, True, True, True]))
    columns = load_columns(main, replicate)
    assert columns.keys == (("p1", 0), ("p2", 0), ("p2", 1))
    # p1 keeps one draw, p2 keeps two: (1.0 + 1.0) / 2 in every column.
    assert columns.theta_a == pytest.approx(1.0)
    assert columns.theta_a_prime == pytest.approx(1.0)
    assert columns.theta_b == pytest.approx(1.0)


def test_the_intersection_cost_is_reported(tmp_path: Path):
    main, replicate = _setup(
        tmp_path,
        a=_pair([True, True, True, True]),
        a_prime=_pair([True, None, True, True]),
        b=_pair([None, None, True, True]))
    columns = load_columns(main, replicate)
    assert columns.dropped == {"a": 2, "a_prime": 1, "b": 0}


def test_the_marginals_are_each_column_s_own_keys(tmp_path: Path):
    # An intersection that throws a lot away is a fact about the run, so the
    # rate each column would have reported alone travels beside the shared one.
    main, replicate = _setup(
        tmp_path,
        a=_pair([True, False, True, False]),
        a_prime=_pair([True, None, True, None]),
        b=_pair([True, True, True, True]))
    columns = load_columns(main, replicate)
    assert columns.marginal["a"] == pytest.approx(0.5)
    assert columns.marginal["a_prime"] == pytest.approx(1.0)
    assert columns.theta_a == pytest.approx(1.0)


def test_theta_is_the_prompt_mean_not_the_flat_mean():
    """Deviation 14.1's second half, which the first version of it left open.

    Endpoint 1 averages within a prompt and then across prompts (deviation
    10.2 point 2). While every prompt keeps all its draws the two agree, so a
    fixture with equal counts cannot tell them apart -- this one has an unequal
    count on purpose.
    """

    column = {("p1", 0): True, ("p1", 1): False,
              ("p2", 0): True, ("p2", 1): True, ("p2", 2): True}
    assert prompt_mean(column, sorted(column)) == pytest.approx(0.75)
    assert sum(column.values()) / len(column) == pytest.approx(0.8)


# --- the inequality ---------------------------------------------------------


def test_the_effect_is_signed_and_the_drift_is_not(tmp_path: Path):
    """B below A' cannot clear a bound, and an absolute value would hide it."""

    main, replicate = _setup(
        tmp_path,
        a=_pair([True, True, True, True]),
        a_prime=_pair([True, True, True, True]),
        b=_pair([False, False, False, False]))
    columns = load_columns(main, replicate)
    assert columns.drift == pytest.approx(0.0)
    assert columns.effect == pytest.approx(-1.0)
    assert columns.separable is False
    assert columns.wording == NOT_SEPARABLE


def test_a_drift_equal_to_the_effect_is_not_separable(tmp_path: Path):
    # 7.2 writes "<" on one side and ">=" on the other, so the boundary belongs
    # to the unfavourable branch. Writing it as "<=" would move a registered
    # failure into a registered success, which is why the two rates here are
    # arranged to land exactly on it: 0.75 - 0.50 on both sides.
    main, replicate = _setup(
        tmp_path,
        a=_pair([True, True, True, False]),
        a_prime=_pair([True, True, False, False]),
        b=_pair([True, True, True, False]))
    columns = load_columns(main, replicate)
    assert columns.drift == pytest.approx(0.25)
    assert columns.effect == pytest.approx(0.25)
    assert columns.separable is False


def test_a_drift_just_under_the_effect_is_separable(tmp_path: Path):
    # The other side of the same boundary, so the test above is pinning where
    # the line falls rather than that nothing ever clears it.
    main, replicate = _setup(
        tmp_path,
        a=_pair([True, True, True, False]),
        a_prime=_pair([True, True, False, False]),
        b=_pair([True, True, True, True]))
    columns = load_columns(main, replicate)
    assert columns.drift == pytest.approx(0.25)
    assert columns.effect == pytest.approx(0.5)
    assert columns.separable is True


# --- the guards -------------------------------------------------------------


def test_two_runs_with_different_splits_are_refused(tmp_path: Path):
    main = _run(tmp_path, "main")
    replicate = _run(tmp_path, "e3-s20260906", split={"outcome": ["p9"]})
    for arm in ("naive", "rfo_gold"):
        _column(main, arm, 8, _pair([True] * 4))
    for arm in ("naive", "blind_self"):
        _column(replicate, arm, 8, _pair([True] * 4))
    with pytest.raises(ValueError, match="different splits"):
        load_columns(main, replicate)


def test_a_run_without_a_split_is_refused(tmp_path: Path):
    run = tmp_path / "nameless"
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="cannot confirm"):
        split_digest(run)


def test_the_split_digest_ignores_key_order(tmp_path: Path):
    first = _run(tmp_path, "a", split={"train": ["p1"], "outcome": ["p2"]})
    second = _run(tmp_path, "b", split={"outcome": ["p2"], "train": ["p1"]})
    assert split_digest(first) == split_digest(second)


def test_the_final_step_is_the_smallest_of_the_three_final_steps(tmp_path: Path):
    main = _run(tmp_path, "main")
    replicate = _run(tmp_path, "e3-s20260906")
    for step in (0, 8, 16):
        _column(main, "naive", step, _pair([True] * 4))
        _column(replicate, "naive", step, _pair([True] * 4))
    for step in (0, 8):
        _column(replicate, "blind_self", step, _pair([True] * 4))
    assert final_common_step([(main, "naive"), (replicate, "naive"),
                              (replicate, "blind_self")]) == 8


def test_a_gapped_ladder_is_refused(tmp_path: Path):
    """One column missing the middle of the ladder is not a shorter ladder.

    Deviation 14.4 says "the smallest of the three final steps", which assumes
    the ladders are prefixes of each other. They should be. When they are not,
    something is wrong with the run, and sliding down to a step all three
    happen to share would hide it behind a plausible number.
    """

    main = _run(tmp_path, "main")
    replicate = _run(tmp_path, "e3-s20260906")
    for step in (0, 8, 16):
        _column(main, "naive", step, _pair([True] * 4))
    for step in (0, 16):
        _column(replicate, "naive", step, _pair([True] * 4))
    for step in (0, 8):
        _column(replicate, "blind_self", step, _pair([True] * 4))
    # min(16, 16, 8) = 8, which the replicate's own naive column skipped.
    with pytest.raises(ValueError, match="gapped ladder"):
        final_common_step([(main, "naive"), (replicate, "naive"),
                           (replicate, "blind_self")])


def test_an_arm_with_no_adjudicated_checkpoint_is_refused(tmp_path: Path):
    main = _run(tmp_path, "main")
    replicate = _run(tmp_path, "e3-s20260906")
    _column(main, "naive", 8, _pair([True] * 4))
    _column(replicate, "naive", 8, _pair([True] * 4))
    with pytest.raises(ValueError, match="no adjudicated checkpoint"):
        final_common_step([(main, "naive"), (replicate, "naive"),
                           (replicate, "blind_self")])


def test_no_shared_key_is_refused(tmp_path: Path):
    # Not a small denominator -- no denominator. 7.2 compares two rates, and
    # numpy would hand back nan < nan = False, which prints as "not separable"
    # and reads like a measurement.
    main, replicate = _setup(
        tmp_path,
        a=_pair([True, True, None, None]),
        a_prime=_pair([None, None, True, True]),
        b=_pair([True, True, True, True]))
    with pytest.raises(ValueError, match="no key is adjudicated in all three"):
        load_columns(main, replicate)


def test_the_unadjudicated_rule_is_endpoint_1_s(tmp_path: Path):
    # `resolution` in UNADJUDICATED and a non-bool image_correct both mean the
    # same thing to endpoint 1, and have to mean it here.
    main, replicate = _setup(
        tmp_path,
        a=_pair([True, True, True, True]),
        a_prime=_pair([True, True, True, True]),
        b=_pair([True, None, True, True]))
    columns = load_columns(main, replicate)
    assert ("p1", 1) not in columns.keys


# --- the round axis ---------------------------------------------------------


def test_step_zero_is_excluded_from_the_round_bootstrap():
    """Deviation 14.5. Both arms are the same model on the same latents at
    step 0, so their verdicts are bit-identical and the difference is 0 by
    construction. Averaging it in dilutes the mean with a non-measurement."""

    result = round_bootstrap({BASE_STEP: 0.0, 8: 0.2, 16: 0.2}, resamples=50, seed=1)
    assert result.steps == (8, 16)
    assert result.mean == pytest.approx(0.2)


def test_the_bootstrap_resamples_rounds():
    varied = round_bootstrap({8: 0.0, 16: 0.4, 24: 0.8}, resamples=200, seed=2)
    assert len(set(varied.differences)) == 3
    assert varied.interval[0] < varied.mean < varied.interval[1]


def test_one_round_has_no_spread_to_find():
    only = round_bootstrap({8: 0.3}, resamples=50, seed=3)
    assert only.interval == pytest.approx((0.3, 0.3))
    assert only.spans_zero is False


def test_a_ladder_with_only_step_zero_has_no_round(tmp_path: Path):
    with pytest.raises(ValueError, match="step 0 is excluded"):
        round_bootstrap({BASE_STEP: 0.1})


def test_the_round_bootstrap_defaults_are_the_final_analysis_branch():
    # Deviation 6.4 split the bootstrap constants into a descriptive branch
    # (20260906 / 2000, still running inside the report) and a final-analysis
    # branch. Deviation 14.5 puts this on the second one.
    assert FINAL_RESAMPLES == 20000
    assert FINAL_BOOTSTRAP_SEED == 20260908
    taken = round_bootstrap({8: 0.1, 16: 0.2})
    assert (taken.resamples, taken.seed) == (FINAL_RESAMPLES, FINAL_BOOTSTRAP_SEED)


def test_the_interval_is_the_registered_2_5_and_97_5_tail():
    """Three rounds, one of them 1.0, and the tail probabilities do the work.

    A resample that draws the single 1.0 round three times has probability
    1/27 = 3.7%, so it sits inside the top 2.5% tail and the upper end is 1.0.
    The 95th percentile would fall in the 2/3 block instead, so a widened tail
    shows up here rather than being a matter of taste. The lower end is 0.0 by
    the same argument at the bottom, which also makes the interval one that
    touches zero -- deviation 7.3 reports that and attaches nothing to it.
    """

    result = round_bootstrap({8: 0.0, 16: 0.0, 24: 1.0})
    assert result.interval == pytest.approx((0.0, 1.0))
    assert result.spans_zero is True


def test_each_resample_draws_a_whole_ladder_of_rounds():
    # Four rounds, one of them 0.0: drawing it four times has probability
    # 1/256, below the tail, so the lower end sits above zero. Resampling one
    # round per draw instead would put a 0.0 in a quarter of them.
    result = round_bootstrap({8: 0.0, 16: 0.4, 24: 0.4, 32: 0.4})
    assert result.interval[0] > 0.0
    assert result.mean == pytest.approx(0.3)


def test_the_seed_moves_the_interval_and_not_the_point():
    ladder = {step: value / 10 for step, value in zip(range(8, 72, 8), range(8))}
    first = round_bootstrap(ladder, resamples=101, seed=1)
    second = round_bootstrap(ladder, resamples=101, seed=2)
    assert first.seed == 1
    assert first.interval != second.interval
    assert first.mean == pytest.approx(second.mean)


# --- against endpoint 1 -----------------------------------------------------


def test_the_effect_is_endpoint_1_s_paired_difference(tmp_path: Path):
    """Run both and compare, rather than assert the calibers match.

    When the shared key set is the replicate's own paired set, theta_B minus
    theta_A' is exactly what endpoint 1 computes for that checkpoint. Any
    disagreement -- flat versus prompt mean, a different deletion rule, a
    transposed arm -- lands here.
    """

    main, replicate = _setup(
        tmp_path,
        a=_pair([True, False, True, False]),
        a_prime=_pair([True, False, False, False]),
        b=_pair([True, True, True, False]))
    columns = load_columns(main, replicate)
    endpoint = paired_difference(load_checkpoint(replicate, 8, "naive", "blind_self"))
    assert columns.effect == pytest.approx(endpoint)
    assert columns.effect == pytest.approx(0.5)


def test_two_runs_split_at_different_times_have_the_same_identity(tmp_path: Path):
    """`created` is a wall clock stamp, and the guard must not read it.

    Every real split.json carries one -- the main run's says
    2026-09-08T14:33:14Z -- and two runs that hold out the same prompts under
    the same config write it at two different times by construction. A digest
    that includes it makes deviation 7.2's guard fire on every legitimate
    pair, at analysis time, after the GPU work is already spent.
    """

    first = _run(tmp_path, "main", split={**SPLIT, "created": "2026-09-08T14:33:14Z"})
    second = _run(tmp_path, "replicate", split={**SPLIT, "created": "2026-09-11T02:07:55Z"})
    assert split_digest(first) == split_digest(second)


def test_the_recorded_digest_alone_is_not_what_is_compared(tmp_path: Path):
    """Same recipe hash, different prompt lists, and the guard must still fire.

    `split.json` carries a `digest` field and reading it would be the cheaper
    check. That field hashes (runs, seed, outcome, probe) -- the recipe -- so
    a file whose prompt lists were edited afterwards keeps it. Deviation 7.2
    needs the two runs to have measured the same questions, not to have been
    configured the same way.
    """

    first = _run(tmp_path, "main",
                 split={**SPLIT, "created": "2026-09-08T14:33:14Z", "outcome": ["p3", "p4"]})
    second = _run(tmp_path, "replicate",
                  split={**SPLIT, "created": "2026-09-11T02:07:55Z", "outcome": ["p3", "p9"]})
    assert first.name != second.name
    assert split_digest(first) != split_digest(second)


def test_the_train_side_is_in_the_identity_too(tmp_path: Path):
    # Held-in prompts decide what the two runs trained on; two runs that split
    # the same corpus differently are not comparable even if their evaluation
    # lists happen to coincide.
    first = _run(tmp_path, "main", split={**SPLIT, "train": ["p1", "p2"]})
    second = _run(tmp_path, "replicate", split={**SPLIT, "train": ["p1", "p5"]})
    assert split_digest(first) != split_digest(second)
