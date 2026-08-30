"""Bounded candidate-bank search and Gate A supply accounting."""

from __future__ import annotations

import pytest

from selfsight.v3.bank import (
    GRADIENT_SEED_BASE,
    BankCandidate,
    assert_disjoint_seed_domains,
    assert_pools_identical_across_arms,
    bank_supply_report,
    build_balanced_pool,
    paired_pool_assignment,
    search_seeds,
)


def _bank(scores, *, abstained=()):
    return [
        BankCandidate(
            candidate_id=f"c{index:02d}",
            sampling_seed=700_000_000 + index,
            gold_score=score,
            abstained=index in abstained,
        )
        for index, score in enumerate(scores)
    ]


def test_balanced_pool_takes_two_correct_and_two_incorrect():
    pool = build_balanced_pool("p1", "color", _bank([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    assert pool.balanced is True
    assert pool.informative is True
    assert len(pool.candidate_ids) == 4
    assert len(pool.correct_ids) == 2
    assert len(pool.incorrect_ids) == 2
    assert pool.reason == "balanced"


def test_selection_is_deterministic_by_seed_then_id():
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    first = build_balanced_pool("p1", "color", _bank(scores))
    shuffled = list(reversed(_bank(scores)))
    second = build_balanced_pool("p1", "color", shuffled)

    assert first.candidate_ids == second.candidate_ids


def test_degenerate_pools_are_rejected_not_padded():
    """79% of natural v2.3 pools were all-correct or all-wrong; those must drop out."""

    all_correct = build_balanced_pool("p1", "existence", _bank([1.0, 1.0, 1.0, 1.0]))
    assert all_correct.informative is False
    assert all_correct.reason == "no_verifier_incorrect_candidate"
    assert all_correct.candidate_ids == ()

    all_wrong = build_balanced_pool("p2", "existence", _bank([0.0, 0.0, 0.0, 0.0]))
    assert all_wrong.informative is False
    assert all_wrong.reason == "no_verifier_correct_candidate"


def test_unbalanced_but_informative_pool_still_reaches_k():
    pool = build_balanced_pool("p1", "spatial", _bank([1.0, 0.0, 0.0, 0.0, 0.0]))

    assert pool.informative is True
    assert pool.balanced is False
    assert pool.reason == "informative_but_unbalanced"
    assert len(pool.candidate_ids) == 4
    assert len(pool.correct_ids) == 1


def test_abstentions_reduce_usable_supply():
    pool = build_balanced_pool(
        "p1", "color", _bank([1.0, 0.0, 0.0, 0.0, 0.0], abstained={1, 2, 3})
    )
    assert pool.abstained == 3
    assert pool.informative is False
    assert pool.reason == "too_many_abstentions"


def test_search_seeds_never_collide_with_gradient_seeds():
    seeds = search_seeds("v3-existence-0005", bank_size=16)
    assert len(set(seeds)) == 16
    assert max(seeds) < GRADIENT_SEED_BASE
    # Keyed by scene id, so sharding or reordering cannot reuse a block.
    assert search_seeds("v3-existence-0005", 16) == seeds
    assert set(search_seeds("v3-color-0005", 16)).isdisjoint(seeds)

    assert_disjoint_seed_domains(seeds, [GRADIENT_SEED_BASE, GRADIENT_SEED_BASE + 1])
    with pytest.raises(ValueError, match="collide"):
        assert_disjoint_seed_domains(seeds, [seeds[0]])


def test_gate_a_fails_at_the_measured_natural_rate():
    """9 informative of 42 prompts is the real v2.3 number; Gate A must reject it."""

    pools = [
        build_balanced_pool(f"p{index}", "existence", _bank([1.0, 0.0, 0.0, 0.0]))
        if index < 9
        else build_balanced_pool(f"p{index}", "existence", _bank([1.0, 1.0, 1.0, 1.0]))
        for index in range(42)
    ]
    report = bank_supply_report(pools, families=["existence"])

    assert report["passed"] is False
    assert report["informative_pools"] == 9
    assert report["informative_rate"] == pytest.approx(9 / 42)
    assert report["informative_supply_ok"] is False
    assert "capability-floor" in report["action_if_failed"]


def test_gate_a_passes_with_sufficient_balanced_supply():
    pools = [
        build_balanced_pool(f"p{index}", "color", _bank([1.0, 1.0, 0.0, 0.0, 0.0]))
        for index in range(140)
    ]
    report = bank_supply_report(
        pools, families=["color"], min_informative_pools=128, natural_informative_rate=0.21
    )

    assert report["passed"] is True
    assert report["balanced_rate"] == pytest.approx(1.0)
    assert report["bank_lift"] == pytest.approx(0.79)
    assert report["by_family"]["color"]["balanced"] == 140


def test_gate_a_reports_rejection_reasons_per_family():
    pools = [
        build_balanced_pool("p0", "existence", _bank([1.0, 1.0, 0.0, 0.0])),
        build_balanced_pool("p1", "color", _bank([1.0, 1.0, 1.0, 1.0])),
        build_balanced_pool("p2", "spatial", _bank([0.0, 0.0, 0.0, 0.0])),
    ]
    report = bank_supply_report(pools, families=["existence", "color", "spatial"])

    assert report["rejection_reasons"]["no_verifier_incorrect_candidate"] == 1
    assert report["rejection_reasons"]["no_verifier_correct_candidate"] == 1
    assert report["by_family"]["color"]["balanced"] == 0
    assert report["by_family"]["existence"]["balanced"] == 1


def test_arms_receive_an_identical_prompt_set():
    pools = [
        build_balanced_pool("keep", "color", _bank([1.0, 1.0, 0.0, 0.0])),
        build_balanced_pool("drop", "color", _bank([1.0, 1.0, 1.0, 1.0])),
    ]
    assignment = paired_pool_assignment(pools, ["naive", "rfo_gold", "rfo_self"])

    assert assignment["naive"] == ["keep"]
    assert assignment["naive"] == assignment["rfo_gold"] == assignment["rfo_self"]


def test_pool_mismatch_across_arms_fails_loudly():
    good = {"naive": [["a", "b"]], "rfo_gold": [["a", "b"]]}
    assert_pools_identical_across_arms(good)

    bad = {"naive": [["a", "b"]], "rfo_gold": [["a", "c"]]}
    with pytest.raises(ValueError, match="different candidate pools"):
        assert_pools_identical_across_arms(bad)
