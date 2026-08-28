from __future__ import annotations

from selfsight.schemas import SelectionDecision
from selfsight.v23.selection import (
    V23_ARMS,
    finite_score_span,
    gold_selection_advantage,
    select_common_informative,
)


def _decision(
    prompt: str,
    arm: str,
    selected: str | None,
    scores: dict[str, float],
    *,
    pool: tuple[str, ...] = ("a", "b", "c", "d"),
) -> SelectionDecision:
    return SelectionDecision(
        prompt_id=prompt,
        arm=arm,
        candidate_pool_ids=pool,
        selected_candidate_id=selected,
        scores=scores,
        selector_id=arm,
        observer_revision="test",
        abstain=selected is None,
    )


def test_common_informative_filter_is_symmetric_and_ordered():
    scores_good = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}
    scores_flat = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    decisions = {
        arm: [
            _decision("p2", arm, "b", scores_good),
            _decision("p1", arm, "a", scores_flat),
        ]
        for arm in V23_ARMS
    }
    paired, report = select_common_informative(
        ("p1", "p2"), decisions, minimum_gold_gap=1.0
    )
    assert report["accepted_prompt_ids"] == ["p2"]
    assert report["paired_pool_order_identical"] is True
    assert [value.prompt_id for value in paired["naive"]] == ["p2"]
    assert [value.prompt_id for value in paired["rfo_self"]] == ["p2"]
    assert [value.prompt_id for value in paired["rfo_gold"]] == ["p2"]


def test_common_filter_fails_loudly_on_pool_mismatch():
    scores = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}
    decisions = {
        "naive": [_decision("p", "naive", "a", scores)],
        "rfo_self": [_decision("p", "rfo_self", "a", scores)],
        "rfo_gold": [
            _decision("p", "rfo_gold", "b", scores, pool=("a", "b", "c", "x"))
        ],
    }
    try:
        select_common_informative(("p",), decisions, minimum_gold_gap=1.0)
    except RuntimeError as error:
        assert "Candidate-pool mismatch" in str(error)
    else:
        raise AssertionError("Expected a fail-loud candidate-pool mismatch")


def test_gold_score_span_and_selection_advantage():
    scores = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}
    naive = _decision("p", "naive", "a", scores)
    gold = _decision("p", "rfo_gold", "b", scores)
    assert finite_score_span(gold) == 1.0
    assert gold_selection_advantage((naive,), (gold,)) == 1.0
