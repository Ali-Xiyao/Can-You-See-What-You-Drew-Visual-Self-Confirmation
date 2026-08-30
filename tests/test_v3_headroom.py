"""Selection headroom, the step-0 pre-check for Gate C.

The quantity is `Oracle@K - Naive@K`. Its two failure modes are silent, so both
are pinned here: reading a different K than the bank probe's natural draw, and
letting verifier abstentions count as incorrect (which would inflate the oracle
gap and turn a red pre-check green).
"""

from __future__ import annotations

import json
import math

import pytest

from selfsight.v3.headroom import (
    GO_THRESHOLD,
    STOP_THRESHOLD,
    PromptHeadroom,
    attach_selections,
    headroom_report,
    load_natural_pools,
)

NAN = float("nan")


def _row(prompt_id, scores, *, family="existence", selected=None):
    return PromptHeadroom(
        prompt_id=prompt_id,
        family=family,
        gold_scores=tuple(scores),
        candidate_ids=tuple(f"{prompt_id}-c{i}" for i in range(len(scores))),
        selected_candidate_id=selected,
    )


def _packet(tmp_path, index, prompt_id, scores, family="existence", extra=0):
    scored = [
        {
            "candidate_id": f"{prompt_id}-c{i}",
            "sampling_seed": 700_000_000 + i,
            "gold_score": value,
            "abstained": math.isnan(value),
        }
        for i, value in enumerate([*scores, *([1.0] * extra)])
    ]
    path = tmp_path / f"{index:04d}-{prompt_id}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": prompt_id,
                "family": family,
                "search_seeds": [],
                "candidates": [],
                "scored": scored,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_oracle_is_one_when_any_candidate_is_correct():
    assert _row("p", [0.0, 0.0, 1.0, 0.0]).oracle == 1.0
    assert _row("p", [0.0, 0.0, 0.0, 0.0]).oracle == 0.0


def test_natural_rate_averages_the_pool_not_the_first_draw():
    row = _row("p", [1.0, 0.0, 0.0, 0.0])

    assert row.natural == 0.25
    assert row.first_candidate == 1.0


def test_abstentions_are_excluded_rather_than_scored_as_incorrect():
    row = _row("p", [NAN, NAN, 1.0, 1.0])

    assert row.natural == 1.0
    assert row.oracle == 1.0
    assert row.informative is False


def test_a_fully_abstained_pool_is_unusable():
    row = _row("p", [NAN, NAN, NAN, NAN])

    assert row.usable is False
    assert row.oracle == 0.0


def test_informative_requires_both_a_correct_and_an_incorrect_candidate():
    assert _row("p", [1.0, 0.0, 1.0, 0.0]).informative is True
    assert _row("p", [1.0, 1.0, 1.0, 1.0]).informative is False
    assert _row("p", [0.0, 0.0, 0.0, 0.0]).informative is False


def test_selected_score_reads_through_the_candidate_id():
    row = _row("p", [0.0, 1.0, 0.0, 0.0], selected="p-c1")

    assert row.selected == 1.0
    assert _row("p", [0.0, 1.0], selected=None).selected is None


def test_load_reads_only_the_natural_k_draw(tmp_path):
    _packet(tmp_path, 0, "p0", [0.0, 0.0, 0.0, 0.0], extra=12)

    rows = load_natural_pools([tmp_path], candidate_k=4)

    assert len(rows) == 1
    assert rows[0].gold_scores == (0.0, 0.0, 0.0, 0.0)
    assert rows[0].oracle == 0.0


def test_load_rejects_a_short_pool(tmp_path):
    _packet(tmp_path, 0, "p0", [1.0, 0.0])

    with pytest.raises(ValueError, match="fewer than 4"):
        load_natural_pools([tmp_path], candidate_k=4)


def test_load_rejects_a_prompt_appearing_in_two_shards(tmp_path):
    left = tmp_path / "gpu0"
    right = tmp_path / "gpu1"
    left.mkdir()
    right.mkdir()
    _packet(left, 0, "p0", [1.0, 0.0, 1.0, 0.0])
    _packet(right, 0, "p0", [1.0, 0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="Duplicate prompt"):
        load_natural_pools([left, right], candidate_k=4)


def test_report_separates_the_ceiling_from_the_headroom():
    rows = [
        _row("p0", [1.0, 0.0, 0.0, 0.0], selected="p0-c0"),
        _row("p1", [1.0, 0.0, 0.0, 0.0], selected="p1-c1"),
    ]

    report = headroom_report(rows, candidate_k=4)
    overall = report["overall"]

    assert overall["natural_rate"] == 0.25
    assert overall["oracle_at_k"] == 1.0
    assert overall["selection_ceiling"] == 0.75
    assert overall["naive_cycle_rate"] == 0.5
    assert overall["headroom_over_naive"] == 0.5


def test_verdict_follows_the_registered_thresholds():
    def verdict_for(naive_hits):
        rows = [
            _row(f"p{i}", [1.0, 0.0, 0.0, 0.0], selected=f"p{i}-c{0 if hit else 1}")
            for i, hit in enumerate(naive_hits)
        ]
        return headroom_report(rows, candidate_k=4)["verdict"]

    assert verdict_for([False] * 100) == "go"
    assert verdict_for([True] * 100) == "stop"
    assert verdict_for([True] * 96 + [False] * 4) == "enlarge"
    assert STOP_THRESHOLD < 0.04 < GO_THRESHOLD


def test_verdict_is_pending_until_the_naive_arm_is_scored():
    report = headroom_report([_row("p0", [1.0, 0.0, 0.0, 0.0])], candidate_k=4)

    assert report["overall"]["naive_cycle_rate"] is None
    assert report["overall"]["headroom_over_naive"] is None
    assert report["verdict"] == "pending_naive_selection"


def test_report_is_split_by_family():
    rows = [
        _row("p0", [1.0, 1.0, 1.0, 1.0], family="existence"),
        _row("p1", [0.0, 0.0, 0.0, 0.0], family="spatial"),
    ]

    report = headroom_report(rows, candidate_k=4)

    assert set(report["by_family"]) == {"existence", "spatial"}
    assert report["by_family"]["existence"]["oracle_at_k"] == 1.0
    assert report["by_family"]["spatial"]["oracle_at_k"] == 0.0


def test_attach_selections_leaves_unscored_prompts_pending():
    rows = [_row("p0", [1.0, 0.0, 0.0, 0.0]), _row("p1", [1.0, 0.0, 0.0, 0.0])]

    attached = attach_selections(rows, {"p0": "p0-c0"})

    assert attached[0].selected == 1.0
    assert attached[1].selected is None
    assert headroom_report(attached, candidate_k=4)["overall"][
        "naive_scored_prompts"
    ] == 1


def test_empty_report_raises():
    with pytest.raises(ValueError, match="No prompts"):
        headroom_report([], candidate_k=4)
