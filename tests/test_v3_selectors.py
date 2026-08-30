"""Restored Naive cycle criterion and the section-5 observation ladder.

The v2.3 Naive arm asked the atomic question, which is the RFO question form, so
Naive and RFO-Self were the same function at step 0 (agreement exactly 1.000).
These tests pin the two properties that made that failure invisible: the cycle
criterion must rank on the prompt likelihood rather than on atomic answers, and
every ladder condition must leave the scoring path untouched.
"""

from __future__ import annotations

import pytest

from selfsight.schemas import AtomicQuestion, CandidateRecord, QuestionFamily
from selfsight.v3.observation_ladder import (
    COUNTERFACTUAL,
    EXTERNAL,
    FRESH_IMAGE_ONLY,
    FRESH_PROMPTED,
    LADDER,
    PROVENANCE_TEMPLATE,
    SAME_SESSION,
    contextualized_question,
    descending_rungs,
)
from selfsight.v3.selectors import (
    cycle_scores,
    cycle_selection,
    select_by_score,
    selection_agreement,
)

PROMPT = "a blue box to the left of a red box"


def _candidates(count=4, *, prompt_id="p1"):
    return [
        CandidateRecord(
            candidate_id=f"c{index:02d}",
            prompt_id=prompt_id,
            scene_id=prompt_id,
            sampling_seed=700_000_000 + index,
            image_path=f"/tmp/{prompt_id}-{index}.png",
            rgb_sha256=f"{index:064x}",
            generator_id="showlab/show-o2-1.5B-HQ",
            generator_revision="d3a220ec",
            checkpoint_id="step-0",
        )
        for index in range(count)
    ]


def _question(text="What color is the left box?"):
    return AtomicQuestion("q1", "a1", QuestionFamily.COLOR, text, "blue")


class _Scorer:
    """Records every call so the tests can assert what the criterion consumed."""

    def __init__(self, by_path):
        self.by_path = by_path
        self.calls = []

    def cycle_consistency_score(self, image_path, prompt):
        self.calls.append((image_path, prompt))
        return self.by_path[image_path]


def test_cycle_selection_picks_the_highest_prompt_likelihood():
    candidates = _candidates()
    scorer = _Scorer(
        {
            candidates[0].image_path: -3.0,
            candidates[1].image_path: -1.5,
            candidates[2].image_path: -9.0,
            candidates[3].image_path: -2.0,
        }
    )

    decision = cycle_selection(
        prompt_id="p1",
        candidates=candidates,
        prompt=PROMPT,
        scorer=scorer,
        selector_id="showo2/cycle",
        observer_revision="d3a220ec",
    )

    assert decision.selected_candidate_id == "c01"
    assert decision.arm == "naive_cycle"
    assert decision.abstain is False


def test_cycle_scoring_uses_the_same_target_text_for_every_candidate():
    candidates = _candidates()
    scorer = _Scorer({candidate.image_path: -1.0 for candidate in candidates})

    cycle_scores(candidates, PROMPT, scorer)

    assert [prompt for _, prompt in scorer.calls] == [PROMPT] * 4
    assert [path for path, _ in scorer.calls] == [c.image_path for c in candidates]


def test_cycle_scoring_requires_the_prompt():
    with pytest.raises(ValueError, match="original prompt"):
        cycle_scores(_candidates(), "", _Scorer({}))


def test_ties_break_by_seed_then_id_like_the_rfo_selector():
    candidates = _candidates()
    scores = {candidate.candidate_id: -1.0 for candidate in candidates}

    decision = select_by_score(
        prompt_id="p1",
        arm="naive_cycle",
        candidates=list(reversed(candidates)),
        scores=scores,
        selector_id="showo2/cycle",
        observer_revision="d3a220ec",
        reason="tie",
    )

    assert decision.selected_candidate_id == "c00"


def test_non_finite_scores_sort_last_but_do_not_abstain():
    candidates = _candidates(count=2)
    scores = {"c00": float("nan"), "c01": -8.0}

    decision = select_by_score(
        prompt_id="p1",
        arm="naive_cycle",
        candidates=candidates,
        scores=scores,
        selector_id="showo2/cycle",
        observer_revision="d3a220ec",
        reason="partial",
    )

    assert decision.selected_candidate_id == "c01"
    assert decision.abstain is False


def test_all_non_finite_scores_abstain():
    candidates = _candidates(count=2)
    scores = {"c00": float("nan"), "c01": float("nan")}

    decision = select_by_score(
        prompt_id="p1",
        arm="naive_cycle",
        candidates=candidates,
        scores=scores,
        selector_id="showo2/cycle",
        observer_revision="d3a220ec",
        reason="partial",
    )

    assert decision.selected_candidate_id is None
    assert decision.abstain is True
    assert decision.reason == "all candidates scored non-finite"


def test_missing_score_raises_rather_than_reordering_one_arm():
    with pytest.raises(KeyError):
        select_by_score(
            prompt_id="p1",
            arm="naive_cycle",
            candidates=_candidates(count=2),
            scores={"c00": -1.0},
            selector_id="showo2/cycle",
            observer_revision="d3a220ec",
            reason="incomplete",
        )


def test_selection_agreement_reproduces_the_v23_identity_and_detects_a_change():
    candidates = _candidates()

    def favouring(winner):
        return _Scorer(
            {
                candidate.image_path: 0.0 if candidate.candidate_id == winner else -1.0
                for candidate in candidates
            }
        )

    identical = favouring("c02")
    other = favouring("c03")

    left = [
        cycle_selection(
            prompt_id="p1",
            candidates=candidates,
            prompt=PROMPT,
            scorer=identical,
            selector_id="a",
            observer_revision="r",
        )
    ]
    same = [
        cycle_selection(
            prompt_id="p1",
            candidates=candidates,
            prompt=PROMPT,
            scorer=identical,
            selector_id="b",
            observer_revision="r",
            arm="rfo_self",
        )
    ]
    different = [
        cycle_selection(
            prompt_id="p1",
            candidates=candidates,
            prompt=PROMPT,
            scorer=other,
            selector_id="b",
            observer_revision="r",
            arm="rfo_self",
        )
    ]

    assert selection_agreement(left, same) == 1.0
    assert selection_agreement(left, different) == 0.0


def test_ladder_is_ordered_by_reachable_intent():
    assert [condition.condition_id for condition in descending_rungs()] == [
        SAME_SESSION.condition_id,
        FRESH_PROMPTED.condition_id,
        FRESH_IMAGE_ONLY.condition_id,
    ]


def test_adjacent_rungs_differ_in_exactly_one_factor():
    top, middle, bottom = descending_rungs()

    assert (top.prompt_access, top.observer) == (middle.prompt_access, middle.observer)
    assert (top.same_session, top.sees_generation_state) != (
        middle.same_session,
        middle.sees_generation_state,
    )
    assert (middle.same_session, middle.sees_generation_state) == (
        bottom.same_session,
        bottom.sees_generation_state,
    )
    assert middle.prompt_access != bottom.prompt_access


def test_external_control_matches_blind_information_but_not_the_observer():
    assert EXTERNAL.prompt_access == FRESH_IMAGE_ONLY.prompt_access
    assert EXTERNAL.same_session == FRESH_IMAGE_ONLY.same_session
    assert EXTERNAL.sees_generation_state == FRESH_IMAGE_ONLY.sees_generation_state
    assert EXTERNAL.observer != FRESH_IMAGE_ONLY.observer
    assert EXTERNAL.on_ladder is False


def test_prompted_condition_only_changes_the_question_text():
    question = _question()

    asked = contextualized_question(question, FRESH_PROMPTED, prompt=PROMPT)

    assert asked.question_id == question.question_id
    assert asked.atom_id == question.atom_id
    assert asked.expected_answer == question.expected_answer
    assert asked.question_format == question.question_format
    assert question.text in asked.text
    assert PROMPT in asked.text


def test_blind_conditions_receive_the_bare_question():
    question = _question()

    for condition in (FRESH_IMAGE_ONLY, EXTERNAL):
        assert contextualized_question(question, condition) is question


def test_blind_conditions_reject_any_prompt_text():
    with pytest.raises(ValueError, match="must not receive any prompt"):
        contextualized_question(_question(), FRESH_IMAGE_ONLY, prompt=PROMPT)


def test_prompted_condition_requires_a_prompt():
    with pytest.raises(ValueError, match="requires the original prompt"):
        contextualized_question(_question(), FRESH_PROMPTED)


def test_counterfactual_requires_a_conflicting_prompt():
    with pytest.raises(ValueError, match="requires a conflicting prompt"):
        contextualized_question(_question(), COUNTERFACTUAL)

    with pytest.raises(ValueError, match="must differ"):
        contextualized_question(
            _question(), COUNTERFACTUAL, prompt=PROMPT, conflicting_prompt=PROMPT
        )


def test_counterfactual_supplies_the_false_provenance():
    conflicting = "a red box to the left of a blue box"

    asked = contextualized_question(
        _question(), COUNTERFACTUAL, conflicting_prompt=conflicting
    )

    assert conflicting in asked.text
    assert PROMPT not in asked.text


def test_provenance_wording_is_not_evaluative():
    """The prefix states provenance only.

    It necessarily repeats the target attributes, because the prompt describes
    the scene -- that is the manipulation. What it must not do is ask the model
    to judge the image, which would turn this rung into a same-context judge.
    """

    template = PROVENANCE_TEMPLATE.format(prompt="").lower()

    assert "match" not in template
    assert "correct" not in template
    assert "satisf" not in template
    assert "?" not in template


def test_prefix_adds_nothing_beyond_the_template_and_the_prompt():
    question = _question()

    asked = contextualized_question(question, FRESH_PROMPTED, prompt=PROMPT)

    assert asked.text == f"{PROVENANCE_TEMPLATE.format(prompt=PROMPT)}\n{question.text}"


def test_every_condition_id_is_unique():
    ids = [condition.condition_id for condition in LADDER]

    assert len(ids) == len(set(ids)) == 5
