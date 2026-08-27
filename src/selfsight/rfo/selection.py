"""Selection functions over a fixed, paired candidate pool."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from selfsight.schemas import (
    AtomicQuestion,
    CandidateRecord,
    ObservationResult,
    SelectionDecision,
)


def observation_score(
    observation: ObservationResult,
    questions: Mapping[str, AtomicQuestion],
) -> tuple[float, bool]:
    correct = 0
    available = 0
    for answer in observation.answers:
        question = questions.get(answer.question_id)
        if question is None:
            raise KeyError(f"Unknown question in observation: {answer.question_id}")
        if answer.abstain or answer.normalized_answer is None:
            continue
        available += 1
        correct += answer.normalized_answer == question.expected_answer
    return (correct / available if available else float("nan"), available == 0)


def select_candidate(
    *,
    prompt_id: str,
    arm: str,
    candidates: Sequence[CandidateRecord],
    observations: Mapping[str, ObservationResult],
    questions: Sequence[AtomicQuestion],
    selector_id: str,
    observer_revision: str,
) -> SelectionDecision:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate pool")
    if any(candidate.prompt_id != prompt_id for candidate in candidates):
        raise ValueError("Candidate pool spans multiple prompt IDs")
    question_map = {question.question_id: question for question in questions}
    scores: dict[str, float] = {}
    all_abstain = True
    for candidate in candidates:
        if candidate.candidate_id not in observations:
            raise KeyError(f"Missing observation for {candidate.candidate_id}")
        score, abstain = observation_score(observations[candidate.candidate_id], question_map)
        scores[candidate.candidate_id] = score
        all_abstain = all_abstain and abstain
    if all_abstain:
        selected = None
    else:
        selected = max(
            candidates,
            key=lambda candidate: (
                float("-inf") if scores[candidate.candidate_id] != scores[candidate.candidate_id]
                else scores[candidate.candidate_id],
                -candidate.sampling_seed,
                candidate.candidate_id,
            ),
        ).candidate_id
    return SelectionDecision(
        prompt_id=prompt_id,
        arm=arm,
        candidate_pool_ids=tuple(candidate.candidate_id for candidate in candidates),
        selected_candidate_id=selected,
        scores=scores,
        selector_id=selector_id,
        observer_revision=observer_revision,
        abstain=all_abstain,
        reason="all questions abstained" if all_abstain else "highest mean atomic accuracy",
    )


def balance_paired_decisions(
    left: Iterable[SelectionDecision], right: Iterable[SelectionDecision]
) -> tuple[list[SelectionDecision], list[SelectionDecision]]:
    left_by_prompt = {decision.prompt_id: decision for decision in left if not decision.abstain}
    right_by_prompt = {decision.prompt_id: decision for decision in right if not decision.abstain}
    common = sorted(set(left_by_prompt).intersection(right_by_prompt))
    return ([left_by_prompt[prompt] for prompt in common], [right_by_prompt[prompt] for prompt in common])
