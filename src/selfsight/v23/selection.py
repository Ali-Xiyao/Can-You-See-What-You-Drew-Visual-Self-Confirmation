"""RFO-Gold scoring and symmetric informative-pool filtering for v2.3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from selfsight.data.generated_verifier import verify_generated_image
from selfsight.schemas import (
    Atom,
    AtomicObservation,
    AtomicQuestion,
    CandidateRecord,
    ObservationResult,
    SelectionDecision,
)

V23_ARMS = ("naive", "rfo_self", "rfo_gold")


def gold_observation(
    candidate: CandidateRecord,
    question: AtomicQuestion,
    atom: Atom,
) -> ObservationResult:
    """Score only candidate pixels using the tolerant generated-image verifier."""

    answer = verify_generated_image(candidate.image_path, (atom,)).answers[atom.atom_id]
    return ObservationResult(
        request_id=f"v23-gold-{candidate.candidate_id}",
        observer_id="selfsight/generated-pixel-verifier",
        observer_revision="v2.3-box-tolerant-0.5-2.0",
        rgb_sha256=candidate.rgb_sha256,
        answers=(
            AtomicObservation(
                question_id=question.question_id,
                raw_answer=answer or "unparsed",
                normalized_answer=answer,
                abstain=answer is None,
            ),
        ),
    )


def finite_score_span(decision: SelectionDecision) -> float | None:
    values = [float(value) for value in decision.scores.values() if math.isfinite(float(value))]
    return max(values) - min(values) if len(values) >= 2 else None


def select_common_informative(
    prompt_order: Sequence[str],
    decisions_by_arm: Mapping[str, Sequence[SelectionDecision]],
    *,
    minimum_gold_gap: float,
    limit: int | None = None,
) -> tuple[dict[str, list[SelectionDecision]], dict[str, Any]]:
    """Keep the same non-abstaining, informative pools in all three arms.

    Informativeness is determined solely from RFO-Gold scores. No arm may gain a
    different prompt count, candidate pool, or prompt order through abstention.
    """

    if tuple(decisions_by_arm) != V23_ARMS:
        raise ValueError(f"Expected decisions in exact v2.3 arm order {V23_ARMS}")
    if minimum_gold_gap < 0:
        raise ValueError("minimum_gold_gap must be nonnegative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")

    maps: dict[str, dict[str, SelectionDecision]] = {}
    for arm in V23_ARMS:
        values = list(decisions_by_arm[arm])
        by_prompt = {decision.prompt_id: decision for decision in values}
        if len(by_prompt) != len(values):
            raise ValueError(f"Duplicate prompt decisions in {arm}")
        maps[arm] = by_prompt

    accepted: list[str] = []
    rows: list[dict[str, Any]] = []
    for prompt_id in prompt_order:
        missing = [arm for arm in V23_ARMS if prompt_id not in maps[arm]]
        if missing:
            rows.append({"prompt_id": prompt_id, "accepted": False, "reason": "missing_arm"})
            continue
        arm_values = {arm: maps[arm][prompt_id] for arm in V23_ARMS}
        pools = {decision.candidate_pool_ids for decision in arm_values.values()}
        if len(pools) != 1:
            raise RuntimeError(f"Candidate-pool mismatch across v2.3 arms: {prompt_id}")
        if any(
            decision.abstain or decision.selected_candidate_id is None
            for decision in arm_values.values()
        ):
            rows.append({"prompt_id": prompt_id, "accepted": False, "reason": "symmetric_abstain"})
            continue
        gap = finite_score_span(arm_values["rfo_gold"])
        informative = gap is not None and gap >= minimum_gold_gap
        rows.append(
            {
                "prompt_id": prompt_id,
                "accepted": informative,
                "reason": "gold_gap_pass" if informative else "gold_gap_below_threshold",
                "gold_score_gap": gap,
                "candidate_pool_ids": list(next(iter(pools))),
            }
        )
        if informative and (limit is None or len(accepted) < limit):
            accepted.append(prompt_id)

    paired = {arm: [maps[arm][prompt_id] for prompt_id in accepted] for arm in V23_ARMS}
    report = {
        "schema_version": 1,
        "prompt_count": len(prompt_order),
        "accepted_count": len(accepted),
        "rejected_count": len(prompt_order) - len(accepted),
        "minimum_gold_gap": minimum_gold_gap,
        "limit": limit,
        "accepted_prompt_ids": accepted,
        "rows": rows,
        "paired_pool_order_identical": all(
            [decision.candidate_pool_ids for decision in paired[arm]]
            == [decision.candidate_pool_ids for decision in paired[V23_ARMS[0]]]
            for arm in V23_ARMS[1:]
        ),
    }
    return paired, report


def gold_selection_advantage(
    naive: Sequence[SelectionDecision],
    gold: Sequence[SelectionDecision],
) -> float:
    """Mean oracle-score advantage of Gold-selected over Naive-selected candidates."""

    naive_map = {decision.prompt_id: decision for decision in naive}
    differences: list[float] = []
    for gold_decision in gold:
        naive_decision = naive_map.get(gold_decision.prompt_id)
        if naive_decision is None:
            continue
        naive_id = naive_decision.selected_candidate_id
        gold_id = gold_decision.selected_candidate_id
        if naive_id is None or gold_id is None:
            continue
        oracle_scores = gold_decision.scores
        naive_score = float(oracle_scores[naive_id])
        gold_score = float(oracle_scores[gold_id])
        if math.isfinite(naive_score) and math.isfinite(gold_score):
            differences.append(gold_score - naive_score)
    if not differences:
        return float("nan")
    return sum(differences) / len(differences)
