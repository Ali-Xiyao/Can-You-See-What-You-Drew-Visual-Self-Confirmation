"""Selection criteria for v3, including the restored Naive cycle arm.

v2.3 implemented the Naive arm as `adapter.observe_atoms(image, atomic_question)`
and RFO-Self as the same atomic question sent to a frozen step-0 copy. The two
arms therefore received identical inputs and differed only by live-vs-frozen
weights, which are the same parameters at step 0. The measured consequence was a
selection agreement of exactly 1.000 and a gradient cosine of exactly 1.000
(EVIDENCE_LOG 4.2) -- an identity, not a null result. Naive vs Gold on the same
pools was 0.444, so the pools themselves were discriminative.

Proposal 7.1 registers the Naive arm as `prompt -> image -> recover prompt`.
`cycle_selection` implements that criterion; `blind_atomic_selection` keeps the
RFO question form. Both produce a `SelectionDecision` through the same
tie-breaking rule as `selfsight.rfo.selection.select_candidate`, so an observed
arm difference cannot come from the ordering.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from selfsight.schemas import CandidateRecord, SelectionDecision

V3_ARMS = ("naive_cycle", "rfo_self", "rfo_gold")


class CycleScorer(Protocol):
    """The subset of the backbone contract the cycle criterion needs."""

    def cycle_consistency_score(self, image_path: str, prompt: str) -> float: ...


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def select_by_score(
    *,
    prompt_id: str,
    arm: str,
    candidates: Sequence[CandidateRecord],
    scores: Mapping[str, float],
    selector_id: str,
    observer_revision: str,
    reason: str,
) -> SelectionDecision:
    """Build a decision from raw per-candidate scores.

    Mirrors `selfsight.rfo.selection.select_candidate` exactly: highest score
    first, then lowest sampling seed, then candidate id. Non-finite scores sort
    last rather than raising, and a pool whose scores are all non-finite
    abstains, so a scorer that fails on one candidate cannot silently reorder
    the pool for one arm only.
    """

    if not candidates:
        raise ValueError("Cannot select from an empty candidate pool")
    if any(candidate.prompt_id != prompt_id for candidate in candidates):
        raise ValueError("Candidate pool spans multiple prompt IDs")
    missing = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id not in scores
    ]
    if missing:
        raise KeyError(f"Missing cycle score for {missing}")

    values = {
        candidate.candidate_id: float(scores[candidate.candidate_id])
        for candidate in candidates
    }
    all_abstain = not any(_finite(value) for value in values.values())
    if all_abstain:
        selected = None
    else:
        selected = max(
            candidates,
            key=lambda candidate: (
                values[candidate.candidate_id]
                if _finite(values[candidate.candidate_id])
                else float("-inf"),
                -candidate.sampling_seed,
                candidate.candidate_id,
            ),
        ).candidate_id
    return SelectionDecision(
        prompt_id=prompt_id,
        arm=arm,
        candidate_pool_ids=tuple(candidate.candidate_id for candidate in candidates),
        selected_candidate_id=selected,
        scores=values,
        selector_id=selector_id,
        observer_revision=observer_revision,
        abstain=all_abstain,
        reason="all candidates scored non-finite" if all_abstain else reason,
    )


def cycle_scores(
    candidates: Sequence[CandidateRecord],
    prompt: str,
    scorer: CycleScorer,
) -> dict[str, float]:
    """Score `log p(prompt | image)` for every candidate in one pool.

    The same target text is scored against every candidate, so the model's
    language prior over that text is a constant offset within the pool and does
    not affect the ranking.
    """

    if not prompt:
        raise ValueError("Cycle scoring requires the original prompt text")
    return {
        candidate.candidate_id: float(
            scorer.cycle_consistency_score(str(candidate.image_path), prompt)
        )
        for candidate in candidates
    }


def cycle_selection(
    *,
    prompt_id: str,
    candidates: Sequence[CandidateRecord],
    prompt: str,
    scorer: CycleScorer,
    selector_id: str,
    observer_revision: str,
    arm: str = "naive_cycle",
) -> SelectionDecision:
    """The registered Naive criterion: pick the candidate that best recovers the prompt."""

    return select_by_score(
        prompt_id=prompt_id,
        arm=arm,
        candidates=candidates,
        scores=cycle_scores(candidates, prompt, scorer),
        selector_id=selector_id,
        observer_revision=observer_revision,
        reason="highest mean per-token log-likelihood of the original prompt",
    )


def selection_agreement(
    left: Sequence[SelectionDecision],
    right: Sequence[SelectionDecision],
) -> float:
    """Fraction of shared, non-abstaining prompts where both arms picked the same candidate.

    This is the quantity that read exactly 1.000 for Naive vs RFO-Self in v2.3.
    It is the cheapest check that the restored cycle criterion actually changed
    the arm: if it still reads 1.000 on the same pools, the fix did not take.
    """

    left_by_prompt = {
        decision.prompt_id: decision for decision in left if not decision.abstain
    }
    right_by_prompt = {
        decision.prompt_id: decision for decision in right if not decision.abstain
    }
    common = sorted(set(left_by_prompt).intersection(right_by_prompt))
    if not common:
        raise ValueError("No shared non-abstaining prompts to compare")
    agree = sum(
        left_by_prompt[prompt].selected_candidate_id
        == right_by_prompt[prompt].selected_candidate_id
        for prompt in common
    )
    return agree / len(common)
