"""The five observation conditions of proposal section 5, as a leakage ladder.

Every condition asks the *same* atomic question about the *same* rendered RGB.
Only the amount of the model's own intent that is reachable changes. That makes
the differences between adjacent conditions additive, and each difference
isolates one named mechanism:

    same_session   - fresh_prompted   session history and generation state
    fresh_prompted - fresh_image_only the original prompt alone
    fresh_image_only - external       the observer identity, i.e. the shared
                                      perceptual blind spot of section 4

`counterfactual` is deliberately off the ladder. It holds the pixels fixed and
supplies a *false* provenance, so a flipped answer is direct evidence that the
supplied text -- not the image -- drove the response. It is the mirror image of
the Tier B pixel counterfactual, which holds the text fixed and edits pixels.

Nothing here commits to *why* a condition leaks. `fresh_image_only` still shares
an encoder with the generator, so a gap that survives every context ablation is
evidence for a representational or steganographic shortcut rather than for
intent recall. The ladder measures the decomposition; it does not presuppose it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from selfsight.schemas import AtomicQuestion

PromptAccess = Literal["none", "true", "conflicting"]
ObserverRuntime = Literal["live_self", "frozen_self", "external"]

PROVENANCE_TEMPLATE = 'The image was generated from this description: "{prompt}"'
"""Frozen before any ladder measurement.

It states provenance and nothing else. An evaluative wording ("does the image
match ...") would ask a different question, and naming the target attributes
would leak the answer rather than the intent.
"""


@dataclass(frozen=True)
class ObservationCondition:
    """One rung of the ladder, or one off-ladder control."""

    condition_id: str
    prompt_access: PromptAccess
    same_session: bool
    sees_generation_state: bool
    observer: ObserverRuntime
    rung: int | None
    role: str

    @property
    def on_ladder(self) -> bool:
        return self.rung is not None


SAME_SESSION = ObservationCondition(
    condition_id="same_session_self_check",
    prompt_access="true",
    same_session=True,
    sees_generation_state=True,
    observer="live_self",
    rung=2,
    role="maximum reachable intent; closest to deployed self-consistency rewards",
)
FRESH_PROMPTED = ObservationCondition(
    condition_id="fresh_context_prompted",
    prompt_access="true",
    same_session=False,
    sees_generation_state=False,
    observer="live_self",
    rung=1,
    role="prompt only; the cleanest single-variable leakage manipulation",
)
FRESH_IMAGE_ONLY = ObservationCondition(
    condition_id="fresh_context_image_only",
    prompt_access="none",
    same_session=False,
    sees_generation_state=False,
    observer="frozen_self",
    rung=0,
    role="isolated blind observation; the RFO-Self training criterion",
)
COUNTERFACTUAL = ObservationCondition(
    condition_id="counterfactual_prompt",
    prompt_access="conflicting",
    same_session=False,
    sees_generation_state=False,
    observer="frozen_self",
    rung=None,
    role="flip test; a prompt-following answer is direct causal evidence",
)
EXTERNAL = ObservationCondition(
    condition_id="external_frozen_observer",
    prompt_access="none",
    same_session=False,
    sees_generation_state=False,
    observer="external",
    rung=None,
    role="same information as fresh_image_only; isolates the observer itself",
)

LADDER: tuple[ObservationCondition, ...] = (
    SAME_SESSION,
    FRESH_PROMPTED,
    FRESH_IMAGE_ONLY,
    COUNTERFACTUAL,
    EXTERNAL,
)

BY_ID: dict[str, ObservationCondition] = {
    condition.condition_id: condition for condition in LADDER
}


def descending_rungs() -> tuple[ObservationCondition, ...]:
    """Ladder conditions from most to least reachable intent.

    SCFR is expected to decrease along this order. Reporting the order
    explicitly keeps the monotonicity claim falsifiable rather than assumed.
    """

    rungs = [condition for condition in LADDER if condition.on_ladder]
    return tuple(sorted(rungs, key=lambda condition: -int(condition.rung or 0)))


def contextualized_question(
    question: AtomicQuestion,
    condition: ObservationCondition,
    *,
    prompt: str | None = None,
    conflicting_prompt: str | None = None,
) -> AtomicQuestion:
    """Return the question text this condition presents to its observer.

    Only `text` changes. `question_id`, `atom_id` and `expected_answer` are
    preserved so that answer normalization and scoring are identical across
    conditions -- otherwise a measured difference could come from the scoring
    path rather than from the manipulation.
    """

    if condition.prompt_access == "none":
        if prompt is not None or conflicting_prompt is not None:
            raise ValueError(
                f"{condition.condition_id} must not receive any prompt text"
            )
        return question
    if condition.prompt_access == "true":
        if not prompt:
            raise ValueError(f"{condition.condition_id} requires the original prompt")
        if conflicting_prompt is not None:
            raise ValueError(
                f"{condition.condition_id} must not receive a conflicting prompt"
            )
        supplied = prompt
    else:
        if not conflicting_prompt:
            raise ValueError(
                f"{condition.condition_id} requires a conflicting prompt"
            )
        if prompt is not None and prompt == conflicting_prompt:
            raise ValueError("Counterfactual prompt must differ from the original")
        supplied = conflicting_prompt
    prefix = PROVENANCE_TEMPLATE.format(prompt=supplied)
    return replace(question, text=f"{prefix}\n{question.text}")
