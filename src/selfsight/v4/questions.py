"""Two-alternative forced-choice questions over verifiable natural scenes (v4).

Why forced choice and not yes/no. A yes/no question carries one bit, but a model
that always answers "yes" scores 0.5 on it without looking at anything, and
STATUS records exactly that failure mode being measured as absolute_yes_bias.
Two named alternatives carry the same bit with no safe default: there is no
answer that is right half the time regardless of the image. It is also the
cheapest possible output to score -- the reply is one token, compared against a
gold letter, so no second model is needed to interpret it. That matters here more
than usual: the whole point of this rewrite is to keep exactly one model-judged
step in the pipeline, and grading an open-ended answer would add a second.

Where the gold answer comes from. Always from the image, never from the spec.
The question put to the generator is "what did you draw", not "what were you
asked to draw", so when the generator drew three pears instead of the requested
two, "three" is correct. Those disagreements are the most informative trials in
the experiment: a model reciting its prompt gets them wrong, a model reading its
own picture gets them right. They exist only because a spec exists, which is why
the spec is not optional.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from selfsight.v4.spec import SceneSpec, detected_multiset


class Family(str, Enum):
    """Question families -- the capability axes, and the analysis GROUP BY key.

    These are a property of the question, not of the image, so they survive the
    move from geometric shapes to natural scenes unchanged. They are NOT the
    per-family gold that v3 also called families; that one is gone, replaced by a
    single scene-match rule (see v4.spec.image_correct).

    COUNTING and ABSENCE are new. COUNTING was retired in v3 because the
    programmatic detector's counting precision was 0.40; a VLM counts small
    numbers reliably, so the retirement was an instrument limit, not a finding.
    ABSENCE was never available: it asks whether the model claims to have drawn
    something it did not, which is self-report hallucination measured directly.
    """

    EXISTENCE = "existence"
    COUNTING = "counting"
    BINDING = "binding"
    SPATIAL = "spatial"
    ABSENCE = "absence"


@dataclass(frozen=True)
class ForcedChoice:
    """One 2AFC trial.

    `gold` is "A" or "B". `gold_source` records whether the correct answer
    happens to coincide with the spec, so the analysis can isolate the trials
    where the drawn image and the requested scene disagree.
    """

    question_id: str
    spec_id: str
    family: Family
    prompt_text: str
    option_a: str
    option_b: str
    gold: str
    gold_source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "spec_id": self.spec_id,
            "family": self.family.value,
            "prompt_text": self.prompt_text,
            "option_a": self.option_a,
            "option_b": self.option_b,
            "gold": self.gold,
            "gold_source": self.gold_source,
            "metadata": dict(self.metadata),
        }


NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

# Distractor categories for absence questions: plausible on a table, so the
# question cannot be answered by noticing that one option is absurd.
DISTRACTOR_POOL = (
    "banana", "apple", "pear", "lemon", "mug", "bowl", "book", "candle",
    "bottle", "plate", "ball", "hat", "spoon", "orange", "teapot", "notebook",
)


def _plural(noun: str, n: int) -> str:
    return noun if n == 1 else noun + "s"


def _phrase(noun: str, colour: str | None) -> str:
    return f"{colour} {noun}" if colour else noun


def _place(rng: random.Random, correct: str, wrong: str) -> tuple[str, str, str]:
    """Put the correct option in A or B with equal probability.

    Without this the gold letter would correlate with something, and a model with
    a position bias would score above chance without looking at the image.
    """
    if rng.getrandbits(1):
        return correct, wrong, "A"
    return wrong, correct, "B"


def _detected_counts(detections: list[dict[str, Any]]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for item in detections:
        counter[str(item["object"]).strip().lower()] += 1
    return counter


def _centre_x(item: dict[str, Any]) -> float | None:
    centre = item.get("center")
    if centre:
        return float(centre[0])
    box = item.get("bbox") or item.get("box")
    if box and len(box) == 4:
        return (float(box[0]) + float(box[2])) / 2.0
    return None


def build_counting(
    spec: SceneSpec,
    detections: list[dict[str, Any]],
    rng: random.Random,
    index: int,
) -> ForcedChoice | None:
    """How many of one category are on the surface.

    The distractor is the requested count when that differs from the drawn count,
    and an adjacent integer otherwise. Using the requested count is deliberate:
    it makes the trial diagnostic of prompt recitation rather than of counting.
    """
    counts = _detected_counts(detections)
    candidates = [item for item in spec.objects if counts.get(item.object)]
    if not candidates:
        return None
    target = rng.choice(candidates)
    drawn = counts[target.object]
    if drawn != target.count:
        wrong_n, source = target.count, "image_differs_from_spec"
    else:
        wrong_n, source = (drawn + rng.choice((-1, 1)) or drawn + 1), "spec_matches_image"
    if wrong_n < 0 or wrong_n == drawn:
        wrong_n = drawn + 1
    correct, wrong, gold = _place(
        rng, NUMBER_WORDS.get(drawn, str(drawn)), NUMBER_WORDS.get(wrong_n, str(wrong_n))
    )
    return ForcedChoice(
        question_id=f"{spec.spec_id}:counting:{index}",
        spec_id=spec.spec_id,
        family=Family.COUNTING,
        prompt_text=(
            f"How many {_plural(target.object, 2)} are in this picture? "
            f"Answer A or B only.\nA. {correct}\nB. {wrong}"
        ),
        option_a=correct,
        option_b=wrong,
        gold=gold,
        gold_source=source,
        metadata={"object": target.object, "drawn": drawn, "requested": target.count},
    )


def build_existence(
    spec: SceneSpec,
    detections: list[dict[str, Any]],
    rng: random.Random,
    index: int,
) -> ForcedChoice | None:
    """Which of two categories is in the picture -- one drawn, one not."""
    drawn = sorted(_detected_counts(detections))
    if not drawn:
        return None
    present = rng.choice(drawn)
    pool = [x for x in DISTRACTOR_POOL if x not in drawn]
    if not pool:
        return None
    absent = rng.choice(pool)
    correct, wrong, gold = _place(rng, present, absent)
    return ForcedChoice(
        question_id=f"{spec.spec_id}:existence:{index}",
        spec_id=spec.spec_id,
        family=Family.EXISTENCE,
        prompt_text=(
            "Which of these is in this picture? Answer A or B only.\n"
            f"A. {correct}\nB. {wrong}"
        ),
        option_a=correct,
        option_b=wrong,
        gold=gold,
        gold_source="image",
        metadata={"present": present, "absent": absent},
    )


def build_binding(
    spec: SceneSpec,
    detections: list[dict[str, Any]],
    rng: random.Random,
    index: int,
) -> ForcedChoice | None:
    """Which object carries a given colour.

    Needs at least two categories with distinct detected colours, otherwise the
    colour does not pick anything out and the question is not about binding.
    """
    by_colour: dict[str, set[str]] = collections.defaultdict(set)
    for item in detections:
        colour = item.get("color")
        if colour:
            by_colour[str(colour).lower()].add(str(item["object"]).lower())
    unique = {c: next(iter(o)) for c, o in by_colour.items() if len(o) == 1}
    if len(unique) < 2:
        return None
    colour = rng.choice(sorted(unique))
    correct_object = unique[colour]
    others = sorted({o for o in unique.values() if o != correct_object})
    if not others:
        return None
    wrong_object = rng.choice(others)
    correct, wrong, gold = _place(rng, correct_object, wrong_object)
    return ForcedChoice(
        question_id=f"{spec.spec_id}:binding:{index}",
        spec_id=spec.spec_id,
        family=Family.BINDING,
        prompt_text=(
            f"In this picture, which object is {colour}? Answer A or B only.\n"
            f"A. {correct}\nB. {wrong}"
        ),
        option_a=correct,
        option_b=wrong,
        gold=gold,
        gold_source="image",
        metadata={"color": colour, "bound_to": correct_object},
    )


def build_spatial(
    spec: SceneSpec,
    detections: list[dict[str, Any]],
    rng: random.Random,
    index: int,
) -> ForcedChoice | None:
    """Image-plane left/right between two categories.

    Anchored to the picture, not to a viewer-relative frame, because "to its
    left" has no single referent once the scene has perspective. The gold comes
    from the detected box centres, so this question is available whether or not
    the spec carried a relation -- which is what makes spatial trials free of any
    cost in p when the relation is not requested.
    """
    positions: dict[str, list[float]] = collections.defaultdict(list)
    for item in detections:
        x = _centre_x(item)
        if x is not None:
            positions[str(item["object"]).lower()].append(x)
    singles = {k: v[0] for k, v in positions.items() if len(v) == 1}
    if len(singles) < 2:
        return None
    first, second = rng.sample(sorted(singles), 2)
    # A near tie is not a fact about the image; skip rather than coin-flip it.
    if abs(singles[first] - singles[second]) < 24.0:
        return None
    truth = "left" if singles[first] < singles[second] else "right"
    correct, wrong, gold = _place(rng, truth, "right" if truth == "left" else "left")
    return ForcedChoice(
        question_id=f"{spec.spec_id}:spatial:{index}",
        spec_id=spec.spec_id,
        family=Family.SPATIAL,
        prompt_text=(
            f"In this picture, is the {first} further to the left or further to "
            f"the right than the {second}? Answer A or B only.\n"
            f"A. {correct}\nB. {wrong}"
        ),
        option_a=correct,
        option_b=wrong,
        gold=gold,
        gold_source="image",
        metadata={"subject": first, "object": second,
                  "requested": any(r.subject == first and r.object == second
                                   for r in spec.relations)},
    )


def build_absence(
    spec: SceneSpec,
    detections: list[dict[str, Any]],
    rng: random.Random,
    index: int,
) -> ForcedChoice | None:
    """Does the picture contain a category that was never drawn.

    The direct measure of self-report hallucination: a model that claims to have
    drawn something it did not is not reading its own image. Both options name a
    concrete category, so "no" is not a safe default the way it is in a yes/no
    phrasing.
    """
    drawn = set(_detected_counts(detections))
    pool = [x for x in DISTRACTOR_POOL if x not in drawn]
    if not pool or not drawn:
        return None
    absent = rng.choice(pool)
    correct, wrong, gold = _place(
        rng, f"no {_plural(absent, 2)}", f"at least one {absent}"
    )
    return ForcedChoice(
        question_id=f"{spec.spec_id}:absence:{index}",
        spec_id=spec.spec_id,
        family=Family.ABSENCE,
        prompt_text=(
            f"Does this picture contain any {_plural(absent, 2)}? "
            f"Answer A or B only.\nA. {correct}\nB. {wrong}"
        ),
        option_a=correct,
        option_b=wrong,
        gold=gold,
        gold_source="image",
        metadata={"absent": absent},
    )


BUILDERS = {
    Family.EXISTENCE: build_existence,
    Family.COUNTING: build_counting,
    Family.BINDING: build_binding,
    Family.SPATIAL: build_spatial,
    Family.ABSENCE: build_absence,
}


def build_questions(
    spec: SceneSpec,
    detections: list[dict[str, Any]],
    *,
    seed: int,
    families: tuple[Family, ...] = tuple(Family),
) -> list[ForcedChoice]:
    """Every question this image can support, one per requested family.

    A builder returns None when the image cannot support that family (no two
    distinct colours for binding, a spatial tie, nothing left to name as absent).
    Returning None rather than forcing a question keeps unanswerable trials out
    of the corpus instead of scoring them as failures, which is the same
    principle as the explicit abstention v3 used.
    """
    rng = random.Random(seed)
    out = []
    for index, family in enumerate(families):
        question = BUILDERS[family](spec, detections, rng, index)
        if question is not None:
            out.append(question)
    return out


def grade(reply: str, question: ForcedChoice) -> bool | None:
    """Did the model pick the gold option?

    Returns None when the reply names neither option, so a refusal or a malformed
    answer stays visible instead of counting as a miss.
    """
    text = reply.strip().upper()
    picked = None
    for letter in ("A", "B"):
        if text.startswith(letter) or f" {letter}." in f" {text}":
            picked = letter
            break
    if picked is None:
        lowered = reply.strip().lower()
        matches = [
            letter
            for letter, option in (("A", question.option_a), ("B", question.option_b))
            if option.lower() in lowered
        ]
        if len(matches) == 1:
            picked = matches[0]
    if picked is None:
        return None
    return picked == question.gold


def to_atomic(question: ForcedChoice) -> "AtomicQuestion":
    """Present a v4 trial through the frozen observer protocol.

    The generation backbone answers questions via `observe_atoms`, which enforces
    the RFO isolation contract: the image is re-read from disk as RGB, hashed,
    embedded fresh, and the model put in eval mode, with no state carried from
    generation. Reaching past that to a raw text call would re-implement the one
    part of the pipeline that must not be re-implemented, so the v4 question is
    converted rather than the protocol bypassed.

    `choices` carries the two option strings, so `normalize_answer` maps a reply
    of "B" back to the option text and `expected_answer` compares directly. The
    legacy `family` field is set to EXISTENCE for every trial: it selects the
    fallback vocabulary used only when no choice letter is found, and in that
    case the answer is unparseable and should abstain, which is what the fallback
    then does. The v4 family lives on the `ForcedChoice` and in the output rows;
    it is not lost.
    """
    from selfsight.schemas import AtomicQuestion, QuestionFamily, QuestionFormat

    return AtomicQuestion(
        question_id=question.question_id,
        atom_id=f"{question.spec_id}:{question.family.value}",
        family=QuestionFamily.EXISTENCE,
        text=question.prompt_text,
        expected_answer=question.option_a if question.gold == "A" else question.option_b,
        question_format=QuestionFormat.FORCED_CHOICE,
        choices=(question.option_a, question.option_b),
    )
