"""Atomic fact/question generation and conservative answer normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass

from selfsight.schemas import (
    Atom,
    AtomicQuestion,
    Color,
    QuestionFamily,
    QuestionFormat,
    SceneSpec,
    Size,
)

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4"}
ABSTAIN_MARKERS = {"unknown", "unclear", "cannot tell", "can't tell", "not sure", "ambiguous"}


def build_primary_atom(scene: SceneSpec) -> Atom:
    family = scene.family
    metadata = scene.metadata
    atom_id = f"{scene.scene_id}:primary"
    if family == QuestionFamily.EXISTENCE:
        shape = str(metadata["target_shape"])
        color = str(metadata["target_color"])
        return Atom(
            atom_id,
            family,
            f"shape={shape};color={color}",
            "exists",
            "yes" if metadata["positive"] else "no",
        )
    if family == QuestionFamily.COUNT:
        shape = str(metadata["target_shape"])
        return Atom(atom_id, family, f"shape={shape}", "count", str(metadata["count"]))
    if family == QuestionFamily.BINDING:
        # Size is part of the reference, not decoration: two objects share the
        # shape, so `shape=X` alone would resolve to two detections and the
        # verifier would abstain. Size is read off the target object rather than
        # the metadata so scenes built before binding carried `target_size`
        # still resolve to a single detection.
        shape = str(metadata["target_shape"])
        target = next(
            item for item in scene.objects if item.object_id == metadata["target_object_id"]
        )
        size = target.size.value
        return Atom(
            atom_id,
            family,
            f"shape={shape};size={size}",
            "color",
            target.color.value,
            (target.object_id,),
        )
    if family == QuestionFamily.COLOR:
        shape = str(metadata["target_shape"])
        target = next(
            item for item in scene.objects if item.object_id == metadata["target_object_id"]
        )
        return Atom(
            atom_id, family, f"shape={shape}", "color", target.color.value, (target.object_id,)
        )
    if family == QuestionFamily.SIZE:
        shape = str(metadata["target_shape"])
        target = next(
            item for item in scene.objects if item.object_id == metadata["target_object_id"]
        )
        return Atom(
            atom_id, family, f"shape={shape}", "size", target.size.value, (target.object_id,)
        )
    relation = str(metadata["relation"])
    subject = f"shape={metadata['subject_shape']}|shape={metadata['object_shape']}"
    return Atom(atom_id, family, subject, relation, "yes" if metadata["truth"] else "no")


def _question_text(atom: Atom) -> str:
    fields = parse_subject(atom.subject)
    shape = fields[0].get("shape", "object")
    if atom.predicate == "exists":
        return f"Is there a {fields[0]['color']} {shape} in the image? Answer yes or no."
    if atom.predicate == "count":
        return f"How many {shape}s are in the image? Answer with one number."
    if atom.predicate == "color":
        size = fields[0].get("size")
        qualifier = f"{size} " if size else ""
        return f"What color is the {qualifier}{shape}? Answer with one color word."
    if atom.predicate == "size":
        return f"Is the {shape} small or large? Answer with one word."
    other = fields[1].get("shape", "object")
    relation_text = {
        "left_of": "to the left of",
        "above": "above",
        "larger_than": "larger than",
    }[atom.predicate]
    return f"Is the {shape} {relation_text} the {other}? Answer yes or no."


def choices_for_atom(atom: Atom, reverse: bool = False) -> tuple[str, ...]:
    if atom.predicate in {"exists", "left_of", "above", "larger_than"}:
        values = ("yes", "no")
    elif atom.predicate == "size":
        values = ("small", "large")
    elif atom.predicate == "color":
        values = tuple(color.value for color in Color)
    else:
        values = ("0", "1", "2", "3", "4")
    return tuple(reversed(values)) if reverse else values


def build_question(
    atom: Atom,
    question_format: QuestionFormat = QuestionFormat.OPEN,
    choice_order_seed: int = 0,
) -> AtomicQuestion:
    choices = ()
    text = _question_text(atom)
    if question_format == QuestionFormat.FORCED_CHOICE:
        choices = choices_for_atom(atom, reverse=bool(choice_order_seed % 2))
        labels = ", ".join(f"{chr(65 + index)}) {choice}" for index, choice in enumerate(choices))
        text = f"{text} Choose exactly one: {labels}"
    return AtomicQuestion(
        question_id=f"{atom.atom_id}:{question_format.value}:{choice_order_seed}",
        atom_id=atom.atom_id,
        family=atom.family,
        text=text,
        expected_answer=atom.answer,
        question_format=question_format,
        choices=choices,
        choice_order_seed=choice_order_seed,
    )


def build_gold_atoms(scene: SceneSpec) -> tuple[Atom, ...]:
    """Atoms that score a *generated* image, as opposed to questioning the model.

    SUPERSEDED by `selfsight.v4.spec.image_correct` (2026-08-31). Retained only
    because the frozen v2.3 path imports this module; not for new work.

    The defect is structural, not a bug in any one branch: each family's gold is
    the *minimal* claim the old geometric detector could still resolve, and a
    minimal claim is cheap to satisfy by drawing less. Measured on the v3 natural
    pool, the share of "correct" verdicts that came from omission was 15%
    (existence: a single atom about one of three objects), 37.4% (spatial: the
    docstring below claims grounding closes the loophole, but only the subject is
    grounded, never the object) and 44.8% (binding: the competitor is a negative
    claim, satisfied by not drawing it). Worse, the same image was held to a
    different standard depending on which question happened to be asked of it.

    v4 replaces all of it with one family-independent rule -- the detected
    (object, colour) multiset must equal the spec's -- which contains no negative
    claim and therefore has no omission loophole to close.

    `build_primary_atom` produces the question put to the model and must stay
    balanced over yes/no. Gold scoring has two different requirements, both
    established by the v3 calibration sweep:

    * **Total.** `color` and the strict relations abstain unless the subject
      resolves to exactly one detection. That holds on every reference render
      and on 12-40% of generated ones, so 60-85% of binding and spatial
      candidates came back unscoreable -- not wrong, unscoreable. The sweep was
      measuring verifier resolvability rather than task difficulty.
    * **Positive.** A negative claim is close to free here, because a model that
      simply omits an object satisfies it. Existence negatives scored p=0.82-0.93
      against 0.25-0.42 for positives, pulling the family above the productive
      band and emptying its pools.

    Families not listed fall back to the primary atom, so pre-v3 manifests keep
    their existing semantics.
    """

    family = scene.family
    metadata = scene.metadata

    def gold_id(index: int) -> str:
        return f"{scene.scene_id}:gold{index}"

    if family == QuestionFamily.EXISTENCE:
        # A "no" scene has no target to assert, so the gold claim is that a real
        # object of the scene is present. The question keeps its "no" answer.
        shape = str(metadata["target_shape"])
        color = str(metadata["target_color"])
        anchor_object = next(
            (
                item
                for item in scene.objects
                if item.shape.value == shape and item.color.value == color
            ),
            scene.objects[0],
        )
        return (
            Atom(
                gold_id(0),
                family,
                f"shape={anchor_object.shape.value};color={anchor_object.color.value}",
                "exists",
                "yes",
                (anchor_object.object_id,),
            ),
        )

    if family == QuestionFamily.BINDING:
        target = next(
            item for item in scene.objects if item.object_id == metadata["target_object_id"]
        )
        competitor = next(
            item
            for item in scene.objects
            if item.shape == target.shape and item.object_id != target.object_id
        )
        reference = f"shape={target.shape.value};size={target.size.value}"
        return (
            Atom(
                gold_id(0),
                family,
                f"{reference};color={target.color.value}",
                "exists",
                "yes",
                (target.object_id,),
            ),
            # The miscombination check: the target's shape and size carrying the
            # *other* same-shape object's colour must not appear.
            Atom(
                gold_id(1),
                family,
                f"{reference};color={competitor.color.value}",
                "exists",
                "no",
                (competitor.object_id,),
            ),
        )

    if family == QuestionFamily.SPATIAL:
        relation = str(metadata["relation"])
        subject_shape = str(metadata["subject_shape"])
        object_shape = str(metadata["object_shape"])
        return (
            Atom(
                gold_id(0),
                family,
                f"shape={subject_shape}|shape={object_shape}",
                f"exists_{relation}",
                "yes" if metadata["truth"] else "no",
            ),
            # Grounding. Without it an image missing the subject satisfies every
            # relation whose truth is "no", which is the omission loophole.
            Atom(gold_id(1), family, f"shape={subject_shape}", "exists", "yes"),
        )

    return (build_primary_atom(scene),)


def parse_subject(subject: str) -> list[dict[str, str]]:
    groups = []
    for group in subject.split("|"):
        fields = {}
        for item in group.split(";"):
            key, value = item.split("=", maxsplit=1)
            fields[key] = value
        groups.append(fields)
    return groups


def normalize_answer(raw: str, question: AtomicQuestion) -> str | None:
    text = re.sub(r"\s+", " ", raw.strip().lower())
    if not text or any(marker in text for marker in ABSTAIN_MARKERS):
        return None

    if question.question_format == QuestionFormat.FORCED_CHOICE:
        label_match = re.search(r"(?:^|\b)([a-e])(?:\b|[.)])", text)
        if label_match:
            index = ord(label_match.group(1)) - ord("a")
            if index < len(question.choices):
                return question.choices[index]

    allowed: set[str]
    family = question.family
    if family in {QuestionFamily.EXISTENCE, QuestionFamily.SPATIAL}:
        allowed = {"yes", "no"}
    elif family in {QuestionFamily.COLOR, QuestionFamily.BINDING}:
        allowed = {color.value for color in Color}
    elif family == QuestionFamily.SIZE:
        allowed = {size.value for size in Size}
    else:
        for word, number in NUMBER_WORDS.items():
            text = re.sub(rf"\b{word}\b", number, text)
        # Reference scenes use the registered 0--4 ontology, but a blind audit of
        # generated pixels can legitimately contain any non-negative count.
        # Keeping the normalizer open-ended prevents a visible count such as six
        # from being converted into an abstention merely because generation left
        # the prompt ontology.
        matches = set(re.findall(r"(?<![\w-])\d+(?!\w)", text))
        if len(matches) == 1:
            return str(int(next(iter(matches))))
        return None

    matches = {token for token in allowed if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text)}
    if len(matches) == 1:
        return next(iter(matches))
    return None


@dataclass(frozen=True)
class ScoredAnswer:
    normalized: str | None
    correct: bool
    abstain: bool


def score_answer(raw: str, question: AtomicQuestion) -> ScoredAnswer:
    normalized = normalize_answer(raw, question)
    return ScoredAnswer(
        normalized=normalized,
        correct=normalized == question.expected_answer,
        abstain=normalized is None,
    )
