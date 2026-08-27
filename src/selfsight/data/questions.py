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
        return Atom(atom_id, family, f"shape={shape};color={color}", "exists", "yes" if metadata["positive"] else "no")
    if family == QuestionFamily.COUNT:
        shape = str(metadata["target_shape"])
        return Atom(atom_id, family, f"shape={shape}", "count", str(metadata["count"]))
    if family in {QuestionFamily.COLOR, QuestionFamily.BINDING}:
        shape = str(metadata["target_shape"])
        target = next(item for item in scene.objects if item.object_id == metadata["target_object_id"])
        return Atom(atom_id, family, f"shape={shape}", "color", target.color.value, (target.object_id,))
    if family == QuestionFamily.SIZE:
        shape = str(metadata["target_shape"])
        target = next(item for item in scene.objects if item.object_id == metadata["target_object_id"])
        return Atom(atom_id, family, f"shape={shape}", "size", target.size.value, (target.object_id,))
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
        return f"What color is the {shape}? Answer with one color word."
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
        allowed = {str(index) for index in range(5)}
        for word, number in NUMBER_WORDS.items():
            text = re.sub(rf"\b{word}\b", number, text)

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
