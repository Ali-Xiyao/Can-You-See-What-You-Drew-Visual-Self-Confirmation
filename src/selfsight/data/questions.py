"""Conservative normalization of a free-text observer reply.

The atom and question builders that used to live here were geometric: they
addressed objects by shape, colour and size over a fixed four-colour palette.
v4 authors its scenes with an LLM and asks about them in 2AFC form, so the
builders are gone and `selfsight.v4.questions` replaces them. What survives is
the reply normalizer, because it is about parsing text, not about shapes: it
reads the letter of a forced choice, treats hedging as abstention rather than
as a wrong answer, and refuses to guess when a reply names two candidates.

Deleted with the rest of the v3 instrument, recoverable from git history:
`build_primary_atom`, `build_question`, `choices_for_atom`, `build_gold_atoms`.
"""

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
