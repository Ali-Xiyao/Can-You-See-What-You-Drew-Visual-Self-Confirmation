"""Appearance-tolerant display vocabulary for v3 scene text.

The programmatic verifier classifies a four-vertex contour as `Shape.SQUARE`
whenever its axis-aligned bounding box has aspect ratio 0.5-2.0
(`data/generated_verifier.py`). Calling that category "square" in text shown to a
model or a human reviewer asks them to judge a construct the verifier does not
measure: a 2:1 rectangle is verifier-correct and square-incorrect. v2.3 removed
the ambiguity by renaming the visible category to `box` and auditing that no
visible string still said "square" (`v23/data.py`, `v23/audit.py`); the blind
human precision of 1.00 on 28 labels was obtained under that reading and does
not transfer to a strict one (EVIDENCE_LOG 1.3).

v3 kept the metadata claim `{"square_display_name": "box"}` but dropped the
rewrite, so 183 of 256 rows of `tier_a_probe` still say "square" in the prompt or
the question. That mismatch is not neutral: the gold verifier reads the tolerant
category while every model-based arm is asked the strict word, which inflates
`oracle - naive` -- the direction that would turn a red pre-check green.

This module owns the rule and the audit so the claim and the content cannot
drift apart again. It differs from `v23.data.display_text` only by preserving
capitalisation; the tolerant category and its aspect range are identical.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from selfsight.data.generated_verifier import QUADRILATERAL_ASPECT_RANGE

INTERNAL_NAME = "square"
DISPLAY_NAME = "box"

ASPECT_RATIO_RANGE = QUADRILATERAL_ASPECT_RANGE
"""Re-exported, never restated.

The verifier owns this range. A second literal here is how the claim and the
rule drift apart, which is the defect in EVIDENCE_LOG section 11.
"""

_PATTERN = re.compile(r"\b(squares|square)\b", re.IGNORECASE)


def _replace(match: re.Match[str]) -> str:
    word = match.group(0)
    target = "boxes" if word.lower() == "squares" else DISPLAY_NAME
    if word.isupper():
        return target.upper()
    if word[0].isupper():
        return target.capitalize()
    return target


def display_text(text: str) -> str:
    """Rewrite the internal shape name to the tolerant display name."""

    return _PATTERN.sub(_replace, text)


def has_internal_wording(text: str) -> bool:
    return _PATTERN.search(text) is not None


def scene_vocabulary_metadata() -> dict[str, Any]:
    """The alias and tolerance a reader needs to interpret a scene's shape words."""

    return {
        "shape_display_alias": {INTERNAL_NAME: DISPLAY_NAME},
        "quadrilateral_aspect_ratio_range": list(ASPECT_RATIO_RANGE),
    }


def visible_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Every string in a manifest row that a model or a human reviewer can read.

    Internal geometry codes (`atom.subject`, `scene.metadata.*_shape`) are
    deliberately excluded: they are never rendered into a prompt or a question,
    and rewriting them would break the verifier's shape lookup.
    """

    texts: list[str] = []
    scene = row.get("scene")
    if isinstance(scene, Mapping) and scene.get("prompt"):
        texts.append(str(scene["prompt"]))
    questions = row.get("questions") or []
    if isinstance(questions, Mapping):
        questions = [questions]
    for question in questions:
        if isinstance(question, Mapping):
            if question.get("text"):
                texts.append(str(question["text"]))
            for choice in question.get("choices") or ():
                texts.append(str(choice))
    return tuple(texts)


def claims_display_vocabulary(row: Mapping[str, Any]) -> bool:
    vocabulary = row.get("vocabulary")
    if not isinstance(vocabulary, Mapping):
        return False
    return str(vocabulary.get(f"{INTERNAL_NAME}_display_name", "")) == DISPLAY_NAME


def audit_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Report whether a manifest's visible text matches the vocabulary it claims.

    The failure this catches is a row that *claims* the display vocabulary while
    its prompt or question still uses the internal name -- silent, because every
    downstream consumer trusts the claim.
    """

    total = 0
    claiming = 0
    offending: list[dict[str, Any]] = []
    for row in rows:
        total += 1
        claims = claims_display_vocabulary(row)
        claiming += int(claims)
        bad = [text for text in visible_texts(row) if has_internal_wording(text)]
        if bad:
            scene = row.get("scene") or {}
            offending.append(
                {
                    "scene_id": str(scene.get("scene_id", "?")),
                    "claims_display_vocabulary": claims,
                    "texts": bad,
                }
            )
    return {
        "rows": total,
        "rows_claiming_display_vocabulary": claiming,
        "rows_with_internal_wording": len(offending),
        "rows_claiming_but_violating": sum(
            int(item["claims_display_vocabulary"]) for item in offending
        ),
        "clean": not offending,
        "internal_name": INTERNAL_NAME,
        "display_name": DISPLAY_NAME,
        "aspect_ratio_range": list(ASPECT_RATIO_RANGE),
        "examples": offending[:5],
    }


def assert_display_vocabulary(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed before a manifest is written or consumed."""

    report = audit_rows(rows)
    if not report["clean"]:
        first = report["examples"][0]
        raise ValueError(
            f"{report['rows_with_internal_wording']}/{report['rows']} rows still use "
            f"'{INTERNAL_NAME}' in visible text, e.g. {first['scene_id']}: "
            f"{first['texts'][0]!r}"
        )
