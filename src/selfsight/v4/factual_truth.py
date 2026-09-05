"""Question-level facts from verifier records, retaining unresolved uncertainty.

An agreed image verdict is not necessarily a complete object list. In particular,
``agreed_verdict`` deletes disputed keys from its core; those keys are unknown,
not absent. All diagnostics use the frozen question text to determine its scope.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from selfsight.data.questions import NUMBER_WORDS
from selfsight.v4.spec import canonical_noun, detected_multiset, has_unnameable


@dataclass(frozen=True)
class QuestionScope:
    kind: str
    noun: str
    colour: str | None

    def matches(self, noun: str, colour: str | None) -> bool:
        return self.noun == canonical_noun(noun) and (
            self.colour is None or self.colour == colour)


@dataclass(frozen=True)
class FactualAnswer:
    answer: str | None
    known: bool
    reason: str


def question_scope(question: Mapping[str, Any]) -> QuestionScope | None:
    """Parse the generated templates; never infer colour from legacy atom IDs."""
    text = str(question.get("text", ""))
    for kind, pattern in (
        ("count", r"How many (.+?) are in this picture\?"),
        ("existence", r"Is there an? (.+?) in this picture\?"),
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # These templates use a canonical single-word noun preceded by the
            # optional colour. Legacy atom IDs omit that colour on existence.
            words = match.group(1).strip().lower().split()
            noun = canonical_noun(words[-1])
            colour = " ".join(words[:-1]) or None
            return QuestionScope(kind, noun, colour)
    return None


def canonical_answer(question: Mapping[str, Any], answer: Any) -> str | None:
    """Compare number words from legacy forced choices with numeric facts."""
    if answer is None:
        return None
    text = str(answer).strip().lower()
    scope = question_scope(question)
    if scope is not None and scope.kind == "count":
        text = NUMBER_WORDS.get(text, text)
        if text.isdigit():
            return str(int(text))
    return text


def factual_answer(question: Mapping[str, Any],
                   verification: Mapping[str, Any] | None) -> FactualAnswer:
    if verification is None or verification.get("detections") is None:
        return FactualAnswer(None, False, "missing_verification")
    items = verification["detections"]
    if has_unnameable(items):
        return FactualAnswer(None, False, "unnameable")
    if verification.get("resolution") in {"pending_human", "unnameable"}:
        return FactualAnswer(None, False, "unresolved_verification")
    scope = question_scope(question)
    if scope is None:
        return FactualAnswer(None, False, "unsupported_question")
    if verification.get("resolution") == "agreed_verdict":
        for disputed in verification.get("disputed", []):
            colour = disputed.get("color")
            colour = str(colour).strip().lower() if colour else None
            if scope.matches(disputed["object"], colour):
                return FactualAnswer(None, False, "disputed_scope")
    # resolved_by_crop may retain its historical disputed keys. Its settled
    # detections are usable; only agreed_verdict above retains unknown keys.
    seen = sum(n for (noun, colour), n in detected_multiset(items).items()
               if scope.matches(noun, colour))
    answer = str(seen) if scope.kind == "count" else ("yes" if seen else "no")
    return FactualAnswer(answer, True, "verified")


def load_verifications(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Load complete records, including their resolution and disputed keys."""
    runs = json.loads((out_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    found: dict[str, dict[str, Any]] = {}
    for name in runs:
        for line in (Path(name) / "verified.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found[row["image_path"]] = row
    return found
