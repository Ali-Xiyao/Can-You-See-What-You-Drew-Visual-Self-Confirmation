from __future__ import annotations

from pathlib import Path

import pytest

from selfsight.analysis.e1 import _filter_eligible_records, _model_answer
from selfsight.schemas import AtomicObservation, AtomicQuestion, ObservationResult, QuestionFamily


class _PublicAdapter:
    def __init__(self) -> None:
        self.questions = []

    def observe_atoms(self, image_path, questions):
        self.questions.extend(questions)
        return ObservationResult(
            request_id="test",
            observer_id="fake",
            observer_revision="0",
            rgb_sha256="0" * 64,
            answers=(
                AtomicObservation(
                    question_id=questions[0].question_id,
                    raw_answer="yes",
                    normalized_answer="yes",
                    abstain=False,
                ),
            ),
        )


def test_e1_model_answer_uses_public_observe_contract() -> None:
    adapter = _PublicAdapter()
    question = AtomicQuestion("q", "a", QuestionFamily.EXISTENCE, "Is there a circle?", "yes")
    assert _model_answer(adapter, Path("image.png"), "contextual question", question) == "yes"
    assert adapter.questions[0].text == "contextual question"
    assert adapter.questions[0].question_id != question.question_id


def test_e1_family_filter_is_fail_closed() -> None:
    records = [
        {"pair": {"source": {"family": "existence"}}},
        {"pair": {"source": {"family": "size"}}},
    ]
    retained, excluded = _filter_eligible_records(records, ("existence",))
    assert retained == records[:1]
    assert excluded == ["size"]
    with pytest.raises(ValueError, match="absent"):
        _filter_eligible_records(records, ("binding",))
    retained, excluded = _filter_eligible_records(
        records, ("existence", "binding"), require_all_eligible_present=False
    )
    assert retained == records[:1]
    assert excluded == ["size"]
