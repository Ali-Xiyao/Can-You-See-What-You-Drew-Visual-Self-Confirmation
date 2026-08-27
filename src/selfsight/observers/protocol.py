"""JSONL wire protocol with strict context allow-listing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from selfsight.data.questions import normalize_answer
from selfsight.observers.base import BaseObserver
from selfsight.schemas import (
    AtomicObservation,
    BlindObservationRequest,
    ObservationResult,
    as_serializable,
)
from selfsight.utils.hashing import rgb_sha256

FORBIDDEN_KEYS = {
    "prompt",
    "original_prompt",
    "source_label",
    "generation_state",
    "hidden_state",
    "scene_graph",
    "expected_answer",
    "ground_truth",
    "history",
}


def assert_blind_wire_payload(payload: dict[str, Any]) -> None:
    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in FORBIDDEN_KEYS:
                    raise ValueError(f"Forbidden context key at {location}.{key}")
                walk(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]")

    walk(payload, "request")
    allowed_top = {"schema_version", "request_id", "image_path", "rgb_sha256", "questions"}
    if set(payload) != allowed_top:
        raise ValueError(f"Unexpected wire fields: {sorted(set(payload).difference(allowed_top))}")
    path = Path(str(payload["image_path"]))
    if not path.is_absolute() or path.suffix.lower() != ".png":
        raise ValueError("Observer image_path must be an absolute PNG path")


def encode_request(request: BlindObservationRequest) -> str:
    payload = request.to_wire()
    assert_blind_wire_payload(payload)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def decode_request(line: str) -> BlindObservationRequest:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise TypeError("Observer request must be a JSON object")
    assert_blind_wire_payload(payload)
    request = BlindObservationRequest.from_wire(payload)
    actual_hash = rgb_sha256(request.image_path)
    if actual_hash != request.rgb_sha256:
        raise ValueError(f"RGB hash mismatch: expected {request.rgb_sha256}, got {actual_hash}")
    return request


def execute_request(observer: BaseObserver, request: BlindObservationRequest) -> ObservationResult:
    started_at = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    raw_answers = observer.answer(request.image_path, request.questions)
    elapsed_ms = (perf_counter() - start) * 1000.0
    if len(raw_answers) != len(request.questions):
        raise ValueError("Observer returned a different number of answers than questions")
    per_answer_ms = elapsed_ms / max(len(raw_answers), 1)
    answers = tuple(
        AtomicObservation(
            question_id=question.question_id,
            raw_answer=raw,
            normalized_answer=normalize_answer(raw, question),
            abstain=normalize_answer(raw, question) is None,
            latency_ms=per_answer_ms,
        )
        for question, raw in zip(request.questions, raw_answers)
    )
    return ObservationResult(
        request_id=request.request_id,
        observer_id=observer.observer_id,
        observer_revision=observer.revision,
        rgb_sha256=request.rgb_sha256,
        answers=answers,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def encode_result(result: ObservationResult) -> str:
    return json.dumps(as_serializable(result), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def decode_result(line: str) -> ObservationResult:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise TypeError("Observer result must be a JSON object")
    return ObservationResult.from_dict(payload)
