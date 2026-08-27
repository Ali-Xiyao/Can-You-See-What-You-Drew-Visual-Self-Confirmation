"""Typed records shared by data, observation, selection, and training code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class QuestionFamily(str, Enum):
    EXISTENCE = "existence"
    COUNT = "count"
    COLOR = "color"
    SIZE = "size"
    SPATIAL = "spatial"
    BINDING = "binding"


class Shape(str, Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"


class Size(str, Enum):
    SMALL = "small"
    LARGE = "large"


class QuestionFormat(str, Enum):
    OPEN = "open"
    FORCED_CHOICE = "forced_choice"
    BINARY = "binary"


class ObservationContext(str, Enum):
    RGB_ONLY = "rgb_only"
    PROMPT_ONLY = "prompt_only"
    RGB_AND_PROMPT = "rgb_and_prompt"
    HARD_RENDER_RGB = "hard_render_rgb"
    PIXEL_COUNTERFACTUAL = "pixel_counterfactual"


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    shape: Shape
    color: Color
    size: Size
    center: tuple[int, int]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneObject:
        return cls(
            object_id=str(value["object_id"]),
            shape=Shape(value["shape"]),
            color=Color(value["color"]),
            size=Size(value["size"]),
            center=(int(value["center"][0]), int(value["center"][1])),
        )


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    split: str
    family: QuestionFamily
    prompt: str
    objects: tuple[SceneObject, ...]
    canvas_size: int = 512
    generator_seed: int = 0
    template_id: str = ""
    signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneSpec:
        return cls(
            scene_id=str(value["scene_id"]),
            split=str(value["split"]),
            family=QuestionFamily(value["family"]),
            prompt=str(value["prompt"]),
            objects=tuple(SceneObject.from_dict(item) for item in value["objects"]),
            canvas_size=int(value.get("canvas_size", 512)),
            generator_seed=int(value.get("generator_seed", 0)),
            template_id=str(value.get("template_id", "")),
            signature=str(value.get("signature", "")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class Atom:
    atom_id: str
    family: QuestionFamily
    subject: str
    predicate: str
    answer: str
    reference_object_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Atom:
        return cls(
            atom_id=str(value["atom_id"]),
            family=QuestionFamily(value["family"]),
            subject=str(value["subject"]),
            predicate=str(value["predicate"]),
            answer=str(value["answer"]),
            reference_object_ids=tuple(value.get("reference_object_ids", ())),
        )


@dataclass(frozen=True)
class AtomicQuestion:
    question_id: str
    atom_id: str
    family: QuestionFamily
    text: str
    expected_answer: str
    question_format: QuestionFormat = QuestionFormat.OPEN
    choices: tuple[str, ...] = ()
    choice_order_seed: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AtomicQuestion:
        return cls(
            question_id=str(value["question_id"]),
            atom_id=str(value["atom_id"]),
            family=QuestionFamily(value["family"]),
            text=str(value["text"]),
            expected_answer=str(value["expected_answer"]),
            question_format=QuestionFormat(value.get("question_format", "open")),
            choices=tuple(value.get("choices", ())),
            choice_order_seed=int(value.get("choice_order_seed", 0)),
        )


@dataclass(frozen=True)
class ObservationRequest:
    """General observation request; not safe for blind RFO transport."""

    request_id: str
    image_path: str
    questions: tuple[AtomicQuestion, ...]
    context: ObservationContext
    prompt: str | None = None
    source_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlindObservationRequest:
    """Allow-listed payload for an isolated observer subprocess."""

    request_id: str
    image_path: str
    rgb_sha256: str
    questions: tuple[AtomicQuestion, ...]

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "image_path": self.image_path,
            "rgb_sha256": self.rgb_sha256,
            "questions": [
                {
                    "question_id": question.question_id,
                    "family": question.family.value,
                    "text": question.text,
                    "question_format": question.question_format.value,
                    "choices": list(question.choices),
                    "choice_order_seed": question.choice_order_seed,
                }
                for question in self.questions
            ],
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> BlindObservationRequest:
        allowed = {"schema_version", "request_id", "image_path", "rgb_sha256", "questions"}
        extras = sorted(set(value).difference(allowed))
        if extras:
            raise ValueError(f"Blind request includes forbidden fields: {extras}")
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported blind request schema")
        return cls(
            request_id=str(value["request_id"]),
            image_path=str(value["image_path"]),
            rgb_sha256=str(value["rgb_sha256"]),
            questions=tuple(
                AtomicQuestion(
                    question_id=str(item["question_id"]),
                    atom_id="",
                    family=QuestionFamily(item["family"]),
                    text=str(item["text"]),
                    expected_answer="",
                    question_format=QuestionFormat(item.get("question_format", "open")),
                    choices=tuple(item.get("choices", ())),
                    choice_order_seed=int(item.get("choice_order_seed", 0)),
                )
                for item in value["questions"]
            ),
        )


@dataclass(frozen=True)
class AtomicObservation:
    question_id: str
    raw_answer: str
    normalized_answer: str | None
    abstain: bool
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class ObservationResult:
    request_id: str
    observer_id: str
    observer_revision: str
    rgb_sha256: str
    answers: tuple[AtomicObservation, ...]
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ObservationResult:
        return cls(
            request_id=str(value["request_id"]),
            observer_id=str(value["observer_id"]),
            observer_revision=str(value["observer_revision"]),
            rgb_sha256=str(value["rgb_sha256"]),
            answers=tuple(AtomicObservation(**item) for item in value["answers"]),
            started_at=str(value.get("started_at", "")),
            finished_at=str(value.get("finished_at", "")),
        )


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    prompt_id: str
    scene_id: str
    sampling_seed: int
    image_path: str
    rgb_sha256: str
    generator_id: str
    generator_revision: str
    checkpoint_id: str
    atom_answers: dict[str, str] = field(default_factory=dict)
    verifier_answers: dict[str, str] = field(default_factory=dict)
    abstain_question_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CandidateRecord:
        return cls(
            candidate_id=str(value["candidate_id"]),
            prompt_id=str(value["prompt_id"]),
            scene_id=str(value["scene_id"]),
            sampling_seed=int(value["sampling_seed"]),
            image_path=str(value["image_path"]),
            rgb_sha256=str(value["rgb_sha256"]),
            generator_id=str(value["generator_id"]),
            generator_revision=str(value["generator_revision"]),
            checkpoint_id=str(value["checkpoint_id"]),
            atom_answers=dict(value.get("atom_answers", {})),
            verifier_answers=dict(value.get("verifier_answers", {})),
            abstain_question_ids=tuple(value.get("abstain_question_ids", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class SelectionDecision:
    prompt_id: str
    arm: str
    candidate_pool_ids: tuple[str, ...]
    selected_candidate_id: str | None
    scores: dict[str, float]
    selector_id: str
    observer_revision: str
    abstain: bool = False
    reason: str = ""


def as_serializable(value: Any) -> Any:
    """Convert nested dataclasses/enums/paths into JSON-compatible values."""

    if hasattr(value, "__dataclass_fields__"):
        return {key: as_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [as_serializable(item) for item in value]
    if isinstance(value, list):
        return [as_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_serializable(item) for key, item in value.items()}
    return value
