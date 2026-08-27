"""Deterministic pixel verifier for the programmatic geometric domain."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from selfsight.data.questions import parse_subject
from selfsight.data.renderer import RGB_PALETTE
from selfsight.schemas import Atom, Color, Shape, Size


@dataclass(frozen=True)
class DetectedObject:
    color: Color
    shape: Shape
    size: Size
    center: tuple[float, float]
    area: int
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class VerificationResult:
    answers: dict[str, str | None]
    detections: tuple[DetectedObject, ...]
    coverage: float
    parse_errors: tuple[str, ...]


def _shape_from_component(area: int, width: int, height: int) -> tuple[Shape, float]:
    fill_ratio = area / max(width * height, 1)
    if fill_ratio >= 0.90:
        return Shape.SQUARE, min(1.0, fill_ratio)
    if fill_ratio <= 0.64:
        return Shape.TRIANGLE, max(0.0, 1.0 - abs(fill_ratio - 0.50))
    return Shape.CIRCLE, max(0.0, 1.0 - abs(fill_ratio - 0.785))


def detect_objects(image_or_path: Image.Image | str | Path) -> tuple[DetectedObject, ...]:
    if isinstance(image_or_path, Image.Image):
        image = image_or_path.convert("RGB")
    else:
        with Image.open(image_or_path) as opened:
            image = opened.convert("RGB")
    rgb = np.asarray(image, dtype=np.int16)
    detections: list[DetectedObject] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for color, palette_rgb in RGB_PALETTE.items():
        difference = rgb - np.asarray(palette_rgb, dtype=np.int16)
        distance = np.sqrt(np.sum(difference.astype(np.float32) ** 2, axis=-1))
        mask = distance <= 65.0
        labels, count = ndimage.label(mask, structure=structure)
        for component_index in range(1, count + 1):
            ys, xs = np.nonzero(labels == component_index)
            area = len(xs)
            if area < 250:
                continue
            min_x, max_x = int(xs.min()), int(xs.max())
            min_y, max_y = int(ys.min()), int(ys.max())
            width, height = max_x - min_x + 1, max_y - min_y + 1
            shape, shape_confidence = _shape_from_component(area, width, height)
            size = Size.SMALL if max(width, height) < 95 else Size.LARGE
            color_confidence = float(np.clip(1.0 - distance[ys, xs].mean() / 65.0, 0.0, 1.0))
            detections.append(
                DetectedObject(
                    color=color,
                    shape=shape,
                    size=size,
                    # Use the visual bounding-box center for layout relations. A triangle's
                    # mass centroid is intentionally below its declared layout center.
                    center=((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
                    area=area,
                    bbox=(min_x, min_y, max_x, max_y),
                    confidence=(shape_confidence + color_confidence) / 2.0,
                )
            )
    return tuple(sorted(detections, key=lambda item: (item.center[1], item.center[0])))


def _matches(detections: Iterable[DetectedObject], fields: dict[str, str]) -> list[DetectedObject]:
    output = []
    for detection in detections:
        if "shape" in fields and detection.shape.value != fields["shape"]:
            continue
        if "color" in fields and detection.color.value != fields["color"]:
            continue
        if "size" in fields and detection.size.value != fields["size"]:
            continue
        output.append(detection)
    return output


def evaluate_atom(detections: tuple[DetectedObject, ...], atom: Atom) -> str | None:
    subjects = parse_subject(atom.subject)
    first = _matches(detections, subjects[0])
    if atom.predicate == "exists":
        return "yes" if first else "no"
    if atom.predicate == "count":
        return str(len(first))
    if atom.predicate == "color":
        return first[0].color.value if len(first) == 1 else None
    if atom.predicate == "size":
        return first[0].size.value if len(first) == 1 else None
    if len(subjects) != 2:
        return None
    second = _matches(detections, subjects[1])
    if len(first) != 1 or len(second) != 1:
        return None
    if atom.predicate == "left_of":
        truth = first[0].center[0] < second[0].center[0]
    elif atom.predicate == "above":
        truth = first[0].center[1] < second[0].center[1]
    elif atom.predicate == "larger_than":
        truth = first[0].size == Size.LARGE and second[0].size == Size.SMALL
    else:
        return None
    return "yes" if truth else "no"


def verify_image(image_or_path: Image.Image | str | Path, atoms: Iterable[Atom]) -> VerificationResult:
    detections = detect_objects(image_or_path)
    answers: dict[str, str | None] = {}
    errors: list[str] = []
    atoms = tuple(atoms)
    for atom in atoms:
        try:
            answer = evaluate_atom(detections, atom)
        except (KeyError, ValueError, IndexError) as exc:
            answer = None
            errors.append(f"{atom.atom_id}:{type(exc).__name__}:{exc}")
        answers[atom.atom_id] = answer
    parsed = sum(answer is not None for answer in answers.values())
    return VerificationResult(
        answers=answers,
        detections=detections,
        coverage=parsed / len(atoms) if atoms else 1.0,
        parse_errors=tuple(errors),
    )
