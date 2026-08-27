"""Deterministic contour verifier for approximate geometric model generations.

The exact-palette verifier remains authoritative for program-rendered references. This
variant tolerates antialiasing, gradients, rotation, and hollow/nested shapes, but it does
not use the prompt, intended answer, scene graph, or a learned model.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from selfsight.data.verifier import (
    DetectedObject,
    VerificationResult,
    evaluate_atom,
)
from selfsight.schemas import Atom, Color, Shape, Size

MIN_SATURATION = 55
MIN_VALUE = 45
MAX_HUE_DISTANCE = 32.0
MIN_COMPONENT_AREA = 220.0
SIZE_BOUNDARY_PX = 95


def _circular_hue_distance(values: np.ndarray, target: float) -> np.ndarray:
    direct = np.abs(values - target)
    return np.minimum(direct, 180.0 - direct)


def _shape_from_contour(contour: np.ndarray) -> tuple[Shape, float]:
    import cv2

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    del x, y
    if perimeter <= 0 or width <= 0 or height <= 0:
        return Shape.CIRCLE, 0.0
    approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    vertices = len(approximation)
    fill_ratio = area / (width * height)
    circularity = float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))
    aspect = width / height

    if vertices == 3:
        return Shape.TRIANGLE, 0.95
    if vertices == 4 and 0.50 <= aspect <= 2.0:
        # Rotated squares have a fill ratio near 0.5, so vertices carry more weight here.
        return Shape.SQUARE, 0.90
    if circularity >= 0.72 and 0.65 <= aspect <= 1.55:
        return Shape.CIRCLE, min(1.0, circularity + 0.15)
    if fill_ratio <= 0.62:
        return Shape.TRIANGLE, max(0.45, 1.0 - abs(fill_ratio - 0.50))
    if fill_ratio >= 0.86 and 0.60 <= aspect <= 1.70:
        return Shape.SQUARE, min(1.0, fill_ratio)
    return Shape.CIRCLE, max(0.35, 1.0 - abs(fill_ratio - np.pi / 4.0))


def detect_generated_objects(
    image_or_path: Image.Image | str | Path,
) -> tuple[DetectedObject, ...]:
    """Detect saturated primary-color contours without consulting intended content."""

    import cv2

    if isinstance(image_or_path, Image.Image):
        image = image_or_path.convert("RGB")
    else:
        with Image.open(image_or_path) as opened:
            image = opened.convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    palette_hues = {
        Color.RED: 0.0,
        Color.YELLOW: 24.0,
        Color.GREEN: 67.0,
        Color.BLUE: 113.0,
    }
    distances = np.stack(
        [_circular_hue_distance(hue, target) for target in palette_hues.values()], axis=-1
    )
    assignments = np.argmin(distances, axis=-1)
    nearest_distance = np.min(distances, axis=-1)
    eligible = (
        (saturation >= MIN_SATURATION)
        & (value >= MIN_VALUE)
        & (nearest_distance <= MAX_HUE_DISTANCE)
    )
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    kernel_close = np.ones((5, 5), dtype=np.uint8)
    detections: list[DetectedObject] = []
    colors = tuple(palette_hues)
    height_total, width_total = rgb.shape[:2]
    for color_index, color in enumerate(colors):
        mask = ((assignments == color_index) & eligible).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        # RETR_TREE is required for a colored object nested inside a differently colored
        # object/background. Negative oriented area identifies foreground boundaries;
        # positive boundaries are holes and would otherwise become duplicate objects.
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour, oriented=True) >= 0:
                continue
            area = float(cv2.contourArea(contour))
            if area < MIN_COMPONENT_AREA:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < 12 or height < 12:
                continue
            touches = sum(
                (
                    x <= 1,
                    y <= 1,
                    x + width >= width_total - 1,
                    y + height >= height_total - 1,
                )
            )
            if touches >= 3 or area >= 0.80 * width_total * height_total:
                continue
            if max(width / max(height, 1), height / max(width, 1)) > 6.0:
                continue
            shape, shape_confidence = _shape_from_contour(contour)
            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
            component = component_mask.astype(bool)
            hue_confidence = float(
                np.clip(1.0 - nearest_distance[component].mean() / MAX_HUE_DISTANCE, 0.0, 1.0)
            )
            saturation_confidence = float(saturation[component].mean() / 255.0)
            color_confidence = (hue_confidence + saturation_confidence) / 2.0
            detections.append(
                DetectedObject(
                    color=color,
                    shape=shape,
                    size=Size.SMALL if max(width, height) < SIZE_BOUNDARY_PX else Size.LARGE,
                    center=(x + (width - 1) / 2.0, y + (height - 1) / 2.0),
                    area=round(area),
                    bbox=(x, y, x + width - 1, y + height - 1),
                    confidence=(shape_confidence + color_confidence) / 2.0,
                )
            )
    return tuple(
        sorted(detections, key=lambda item: (item.center[1], item.center[0], -item.confidence))
    )


def verify_generated_image(
    image_or_path: Image.Image | str | Path, atoms: Iterable[Atom]
) -> VerificationResult:
    detections = detect_generated_objects(image_or_path)
    atoms = tuple(atoms)
    answers: dict[str, str | None] = {}
    errors: list[str] = []
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
