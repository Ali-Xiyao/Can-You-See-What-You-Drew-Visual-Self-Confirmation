"""Deterministic RGB-only observer for integration tests; never scientific evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from selfsight.data.verifier import DetectedObject, detect_objects
from selfsight.observers.base import BaseObserver
from selfsight.schemas import AtomicQuestion, Shape, Size


class MockPixelObserver(BaseObserver):
    observer_id = "mock/pixel-observer"
    revision = "deterministic-v1-non-scientific"

    @staticmethod
    def _by_shape(detections: tuple[DetectedObject, ...], shape: str) -> list[DetectedObject]:
        return [item for item in detections if item.shape.value == shape]

    def _answer_one(self, detections: tuple[DetectedObject, ...], text: str) -> str:
        lower = text.lower()
        shapes = "|".join(shape.value for shape in Shape)
        match = re.search(rf"is there a (red|blue|green|yellow) ({shapes})", lower)
        if match:
            color, shape = match.groups()
            return "yes" if any(item.shape.value == shape and item.color.value == color for item in detections) else "no"
        match = re.search(rf"how many ({shapes})s?\b", lower)
        if match:
            return str(len(self._by_shape(detections, match.group(1))))
        match = re.search(rf"what color is the ({shapes})", lower)
        if match:
            items = self._by_shape(detections, match.group(1))
            return items[0].color.value if len(items) == 1 else "unknown"
        match = re.search(rf"is the ({shapes}) small or large", lower)
        if match:
            items = self._by_shape(detections, match.group(1))
            return items[0].size.value if len(items) == 1 else "unknown"
        match = re.search(
            rf"is the ({shapes}) (to the left of|above|larger than) the ({shapes})", lower
        )
        if match:
            first = self._by_shape(detections, match.group(1))
            second = self._by_shape(detections, match.group(3))
            if len(first) != 1 or len(second) != 1:
                return "unknown"
            relation = match.group(2)
            if relation == "to the left of":
                truth = first[0].center[0] < second[0].center[0]
            elif relation == "above":
                truth = first[0].center[1] < second[0].center[1]
            else:
                truth = first[0].size == Size.LARGE and second[0].size == Size.SMALL
            return "yes" if truth else "no"
        return "unknown"

    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        detections = detect_objects(image_path)
        return [self._answer_one(detections, question.text) for question in questions]
