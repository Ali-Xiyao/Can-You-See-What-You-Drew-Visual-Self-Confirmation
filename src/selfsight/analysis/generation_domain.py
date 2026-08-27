"""Diagnostics for whether generated RGBs remain inside the deterministic verifier domain."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from selfsight.data.verifier import DetectedObject
from selfsight.schemas import SceneObject

Descriptor = tuple[str, str, str]


def scene_descriptors(objects: Iterable[SceneObject]) -> Counter[Descriptor]:
    """Return the intended (shape, color, size) object multiset."""

    return Counter((item.shape.value, item.color.value, item.size.value) for item in objects)


def detection_descriptors(detections: Iterable[DetectedObject]) -> Counter[Descriptor]:
    """Return the verifier-parsed (shape, color, size) object multiset."""

    return Counter((item.shape.value, item.color.value, item.size.value) for item in detections)


def descriptor_quality(
    expected: Counter[Descriptor], detected: Counter[Descriptor]
) -> dict[str, float | int | bool]:
    """Compare multisets without allowing duplicate detections to inflate recall."""

    overlap = sum((expected & detected).values())
    expected_count = sum(expected.values())
    detected_count = sum(detected.values())
    return {
        "expected_objects": expected_count,
        "detected_objects": detected_count,
        "matched_objects": overlap,
        "intended_object_recall": overlap / expected_count if expected_count else 1.0,
        "detected_object_precision": overlap / detected_count if detected_count else 0.0,
        "all_intended_objects_detected": overlap == expected_count,
        "intended_scene_exact": expected == detected,
        "spurious_objects": max(detected_count - overlap, 0),
    }


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize_generation_rows(
    rows: Iterable[Mapping[str, Any]], *, parseability_min: float
) -> dict[str, Any]:
    """Aggregate strict scene parsing and primary-atom diagnostics by family."""

    materialized = list(rows)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_family[str(row["family"])].append(row)

    def metrics(items: list[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "samples": len(items),
            "primary_answer_coverage": _mean(items, "primary_answer_covered"),
            "primary_verifier_accuracy": _mean(items, "primary_correct"),
            "mean_intended_object_recall": _mean(items, "intended_object_recall"),
            "mean_detected_object_precision": _mean(items, "detected_object_precision"),
            "intended_scene_recovery_rate": _mean(items, "all_intended_objects_detected"),
            "intended_scene_exact_rate": _mean(items, "intended_scene_exact"),
            "mean_detected_objects": _mean(items, "detected_objects"),
            "mean_spurious_objects": _mean(items, "spurious_objects"),
        }

    overall = metrics(materialized)
    parseability = float(overall["primary_answer_coverage"])
    return {
        "overall": overall,
        "by_family": {family: metrics(items) for family, items in sorted(by_family.items())},
        "parseability_min": parseability_min,
        "parseability_gate_basis": "primary_answer_coverage",
        "gate_generated_domain_pass": bool(materialized) and parseability >= parseability_min,
    }
