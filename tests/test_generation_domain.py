from collections import Counter

from selfsight.analysis.generation_domain import descriptor_quality, summarize_generation_rows


def test_descriptor_quality_uses_multiset_overlap() -> None:
    expected = Counter({("square", "blue", "large"): 1, ("circle", "red", "small"): 1})
    detected = Counter({("square", "blue", "large"): 2, ("triangle", "red", "small"): 1})
    quality = descriptor_quality(expected, detected)
    assert quality["matched_objects"] == 1
    assert quality["intended_object_recall"] == 0.5
    assert quality["detected_object_precision"] == 1 / 3
    assert quality["spurious_objects"] == 2
    assert quality["all_intended_objects_detected"] is False


def test_generation_summary_gates_on_primary_answer_coverage() -> None:
    base = {
        "primary_answer_covered": True,
        "primary_correct": True,
        "intended_object_recall": 1.0,
        "detected_object_precision": 1.0,
        "intended_scene_exact": True,
        "detected_objects": 2,
        "spurious_objects": 0,
    }
    rows = [
        {**base, "family": "color", "all_intended_objects_detected": True},
        {
            **base,
            "family": "size",
            "all_intended_objects_detected": False,
            "intended_object_recall": 0.5,
            "primary_answer_covered": False,
            "primary_correct": False,
        },
    ]
    report = summarize_generation_rows(rows, parseability_min=0.95)
    assert report["overall"]["intended_scene_recovery_rate"] == 0.5
    assert report["overall"]["primary_answer_coverage"] == 0.5
    assert report["gate_generated_domain_pass"] is False
