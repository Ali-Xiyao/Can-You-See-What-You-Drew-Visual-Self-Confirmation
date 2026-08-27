from __future__ import annotations

from selfsight.analysis.readiness_audit import (
    summarize_generated_rows,
    summarize_reference_rows,
)


def test_reference_summary_enforces_all_registered_controls():
    rows = []
    for family in ("existence", "count", "color", "size", "spatial", "binding"):
        for index in range(20):
            expected = "yes" if index % 2 == 0 else "no"
            rows.append(
                {
                    "family": family,
                    "expected": expected,
                    "open_prediction": expected,
                    "open_repeat_prediction": expected,
                    "forced_predictions": [expected, expected],
                    "forced_expected": [expected, expected],
                }
            )
    thresholds = {
        "family_open_accuracy_min": 0.8,
        "families_passing_min": 4,
        "yes_bias_points_max": 10,
        "repeat_agreement_min": 0.9,
        "abstain_rate_max": 0.2,
    }
    report = summarize_reference_rows(rows, thresholds)
    assert report["passed"]
    assert report["absolute_yes_bias_points"] == 0.0
    assert report["repeat_agreement"] == 1.0


def test_generated_summary_uses_k1_coverage_and_k4_oracle():
    rows = []
    families = ("existence", "count")
    for family in families:
        for prompt in range(10):
            for candidate in range(4):
                rows.append(
                    {
                        "scene_id": f"{family}-{prompt}",
                        "family": family,
                        "candidate_index": candidate,
                        "primary_answer_covered": candidate != 0 or prompt != 0,
                        "primary_correct": candidate == 3,
                    }
                )
    report = summarize_generated_rows(rows, families=families, oracle_families=("existence",))
    assert report["overall_coverage"] == 0.9
    assert report["overall_oracle_at_4"] == 1.0
    assert report["family_coverage"] == {"existence": 0.9, "count": 0.9}
    assert report["oracle_evaluation_families"] == ["existence"]
