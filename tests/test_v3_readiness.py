"""L0: the v3.0 three-family floor must be an open change, not a silent one."""

from __future__ import annotations

import json

import pytest

from selfsight.analysis.readiness import require_joint_readiness
from selfsight.v3.readiness import evaluate_family_eligibility, recompute_v3_decision

THRESHOLDS = {
    "family_open_accuracy_min": 0.80,
    "family_coverage_min": 0.70,
    "verifier_precision_min": 0.95,
    "oracle_at_4_min": 0.70,
}

# The real frozen Show-o2-1.5B-HQ numbers.
HQ_ACCURACY = {"existence": 1.0, "color": 1.0, "spatial": 1.0, "count": 0.85, "binding": 1.0, "size": 0.6}
HQ_COVERAGE = {"existence": 1.0, "color": 1.0, "spatial": 0.8, "count": 1.0, "binding": 0.9, "size": 0.2}
HQ_PRECISION = {"existence": 1.0, "color": 1.0, "spatial": 1.0, "count": 0.4, "binding": 0.7777777777777778, "size": 1.0}
HQ_ORACLE = {"existence": 1.0, "color": 1.0, "spatial": 1.0, "count": 0.5, "binding": 1.0, "size": 0.2}


def _eligibility(families):
    return evaluate_family_eligibility(
        families,
        reference_accuracy=HQ_ACCURACY,
        generated_coverage=HQ_COVERAGE,
        family_precision=HQ_PRECISION,
        oracle_at_4=HQ_ORACLE,
        thresholds=THRESHOLDS,
    )


def test_three_primary_families_are_eligible_on_real_hq_numbers():
    result = _eligibility(["existence", "color", "spatial"])
    assert result["eligible"] == ["existence", "color", "spatial"]


def test_no_fourth_family_qualifies_which_is_why_v22_went_red():
    result = _eligibility(["existence", "color", "spatial", "count", "binding", "size"])
    assert result["eligible"] == ["existence", "color", "spatial"]
    # count fails on verifier precision and Oracle@4; binding on precision; size on coverage.
    assert result["per_family"]["count"]["checks"]["verifier_precision"] is False
    assert result["per_family"]["count"]["checks"]["oracle_at_4"] is False
    assert result["per_family"]["binding"]["checks"]["verifier_precision"] is False
    assert result["per_family"]["size"]["checks"]["generated_coverage"] is False


def test_missing_evidence_is_reported_not_assumed():
    result = evaluate_family_eligibility(
        ["existence"],
        reference_accuracy=HQ_ACCURACY,
        generated_coverage={},
        family_precision=HQ_PRECISION,
        oracle_at_4=HQ_ORACLE,
        thresholds=THRESHOLDS,
    )
    assert result["eligible"] == []
    assert result["per_family"]["existence"]["missing_evidence"] == ["generated_coverage"]


def _fixture(tmp_path, *, a4_passed=True, frozen_passed=False):
    frozen = {
        "gate": "minus_2_joint_readiness",
        "passed": frozen_passed,
        "model_id": "showlab/show-o2-1.5B-HQ",
        "revision": "d3a220ec",
        "native_resolution": 512,
        "candidate_rank": 2,
        "source": {"repository_id": "showlab/Show-o"},
        "dependency_revisions": {},
        "subchecks": {"minus_2a": {"same_checkpoint_generate_and_observe": True}},
        "thresholds": {
            "reference": {
                "family_open_accuracy_min": 0.80,
                "yes_bias_points_max": 10.0,
                "repeat_agreement_min": 0.90,
                "abstain_rate_max": 0.20,
            },
            "generated": {
                "family_coverage_min": 0.70,
                "verifier_precision_min": 0.95,
                "oracle_at_4_min": 0.70,
            },
            "joint": {"families_passing_min": 4},
        },
    }
    files = {
        "frozen": frozen,
        "reference": {
            "family_open_accuracy": HQ_ACCURACY,
            "absolute_yes_bias_points": 2.0,
            "repeat_agreement": 0.98,
            "abstain_rate": 0.02,
        },
        "generated": {"family_coverage": HQ_COVERAGE, "family_oracle_at_4": HQ_ORACLE},
        "human": {"family_precision": HQ_PRECISION},
        "a4": {"passed": a4_passed, "non_formal": True},
    }
    paths = {}
    for name, payload in files.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    return paths


def test_recomputed_decision_is_green_and_declares_both_changes(tmp_path):
    paths = _fixture(tmp_path)
    output = tmp_path / "decision-v3.json"
    report = recompute_v3_decision(
        frozen_decision_path=paths["frozen"],
        reference_report_path=paths["reference"],
        generated_report_path=paths["generated"],
        human_report_path=paths["human"],
        a4_report_path=paths["a4"],
        output_path=output,
    )

    assert report["passed"] is True
    assert report["selected_eligible_families"] == ["existence", "color", "spatial"]
    assert report["derived_measurement"] is False

    changes = {item["change"]: item for item in report["declared_changes"]}
    assert changes["main_family_floor"]["from"] == 4
    assert changes["main_family_floor"]["to"] == 3
    assert changes["main_family_floor"]["declared_before_any_v3_result"] is True
    assert changes["minus_2a_lora_backward_resume"]["from"] == "skipped_by_stop_rule"
    assert changes["minus_2a_lora_backward_resume"]["evidence_is_non_preregistered"] is True

    # Every input is hash-bound and the frozen predecessor is untouched.
    assert set(report["evidence"]) >= {"predecessor_frozen_v22_decision", "a4_lora_canary"}
    assert report["immutability"]["frozen_v22_decision_rewritten"] is False


def test_recompute_requires_a_red_predecessor(tmp_path):
    paths = _fixture(tmp_path, frozen_passed=True)
    with pytest.raises(RuntimeError, match="frozen red v2.2"):
        recompute_v3_decision(
            frozen_decision_path=paths["frozen"],
            reference_report_path=paths["reference"],
            generated_report_path=paths["generated"],
            human_report_path=paths["human"],
            a4_report_path=paths["a4"],
            output_path=tmp_path / "out.json",
        )


def test_a4_cannot_be_filled_in_from_a_failed_canary(tmp_path):
    paths = _fixture(tmp_path, a4_passed=False)
    with pytest.raises(RuntimeError, match="failed A4 canary"):
        recompute_v3_decision(
            frozen_decision_path=paths["frozen"],
            reference_report_path=paths["reference"],
            generated_report_path=paths["generated"],
            human_report_path=paths["human"],
            a4_report_path=paths["a4"],
            output_path=tmp_path / "out.json",
        )


def test_validator_default_still_enforces_the_v22_contract(tmp_path):
    """Lowering the floor for v3 must not retroactively loosen v2.2 consumers."""

    paths = _fixture(tmp_path)
    output = tmp_path / "decision-v3.json"
    recompute_v3_decision(
        frozen_decision_path=paths["frozen"],
        reference_report_path=paths["reference"],
        generated_report_path=paths["generated"],
        human_report_path=paths["human"],
        a4_report_path=paths["a4"],
        output_path=output,
    )

    assert require_joint_readiness(output, minimum_families=3)["passed"] is True
    with pytest.raises(RuntimeError, match="fewer than 4"):
        require_joint_readiness(output)
