import json
from pathlib import Path

from selfsight.analysis.readiness_figure import render_readiness_matrix

FAMILIES = ("existence", "count", "color", "size", "spatial", "binding")


def _decision(path: Path, *, rank: int, passed: bool) -> Path:
    family = {name: (0.9 if passed or index < 4 else 0.6) for index, name in enumerate(FAMILIES)}
    precision = {
        name: (0.98 if passed or index < 4 else 0.6) for index, name in enumerate(FAMILIES)
    }
    eligible = list(FAMILIES if passed else FAMILIES[:4])
    report = {
        "gate": "minus_2_joint_readiness",
        "candidate_rank": rank,
        "model_id": "showlab/show-o2-1.5B" if rank == 1 else "showlab/show-o2-1.5B-HQ",
        "passed": passed,
        "checks": {
            "minus_2a_unified_functionality": True,
            "minus_2b_reference_observation": True,
            "minus_2c_generated_measurability": passed,
            "minus_2d_joint_families": True,
        },
        "selected_eligible_families": eligible,
        "metrics": {
            "family_open_accuracy": family,
            "family_coverage": family,
            "family_precision": precision,
            "family_oracle_at_4": family,
        },
        "thresholds": {
            "reference": {"family_open_accuracy_min": 0.8},
            "generated": {
                "family_coverage_min": 0.7,
                "verifier_precision_min": 0.95,
                "oracle_at_4_min": 0.7,
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_readiness_figure_exports_exact_matrices_and_qa(tmp_path: Path) -> None:
    first = _decision(tmp_path / "rank1.json", rank=1, passed=False)
    second = _decision(tmp_path / "rank2.json", rank=2, passed=True)
    outputs = render_readiness_matrix(
        [first, second], tmp_path / "figure", evidence_status="synthetic test"
    )
    for path in outputs.values():
        assert Path(path).is_file()
    qa = json.loads(Path(outputs["qa"]).read_text(encoding="utf-8"))
    assert qa["passed"]
    assert qa["profile"]["candidate_count"] == 2
    assert qa["profile"]["family_metric_rows"] == 48


def test_readiness_figure_distinguishes_upstream_stop_from_failure(tmp_path: Path) -> None:
    decision = _decision(tmp_path / "rank1-stop.json", rank=1, passed=False)
    report = json.loads(decision.read_text(encoding="utf-8"))
    report["decision_mode"] = "upstream_stop_before_human_and_a4"
    report["skipped_by_stop_rule"] = ["blind_human_precision", "a4_lora_backward_resume"]
    report["metrics"]["family_precision"] = None
    report["selected_eligible_families"] = []
    decision.write_text(json.dumps(report), encoding="utf-8")

    outputs = render_readiness_matrix(
        [decision], tmp_path / "stop-figure", evidence_status="synthetic upstream stop"
    )
    qa = json.loads(Path(outputs["qa"]).read_text(encoding="utf-8"))
    assert qa["passed"]
    assert qa["profile"]["not_tested_cells"] == len(FAMILIES)
    rows = Path(outputs["family_csv"]).read_text(encoding="utf-8")
    assert rows.count(",precision,,") == len(FAMILIES)
    gate_rows = Path(outputs["gate_csv"]).read_text(encoding="utf-8")
    assert "minus_2a_unified_functionality,False,False" in gate_rows
    assert "minus_2d_joint_families,False,False" in gate_rows
