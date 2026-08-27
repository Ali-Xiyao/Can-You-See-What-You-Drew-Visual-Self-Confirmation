from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from selfsight.analysis.readiness import finalize_joint_readiness, require_joint_readiness

FAMILIES = ("existence", "count", "color", "size", "spatial", "binding")


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _evidence(tmp_path: Path, *, weak_families: int = 0) -> dict[str, Path]:
    model_id = "showlab/show-o2-1.5B"
    revision = "07ec16589d4fc5422a74dddbbc4b2cd11e551039"
    weak = set(FAMILIES[-weak_families:]) if weak_families else set()
    reference_accuracy = {family: 0.75 if family in weak else 0.90 for family in FAMILIES}
    family_coverage = {family: 0.65 if family in weak else 0.85 for family in FAMILIES}
    family_oracle = {family: 0.60 if family in weak else 0.80 for family in FAMILIES}
    family_precision = {family: 0.94 if family in weak else 0.98 for family in FAMILIES}
    backbone = yaml.safe_load(
        Path("configs/backbones/showo2_1p5b.yaml").read_text(encoding="utf-8")
    )
    common = {
        "model_id": model_id,
        "revision": revision,
        "source_revision": backbone["source"]["revision"],
        "dependency_revisions": backbone["dependencies"],
    }
    return {
        "canary": _write_json(tmp_path / "canary.json", {**common, "passed": True}),
        "reference": _write_json(
            tmp_path / "reference.json",
            {
                **common,
                "family_open_accuracy": reference_accuracy,
                "absolute_yes_bias_points": 4.0,
                "repeat_agreement": 0.95,
                "abstain_rate": 0.10,
            },
        ),
        "generated": _write_json(
            tmp_path / "generated.json",
            {
                **common,
                "overall_coverage": 0.85,
                "family_coverage": family_coverage,
                "overall_oracle_at_4": 0.80,
                "family_oracle_at_4": family_oracle,
                "fixed_seed_coverage_swing_points": 5.0,
            },
        ),
        "human": _write_json(
            tmp_path / "human.json",
            {
                **common,
                "overall_precision": 0.98,
                "family_precision": family_precision,
            },
        ),
        "lora": _write_json(
            tmp_path / "lora.json",
            {**common, "passed": True, "frozen_step0_supported": True},
        ),
    }


def _finalize(tmp_path: Path, evidence: dict[str, Path], output: Path | None = None):
    return finalize_joint_readiness(
        backbone_config_path="configs/backbones/showo2_1p5b.yaml",
        readiness_config_path="configs/readiness_v2.2.yaml",
        canary_report_path=evidence["canary"],
        reference_report_path=evidence["reference"],
        generated_report_path=evidence["generated"],
        human_report_path=evidence["human"],
        lora_report_path=evidence["lora"],
        output_path=output,
    )


def test_joint_readiness_green_requires_and_selects_at_least_four_families(tmp_path: Path):
    evidence = _evidence(tmp_path, weak_families=2)
    decision_path = tmp_path / "decision.json"
    decision = _finalize(tmp_path, evidence, decision_path)
    assert decision["passed"]
    assert decision["selected_eligible_families"] == list(FAMILIES[:4])
    validated = require_joint_readiness(decision_path)
    assert validated["model_id"] == "showlab/show-o2-1.5B"


def test_joint_readiness_red_with_only_three_joint_families(tmp_path: Path):
    decision = _finalize(tmp_path, _evidence(tmp_path, weak_families=3))
    assert not decision["passed"]
    assert decision["checks"]["minus_2d_joint_families"] is False
    assert decision["fallback"]["next_model_id"] == "showlab/show-o2-1.5B-HQ"


def test_joint_readiness_binds_identity_before_writing(tmp_path: Path):
    evidence = _evidence(tmp_path)
    human = json.loads(evidence["human"].read_text(encoding="utf-8"))
    human["revision"] = "0" * 40
    _write_json(evidence["human"], human)
    output = tmp_path / "decision.json"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _finalize(tmp_path, evidence, output)
    assert not output.exists()


def test_joint_readiness_detects_evidence_tamper(tmp_path: Path):
    evidence = _evidence(tmp_path)
    decision_path = tmp_path / "decision.json"
    _finalize(tmp_path, evidence, decision_path)
    with evidence["human"].open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        require_joint_readiness(decision_path)


def test_fallback_requires_failed_predecessor_authorization(tmp_path: Path):
    first_evidence = _evidence(tmp_path / "first", weak_families=3)
    predecessor_path = tmp_path / "first" / "decision.json"
    predecessor_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_joint_readiness(
        backbone_config_path="configs/backbones/showo2_1p5b.yaml",
        readiness_config_path="configs/readiness_v2.2.yaml",
        canary_report_path=first_evidence["canary"],
        reference_report_path=first_evidence["reference"],
        generated_report_path=first_evidence["generated"],
        human_report_path=first_evidence["human"],
        lora_report_path=first_evidence["lora"],
        output_path=predecessor_path,
    )

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_evidence = _evidence(second_root)
    for path in second_evidence.values():
        value = json.loads(path.read_text(encoding="utf-8"))
        value["model_id"] = "showlab/show-o2-1.5B-HQ"
        value["revision"] = "d3a220ec55feaacbdfcb053847edee14edd4e69a"
        _write_json(path, value)
    backbone = yaml.safe_load(
        Path("configs/backbones/showo2_1p5b.yaml").read_text(encoding="utf-8")
    )
    backbone["candidate_rank"] = 2
    backbone["backbone_id"] = "showlab/show-o2-1.5B-HQ"
    backbone["revision"] = "d3a220ec55feaacbdfcb053847edee14edd4e69a"
    del backbone["dependencies"]["showlab/show-o2-1.5B"]
    backbone["dependencies"]["showlab/show-o2-1.5B-HQ"] = (
        "d3a220ec55feaacbdfcb053847edee14edd4e69a"
    )
    backbone["fallback"]["on_failure"] = "showlab/show-o2-7B"
    hq_config = second_root / "hq.yaml"
    hq_config.write_text(yaml.safe_dump(backbone, sort_keys=False), encoding="utf-8")
    for path in second_evidence.values():
        value = json.loads(path.read_text(encoding="utf-8"))
        value["dependency_revisions"] = backbone["dependencies"]
        _write_json(path, value)
    decision = finalize_joint_readiness(
        backbone_config_path=hq_config,
        readiness_config_path="configs/readiness_v2.2.yaml",
        canary_report_path=second_evidence["canary"],
        reference_report_path=second_evidence["reference"],
        generated_report_path=second_evidence["generated"],
        human_report_path=second_evidence["human"],
        lora_report_path=second_evidence["lora"],
        predecessor_path=predecessor_path,
    )
    assert decision["passed"]
    assert decision["evidence"]["predecessor"]["sha256"]
