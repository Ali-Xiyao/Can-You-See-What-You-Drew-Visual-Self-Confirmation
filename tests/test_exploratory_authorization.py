from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from selfsight.analysis import exploratory
from selfsight.utils.hashing import sha256_file

FAMILIES = ("existence", "count", "color", "size", "spatial", "binding")
SELECTED = ("existence", "color", "spatial")


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    backbone_path = tmp_path / "configs" / "backbones" / "showo2_hq.yaml"
    backbone_path.parent.mkdir(parents=True)
    backbone = {
        "backbone_id": "showlab/show-o2-1.5B-HQ",
        "revision": "model-revision",
        "source": {"revision": "source-revision"},
        "dependencies": {"transformers": "4.47.0"},
    }
    backbone_path.write_text(yaml.safe_dump(backbone), encoding="utf-8")
    readiness_path = tmp_path / "configs" / "readiness.yaml"
    readiness = {
        "main_families": list(FAMILIES),
        "thresholds": {
            "reference": {"family_open_accuracy_min": 0.8},
            "generated": {
                "family_coverage_min": 0.7,
                "oracle_at_4_min": 0.7,
                "verifier_precision_min": 0.95,
            },
        },
    }
    readiness_path.write_text(yaml.safe_dump(readiness), encoding="utf-8")
    common = {
        "model_id": backbone["backbone_id"],
        "revision": backbone["revision"],
        "source_revision": backbone["source"]["revision"],
        "dependency_revisions": backbone["dependencies"],
    }
    metrics = {family: 1.0 for family in FAMILIES}
    paths = {
        "backbone_config": backbone_path,
        "readiness_config": readiness_path,
        "canary": _write_json(tmp_path / "evidence" / "canary.json", {**common, "passed": True}),
        "reference": _write_json(
            tmp_path / "evidence" / "reference.json",
            {**common, "family_open_accuracy": metrics},
        ),
        "generated": _write_json(
            tmp_path / "evidence" / "generated.json",
            {**common, "family_coverage": metrics, "family_oracle_at_4": metrics},
        ),
        "human": _write_json(
            tmp_path / "evidence" / "human.json", {**common, "family_precision": metrics}
        ),
    }
    evidence = {
        key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for key, path in paths.items()
    }
    decision = {
        "gate": "minus_2_joint_readiness",
        "passed": False,
        "decision_mode": "stop_after_human_before_a4",
        "skipped_by_stop_rule": ["a4_lora_backward_resume"],
        "evidence": evidence,
    }
    paths["decision"] = _write_json(tmp_path / "evidence" / "decision.json", decision)
    return paths


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict[str, Path], Path]:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(exploratory, "validate_generated_artifacts", lambda report: {})
    monkeypatch.setattr(exploratory, "validate_human_precision_report", lambda *a, **k: None)
    output_root = tmp_path / "runs" / "exploratory-post-gate" / "showo2-hq"
    authorization_path = output_root / "authorization.json"
    report = exploratory.build_exploratory_authorization(
        backbone_config_path=paths["backbone_config"],
        readiness_config_path=paths["readiness_config"],
        decision_path=paths["decision"],
        canary_report_path=paths["canary"],
        reference_report_path=paths["reference"],
        generated_report_path=paths["generated"],
        human_report_path=paths["human"],
        families=SELECTED,
        output_root=output_root,
        output_path=authorization_path,
    )
    return report, paths, authorization_path


def test_exploratory_authorization_is_non_formal_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, paths, authorization_path = _build(tmp_path, monkeypatch)
    assert report["non_formal"] is True
    assert report["model_downloads_allowed"] is False
    assert report["families"] == list(SELECTED)
    validated = exploratory.validate_exploratory_authorization(
        authorization_path,
        stage="a4_lora_backward_resume",
        backbone_config_path=paths["backbone_config"],
        decision_path=paths["decision"],
        canary_report_path=paths["canary"],
        reference_report_path=paths["reference"],
        generated_report_path=paths["generated"],
        human_report_path=paths["human"],
        output_path=authorization_path.parent / "a4.json",
    )
    assert validated == report


def test_exploratory_authorization_rejects_readiness_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _report, paths, authorization_path = _build(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="escapes authorized root"):
        exploratory.validate_exploratory_authorization(
            authorization_path,
            stage="a4_lora_backward_resume",
            backbone_config_path=paths["backbone_config"],
            decision_path=paths["decision"],
            canary_report_path=paths["canary"],
            reference_report_path=paths["reference"],
            generated_report_path=paths["generated"],
            human_report_path=paths["human"],
            output_path=tmp_path / "runs" / "readiness" / "a4.json",
        )


def test_exploratory_authorization_detects_frozen_evidence_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _report, paths, authorization_path = _build(tmp_path, monkeypatch)
    paths["human"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        exploratory.validate_exploratory_authorization(
            authorization_path,
            stage="a4_lora_backward_resume",
            backbone_config_path=paths["backbone_config"],
            decision_path=paths["decision"],
            canary_report_path=paths["canary"],
            reference_report_path=paths["reference"],
            generated_report_path=paths["generated"],
            human_report_path=paths["human"],
            output_path=authorization_path.parent / "a4.json",
        )


def test_exploratory_observer_audit_allows_exact_three_family_subset(tmp_path: Path) -> None:
    audit = _write_json(
        tmp_path / "observer.json",
        {
            "observer_id": "observer/model",
            "observer_revision": "revision",
            "family_open_accuracy": {name: 0.8 for name in SELECTED},
            "absolute_yes_bias": 0.05,
            "abstain_rate": 0.0,
        },
    )
    report = exploratory.validate_exploratory_observer_audit(
        audit,
        model_id="observer/model",
        revision="revision",
        eligible_families=SELECTED,
        family_accuracy_min=0.8,
        yes_bias_max=0.1,
        abstain_rate_max=0.2,
    )
    assert report["family_open_accuracy"]["spatial"] == 0.8
