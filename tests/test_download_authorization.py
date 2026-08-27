import json
from pathlib import Path

import pytest

from scripts.download_models import _validate_fallback_download_authorization
from selfsight.utils.hashing import sha256_file


def _decision(tmp_path: Path) -> Path:
    evidence = {}
    for label in ("backbone_config", "readiness_config", "canary", "reference", "generated"):
        path = tmp_path / f"{label}.json"
        path.write_text(label, encoding="utf-8")
        evidence[label] = {"path": str(path), "sha256": sha256_file(path)}
    evidence.update({"human": None, "lora": None, "predecessor": None})
    report = {
        "gate": "minus_2_joint_readiness",
        "decision_mode": "upstream_stop_before_human_and_a4",
        "model_id": "showlab/show-o2-1.5B",
        "candidate_rank": 1,
        "passed": False,
        "checks": {
            "minus_2a_unified_functionality": False,
            "minus_2b_reference_observation": True,
            "minus_2c_generated_measurability": False,
            "minus_2d_joint_families": False,
        },
        "skipped_by_stop_rule": ["blind_human_precision", "a4_lora_backward_resume"],
        "evidence": evidence,
        "fallback": {"next_model_id": "showlab/show-o2-1.5B-HQ"},
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_hq_download_requires_exact_hashed_red_predecessor(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    authorization = _validate_fallback_download_authorization(
        "readiness_fallback_hq", decision
    )
    assert authorization["authorized_model_id"] == "showlab/show-o2-1.5B-HQ"
    assert authorization["sha256"] == sha256_file(decision)


def test_fallback_download_rejects_missing_or_tampered_evidence(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="predecessor-decision"):
        _validate_fallback_download_authorization("readiness_fallback_hq", None)
    decision = _decision(tmp_path)
    (tmp_path / "generated.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _validate_fallback_download_authorization("readiness_fallback_hq", decision)


def test_fallback_download_rejects_ladder_skip(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    with pytest.raises(RuntimeError, match="candidate rank"):
        _validate_fallback_download_authorization("readiness_fallback_7b", decision)


def test_fallback_download_rejects_incomplete_evidence_set(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    report = json.loads(decision.read_text(encoding="utf-8"))
    del report["evidence"]["reference"]
    decision.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence set"):
        _validate_fallback_download_authorization("readiness_fallback_hq", decision)
