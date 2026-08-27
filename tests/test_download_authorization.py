import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.download_models import _validate_fallback_download_authorization
from selfsight.utils.hashing import rgb_sha256, sha256_file


def _decision(tmp_path: Path) -> Path:
    evidence = {}
    for label in ("backbone_config", "readiness_config", "canary", "reference"):
        path = tmp_path / f"{label}.json"
        path.write_text(label, encoding="utf-8")
        evidence[label] = {"path": str(path), "sha256": sha256_file(path)}
    image_path = tmp_path / "candidate.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(image_path)
    rows_path = tmp_path / "generated-rows.jsonl"
    rows_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-0",
                "image_path": str(image_path),
                "rgb_sha256": rgb_sha256(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generated_path = tmp_path / "generated.json"
    generated_path.write_text(
        json.dumps(
            {
                "rows": str(rows_path),
                "rows_sha256": sha256_file(rows_path),
                "candidates": 1,
                "unique_candidate_ids": 1,
                "unique_image_paths": 1,
            }
        ),
        encoding="utf-8",
    )
    evidence["generated"] = {
        "path": str(generated_path),
        "sha256": sha256_file(generated_path),
    }
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


def test_fallback_download_rejects_nested_a3_rows_tamper(tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    with (tmp_path / "generated-rows.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="rows SHA-256 mismatch"):
        _validate_fallback_download_authorization("readiness_fallback_hq", decision)
