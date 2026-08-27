from __future__ import annotations

import json

import pytest

from selfsight.analysis.prerequisites import (
    require_generated_domain,
    require_selected_detector_audit,
)
from selfsight.utils.hashing import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_generated_gate_cannot_lower_registered_threshold(tmp_path) -> None:
    report = _write(
        tmp_path / "generated.json",
        {
            "parseability_gate_basis": "primary_answer_coverage",
            "parseability_min": 0.50,
            "gate_generated_domain_pass": True,
            "overall": {"primary_answer_coverage": 0.55, "samples": 60},
        },
    )
    with pytest.raises(RuntimeError, match="lowered the configured coverage threshold"):
        require_generated_domain(report, configured_coverage_min=0.95)


def test_red_generated_gate_stays_red_even_above_configured_min(tmp_path) -> None:
    report = _write(
        tmp_path / "generated.json",
        {
            "parseability_gate_basis": "primary_answer_coverage",
            "parseability_min": 0.99,
            "gate_generated_domain_pass": False,
            "overall": {"primary_answer_coverage": 0.97, "samples": 60},
        },
    )
    with pytest.raises(RuntimeError, match="are forbidden"):
        require_generated_domain(report, configured_coverage_min=0.95)


def _green_detector_gate(audit_path):
    return {
        "matched_observer": {
            "matched": True,
            "observer_id": "candidate/model",
            "observer_revision": "b" * 40,
        },
        "evidence_reports": {
            "candidates": [
                {
                    "observer_id": "candidate/model",
                    "sha256": sha256_file(audit_path),
                }
            ]
        },
    }


def test_selected_detector_is_bound_to_hashed_gate_audit(tmp_path) -> None:
    audit = _write(
        tmp_path / "candidate.json",
        {
            "observer_id": "candidate/model",
            "observer_revision": "b" * 40,
            "gate_minus_1_capability_pass": True,
            "gate_minus_1_bias_pass": True,
        },
    )
    gate = _green_detector_gate(audit)
    assert (
        require_selected_detector_audit(
            gate,
            audit,
            model_id="candidate/model",
            revision="b" * 40,
        )["observer_id"]
        == "candidate/model"
    )


def test_selected_detector_rejects_identity_or_audit_tampering(tmp_path) -> None:
    audit = _write(
        tmp_path / "candidate.json",
        {
            "observer_id": "candidate/model",
            "observer_revision": "b" * 40,
            "gate_minus_1_capability_pass": True,
            "gate_minus_1_bias_pass": True,
        },
    )
    gate = _green_detector_gate(audit)
    with pytest.raises(RuntimeError, match="does not match Gate -1 selection"):
        require_selected_detector_audit(
            gate,
            audit,
            model_id="stronger/model",
            revision="c" * 40,
        )
    audit.write_text(audit.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        require_selected_detector_audit(
            gate,
            audit,
            model_id="candidate/model",
            revision="b" * 40,
        )
