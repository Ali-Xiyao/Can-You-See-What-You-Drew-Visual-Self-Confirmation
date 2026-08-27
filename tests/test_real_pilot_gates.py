from __future__ import annotations

import json

import pytest

from selfsight.pilot.real_loop import _assert_joint_prerequisites, _assert_prerequisites
from selfsight.utils.hashing import sha256_file


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _green_gate_minus_one():
    return {
        "gate": "minus_1_pre_e1",
        "passed": True,
        "conditions": {"reference": True, "showo": True, "matched_observer": True},
    }


def test_real_pilot_prerequisites_allow_failed_gradient_gate_fallback(tmp_path) -> None:
    gate = _write_json(tmp_path / "gate.json", _green_gate_minus_one())
    gradient = _write_json(tmp_path / "gradient.json", {"gate": "minus_1b", "passed": False})
    generated = _write_json(
        tmp_path / "generated.json",
        {
            "parseability_gate_basis": "primary_answer_coverage",
            "parseability_min": 0.95,
            "gate_generated_domain_pass": True,
            "overall": {"primary_answer_coverage": 0.97, "samples": 60},
        },
    )
    assert (
        _assert_prerequisites(
            gate,
            gradient,
            generated,
            generated_coverage_min=0.95,
        )
        is False
    )


def test_real_pilot_prerequisites_block_red_generated_domain_gate(tmp_path) -> None:
    gate = _write_json(tmp_path / "gate.json", _green_gate_minus_one())
    gradient = _write_json(tmp_path / "gradient.json", {"gate": "minus_1b", "passed": True})
    generated = _write_json(
        tmp_path / "generated.json",
        {
            "parseability_gate_basis": "primary_answer_coverage",
            "parseability_min": 0.95,
            "gate_generated_domain_pass": False,
            "overall": {"primary_answer_coverage": 0.55, "samples": 60},
        },
    )
    with pytest.raises(RuntimeError, match="E1, Gate -1b, and E2 are forbidden"):
        _assert_prerequisites(
            gate,
            gradient,
            generated,
            generated_coverage_min=0.95,
        )


def test_real_pilot_joint_prerequisites_bind_decision_and_families(tmp_path) -> None:
    families = ["existence", "count", "color", "spatial"]
    decision = _write_json(
        tmp_path / "decision.json",
        {
            "gate": "minus_2_joint_readiness",
            "model_id": "showlab/show-o2-1.5B",
            "revision": "locked",
            "checks": {"minus_2a": True, "minus_2d": True},
            "passed": True,
            "selected_eligible_families": families,
            "evidence": {},
        },
    )
    gradient = _write_json(
        tmp_path / "gradient.json",
        {
            "gate": "minus_1b",
            "model_id": "showlab/show-o2-1.5B",
            "revision": "locked",
            "passed": False,
            "eligible_families": families,
            "evidence_bindings": {
                "joint_readiness_decision": {"sha256": sha256_file(decision)}
            },
        },
    )
    assert _assert_joint_prerequisites(decision, gradient) == (False, tuple(families))
    value = json.loads(gradient.read_text(encoding="utf-8"))
    value["eligible_families"] = families[:-1] + ["binding"]
    _write_json(gradient, value)
    with pytest.raises(RuntimeError, match="eligible families"):
        _assert_joint_prerequisites(decision, gradient)
