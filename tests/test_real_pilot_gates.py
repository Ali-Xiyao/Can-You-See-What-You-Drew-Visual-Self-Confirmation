from __future__ import annotations

import json

import pytest

from selfsight.pilot.real_loop import _assert_prerequisites


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
