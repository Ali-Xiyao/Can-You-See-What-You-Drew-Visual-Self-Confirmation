from __future__ import annotations

import json

from selfsight.analysis.observer_audit import finalize_gate_minus_1


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_gate_minus_one_decision_records_hashed_evidence(tmp_path) -> None:
    reference = _write(tmp_path / "reference.json", {"gate_reference_pass": True})
    showo = _write(
        tmp_path / "showo.json",
        {
            "observer_id": "showo",
            "observer_revision": "a" * 40,
            "macro_open_accuracy": 0.65,
            "gate_minus_1_capability_pass": False,
            "gate_minus_1_bias_pass": True,
        },
    )
    candidate = _write(
        tmp_path / "candidate.json",
        {
            "observer_id": "candidate",
            "observer_revision": "b" * 40,
            "macro_open_accuracy": 0.66,
            "gate_minus_1_capability_pass": True,
            "gate_minus_1_bias_pass": True,
        },
    )
    report = finalize_gate_minus_1(
        reference_audit_path=reference,
        showo_report_path=showo,
        candidate_report_paths=[candidate],
        output_path=tmp_path / "decision.json",
    )
    assert report["passed"] is False
    assert report["matched_observer"]["matched"] is True
    assert report["evidence_reports"]["showo"]["sha256"]
    assert report["evidence_reports"]["candidates"][0]["observer_id"] == "candidate"
