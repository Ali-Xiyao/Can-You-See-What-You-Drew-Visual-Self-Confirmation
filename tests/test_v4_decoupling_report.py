"""Pilot reports must not turn incomplete data or hindsight into an early warning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v4_decoupling_report import (  # noqa: E402
    analyze_arm,
    behavior_windows,
    build_report,
    classify_lead,
    gradient_trajectory,
    paired_changes,
    read_outcome,
)


def _point(step: int, score: float, external: float = 1.0, n: int = 6) -> dict:
    return {"step": step, "arm": "naive", "coverage": {"complete_specs": n},
            "complete_specs": {f"s{i}": {"s_select": score, "external": external,
                                          "n_images": 1} for i in range(n)}}


def _gradient(step: int, cosine: float, *, paired: bool = False, arm: str = "naive") -> dict:
    row = {"checkpoint": {"step": step, "arm": arm},
           "gda_free": {"cosine": cosine, "ci_low": cosine - 0.001,
                        "ci_high": cosine + 0.001}}
    if paired:
        delta = cosine - 0.99
        row["delta_from_reference"] = {"gda_free": {
            "point_difference": delta, "ci_low": delta - 0.001,
            "ci_high": delta + 0.001, "n_common": 6, "reference_step": 0,
            "method": "paired_bootstrap_difference"}}
    return row


def _write_checkpoint(root: Path, step: int, images: list[dict], *,
                      write_verified: bool = True) -> Path:
    path = root / "evaluations" / "naive" / f"step-{step:05d}"
    path.mkdir(parents=True)
    manifest = []
    verified = []
    scores = []
    for index, row in enumerate(images):
        image = f"image-{step}-{index}.png"
        manifest.append({"image_path": image, "spec_id": row["spec"]})
        if row.get("score") is not None:
            scores.append({"image_path": image, "prompt_id": row["spec"],
                           "s_select": row["score"], "available": row.get("available", 4),
                           "total": 4})
        if row.get("verdict") is not None:
            verified.append({"image_path": image, "image_correct": row["verdict"],
                             "resolution": row.get("resolution", "agreed"),
                             "detections": row.get("detections", [])})
    for name, rows in (("manifest", manifest), ("s_select", scores), ("verified", verified)):
        if name == "verified" and not write_verified:
            continue
        (path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows),
                                           encoding="utf-8")
    return path


@pytest.mark.parametrize("points", [[_point(0, 0.5)], [_point(0, 0.5), _point(8, 0.6)]])
def test_baseline_or_two_points_do_not_create_an_event(points):
    result = analyze_arm(points, [], bootstrap_samples=100)
    assert result["status"] == "insufficient_checkpoints"
    assert result["windows"] == []
    assert result["lead"]["status"] == "no_event"
    assert result["lead"]["lead_steps"] is None


def test_behavior_uses_first_and_last_paired_specs_and_records_availability():
    points = [_point(0, 0.2), _point(8, 0.99), _point(16, 0.22)]
    window = behavior_windows(points, bootstrap_samples=100)[0]
    assert window["window_steps"] == [0, 8, 16]
    assert window["available_at_step"] == 16
    assert window["s_select"]["delta"] == pytest.approx(0.02)
    assert window["candidate"] is True
    assert window["interval_supported"] is True
    assert window["n_paired_specs"] == 6
    assert window["exploratory"] is True
    assert window["multiple_looks_adjusted"] is False


def test_both_metrics_improving_and_both_metrics_flat_are_not_decoupling():
    improving = [_point(0, 0.2, 0), _point(8, 0.3, 0), _point(16, 0.4, 1)]
    flat = [_point(step, 0.5) for step in (0, 8, 16)]
    assert not behavior_windows(improving, bootstrap_samples=100)[0]["candidate"]
    assert not behavior_windows(flat, bootstrap_samples=100)[0]["candidate"]


def test_missing_verdict_is_unknown_but_human_unnameable_is_known_failure(tmp_path):
    rows = [{"spec": "known", "score": 0.5, "verdict": True},
            {"spec": "pending", "score": 0.5, "verdict": False,
             "resolution": "pending_human"},
            {"spec": "missing", "score": 0.5, "verdict": None},
            {"spec": "human_x", "score": 0.0, "available": 0, "verdict": False,
             "resolution": "human", "detections": [{"object": "unnameable"}]}]
    result = read_outcome(_write_checkpoint(tmp_path, 0, rows))
    assert set(result["complete_specs"]) == {"known", "human_x"}
    assert result["complete_specs"]["human_x"]["external"] == 0
    assert result["coverage"]["known_verdict_images"] == 2
    assert result["coverage"]["answer_coverage"] == 0.75
    missing = read_outcome(_write_checkpoint(tmp_path, 8, rows, write_verified=False))
    assert missing["complete_specs"] == {}
    assert "verified.jsonl" in missing["missing_files"]
    paired = paired_changes(result, missing, bootstrap_samples=100)
    assert paired["n_paired_specs"] == 0
    assert paired["external"]["delta"] is None


def test_repeated_images_of_one_spec_do_not_count_as_independent_specs(tmp_path):
    before = [{"spec": "a", "score": 0, "verdict": True},
              {"spec": "a", "score": 0, "verdict": True},
              {"spec": "b", "score": 1, "verdict": True}]
    after = [{**row, "score": 1} for row in before]
    start = read_outcome(_write_checkpoint(tmp_path, 0, before))
    end = read_outcome(_write_checkpoint(tmp_path, 8, after))
    result = paired_changes(start, end, bootstrap_samples=100)
    assert result["n_paired_specs"] == 2
    assert result["s_select"]["delta"] == 0.5


def test_incomplete_spec_is_not_partially_averaged_into_a_paired_window(tmp_path):
    rows = [{"spec": "a", "score": 0.5, "verdict": True},
            {"spec": "a", "score": 0.5, "verdict": None},
            {"spec": "b", "score": 0.5, "verdict": True}]
    result = read_outcome(_write_checkpoint(tmp_path, 0, rows))
    assert set(result["complete_specs"]) == {"b"}
    paired = paired_changes(result, result, bootstrap_samples=100)
    assert paired["n_paired_specs"] == 1
    assert paired["ci_status"] == "insufficient_paired_specs"
    assert paired["s_select"]["ci_low"] is None


def test_marginal_gradient_intervals_cannot_support_a_difference_alarm():
    reports = [_gradient(0, 0.99), _gradient(8, 0.97), _gradient(16, 0.96)]
    result = gradient_trajectory(reports, [0, 8, 16])
    assert result["candidate_alarm_step"] == 16
    assert result["interval_supported_alarm_step"] is None
    assert result["points"][-1]["support_status"] == "unsupported_no_paired_difference_ci"
    assert result["points"][-1]["paired_ci_high"] is None


def test_missing_probe_breaks_consecutive_flags_and_alarm_uses_confirmation_time():
    reports = [_gradient(0, 0.99), _gradient(8, 0.97, paired=True),
               _gradient(24, 0.96, paired=True), _gradient(32, 0.95, paired=True)]
    result = gradient_trajectory(reports, [0, 8, 16, 24, 32])
    assert result["points"][2]["missing_report"] is True
    assert result["interval_supported_alarm_step"] == 32
    assert result["candidate_alarm_step"] == 32


def test_late_gradient_alarm_has_negative_lead_not_zero():
    points = [_point(0, 0.5), _point(8, 0.51), _point(16, 0.54), _point(24, 0.55)]
    gradients = [_gradient(0, 0.99), _gradient(8, 0.987, paired=True),
                 _gradient(16, 0.97, paired=True), _gradient(24, 0.96, paired=True)]
    result = analyze_arm(points, gradients, bootstrap_samples=100)
    assert result["first_interval_supported_step"] == 16
    assert result["gradient"]["interval_supported_alarm_step"] == 24
    assert result["lead"]["status"] == "late"
    assert result["lead"]["lead_steps"] == -8


@pytest.mark.parametrize("event,alarm,status,lead", [
    (16, 8, "earlier", 8), (16, 16, "simultaneous", 0),
    (16, 24, "late", -8), (16, None, "missed", None),
    (None, 8, "no_event", None), (None, None, "no_event", None),
])
def test_lead_keeps_missing_and_simultaneous_distinct(event, alarm, status, lead):
    assert classify_lead(event, alarm)["status"] == status
    assert classify_lead(event, alarm)["lead_steps"] == lead


def test_shared_base_probe_and_frozen_protocol_are_used(tmp_path):
    for step, score in [(0, 0.5), (8, 0.51), (16, 0.54)]:
        _write_checkpoint(tmp_path, step, [
            {"spec": f"s{i}", "score": score, "verdict": True} for i in range(6)])
    for path, row in [
        ("base", _gradient(0, 0.99, arm="base")),
        ("naive/round-000", _gradient(8, 0.97, paired=True)),
        ("naive/round-001", _gradient(16, 0.96, paired=True)),
    ]:
        directory = tmp_path / "probes" / path
        directory.mkdir(parents=True)
        (directory / "report.json").write_text(json.dumps(row), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("pilot:\n  internal_change_margin: 0.05\n", encoding="utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen before this run", encoding="utf-8")
    report = build_report(tmp_path, bootstrap_samples=100,
                          config_path=config, protocol_path=protocol)
    assert set(report["arms"]) == {"naive"}
    assert report["arms"]["naive"]["gradient"]["interval_supported_alarm_step"] == 16
    assert report["arms"]["naive"]["first_candidate_step"] is None
    assert len(report["provenance"]["protocol"]["sha256"]) == 64
    (tmp_path / "decoupling_report.json").write_text(json.dumps(report), encoding="utf-8")
    protocol.write_text("changed after looking", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen protocol changed"):
        build_report(tmp_path, bootstrap_samples=100, config_path=config, protocol_path=protocol)


def test_wrong_reference_or_too_few_common_prompts_cannot_support_alarm():
    reports = [_gradient(0, 0.99), _gradient(8, 0.97, paired=True),
               _gradient(16, 0.96, paired=True)]
    reports[1]["delta_from_reference"]["gda_free"]["reference_step"] = 4
    reports[2]["delta_from_reference"]["gda_free"]["n_common"] = 3
    result = gradient_trajectory(reports, [0, 8, 16])
    assert result["candidate_alarm_step"] == 16
    assert result["interval_supported_alarm_step"] is None
