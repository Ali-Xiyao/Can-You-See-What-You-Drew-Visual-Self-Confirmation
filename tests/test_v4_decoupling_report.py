"""Pilot reports must not turn incomplete data or hindsight into an early warning."""

from __future__ import annotations

import json
import pathlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest

_script = Path(__file__).with_name("v4_decoupling_report.py")
if not _script.exists():
    _script = Path(__file__).resolve().parents[1] / "scripts/v4_decoupling_report.py"
_spec = importlib.util.spec_from_file_location("reviewed_v4_decoupling_report", _script)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
analyze_arm = _module.analyze_arm
behavior_windows = _module.behavior_windows
build_report = _module.build_report
classify_lead = _module.classify_lead
gradient_trajectory = _module.gradient_trajectory
paired_changes = _module.paired_changes
paired_binary_bounds = _module.paired_binary_bounds
read_outcome = _module.read_outcome
scene_cluster_changes = _module.scene_cluster_changes
add_scene_sensitivity = _module.add_scene_sensitivity


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


def test_late_registered_rule_lead_is_preserved_but_not_upgraded_to_robust_support():
    points = [_point(0, 0.5), _point(8, 0.51), _point(16, 0.54), _point(24, 0.55)]
    gradients = [_gradient(0, 0.99), _gradient(8, 0.987, paired=True),
                 _gradient(16, 0.97, paired=True), _gradient(24, 0.96, paired=True)]
    result = analyze_arm(points, gradients, bootstrap_samples=100)
    assert result["first_interval_supported_step"] == 16
    assert result["gradient"]["interval_supported_alarm_step"] == 24
    assert result["bootstrap_rule_lead"]["status"] == "late"
    assert result["bootstrap_rule_lead"]["lead_steps"] == -8
    assert result["lead"]["status"] == "no_event"
    assert result["first_robustness_supported_step"] is None


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


def test_zero_external_changes_on_64_specs_do_not_prove_two_point_stagnation():
    points = [_point(0, 0.5, 0.0, n=64), _point(8, 0.516, 0.0, n=64),
              _point(16, 0.532, 0.0, n=64)]
    result = analyze_arm(points, [], bootstrap_samples=2000)
    window = result["windows"][0]
    assert window["bootstrap_rule_supported"] is True
    assert window["bootstrap_degenerate_external_interval"] is True
    assert window["external"]["ci_high"] == 0
    robustness = window["robustness"]
    exact_upper = 1 - 0.0125 ** (1 / 64)
    assert robustness["external_paired_binary_bounds"]["ci_high"] == pytest.approx(exact_upper)
    assert robustness["external_confidence_and_unknown_envelope"][1] > 0.02
    assert window["robustness_supported"] is False
    assert result["first_bootstrap_rule_supported_step"] == 16
    assert result["first_robustness_supported_step"] is None


def test_two_complete_pairs_cannot_represent_64_specs(tmp_path):
    points = []
    for step in (0, 8, 16):
        rows = [{"spec": f"s{i}", "score": 0.5 + step * 0.002,
                 "verdict": False if step < 16 or i < 2 else None} for i in range(64)]
        points.append(read_outcome(_write_checkpoint(tmp_path, step, rows)))
    window = behavior_windows(points, bootstrap_samples=2000)[0]
    assert window["n_paired_specs"] == 2
    assert window["bootstrap_rule_supported"] is True
    robustness = window["robustness"]
    assert robustness["n_population_specs"] == 64
    assert robustness["n_external_unknown_pairs"] == 62
    assert robustness["external_pair_coverage"] == 2 / 64
    assert robustness["external_unknown_completion_delta_bounds"] == [0.0, 62 / 64]
    assert robustness["external_confidence_and_unknown_envelope"][1] >= 62 / 64
    assert window["robustness_supported"] is False


def test_missing_manifest_rows_still_belong_to_the_frozen_population(tmp_path):
    all_ids = [f"s{i}" for i in range(64)]
    directory = _write_checkpoint(tmp_path, 16, [
        {"spec": sid, "score": 0.6, "verdict": False} for sid in all_ids[:2]])
    point = read_outcome(directory, expected_spec_ids=all_ids)
    assert point["coverage"]["missing_manifest_specs"] == 62
    assert len(point["spec_measurements"]) == 64
    assert point["spec_measurements"]["s63"]["external"] is None
    assert point["spec_measurements"]["s63"]["external_high"] == 1
    with pytest.raises(ValueError, match="outside the frozen outcome split"):
        read_outcome(directory, expected_spec_ids=["different"])


def test_a_large_fully_observed_decline_can_pass_the_supplementary_check():
    points = [_point(0, 0.5, 1.0, n=64), _point(8, 0.516, 1.0, n=64),
              _point(16, 0.532, 0.0, n=64), _point(24, 0.54, 0.0, n=64)]
    gradients = [_gradient(0, 0.99), _gradient(8, 0.987, paired=True),
                 _gradient(16, 0.97, paired=True), _gradient(24, 0.96, paired=True)]
    result = analyze_arm(points, gradients, bootstrap_samples=100)
    window = result["windows"][0]
    assert window["bootstrap_rule_supported"] is True
    assert window["robustness_supported"] is True
    assert window["robustness"]["external_paired_binary_bounds"]["deteriorations"] == 64
    assert result["first_robustness_supported_step"] == 16
    assert result["lead"]["status"] == "late"
    assert result["lead"]["lead_steps"] == -8


def test_nonbinary_spec_rates_are_not_silently_sent_to_binomial_inference():
    points = [_point(0, 0.5, 0.5, n=64), _point(8, 0.516, 0.5, n=64),
              _point(16, 0.532, 0.0, n=64)]
    window = behavior_windows(points, bootstrap_samples=100)[0]
    assert window["bootstrap_rule_supported"] is True
    assert window["robustness"]["external_ci_status"] == "nonbinary_spec_rates"
    assert window["robustness_supported"] is False


def test_missing_internal_scores_block_the_supplementary_population_claim(tmp_path):
    points = []
    for step in (0, 8, 16):
        rows = [{"spec": f"s{i}", "score": 0.5 + step * 0.002
                 if step < 16 or i < 2 else None, "verdict": step < 16} for i in range(64)]
        points.append(read_outcome(_write_checkpoint(tmp_path, step, rows)))
    window = behavior_windows(points, bootstrap_samples=100)[0]
    assert window["bootstrap_rule_supported"] is True
    assert window["robustness"]["external_supported"] is True
    assert window["robustness"]["internal_pair_coverage"] == 2 / 64
    assert window["robustness_supported"] is False


def test_clopper_pearson_joint_bounds_cover_small_multinomial_population():
    n, p_plus, p_minus = 5, 0.3, 0.2
    truth = p_plus - p_minus
    coverage = 0.0
    for plus in range(n + 1):
        for minus in range(n - plus + 1):
            unchanged = n - plus - minus
            probability = (math.factorial(n) / math.factorial(plus) / math.factorial(minus)
                           / math.factorial(unchanged) * p_plus ** plus * p_minus ** minus
                           * (1 - p_plus - p_minus) ** unchanged)
            interval = paired_binary_bounds(plus, minus, n)
            if interval["ci_low"] <= truth <= interval["ci_high"]:
                coverage += probability
    assert coverage >= 0.95
    positive = paired_binary_bounds(64, 0, 64)
    negative = paired_binary_bounds(0, 64, 64)
    assert positive["ci_low"] == pytest.approx(-negative["ci_high"])
    assert positive["ci_high"] == pytest.approx(-negative["ci_low"])


def _scene_audit() -> dict:
    groups = [["s0", "s1", "s2"], ["s3", "s4"], ["s5", "s6"], ["s7", "s8"]]
    groups.extend([[f"s{i}"] for i in range(9, 64)])
    return {"created_before_any_outcome_evaluation_artifact": True,
            "within_outcome_clusters": [{"scene_sha256": f"scene-{index}", "spec_ids": ids}
                                         for index, ids in enumerate(groups)],
            "scene_disjoint_outcome_sensitivity": {"spec_ids": [f"s{i}" for i in range(57)]},
            "summary": {"outcome_unique_canonical_scenes": 59}}


def test_cluster_bootstrap_preserves_prompt_weighting_and_paired_scene_membership():
    start = _point(0, 0.0, n=3)
    end = _point(16, 1.0, n=3)
    end["complete_specs"]["s2"]["s_select"] = 0.0
    result = scene_cluster_changes(start, end, {"s0": "duplicate", "s1": "duplicate", "s2": "other"},
                                   bootstrap_samples=2000, seed=17)
    assert result["s_select"]["delta"] == pytest.approx(2 / 3)
    assert result["n_population_scene_clusters"] == 2
    assert result["n_paired_scene_clusters"] == 2
    assert result["s_select"]["ci_low"] == 0.0
    assert result["s_select"]["ci_high"] == 1.0
    assert result["valid_bootstrap_draws"] == 2000


def test_duplicate_scene_screen_does_not_change_registered_64_spec_rule():
    points = [_point(0, 0.5, 1.0, n=64), _point(8, 0.516, 1.0, n=64),
              _point(16, 0.532, 0.0, n=64)]
    report = analyze_arm(points, [], bootstrap_samples=100)
    assert report["windows"][0]["robustness_supported"] is True
    add_scene_sensitivity(report, points, _scene_audit(), bootstrap_samples=100, seed=17,
                          rules=_module.RULES)
    window = report["windows"][0]
    assert window["n_paired_specs"] == 64
    assert window["bootstrap_rule_supported"] is True
    assert report["first_bootstrap_rule_supported_step"] == 16
    assert window["scene_cluster_bootstrap"]["n_population_scene_clusters"] == 59
    assert window["robustness_supported_assuming_independent_specs"] is True
    assert window["robustness_supported"] is False
    assert "blocked_pending_scene" in window["robustness_support_status"]
    assert report["lead"]["status"] == "no_event"
    sensitivity = window["scene_disjoint_sensitivity"]
    assert sensitivity["n_paired_specs"] == 57
    assert sensitivity["requested_sensitivity_n"] == 57
    assert sensitivity["scene_cluster_bootstrap"]["n_population_scene_clusters"] == 52
    assert sensitivity["robustness_supported"] is False
    assert report["scene_disjoint_sensitivity_summary"]["requested_specs"] == 57


def test_scene_audit_loaded_and_its_identity_frozen_across_reports(tmp_path):
    (tmp_path / "split.json").write_text(json.dumps({"outcome": [f"s{i}" for i in range(64)]}),
                                         encoding="utf-8")
    for step in (0, 8, 16):
        _write_checkpoint(tmp_path, step, [{"spec": f"s{i}", "score": 0.5 + step * 0.002,
                                          "verdict": step < 16} for i in range(64)])
    audit_path = tmp_path / "audit-splits" / "scene_overlap.json"
    audit_path.parent.mkdir()
    audit = _scene_audit()
    audit["provenance"] = {"split_sha256": _module.file_identity(tmp_path / "split.json")["sha256"]}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    report = build_report(tmp_path, bootstrap_samples=100)
    assert report["scene_audit_status"] == "prospectively_frozen_supplement_loaded"
    assert report["provenance"]["scene_overlap"]["sha256"] == _module.file_identity(audit_path)["sha256"]
    assert report["arms"]["naive"]["windows"][0]["scene_disjoint_sensitivity"]["n_paired_specs"] == 57
    (tmp_path / "decoupling_report.json").write_text(json.dumps(report), encoding="utf-8")
    audit["scene_disjoint_outcome_sensitivity"]["spec_ids"].pop()
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen scene_overlap changed"):
        build_report(tmp_path, bootstrap_samples=100)


def _run_with_audit(tmp_path, *, recorded_run):
    """A run whose audit says, in its own provenance, where it was built."""

    (tmp_path / "split.json").write_text(json.dumps({"outcome": [f"s{i}" for i in range(64)]}),
                                         encoding="utf-8")
    for step in (0, 8, 16):
        _write_checkpoint(tmp_path, step, [{"spec": f"s{i}", "score": 0.5 + step * 0.002,
                                            "verdict": step < 16} for i in range(64)])
    audit_path = tmp_path / "audit-splits" / "scene_overlap.json"
    audit_path.parent.mkdir()
    audit = _scene_audit()
    audit["provenance"] = {
        "split_sha256": _module.file_identity(tmp_path / "split.json")["sha256"],
        "run": recorded_run}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return audit_path


def test_an_audit_built_in_another_run_is_not_this_runs_freeze(tmp_path):
    """Two runs off one config share a config digest and a split digest, so
    every other check here passes on a copied file. The one that matters is
    that the other run's audit may be honestly prospective there and post hoc
    here -- which is exactly the case arm B and the main run are in."""

    _run_with_audit(tmp_path, recorded_run=str(tmp_path.parent / "decoupling-main-20260908"))
    with pytest.raises(ValueError, match="was built in"):
        build_report(tmp_path, bootstrap_samples=100)


def test_a_run_that_moved_can_still_read_its_own_audit(tmp_path):
    """Compared by name rather than by path on purpose: after outcomes exist
    the audit cannot be rebuilt, so a moved directory must not lose it."""

    _run_with_audit(tmp_path, recorded_run=str(pathlib.PurePosixPath("/somewhere/else")
                                               / tmp_path.name))
    report = build_report(tmp_path, bootstrap_samples=100)
    assert report["scene_audit_status"] == "prospectively_frozen_supplement_loaded"


def test_an_audit_that_does_not_say_where_it_was_built_is_still_read(tmp_path):
    """The pilot's audit predates the field. Requiring it would strand the one
    run that has a hand-checked prospective freeze."""

    _run_with_audit(tmp_path, recorded_run=None)
    report = build_report(tmp_path, bootstrap_samples=100)
    assert report["scene_audit_status"] == "prospectively_frozen_supplement_loaded"


def test_scene_audit_cannot_silently_target_a_different_outcome_population(tmp_path):
    (tmp_path / "split.json").write_text(json.dumps({"outcome": ["foreign"]}), encoding="utf-8")
    path = tmp_path / "audit-splits" / "scene_overlap.json"
    path.parent.mkdir()
    path.write_text(json.dumps(_scene_audit()), encoding="utf-8")
    with pytest.raises(ValueError, match="population does not match frozen split"):
        build_report(tmp_path, bootstrap_samples=100)


def test_one_observed_scene_does_not_get_a_cluster_confidence_interval():
    start, end = _point(0, 0.5, n=3), _point(16, 0.6, n=3)
    result = scene_cluster_changes(start, end, {f"s{i}": "same" for i in range(3)},
                                   bootstrap_samples=100, seed=17)
    assert result["n_paired_scene_clusters"] == 1
    assert result["s_select"]["ci_low"] is None
    assert result["external"]["ci_high"] is None


def test_fixed_missingness_mixture_is_not_promoted_to_population_support(tmp_path):
    points = []
    for step in (0, 8, 16):
        rows = [{"spec": f"s{i}", "score": 0.5 + step * 0.002,
                 "verdict": (i < 36) if step < 16 else False if i < 36 else None}
                for i in range(64)]
        points.append(read_outcome(_write_checkpoint(tmp_path, step, rows)))
    window = behavior_windows(points, bootstrap_samples=100)[0]
    robust = window["robustness"]
    assert robust["external_confidence_and_unknown_envelope"][1] < 0.02
    assert robust["external_fixed_cohort_screen"] is True
    assert robust["external_complete_population"] is False
    assert robust["external_full_cohort_completion_confidence_bounds"][1] > 0.02
    assert robust["external_supported"] is False
    assert window["robustness_supported"] is False


def test_full_cohort_completion_can_support_a_large_decline_with_one_unknown(tmp_path):
    points = []
    for step in (0, 8, 16):
        rows = [{"spec": f"s{i}", "score": 0.5 + step * 0.002,
                 "verdict": True if step < 16 else False if i < 63 else None}
                for i in range(64)]
        points.append(read_outcome(_write_checkpoint(tmp_path, step, rows)))
    window = behavior_windows(points, bootstrap_samples=100)[0]
    robust = window["robustness"]
    assert robust["n_external_unknown_pairs"] == 1
    assert robust["external_complete_population"] is False
    assert robust["external_upper_completion_binary_bounds"]["n"] == 64
    assert robust["external_full_cohort_completion_confidence_bounds"][1] < 0.02
    assert robust["external_supported"] is True
    assert window["robustness_supported"] is True


def test_frozen_one_representative_per_held_out_scene_allows_conditional_screen():
    points = [_point(0, 0.5, 1.0, n=64), _point(8, 0.516, 1.0, n=64),
              _point(16, 0.532, 0.0, n=64)]
    audit = _scene_audit()
    allowed = set(audit["scene_disjoint_outcome_sensitivity"]["spec_ids"])
    representatives = {"spec_ids": sorted(min(group["spec_ids"]) for group in audit["within_outcome_clusters"]
                                          if set(group["spec_ids"]) <= allowed)}
    report = analyze_arm(points, [], bootstrap_samples=100)
    add_scene_sensitivity(report, points, audit, bootstrap_samples=100, seed=17,
                          rules=_module.RULES, representative_audit=representatives)
    window = report["windows"][0]
    assert window["n_paired_specs"] == 64
    assert window["bootstrap_rule_supported"] is True
    assert window["robustness_supported"] is False
    assert window["scene_disjoint_sensitivity"]["n_paired_specs"] == 57
    representative = window["independent_scene_representative_sensitivity"]
    assert representative["n_paired_specs"] == 52
    assert representative["scene_cluster_bootstrap"]["n_population_scene_clusters"] == 52
    assert representative["robustness_supported"] is True
    assert representative["robustness_support_status"] == "conditional_on_independent_canonical_scenes"
    assert report["independent_scene_representative_summary"]["first_robustness_supported_step"] == 16


def test_representative_selection_cannot_change_after_seeing_outcomes():
    report = analyze_arm([], [], bootstrap_samples=100)
    with pytest.raises(ValueError, match="lexicographic scene rule"):
        add_scene_sensitivity(report, [], _scene_audit(), bootstrap_samples=100, seed=17,
                              rules=_module.RULES, representative_audit={"spec_ids": ["s1"]})
