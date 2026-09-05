"""Supplementary scene filtering must preserve both estimand and paired resampling."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v4_gradient_sensitivity import (  # noqa: E402
    build_report, checkpoint_summary, cluster_weights, paired_interval,
    read_main_probe, write_versioned,
)
from selfsight.utils.hashing import sha256_file, sha256_json


def _grams(left, right):
    return {"ll": left @ left.T, "rr": right @ right.T, "lr": left @ right.T}


def _cos(left, right):
    return (left * right).sum(-1) / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1))


def _fixture(root):
    bank_path = root / "probe-bank" / "bank.json"
    bank_path.parent.mkdir()
    pools = [{"spec_id": f"s{i}", "prompt_id": f"pool:s{i}"} for i in range(6)]
    bank = {"pools": pools}
    bank["fingerprint"] = sha256_json(bank)
    bank_path.write_text(json.dumps(bank))
    groups = [{"canonical_scene": [[f"object{i}", "red", 1]], "spec_ids": [f"s{i}"]} for i in range(6)]
    for group in groups:
        group["scene_sha256"] = sha256_json(group["canonical_scene"])
    audit = {"provenance": {"bank_sha256": sha256_file(bank_path), "bank_fingerprint": bank["fingerprint"]},
        "within_probe_bank_clusters": groups,
        "overlap": {"actual_probe_bank": [
            {"spec_id": group["spec_ids"][0], "scene_sha256": group["scene_sha256"]} for group in groups[:2]]},
        "summary": {"actual_probe_bank_overlap_n": 2}, "canonical_scene_key_definition": "test scene definition"}
    audit_path = root / "audit-splits" / "scene_overlap.json"
    audit_path.parent.mkdir()
    audit_path.write_text(json.dumps(audit))
    return bank


def _write_probe(root, bank, *, arm="base", step=0):
    output = root / "gradient-probes" / arm / f"step-{step:05}"
    output.mkdir(parents=True)
    rng = np.random.default_rng(step)
    left, right = rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
    arrays = {"prompt_ids": np.asarray([p["prompt_id"] for p in bank["pools"]])}
    for name in ("gda_free", "gda_gold"):
        arrays.update({f"{name}_{part}": matrix for part, matrix in _grams(left, right).items()})
    np.savez_compressed(output / "grams.npz", **arrays)
    report = {"bank_fingerprint": bank["fingerprint"], "checkpoint": {"arm": arm, "step": step},
              "prompt_ids": arrays["prompt_ids"].tolist(), "n_retained": 6,
              "grams_sha256": sha256_file(output / "grams.npz")}
    path = output / "report.json"
    path.write_text(json.dumps(report))
    return path


def test_scene_members_share_bootstrap_multiplicity():
    scenes = ["same", "other", "same", "third"]
    weights = cluster_weights(scenes, resamples=200, seed=91)
    assert np.array_equal(weights[:, 0], weights[:, 2])
    assert np.all(weights[:, 0] + weights[:, 1] + weights[:, 3] == 3)
    assert np.unique(weights.sum(1)).size > 1  # cluster sizes vary; do not reweight point to scene means.


def test_point_stays_prompt_weighted_when_scene_sizes_differ():
    left = np.array([[100., 0], [100, 0], [0, 1]])
    right = np.array([[1., 0], [1, 0], [0, 100]])
    summary = paired_interval(_grams(left, right), ["a", "a", "b"],
                              resamples=200, seed=19, minimum_clusters=2)
    expected = _cos(left.mean(0), right.mean(0))
    scene_weighted = _cos(np.mean([left[:2].mean(0), left[2]], axis=0),
                         np.mean([right[:2].mean(0), right[2]], axis=0))
    assert summary["point"] == pytest.approx(expected)
    assert abs(summary["point"] - scene_weighted) > .001


def test_filtered_reordered_paired_difference_matches_direct_vectors():
    rng = np.random.default_rng(3)
    base_ids = ("s5", "s2", "s0", "s4", "s1", "s3")
    current_ids = ("s4", "s1", "s5", "s2", "s0")
    base_l, base_r = rng.normal(size=(6, 10)), rng.normal(size=(6, 10))
    now_l, now_r = rng.normal(size=(5, 10)), rng.normal(size=(5, 10))
    base = {"prompt_ids": base_ids, "grams": {name: _grams(base_l, base_r) for name in ("gda_free", "gda_gold")}}
    current = {"checkpoint": {"arm": "naive", "step": 8}, "source": {}, "prompt_ids": current_ids,
               "grams": {name: _grams(now_l, now_r) for name in ("gda_free", "gda_gold")}}
    spec_scenes = {f"s{i}": ("same" if i in (1, 2) else f"scene{i}") for i in range(6)}
    report = checkpoint_summary(current, base, prompt_to_spec={s: s for s in base_ids},
        scene_by_spec=spec_scenes, excluded={"s0"},
        rules={"resamples": 2000, "seed": 15, "confidence": .95, "minimum_clusters": 2})
    delta = report["delta_from_base"]["gda_free"]
    common = ("s4", "s1", "s5", "s2")
    ni, bi = [current_ids.index(p) for p in common], [base_ids.index(p) for p in common]
    weights = cluster_weights([spec_scenes[p] for p in common], resamples=2000, seed=15)
    direct = _cos(weights @ now_l[ni], weights @ now_r[ni]) - _cos(weights @ base_l[bi], weights @ base_r[bi])
    point = _cos(now_l[ni].mean(0), now_r[ni].mean(0)) - _cos(base_l[bi].mean(0), base_r[bi].mean(0))
    assert delta["point_difference"] == pytest.approx(point)
    assert [delta["ci_low"], delta["ci_high"]] == pytest.approx(np.quantile(direct, [.025, .975]))
    assert delta["n_common"] == 4 and delta["n_clusters"] == 3
    assert report["missing_scene_disjoint_spec_ids"] == ["s3"]


def test_no_main_data_ignores_runtime_canary_and_reports_frozen_target(tmp_path):
    _fixture(tmp_path)
    runtime = tmp_path / "runtime-check" / "gradient-four-result"
    runtime.mkdir(parents=True)
    (runtime / "report.json").write_text('{"this": "is deliberately not a main report"}')
    report = build_report(tmp_path)
    assert report["status"] == "no_data"
    assert report["n_target_retained"] == 4
    assert report["arms"] == {} and report["input_provenance"] == []


def test_main_trajectory_filters_exposure_and_checks_hash(tmp_path):
    bank = _fixture(tmp_path)
    _write_probe(tmp_path, bank)
    current_path = _write_probe(tmp_path, bank, arm="naive", step=8)
    report = build_report(tmp_path)
    current = report["arms"]["naive"][0]
    assert current["n_scene_disjoint_retained"] == 4
    assert current["removed_spec_ids"] == ["s0", "s1"]
    assert current["delta_from_base"]["gda_free"]["n_common"] == 4
    assert current["delta_from_base"]["gda_free"]["status"] == "ok"
    original = json.loads(current_path.read_text())
    original["grams_sha256"] = "bad-hash"
    current_path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="checksum"):
        read_main_probe(current_path, bank)


def test_insufficient_and_degenerate_data_cannot_produce_valid_intervals():
    identity = np.eye(3)
    small = paired_interval(_grams(identity, identity), ["a", "b", "c"], resamples=200, seed=1)
    assert small["status"] == "insufficient_clusters" and small["ci_high"] is None
    zeros = np.zeros((4, 3))
    bad = paired_interval(_grams(zeros, zeros), list("abcd"), resamples=200, seed=1)
    assert bad["status"] == "degenerate_mean_gradient" and bad["point"] is None


def test_new_data_archives_previous_output_but_rule_change_refuses(tmp_path):
    bank = _fixture(tmp_path)
    output = tmp_path / "gradient_sensitivity.json"
    initial = build_report(tmp_path)
    assert write_versioned(output, initial) == "written"
    first_bytes = output.read_bytes()
    assert write_versioned(output, initial) == "unchanged"
    _write_probe(tmp_path, bank)
    assert write_versioned(output, build_report(tmp_path)) == "written"
    archives = list((tmp_path / "gradient-sensitivity-history").glob("*.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == first_bytes
    with pytest.raises(ValueError, match="rules/audit/bank"):
        write_versioned(output, build_report(tmp_path, seed=999))
