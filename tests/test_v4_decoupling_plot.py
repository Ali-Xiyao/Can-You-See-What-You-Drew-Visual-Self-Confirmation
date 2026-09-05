"""Plotting must preserve missingness, paired-CI meaning, and canary isolation."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "v4_decoupling_plot.py"
    spec = importlib.util.spec_from_file_location("v4_decoupling_plot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_run(path):
    (path / "checkpoint_metrics.csv").write_text(
        "arm,step,s_select,external_correct,external_n,external_unadjudicated,s_select_available,s_select_total\n"
        "naive,0,0.6,0.5,10,0,40,40\nnaive,8,0.7,0.4,8,2,38,40\n"
        "runtime-canary,999,1,1,1,0,1,1\n", encoding="utf-8")
    reports = [
        ("base", {"checkpoint": {"arm": "base", "step": 0}, "gda_free": {"cosine": .98}}),
        ("naive/step-008", {"checkpoint": {"arm": "naive", "step": 8}, "gda_free": {"cosine": .95},
         "delta_from_reference": {"gda_free": {"point_difference": -.02, "ci_low": -.04,
             "ci_high": .01, "reference_step": 0, "n_common": 16,
             "method": "paired_bootstrap_difference"}}}),
        ("runtime-canary", {"checkpoint": {"arm": "naive", "step": 999}, "gda_free": {"cosine": .01}}),
    ]
    for name, row in reports:
        directory = path / "gradient-probes" / name
        directory.mkdir(parents=True)
        (directory / "report.json").write_text(json.dumps(row), encoding="utf-8")


def test_no_data_is_successful_noop(tmp_path, capsys):
    mod = module()
    assert mod.render(mod.load_inputs(tmp_path), tmp_path) is None
    assert "No trajectory yet" in capsys.readouterr().out
    assert not (tmp_path / "plots").exists()


def test_canaries_excluded_and_paired_delta_not_replaced_by_marginal(tmp_path):
    mod = module()
    fixture_run(tmp_path)
    data = mod.load_inputs(tmp_path)
    assert data["excluded_canaries"] == 2
    assert set(data["metrics"]) == {"naive"}
    points = mod.gradient_points(data["gradients"]["naive"])
    assert [p["step"] for p in points] == [0, 8]
    assert points[1]["delta"] == -.02  # marginal difference would be -.03
    assert points[1]["paired"] and points[1]["low"] == -.04
    raw = data["gradients"]["naive"][1]
    raw["delta_from_reference"]["gda_free"]["method"] = "independent_intervals"
    assert mod.gradient_points(data["gradients"]["naive"])[1]["low"] is None
    assert math.isnan(mod.verdict_coverage({"external_n": 8, "external_unadjudicated": None}))


def test_render_standalone_png_and_preserve_existing_output(tmp_path):
    pytest.importorskip("matplotlib")
    mod = module()
    fixture_run(tmp_path)
    data = mod.load_inputs(tmp_path)
    output = tmp_path / "fixture.png"
    assert mod.render(data, tmp_path, output) == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads(output.with_suffix(".json").read_text())["exploratory"] is True
    with pytest.raises(ValueError, match="overwrite"):
        mod.render(data, tmp_path, output)


def test_modified_input_aborts_before_writing_plot(tmp_path):
    pytest.importorskip("matplotlib")
    mod = module()
    fixture_run(tmp_path)
    data = mod.load_inputs(tmp_path)
    (tmp_path / "checkpoint_metrics.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Input changed"):
        mod.render(data, tmp_path)
    assert not (tmp_path / "plots").exists()
