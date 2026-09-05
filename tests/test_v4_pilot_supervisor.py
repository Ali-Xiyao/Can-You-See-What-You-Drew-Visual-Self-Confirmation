"""A completed file must not hide absent or invalid optimizer updates."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pilot_supervisor", ROOT / "scripts/run_decoupling_pilot.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("field,value", [
    ("parameter_delta_l2", 0),
    ("parameter_delta_l2", float("nan")),
    ("mean_gradient_norm_before_clip", float("inf")),
    ("mean_t2i_loss", float("nan")),
])
def test_done_with_invalid_updates_is_rejected(tmp_path, field, value):
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.config = {"pilot": {"min_paired_prompts": 2}}
    arm = {"mean_t2i_loss": 1, "mean_gradient_norm_before_clip": .2, "parameter_delta_l2": .1}
    arm[field] = value
    target = tmp_path / "rounds/round-000/DONE.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"paired": 3, "arms": [arm, arm]}))
    with pytest.raises(RuntimeError):
        runner.validate_round(0)


def test_done_with_too_few_paired_prompts_is_rejected(tmp_path):
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.config = {"pilot": {"min_paired_prompts": 2}}
    target = tmp_path / "rounds/round-000/DONE.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"paired": 1, "arms": []}))
    with pytest.raises(RuntimeError, match="too few"):
        runner.validate_round(0)
