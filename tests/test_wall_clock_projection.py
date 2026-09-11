"""The ladder must be derived, and a short ladder must be loud.

Sections 0.12 and 0.21 both projected the finish time off a hand-typed list of
remaining checkpoints, and both lists were one short.  These tests pin the two
things that would have caught it: the last step is 8 * rounds (not
8 * (rounds - 1)), and a landed sequence that disagrees with the derivation
raises instead of projecting.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v4_wall_clock_projection.py"
CONFIG = ROOT / "configs" / "v4_decoupling_main_20260908.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("v4_wall_clock_projection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proj = _module()


def test_the_ladder_ends_at_eight_times_rounds_not_one_short():
    # The exact off-by-one that 0.12 and 0.21 shipped: both stopped at step-80.
    steps = proj.ladder({"training": {"rounds": 11, "optimizer_steps_per_round": 8}})
    assert steps[-1] == 88
    assert steps == [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]


def test_the_ladder_has_one_more_entry_than_rounds():
    for rounds in (1, 3, 11, 20):
        steps = proj.ladder({"training": {"rounds": rounds, "optimizer_steps_per_round": 8}})
        assert len(steps) == rounds + 1, "the untrained baseline gets a report too"


def test_the_ladder_matches_the_live_config():
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    steps = proj.ladder(config)
    assert steps[-1] == config["training"]["rounds"] * config["training"]["optimizer_steps_per_round"]
    # The config comment promises twelve checkpoints "counting round -1".
    assert len(steps) == 12


def test_a_short_ladder_raises_instead_of_projecting():
    with pytest.raises(SystemExit) as excinfo:
        proj.check_prefix([0, 8, 16, 24], [0, 8, 32])
    assert "not a prefix" in str(excinfo.value)


def test_a_correct_prefix_passes():
    proj.check_prefix([0, 8, 16, 24, 32], [0, 8, 16])


def test_the_live_run_is_still_a_prefix_of_its_own_ladder():
    outdir = ROOT / "runs" / "v4" / "decoupling-main-20260908"
    if not (outdir / "state.json").exists():
        pytest.skip("the main run directory is not present")
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seen = [step for step, _ in proj.landed(outdir)]
    proj.check_prefix(proj.ladder(config), seen)
    assert seen, "at least one report has landed"
    assert json.loads((outdir / "state.json").read_text(encoding="utf-8"))["through_round"] == config["training"]["rounds"]
