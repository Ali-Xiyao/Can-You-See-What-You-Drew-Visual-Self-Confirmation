"""The walk from an evaluation directory back to the adapter that drew it.

Getting this mapping wrong is the kind of mistake that produces a clean-looking
curve made of the wrong models: `step-00016` answered by round 1's adapter
instead of round 0's shifts every point by one round and still plots.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _curve():
    path = Path(__file__).resolve().parents[1] / "scripts/v4_context_curve.py"
    spec = importlib.util.spec_from_file_location("v4_context_curve_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(root: Path, steps: list[int], arm: str = "naive") -> Path:
    for step in steps:
        (root / "evaluations" / arm / f"step-{step:05d}").mkdir(parents=True)
    return root


def test_the_untrained_point_wears_the_base_adapter_not_the_arms(tmp_path):
    """Before round 0 the arms are the same model and no arm directory exists."""

    curve = _curve()
    (step, _, checkpoint), = curve.checkpoints(_run(tmp_path, [0]), "naive", 8)
    assert step == 0
    assert checkpoint.parts[-2:] == ("base", "round--01")


def test_step_maps_back_to_the_round_that_produced_it(tmp_path):
    curve = _curve()
    found = curve.checkpoints(_run(tmp_path, [0, 8, 16, 96]), "naive", 8)
    assert [step for step, _, _ in found] == [0, 8, 16, 96]
    assert [path.name for _, _, path in found] == [
        "round--01", "round-000", "round-001", "round-011"]


def test_the_arm_owns_its_checkpoints(tmp_path):
    """Two arms train from the same base and diverge immediately; reading one
    arm's curve out of the other's adapters would show no difference at all."""

    curve = _curve()
    (_, _, base), (_, _, trained) = curve.checkpoints(_run(tmp_path, [0, 8], "blind_self"),
                                                      "blind_self", 8)
    assert "blind_self" in str(trained)
    assert "blind_self" not in str(base)


def test_the_order_is_training_order_not_directory_order(tmp_path):
    """step-00008 sorts before step-00096 as a string only because the names are
    zero padded. The padding is load-bearing and this says so."""

    curve = _curve()
    found = curve.checkpoints(_run(tmp_path, [96, 8, 0]), "naive", 8)
    assert [step for step, _, _ in found] == [0, 8, 96]


def test_a_run_with_no_evaluations_yet_is_empty_not_an_error(tmp_path):
    curve = _curve()
    (tmp_path / "evaluations" / "naive").mkdir(parents=True)
    assert curve.checkpoints(tmp_path, "naive", 8) == []


def test_both_conditions_write_to_files_the_curve_can_tell_apart():
    """If the two condition files collided, every point would be a self-contrast."""

    curve = _curve()
    assert len(set(curve.CONDITIONS.values())) == 2
    for filename in curve.CONDITIONS.values():
        assert filename.endswith(".self.jsonl"), "a curve point must not land in answers.jsonl"
