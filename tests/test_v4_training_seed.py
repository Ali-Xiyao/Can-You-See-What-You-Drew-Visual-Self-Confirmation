"""Replicate seeds must move the training and leave the measurement alone.

Three seeds of one design only mean anything if all three land on the same
partition and the same frozen latents. `v4_train.py` used to take every seed
from one top-level key, so moving it moved the split, the 64 outcome specs and
the evaluation latents at the same time -- three experiments, not three seeds.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v4_train.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("v4_train_training_seed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**training):
    return {"seed": 20260906,
            "data": {"local_outcome": 64, "local_probe": 32},
            "training": dict(training)}


def test_a_config_without_the_new_key_behaves_exactly_as_it_did():
    runner = load_runner()
    config = _config()
    assert runner.training_seed(config) == 20260906
    assert runner.partition_seed(config) == 20260906


def test_the_training_seed_moves_and_the_partition_seed_stays():
    runner = load_runner()
    moved = _config(seed=20260907)
    assert runner.training_seed(moved) == 20260907
    # The whole point. If this ever reads 20260907, the second replicate is
    # measured on a different set of prompts than the first and the paired
    # comparison across seeds is comparing two populations.
    assert runner.partition_seed(moved) == 20260906
    assert runner.split_digest(moved, ("a", "b")) == runner.split_digest(_config(), ("a", "b"))


def test_the_dead_plural_key_is_refused_rather_than_quietly_ignored():
    runner = load_runner()
    with pytest.raises(ValueError, match="never by v4"):
        runner.training_seed(_config(seeds=[20260906, 20260907, 20260909]))


def test_the_plural_key_the_existing_configs_carry_is_still_accepted():
    # configs/v4_decoupling_main_20260908.yaml has `seeds: [20260906]` beside
    # `seed: 20260906`. The guard must not turn the running design into an error.
    runner = load_runner()
    assert runner.training_seed(_config(seeds=[20260906])) == 20260906


# ------------------------------------------------------- the call sites

TRAINING_SIDE = {"build_schedule", "train_arm"}
PARTITION_SIDE = {"split_prompts", "evaluation_seed"}


def _callee(node: ast.Call) -> str:
    target = node.func
    return target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")


def _helper(value: ast.AST) -> str | None:
    return _callee(value) if isinstance(value, ast.Call) else None


def _seed_arguments():
    """(callee, helper) for every seed a call site hands to something."""

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    # Skip the helpers themselves: partition_seed's body is the one place that
    # is allowed to read config["seed"] directly.
    bodies = [node for node in tree.body
              if not (isinstance(node, ast.FunctionDef)
                      and node.name in {"partition_seed", "training_seed"})]
    for body in bodies:
        for node in ast.walk(body):
            if not isinstance(node, ast.Call):
                continue
            name = _callee(node)
            if name == "seed_training" and node.args:
                yield name, _helper(node.args[0])
            for keyword in node.keywords:
                if keyword.arg == "seed":
                    yield name, _helper(keyword.value)


def test_every_seed_a_call_site_hands_out_comes_from_one_of_the_two_helpers():
    seen = list(_seed_arguments())
    # Vacuous-pass guard: an ast walk that matched nothing would agree with
    # every claim below.
    assert len(seen) >= 8, seen
    assert all(helper in {"partition_seed", "training_seed"} for _, helper in seen), seen


@pytest.mark.parametrize("callee", sorted(TRAINING_SIDE | {"seed_training"}))
def test_the_training_side_takes_the_training_seed(callee):
    helpers = [helper for name, helper in _seed_arguments() if name == callee]
    assert helpers, f"no call to {callee} passes a seed"
    assert set(helpers) == {"training_seed"}, {callee: helpers}


@pytest.mark.parametrize("callee", sorted(PARTITION_SIDE))
def test_the_measurement_side_takes_the_partition_seed(callee):
    helpers = [helper for name, helper in _seed_arguments() if name == callee]
    assert helpers, f"no call to {callee} passes a seed"
    assert set(helpers) == {"partition_seed"}, {callee: helpers}
