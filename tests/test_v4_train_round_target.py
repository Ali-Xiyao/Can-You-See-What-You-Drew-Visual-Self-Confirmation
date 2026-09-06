"""Exact round requests must be idempotent across a supervisor restart."""
import argparse
import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

from selfsight.v4.train import pending_rounds


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/v4_train.py"
    spec = importlib.util.spec_from_file_location("v4_train_round_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("done,target,expected", [
    ([], 0, [0]),
    ([0], 1, [1]),
    ([0, 1], 1, []),
    ([0, 1], 0, []),
    ([0, 1], 2, [2]),
    (list(range(10)), 9, []),
])
def test_exact_round_does_not_advance_to_another_pending_round(done, target, expected):
    assert pending_rounds(10, done, None, round_index=target) == expected


@pytest.mark.parametrize("target", [-1, 10])
def test_exact_round_must_be_inside_frozen_schedule(target):
    with pytest.raises(ValueError, match="round-index"):
        pending_rounds(10, [], None, round_index=target)


def test_exact_round_requires_preceding_rounds_and_valid_done_prefix():
    with pytest.raises(ValueError, match="preceding"):
        pending_rounds(10, [0], None, round_index=2)
    with pytest.raises(ValueError, match="contiguous"):
        pending_rounds(10, [0, 2], None, round_index=2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        pending_rounds(10, [0], 1, round_index=1)


def test_completed_target_returns_before_model_imports_or_training_inputs(tmp_path, monkeypatch, capsys):
    runner = load_runner()
    config = tmp_path / "config.yaml"
    config.write_text("training:\n  rounds: 10\n", encoding="utf-8")
    for index in (0, 1):
        done = tmp_path / "rounds" / f"round-{index:03d}" / "DONE.json"
        done.parent.mkdir(parents=True)
        done.write_text("{}", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Completed target must not read training data or construct a model")

    monkeypatch.setattr(runner, "read_split", forbidden)
    monkeypatch.setattr(runner, "load_training_corpus", forbidden)
    monkeypatch.setattr(runner, "seed_training", forbidden)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith(("selfsight.backbones", "selfsight.observers")):
            pytest.fail("Completed target must return before importing model runtimes")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runner.stage_train(argparse.Namespace(config=config, outdir=tmp_path,
                                         round_index=1, max_rounds=None))
    assert "Requested round 1 is already complete" in capsys.readouterr().out
    assert not (tmp_path / "rounds/round-002").exists()


def test_cli_accepts_exact_round_index(tmp_path, monkeypatch):
    runner = load_runner()
    received = []
    monkeypatch.setattr(runner, "stage_train", received.append)
    monkeypatch.setattr(sys, "argv", ["v4_train.py", "train", "--outdir", str(tmp_path),
                                     "--round-index", "1"])
    runner.main()
    assert len(received) == 1
    assert received[0].round_index == 1 and received[0].max_rounds is None


def test_cli_round_modes_are_mutually_exclusive(tmp_path, monkeypatch, capsys):
    runner = load_runner()
    monkeypatch.setattr(sys, "argv", ["v4_train.py", "train", "--outdir", str(tmp_path),
                                     "--round-index", "1", "--max-rounds", "1"])
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
