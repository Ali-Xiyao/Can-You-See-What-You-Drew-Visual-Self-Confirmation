"""The verifier has to fail the case that looks right from the outside.

Five directories named after five seeds, five configs each naming a different
`training.seed`, five identical runs underneath. That is what happens when
`v4_train.py` predates the key, and it is the only case worth writing tests
for -- the others announce themselves.
"""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v4_verify_replicates.py"


def load():
    spec = importlib.util.spec_from_file_location("v4_verify_replicates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, name: str, *, training_seed: int, split: str, rounds: int = 2) -> Path:
    run = tmp_path / name
    (run).mkdir(parents=True)
    (run / "split.json").write_text(json.dumps({"digest": split}), encoding="utf-8")
    for index in range(rounds):
        round_dir = run / "rounds" / f"round-{index:03d}"
        round_dir.mkdir(parents=True)
        (round_dir / "done.json").write_text(
            json.dumps({"round": index, "initialization_seed": training_seed}), encoding="utf-8")
    return run


def test_five_directories_that_all_trained_on_one_seed_are_refused(tmp_path):
    # The whole point. Names differ, split is shared, seeds silently do not.
    runs = [_run(tmp_path, f"e3-s{s}", training_seed=20260906, split="abc")
            for s in (20260906, 20260907, 20260909, 20260910, 20260911)]
    report = load().verify(runs)
    assert report["verdict"] == "NOT independent seeds"
    assert "training seeds repeat" in report["reason"]


def test_distinct_seeds_on_different_splits_are_refused_too(tmp_path):
    runs = [_run(tmp_path, "a", training_seed=1, split="abc"),
            _run(tmp_path, "b", training_seed=2, split="def")]
    report = load().verify(runs)
    assert report["verdict"] == "NOT independent seeds"
    assert "do not share a split" in report["reason"]


def test_distinct_seeds_on_one_split_pass(tmp_path):
    runs = [_run(tmp_path, f"e3-s{s}", training_seed=s, split="abc")
            for s in (20260906, 20260907, 20260909, 20260910, 20260911)]
    report = load().verify(runs)
    assert report["verdict"] == "five seeds"
    assert [row["training_seed"] for row in report["runs"]] == [
        20260906, 20260907, 20260909, 20260910, 20260911]


def test_a_run_with_no_finished_round_is_an_error_not_a_pass(tmp_path):
    runs = [_run(tmp_path, "a", training_seed=1, split="abc"),
            _run(tmp_path, "b", training_seed=2, split="abc", rounds=0)]
    with pytest.raises(ValueError, match="no finished round"):
        load().verify(runs)


def test_a_run_that_changed_seed_midway_is_an_error(tmp_path):
    runs = [_run(tmp_path, "a", training_seed=1, split="abc"),
            _run(tmp_path, "b", training_seed=2, split="abc")]
    later = runs[1] / "rounds" / "round-005"
    later.mkdir(parents=True)
    (later / "done.json").write_text(json.dumps({"initialization_seed": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="more than one initialization seed"):
        load().verify(runs)


def test_one_run_cannot_be_checked_against_itself(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        load().verify([_run(tmp_path, "a", training_seed=1, split="abc")])
