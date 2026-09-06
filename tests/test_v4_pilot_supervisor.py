"""A completed file must not hide absent or invalid optimizer updates."""
import importlib.util
import json
from pathlib import Path
import time

import pytest

from selfsight.v4.train import pending_rounds

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
    runner.config = {"pilot": {"min_paired_prompts": 2}, "training": {"optimizer_steps_per_round": 8}}
    arm = {"mean_t2i_loss": 1, "mean_gradient_norm_before_clip": .2, "parameter_delta_l2": .1}
    arm[field] = value
    target = tmp_path / "rounds/round-000/DONE.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"paired": 3, "arms": [
        dict(arm, arm=name, round=0, optimizer_steps=8) for name in MODULE.ARMS]}))
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


@pytest.mark.parametrize("names", [[], ["naive"], ["naive", "naive"]])
def test_done_does_not_hide_missing_independent_training_arm(tmp_path, names):
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.config = {"pilot": {"min_paired_prompts": 2}, "training": {"optimizer_steps_per_round": 8}}
    target = tmp_path / "rounds/round-000/DONE.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"paired": 3, "arms": [{"arm": name} for name in names]}))
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        runner.validate_round(0)


def _resume_after_unrecorded_train(tmp_path, monkeypatch, through_round):
    """Run the real main/execute/run chain with CPU-only stage subprocesses.

    Training round 1 finished, but its supervisor sentinel was not persisted.
    The fake subprocess models both the old additional-round limit and the new
    exact-round target so the regression checks execution order, not just argv.
    """
    runner = object.__new__(MODULE.Pilot)
    runner.out = tmp_path
    runner.config_path = tmp_path / "config.yaml"
    runner.config_path.write_text("fixture: true\n", encoding="utf-8")
    runner.config = {
        "training": {"rounds": 4, "optimizer_steps_per_round": 8},
        "pilot": {"min_paired_prompts": 2, "min_free_gib": 0},
        "gradient_probe": {"source": "fixture", "size": 2, "resamples": 2},
    }
    runner.limit = through_round
    runner.started = time.time()
    runner.deadline = runner.started + 60
    runner.env = {}
    runner.common = ["--outdir", str(tmp_path)]
    runner.frozen = {
        "config_sha256": MODULE.digest(runner.config_path), "source_sha256": {},
    }
    runner.done_stages = tmp_path / "stage-completion"
    runner.done_stages.mkdir()
    (tmp_path / "split.json").write_text("{}", encoding="utf-8")

    def finish_round(index):
        MODULE.write_json(tmp_path / "rounds" / f"round-{index:03d}" / "DONE.json", {
            "paired": 3,
            "arms": [{"arm": arm, "round": index, "optimizer_steps": 8,
                      "mean_t2i_loss": 1, "mean_gradient_norm_before_clip": .2,
                      "parameter_delta_l2": .1} for arm in MODULE.ARMS],
        })

    for index in (0, 1):
        finish_round(index)
    completed = ["freeze-probe", "round-000.train", "base.step-00000.gradient"]
    for step in (0, 8):
        for arm in MODULE.ARMS:
            prefix = f"{arm}.step-{step:05d}"
            completed.extend(f"{prefix}.{suffix}" for suffix in (
                "generate", "detect.qwen3vl", "detect.internvl", "crop", "verify"))
            if step:
                completed.append(f"{prefix}.gradient")
        completed.extend(f"step-{step:05d}.{suffix}" for suffix in (
            "score", "report", "gradient-sensitivity", "plot"))
    for stage in completed:
        MODULE.write_json(runner.done_stages / f"{stage}.json", {"stage": stage})
    assert not (runner.done_stages / "round-001.train.json").exists()

    events = []
    commands = []

    class StageProcess:
        pid = 12345

        def __init__(self, command, **kwargs):
            commands.append(command)
            state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
            events.append(("stage", state["stage"]))
            if Path(command[2]).name != "v4_train.py" or command[3] != "train":
                return
            done = {index for index in range(4)
                    if (tmp_path / "rounds" / f"round-{index:03d}" / "DONE.json").exists()}
            if "--round-index" in command:
                target = int(command[command.index("--round-index") + 1])
                pending = pending_rounds(4, sorted(done), None, round_index=target)
            else:
                count = int(command[command.index("--max-rounds") + 1])
                pending = pending_rounds(4, sorted(done), count)
            for index in pending:
                events.append(("train", index))
                finish_round(index)

        def wait(self, timeout):
            return 0

    def build_runner(args):
        runner.limit = args.through_round
        return runner

    monkeypatch.setattr(MODULE.subprocess, "Popen", StageProcess)
    monkeypatch.setattr(MODULE, "Pilot", build_runner)
    monkeypatch.setattr(MODULE.sys, "argv", ["run_decoupling_pilot.py", "--through-round", str(through_round)])
    MODULE.main()
    return runner, events, commands


def test_resume_measures_completed_round_before_training_next(tmp_path, monkeypatch):
    runner, events, commands = _resume_after_unrecorded_train(tmp_path, monkeypatch, 3)
    next_train = events.index(("train", 2))
    for arm in MODULE.ARMS:
        for suffix in ("generate", "detect.qwen3vl", "detect.internvl", "crop", "verify", "gradient"):
            assert events.index(("stage", f"{arm}.step-00016.{suffix}")) < next_train
    assert events.index(("stage", "step-00016.report")) < next_train
    assert [event for event in events if event[0] == "train"] == [("train", 2)]
    assert not (runner.out / "rounds/round-003/DONE.json").exists()
    trains = [command for command in commands if Path(command[2]).name == "v4_train.py"
              and command[3] == "train"]
    assert [command[command.index("--round-index") + 1] for command in trains] == ["1", "2"]


def test_repeated_completed_round_target_does_not_advance_pending_round(tmp_path, monkeypatch):
    runner, events, commands = _resume_after_unrecorded_train(tmp_path, monkeypatch, 2)
    assert [event for event in events if event[0] == "train"] == []
    assert not (runner.out / "rounds/round-002/DONE.json").exists()
    assert (runner.done_stages / "round-001.train.json").exists()
    assert ("stage", "naive.step-00016.gradient") in events
    assert ("stage", "rfo_gold.step-00016.gradient") in events
    trains = [command for command in commands if Path(command[2]).name == "v4_train.py"
              and command[3] == "train"]
    assert len(trains) == 1
    assert trains[0][trains[0].index("--round-index") + 1] == "1"
