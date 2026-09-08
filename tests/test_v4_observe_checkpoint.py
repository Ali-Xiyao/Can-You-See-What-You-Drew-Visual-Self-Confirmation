"""`observe --checkpoint`: the arm's own adapter answers, not the base model.

Every context result before 2026-09-08 was measured with the untrained backbone
answering about a fixed corpus. That is the right instrument for a cross-section
and the wrong one for a curve: the co-evolution claim is that training changes
what the model can see in its own output, so the answering model has to be the
one that did the training.

Two properties carry the whole change and both are tested here: the curve's
answers must not land in the file the cross-sectional result lives in, and the
adapter must be initialised before the saved state is poured into it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _pipeline():
    path = Path(__file__).resolve().parents[1] / "scripts/v4_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("v4_run_pipeline_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = {"seed": 20260908,
          "training": {"lora": {"rank": 16, "alpha": 32, "dropout": 0.05}}}


class Recorder:
    """A backbone that remembers the order it was operated on."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.model = object()

    def attach_lora(self, **kwargs):
        self.calls.append("attach")
        self.kwargs = kwargs


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """The real `wear_checkpoint`, with the three things it reaches for stubbed."""

    pipeline = _pipeline()
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"target_modules": ["q_proj", "v_proj"]}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "LORA_TARGETS", str(targets))

    backbone = Recorder()
    import selfsight.training.checkpoint as checkpoint_module
    import selfsight.utils.hashing as hashing_module
    import selfsight.v4.train as train_module

    monkeypatch.setattr(train_module, "seed_training",
                        lambda seed: backbone.calls.append(f"seed:{seed}"))
    monkeypatch.setattr(checkpoint_module, "load_checkpoint",
                        lambda *a, **k: backbone.calls.append("load"))
    monkeypatch.setattr(hashing_module, "sha256_json", lambda payload: "digest")
    return pipeline, backbone


# ------------------------------------------------------- where answers land


def test_a_curve_point_never_overwrites_the_cross_sectional_answers(tmp_path):
    """answers.jsonl is read by name by every analysis written before today."""

    pipeline = _pipeline()
    base = pipeline.answer_file(tmp_path, prompted=False, checkpoint=None)
    worn = pipeline.answer_file(tmp_path, prompted=False, checkpoint="ckpt/round-003")
    assert base.name == "answers.jsonl"
    assert worn != base


def test_the_two_conditions_still_have_two_files_when_a_checkpoint_answers():
    """Blind and told must not collide either, or the contrast disappears."""

    pipeline = _pipeline()
    blind = pipeline.answer_file(Path("run"), prompted=False, checkpoint="c")
    told = pipeline.answer_file(Path("run"), prompted=True, checkpoint="c")
    assert blind != told
    assert "prompted" in told.name and "prompted" not in blind.name


# ------------------------------------------------------------- the loading


def test_the_adapter_is_initialised_before_the_saved_state_is_poured_in(wired, tmp_path):
    """Load before attach would name modules that do not exist yet."""

    pipeline, backbone = wired
    checkpoint = tmp_path / "round-003"
    checkpoint.mkdir()
    pipeline.wear_checkpoint(backbone, checkpoint=checkpoint, config=CONFIG)
    assert backbone.calls == [f"seed:{CONFIG['seed']}", "attach", "load"]


def test_the_adapter_is_shaped_by_the_config_it_was_trained_under(wired, tmp_path):
    """A rank mismatch loads a state dict into the wrong tensors, silently."""

    pipeline, backbone = wired
    checkpoint = tmp_path / "round-003"
    checkpoint.mkdir()
    pipeline.wear_checkpoint(backbone, checkpoint=checkpoint, config=CONFIG)
    assert backbone.kwargs["rank"] == 16
    assert backbone.kwargs["alpha"] == 32
    assert backbone.kwargs["target_modules"] == ["q_proj", "v_proj"]
    assert backbone.kwargs["gradient_checkpointing"] is False


def test_a_checkpoint_that_is_not_there_is_refused_rather_than_ignored(wired, tmp_path):
    """Silently answering as the base model is how a flat curve gets published."""

    pipeline, backbone = wired
    with pytest.raises(SystemExit, match="No checkpoint at"):
        pipeline.wear_checkpoint(backbone, checkpoint=tmp_path / "absent", config=CONFIG)
    assert backbone.calls == [], "nothing should be touched on the way out"


# --------------------------------------------------------------- the command


def test_asking_for_a_checkpoint_without_a_config_is_refused(tmp_path):
    """The adapter shape and the digest both come from the config, so guessing
    one would mean loading a state dict into an adapter of the wrong shape."""

    pipeline = _pipeline()
    run = tmp_path / "run"
    run.mkdir()
    (run / "verified.jsonl").write_text("", encoding="utf-8")
    (run / "manifest.jsonl").write_text("", encoding="utf-8")
    args = SimpleNamespace(run=str(run), condition="image_only", device="cpu",
                           overwrite=False, checkpoint="somewhere", config=None)
    with pytest.raises(SystemExit, match="needs --config"):
        pipeline.stage_observe(args)


def test_observe_without_a_checkpoint_is_the_instrument_it_always_was(tmp_path):
    """Every command in the docstrings omits --checkpoint and must keep working."""

    pipeline = _pipeline()
    run = tmp_path / "run"
    run.mkdir()
    (run / "verified.jsonl").write_text("", encoding="utf-8")
    (run / "manifest.jsonl").write_text("", encoding="utf-8")
    args = SimpleNamespace(run=str(run), condition="image_only", device="cpu",
                           overwrite=False)
    pipeline.stage_observe(args)  # "nothing to do", and no attribute error


def test_the_checkpoint_can_be_asked_for_on_the_command_line(monkeypatch, tmp_path):
    pipeline = _pipeline()
    seen: dict = {}
    monkeypatch.setattr(pipeline, "stage_observe", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["v4_run_pipeline.py", "observe", "--run", str(tmp_path),
                                      "--checkpoint", "ck/round-003",
                                      "--config", "configs/x.yaml",
                                      "--condition", "prompted"])
    pipeline.main()
    assert seen["checkpoint"] == "ck/round-003"
    assert seen["config"] == "configs/x.yaml"
    assert seen["condition"] == "prompted"


def test_argparse_still_defaults_the_new_flags_to_the_old_behaviour(monkeypatch, tmp_path):
    pipeline = _pipeline()
    seen: dict = {}
    monkeypatch.setattr(pipeline, "stage_observe", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["v4_run_pipeline.py", "observe", "--run", str(tmp_path)])
    pipeline.main()
    assert seen["checkpoint"] is None
    assert seen["config"] is None
    assert seen["condition"] == "image_only"
