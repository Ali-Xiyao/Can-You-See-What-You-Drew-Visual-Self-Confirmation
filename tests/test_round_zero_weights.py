"""Round 0's second arm starts where the first one did, not where it finished.

The paired design runs both arms through one resident backbone and swaps their
weights in and out around each stage. The swap was written as

    previous = out / "checkpoints" / arm / f"round-{round_index - 1:03d}"
    if previous.exists():
        load_checkpoint(previous, ...)

whose else branch is silence. In round 0 no previous checkpoint exists for
either arm, so nothing is restored, and `naive` -- first in ARMS -- trains and
mutates the shared parameters before `rfo_gold` starts. RFO-Gold's round 0
therefore began from Naive's updated weights, and every later round inherited
that through rfo_gold's own checkpoint. The arms were never two arms.

Nothing in the suite caught it because nothing exercised the cross-arm weight
lifecycle: the training tests are unit tests of schedules, pairing, selection
and manifests, none of which hold parameters.

These tests are CPU-only on purpose, and pin `torch.cuda.is_available` to False
so they stay that way: `_rng_state` calls `torch.cuda.get_rng_state_all()`,
which builds a CUDA context on every visible card just to read a seed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from selfsight.training.checkpoint import (
    capture_base_state,
    lora_state_dict,
    restore_arm_state,
    restore_base_state,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "0" * 64


class TinyLora(torch.nn.Module):
    """Same shape the gradient tests use: two adapter tensors and nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.lora_A = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.block.lora_B = torch.nn.Parameter(torch.tensor([0.5, 0.25]))


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _train(model: TinyLora, amount: float) -> None:
    """Stand-in for train_arm: whatever it does, it moves the adapter."""
    with torch.no_grad():
        model.block.lora_A.add_(amount)
        model.block.lora_B.add_(amount)


def _adapter(model: TinyLora) -> dict[str, torch.Tensor]:
    """Cloned, because lora_state_dict hands back the live parameters on CPU.

    Without the clone every comparison here is a tensor against itself and the
    whole file passes vacuously -- which is how the first version of it passed
    against the unfixed code.
    """
    return {name: tensor.clone() for name, tensor in lora_state_dict(model).items()}


def _same(left, right) -> bool:
    return set(left) == set(right) and all(torch.equal(left[k], right[k]) for k in left)


def _checkpoint(directory: Path, model: TinyLora) -> Path:
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _s: 1.0)
    return save_checkpoint(directory, model=model, optimizer=optimizer, scheduler=scheduler,
                           config_digest=DIGEST, config_values={}, step=1, round_index=0)


# ------------------------------------------------------------------- the defect


def test_the_second_arm_of_round_zero_does_not_inherit_the_first_arms_update(tmp_path):
    """The whole point. Drive round 0 the way stage_train drives it."""
    model = TinyLora()
    base = capture_base_state(model)
    expected = _adapter(model)

    # ARMS = ("naive", "rfo_gold"), and round 0 has no checkpoint for either.
    for arm, amount in (("naive", 10.0), ("rfo_gold", 100.0)):
        source = restore_arm_state(tmp_path / "checkpoints" / arm / "round--01",
                                   model=model, optimizer=None, scheduler=None,
                                   expected_config_digest=DIGEST, base=base)
        assert source == "base"
        if arm == "rfo_gold":
            assert _same(_adapter(model), expected), (
                "rfo_gold started round 0 from naive's updated weights")
        _train(model, amount)

    # And the base itself survived being handed out twice.
    assert _same(base["adapter"], expected)


def test_the_old_shape_of_this_code_is_gone(tmp_path):
    """A regression guard on the pattern, not just on one call site.

    `if previous.exists(): load_checkpoint(...)` is the defect written out. It
    is easy to reintroduce by hand at a fourth site, and it fails silently.
    """
    source = (ROOT / "scripts" / "v4_train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    stage = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "stage_train")
    called = {node.func.id for node in ast.walk(stage)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "load_checkpoint" not in called, (
        "stage_train must swap arms through restore_arm_state, which has an else branch")
    assert sum(1 for node in ast.walk(stage)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "restore_arm_state") == 3, (
        "the three arm swaps are candidate generation, selection, and training")


def test_the_base_is_captured_before_the_round_loop_and_not_inside_it():
    """Captured after attach_lora and before any training -- the only valid moment."""
    tree = ast.parse((ROOT / "scripts" / "v4_train.py").read_text(encoding="utf-8"))
    stage = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "stage_train")
    captures = [node for node in ast.walk(stage)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "capture_base_state"]
    assert len(captures) == 1
    loops = [node for node in stage.body if isinstance(node, ast.For)]
    assert loops, "stage_train has a round loop"
    assert captures[0].lineno < loops[-1].lineno, (
        "capturing inside the loop would capture an already-trained arm")


# ------------------------------------------------------- restore_arm_state itself


def test_a_missing_checkpoint_means_base_rather_than_no_op(tmp_path):
    model = TinyLora()
    base = capture_base_state(model)
    _train(model, 7.0)
    assert restore_arm_state(tmp_path / "nope", model=model, optimizer=None, scheduler=None,
                             expected_config_digest=DIGEST, base=base) == "base"
    assert _same(_adapter(model), base["adapter"])


def test_a_later_round_still_loads_that_arms_own_checkpoint(tmp_path):
    """The rounds >= 1 path must be untouched: base is the fallback, not the rule."""
    model = TinyLora()
    base = capture_base_state(model)
    _train(model, 3.0)
    trained = _adapter(model)
    directory = _checkpoint(tmp_path / "round-000", model)

    _train(model, 50.0)  # the other arm runs and moves the shared parameters
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _s: 1.0)
    source = restore_arm_state(directory, model=model, optimizer=optimizer,
                               scheduler=scheduler, expected_config_digest=DIGEST, base=base)

    assert source == "checkpoint"
    assert _same(_adapter(model), trained)
    assert not _same(_adapter(model), base["adapter"])


def test_a_corrupt_checkpoint_still_raises_rather_than_falling_back_to_base(tmp_path):
    """Falling back would turn a damaged resume into a silently restarted arm."""
    model = TinyLora()
    base = capture_base_state(model)
    directory = _checkpoint(tmp_path / "round-000", model)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["config_digest"] = "f" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError):
        restore_arm_state(directory, model=model, optimizer=None, scheduler=None,
                          expected_config_digest=DIGEST, base=base)


# ----------------------------------------------------------------------- the RNG


def test_the_base_carries_the_rng_so_both_arms_draw_the_same_randomness(tmp_path):
    """From round 1 on each arm restores its own RNG; round 0 must match that.

    Otherwise the two arms see different LoRA dropout masks in the one round
    where they are supposed to be identical.
    """
    model = TinyLora()
    torch.manual_seed(20260901)
    base = capture_base_state(model)

    first = torch.randn(4)
    restore_base_state(model, base)
    second = torch.randn(4)

    assert torch.equal(first, second)


def test_restoring_the_base_does_not_touch_cuda(monkeypatch):
    """A CPU-only caller must not have a CUDA context built under it."""
    def explode(*_a, **_k):
        raise AssertionError("reading the RNG must not initialise CUDA")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", explode)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", explode)

    model = TinyLora()
    base = capture_base_state(model)
    _train(model, 1.0)
    restore_base_state(model, base)
    assert _same(_adapter(model), base["adapter"])
