"""Tests for the shared checkpoint reload used by both re-measurement passes.

Everything here is structural. Actually loading a checkpoint needs a GPU and a
17 GB model, and the property that matters is not "it loads" -- it is that the
model it loads is the one the run had. That property is decided by two lines
of seeding whose omission produces a working pass over a model the run never
trained.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from selfsight.v4 import checkpoint_reload
from selfsight.v4.checkpoint_reload import LORA_TARGETS, checkpoint_for


def _function_ast(name: str) -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(checkpoint_reload))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    found = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if isinstance(function, ast.Name) and function.id == name:
            found.append(child)
        if isinstance(function, ast.Attribute) and function.attr == name:
            found.append(child)
    return found


def test_the_seed_is_set_twice():
    """Once before the adapter exists, once before LoRA A is drawn.

    `scripts/v4_train.py` and `src/selfsight/v4/checkpoint_probe.py` both do
    this, and the second call is the one that is easy to read as redundant. It
    is not: `attach_lora` randomises LoRA A from whatever state the generator
    is in, and constructing the adapter consumes from it.
    """

    assert len(_calls(_function_ast("load_trained_backbone"), "seed_training")) == 2


def test_the_second_seed_is_the_statement_before_attach_lora():
    """Adjacent, not merely both present.

    A seed set before the adapter is constructed and never re-set gives a
    different LoRA A. So does a seed set two statements early with anything
    that touches the generator in between. The check is on adjacency because
    that is what the property is.
    """

    body = _function_ast("load_trained_backbone").body
    attach = [index for index, statement in enumerate(body)
              if _calls(statement, "attach_lora")]
    assert len(attach) == 1, "expected exactly one attach_lora call"
    previous = body[attach[0] - 1]
    assert _calls(previous, "seed_training"), ast.dump(previous)


def test_the_checkpoint_digest_is_checked_against_the_config():
    # load_checkpoint's expected_config_digest is what stops a pass being run
    # with another run's config and producing plausible numbers.
    call = _calls(_function_ast("load_trained_backbone"), "load_checkpoint")
    assert len(call) == 1
    keywords = {keyword.arg for keyword in call[0].keywords}
    assert "expected_config_digest" in keywords


def test_gradient_checkpointing_is_off_for_a_read_only_pass():
    call = _calls(_function_ast("load_trained_backbone"), "attach_lora")
    keywords = {keyword.arg: keyword.value for keyword in call[0].keywords}
    assert isinstance(keywords["gradient_checkpointing"], ast.Constant)
    assert keywords["gradient_checkpointing"].value is False


def test_the_lora_targets_path_is_absolute():
    # A relative "runs/readiness/..." would resolve against the caller's
    # working directory, and a pass launched from elsewhere would silently
    # build a differently shaped adapter -- or fail after loading the model.
    assert LORA_TARGETS.is_absolute()
    assert LORA_TARGETS.exists(), f"{LORA_TARGETS} is missing"


def test_step_zero_is_the_untrained_base_for_both_arms():
    run = Path("run")
    for arm in ("naive", "rfo_gold", "blind_self"):
        assert checkpoint_for(run, arm, 0, 8) == run / "checkpoints/base/round--01"


def test_the_first_trained_checkpoint_is_round_zero():
    run = Path("run")
    assert checkpoint_for(run, "naive", 8, 8) == run / "checkpoints/naive/round-000"
    assert checkpoint_for(run, "blind_self", 88, 8) == run / "checkpoints/blind_self/round-010"


def test_a_step_between_checkpoints_is_refused():
    with pytest.raises(ValueError, match="multiple of 8"):
        checkpoint_for(Path("run"), "naive", 12, 8)
