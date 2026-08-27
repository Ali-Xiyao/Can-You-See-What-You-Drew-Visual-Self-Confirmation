from __future__ import annotations

import torch

from selfsight.training.checkpoint import load_checkpoint, save_checkpoint
from selfsight.training.gradients import (
    collect_lora_gradient,
    collect_lora_gradient_accumulated,
    compare_gradients,
)


class TinyLora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.lora_A = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.block.lora_B = torch.nn.Parameter(torch.tensor([0.5, 0.25]))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)


def _loss(model: TinyLora):
    return (model.block.lora_A.square().sum() + 2.0 * model.block.lora_B.square().sum())


def test_identical_selection_gradient_cosine_is_one():
    model = TinyLora()
    left = collect_lora_gradient(model, lambda: _loss(model), criterion="left", sample_ids=("a",))
    right = collect_lora_gradient(model, lambda: _loss(model), criterion="right", sample_ids=("a",))
    comparison = compare_gradients(left, right)
    assert comparison.cosine > 0.999999
    assert comparison.norm_ratio == 1.0
    assert all(value > 0.999999 for value in comparison.per_block_cosine.values())


def test_accumulated_microbatch_gradient_matches_joint_mean():
    model = TinyLora()
    joint = collect_lora_gradient(
        model,
        lambda: (_loss(model) + 3.0 * _loss(model)) / 2.0,
        criterion="joint",
    )
    accumulated = collect_lora_gradient_accumulated(
        model,
        [lambda: _loss(model), lambda: 3.0 * _loss(model)],
        criterion="accumulated",
    )
    assert compare_gradients(joint, accumulated).cosine > 0.999999
    assert torch.allclose(joint.vector, accumulated.vector)


def test_adapter_checkpoint_resume_restores_parameters_optimizer_and_rng(tmp_path):
    model = TinyLora()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    optimizer.zero_grad()
    _loss(model).backward()
    optimizer.step()
    scheduler.step()
    expected = model.block.lora_A.detach().clone()
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config_digest="a" * 64,
        config_values={"test": True},
        step=7,
        round_index=2,
    )
    with torch.no_grad():
        model.block.lora_A.add_(100)
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_digest="a" * 64,
    )
    assert state == {"step": 7, "round_index": 2}
    assert torch.equal(model.block.lora_A, expected)
