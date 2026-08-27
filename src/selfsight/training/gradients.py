"""LoRA-only gradient vectors, pairwise GDA, and noise controls."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GradientSnapshot:
    criterion: str
    vector: Any
    per_block: dict[str, Any]
    parameter_names: tuple[str, ...]
    loss: float
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class GradientComparison:
    left: str
    right: str
    cosine: float
    norm_ratio: float
    per_block_cosine: dict[str, float]


def _torch():
    import torch

    return torch


def _block_name(parameter_name: str) -> str:
    marker = parameter_name.lower().find(".lora_")
    prefix = parameter_name[:marker] if marker >= 0 else parameter_name
    parts = prefix.split(".")
    for index, part in enumerate(parts):
        if part in {"layers", "layer", "blocks", "block"} and index + 1 < len(parts):
            return ".".join(parts[: index + 2])
    return ".".join(parts[:-1]) or prefix


def collect_lora_gradient(
    model: Any,
    loss_closure: Callable[[], Any],
    *,
    criterion: str,
    sample_ids: Iterable[str] = (),
) -> GradientSnapshot:
    torch = _torch()
    model.zero_grad(set_to_none=True)
    loss = loss_closure()
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"Non-finite loss for {criterion}: {loss}")
    loss.backward()
    names = []
    vectors = []
    blocks: dict[str, list[Any]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name.lower():
            if parameter.grad is None:
                raise RuntimeError(f"Trainable LoRA parameter has no gradient: {name}")
            gradient = parameter.grad.detach().float().cpu().reshape(-1).clone()
            names.append(name)
            vectors.append(gradient)
            blocks.setdefault(_block_name(name), []).append(gradient)
    if not vectors:
        raise RuntimeError("No LoRA gradients were collected")
    return GradientSnapshot(
        criterion=criterion,
        vector=torch.cat(vectors),
        per_block={key: torch.cat(items) for key, items in blocks.items()},
        parameter_names=tuple(names),
        loss=float(loss.detach().cpu()),
        sample_ids=tuple(sample_ids),
    )


def collect_lora_gradient_accumulated(
    model: Any,
    loss_closures: Iterable[Callable[[], Any]],
    *,
    criterion: str,
    sample_ids: Iterable[str] = (),
) -> GradientSnapshot:
    """Collect an equal-weight mean gradient without retaining all microbatch graphs."""

    torch = _torch()
    closures = list(loss_closures)
    if not closures:
        raise ValueError("At least one gradient microbatch is required")
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    for closure in closures:
        loss = closure()
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite loss for {criterion}: {loss}")
        total_loss += float(loss.detach().cpu())
        (loss / len(closures)).backward()
    names = []
    vectors = []
    blocks: dict[str, list[Any]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name.lower():
            if parameter.grad is None:
                raise RuntimeError(f"Trainable LoRA parameter has no gradient: {name}")
            gradient = parameter.grad.detach().float().cpu().reshape(-1).clone()
            names.append(name)
            vectors.append(gradient)
            blocks.setdefault(_block_name(name), []).append(gradient)
    if not vectors:
        raise RuntimeError("No LoRA gradients were collected")
    return GradientSnapshot(
        criterion=criterion,
        vector=torch.cat(vectors),
        per_block={key: torch.cat(items) for key, items in blocks.items()},
        parameter_names=tuple(names),
        loss=total_loss / len(closures),
        sample_ids=tuple(sample_ids),
    )


def cosine(left: Any, right: Any, epsilon: float = 1e-12) -> float:
    torch = _torch()
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.numel() != right.numel():
        raise ValueError(f"Gradient dimensions differ: {left.numel()} vs {right.numel()}")
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= epsilon:
        return float("nan")
    return float(torch.clamp(torch.dot(left, right) / denominator, -1.0, 1.0))


def compare_gradients(left: GradientSnapshot, right: GradientSnapshot) -> GradientComparison:
    torch = _torch()
    left_norm = float(torch.linalg.vector_norm(left.vector.float()))
    right_norm = float(torch.linalg.vector_norm(right.vector.float()))
    shared_blocks = sorted(set(left.per_block).intersection(right.per_block))
    return GradientComparison(
        left=left.criterion,
        right=right.criterion,
        cosine=cosine(left.vector, right.vector),
        norm_ratio=left_norm / right_norm if right_norm else float("nan"),
        per_block_cosine={
            block: cosine(left.per_block[block], right.per_block[block]) for block in shared_blocks
        },
    )


def noise_interval(comparisons: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    values = np.asarray([value for value in comparisons if np.isfinite(value)], dtype=float)
    if values.size < 2:
        raise ValueError("At least two finite split-half comparisons are required")
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1.0 - alpha)),
        "std": float(values.std(ddof=1)),
    }


def exponential_moving_average(values: Iterable[float], alpha: float = 0.35) -> list[float]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1]")
    output: list[float] = []
    state: float | None = None
    for value in values:
        state = value if state is None else alpha * value + (1.0 - alpha) * state
        output.append(state)
    return output
