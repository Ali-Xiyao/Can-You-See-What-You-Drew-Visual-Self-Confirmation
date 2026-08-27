"""PEFT LoRA integration for the Show-o transformer only."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2")
FORBIDDEN_TOKENS = ("embed_tokens", "lm_head", "mm_projector", "vision_tower", "vq_model")


@dataclass(frozen=True)
class LoraSummary:
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float
    trainable_names: tuple[str, ...]
    matched_target_suffixes: tuple[str, ...]


def _transformer(model: Any) -> Any:
    return model.showo if hasattr(model, "showo") else model


def attach_showo_lora(
    model: Any,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
    gradient_checkpointing: bool = True,
) -> tuple[Any, LoraSummary]:
    from peft import LoraConfig, TaskType, get_peft_model

    targets = tuple(target_modules)
    transformer = _transformer(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    available_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, _ in transformer.named_modules()
        if name.rsplit(".", 1)[-1] in targets
    }
    missing = sorted(set(targets).difference(available_suffixes))
    if missing:
        raise ValueError(f"Show-o transformer is missing requested LoRA targets: {missing}")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(targets),
    )
    wrapped = get_peft_model(transformer, lora_config)
    if hasattr(model, "showo"):
        model.showo = wrapped
        output_model = model
    else:
        output_model = wrapped
    if gradient_checkpointing:
        target = wrapped
        if hasattr(target, "gradient_checkpointing_enable"):
            target.gradient_checkpointing_enable()
        if hasattr(target, "enable_input_require_grads"):
            target.enable_input_require_grads()
    summary = summarize_trainables(output_model, available_suffixes)
    assert_only_lora_trainable(output_model)
    return output_model, summary


def assert_only_lora_trainable(model: Any) -> None:
    invalid = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        if "lora_" not in lower or any(token in lower for token in FORBIDDEN_TOKENS):
            invalid.append(name)
    if invalid:
        raise ValueError(f"Non-LoRA or forbidden parameters are trainable: {invalid[:20]}")


def summarize_trainables(model: Any, matched: Iterable[str] = ()) -> LoraSummary:
    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return LoraSummary(
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_fraction=trainable / total if total else 0.0,
        trainable_names=trainable_names,
        matched_target_suffixes=tuple(sorted(matched)),
    )
