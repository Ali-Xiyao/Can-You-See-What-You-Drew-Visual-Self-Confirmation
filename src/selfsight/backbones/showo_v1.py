"""Frozen Show-o v1 negative-control wrapper with the v2.2 backbone surface."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from selfsight.backbones.base import (
    BackboneCapabilities,
    BackboneIdentity,
    LoraTargetAudit,
    ResourceReport,
)
from selfsight.showo_adapter import ShowoAdapter, ShowoReplayBatch, ShowoSFTBatch
from selfsight.training.checkpoint import load_checkpoint, save_checkpoint
from selfsight.training.lora import DEFAULT_TARGET_MODULES


class ShowoV1Adapter(ShowoAdapter):
    """Compatibility name; behavior is identical to the proven v1 adapter."""

    @property
    def identity(self) -> BackboneIdentity:
        return BackboneIdentity(
            model_id=self.model_id,
            revision=self.revision,
            source_repository="showlab/Show-o",
            source_revision="45a5a2de01d1ebd10cd5864d29310a76476cdf23",
            implementation="showo_v1_negative_control",
            native_resolution=512,
        )

    @property
    def capabilities(self) -> BackboneCapabilities:
        return BackboneCapabilities(True, True, True, True, True)

    def discover_lora_targets(self) -> LoraTargetAudit:
        targets = set(DEFAULT_TARGET_MODULES)
        linear = tuple(
            name
            for name, module in self.model.showo.named_modules()
            if module.__class__.__name__ == "Linear"
        )
        suffixes = Counter(name.rsplit(".", 1)[-1] for name in linear)
        candidates = tuple(name for name in linear if name.rsplit(".", 1)[-1] in targets)
        return LoraTargetAudit(
            model_id=self.model_id,
            revision=self.revision,
            linear_modules=linear,
            suffix_counts=dict(sorted(suffixes.items())),
            shared_transformer_candidates=candidates,
            generation_head_candidates=(),
            frozen_or_forbidden=("embed_tokens", "lm_head", "mm_projector", "vision_tower", "vq_model"),
        )

    def generation_loss(self, batch: ShowoSFTBatch) -> Any:
        return self.sft_loss(batch)

    def understanding_replay_loss(self, batch: ShowoReplayBatch) -> Any:
        return self.mmu_replay_loss(batch)

    def collect_gradient(self, batch: ShowoSFTBatch, criterion: str):
        return self.compute_lora_gradient(batch, criterion)

    def save_adapter(self, destination: str | Path, **state: Any) -> Path:
        required = {
            "optimizer",
            "scheduler",
            "config_digest",
            "config_values",
            "step",
            "round_index",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Missing adapter checkpoint state: {missing}")
        return save_checkpoint(
            destination,
            model=self.model,
            optimizer=state["optimizer"],
            scheduler=state["scheduler"],
            config_digest=str(state["config_digest"]),
            config_values=dict(state["config_values"]),
            step=int(state["step"]),
            round_index=int(state["round_index"]),
            metadata=dict(state.get("metadata", {})),
        )

    def load_adapter(self, source: str | Path, **state: Any):
        required = {"optimizer", "scheduler", "expected_config_digest"}
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Missing adapter restore state: {missing}")
        return load_checkpoint(
            source,
            model=self.model,
            optimizer=state["optimizer"],
            scheduler=state["scheduler"],
            expected_config_digest=str(state["expected_config_digest"]),
        )

    def resource_report(self) -> ResourceReport:
        import torch

        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        allocated = reserved = None
        if self.device.type == "cuda":
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
        return ResourceReport(
            device=str(self.device),
            dtype=str(self.dtype).removeprefix("torch."),
            loaded=True,
            total_parameters=total,
            trainable_parameters=trainable,
            allocated_gpu_bytes=allocated,
            reserved_gpu_bytes=reserved,
        )


__all__ = ["ShowoAdapter", "ShowoReplayBatch", "ShowoSFTBatch", "ShowoV1Adapter"]
