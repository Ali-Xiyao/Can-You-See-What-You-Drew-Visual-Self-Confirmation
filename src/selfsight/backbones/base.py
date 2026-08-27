"""Family-neutral contract for trainable unified multimodal backbones."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from selfsight.adapters import GradientResult
from selfsight.schemas import AtomicQuestion, CandidateRecord, ObservationResult


@dataclass(frozen=True)
class BackboneIdentity:
    model_id: str
    revision: str
    source_repository: str
    source_revision: str
    implementation: str
    native_resolution: int


@dataclass(frozen=True)
class BackboneCapabilities:
    text_to_image: bool
    image_to_atomic_qa: bool
    frozen_step0_copy: bool
    generation_lora: bool
    adapter_resume: bool

    @property
    def unified_functionality(self) -> bool:
        return all(
            (
                self.text_to_image,
                self.image_to_atomic_qa,
                self.frozen_step0_copy,
                self.generation_lora,
                self.adapter_resume,
            )
        )


@dataclass(frozen=True)
class LoraTargetAudit:
    model_id: str
    revision: str
    linear_modules: tuple[str, ...]
    suffix_counts: Mapping[str, int]
    shared_transformer_candidates: tuple[str, ...]
    generation_head_candidates: tuple[str, ...]
    frozen_or_forbidden: tuple[str, ...]


@dataclass(frozen=True)
class ResourceReport:
    device: str
    dtype: str
    loaded: bool
    total_parameters: int | None
    trainable_parameters: int | None
    allocated_gpu_bytes: int | None
    reserved_gpu_bytes: int | None


@runtime_checkable
class UnifiedBackbone(Protocol):
    """Complete surface required by Gate -2 and later phenomenon experiments."""

    model_id: str
    revision: str

    @property
    def identity(self) -> BackboneIdentity: ...

    @property
    def capabilities(self) -> BackboneCapabilities: ...

    def generate_images(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        output_dir: str | Path,
        checkpoint_id: str,
    ) -> list[CandidateRecord]: ...

    def observe_atoms(
        self,
        image_path: str | Path,
        questions: Sequence[AtomicQuestion],
    ) -> ObservationResult: ...

    def encode_image_targets(self, images: Sequence[Image.Image | str | Path]) -> Any: ...

    def discover_lora_targets(self) -> LoraTargetAudit: ...

    def attach_lora(self, **kwargs: Any) -> Any: ...

    def generation_loss(self, batch: Any) -> Any: ...

    def understanding_replay_loss(self, batch: Any) -> Any: ...

    def collect_gradient(self, batch: Any, criterion: str) -> GradientResult: ...

    def save_adapter(self, destination: str | Path, **state: Any) -> Path: ...

    def load_adapter(self, source: str | Path) -> Mapping[str, Any]: ...

    def resource_report(self) -> ResourceReport: ...

