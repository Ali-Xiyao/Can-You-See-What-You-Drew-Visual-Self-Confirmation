"""ModelAdapter contract used by generation, observation, targets, and GDA probes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from selfsight.schemas import AtomicQuestion, CandidateRecord, ObservationResult


@dataclass(frozen=True)
class GradientResult:
    criterion: str
    vector: Any
    per_block: dict[str, Any]
    loss: float
    sample_ids: tuple[str, ...]


class ModelAdapter(ABC):
    """Minimal model-facing surface; implementations may live in isolated envs."""

    model_id: str
    revision: str

    @abstractmethod
    def generate_images(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int],
        output_dir: str | Path,
        checkpoint_id: str,
    ) -> list[CandidateRecord]:
        raise NotImplementedError

    @abstractmethod
    def observe_atoms(
        self,
        image_path: str | Path,
        questions: Sequence[AtomicQuestion],
    ) -> ObservationResult:
        raise NotImplementedError

    @abstractmethod
    def encode_image_targets(self, images: Sequence[Image.Image | str | Path]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compute_lora_gradient(self, batch: Any, criterion: str) -> GradientResult:
        raise NotImplementedError
