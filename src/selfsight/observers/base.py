"""Observer backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from selfsight.schemas import AtomicQuestion


class BaseObserver(ABC):
    observer_id: str
    revision: str

    @abstractmethod
    def answer(self, image_path: str | Path, questions: Sequence[AtomicQuestion]) -> list[str]:
        raise NotImplementedError
