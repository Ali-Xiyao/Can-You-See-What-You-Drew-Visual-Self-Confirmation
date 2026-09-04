"""Parking the backbone so the ladder's adjudicator can have the card.

The two-card plan was not available -- the second card had other people's work
on it -- and loading an 8B adjudicator beside a resident 13.5 GB backbone dies
part-way through the shards. On Windows that surfaces as an access violation
rather than a clean CUDA OOM, so it reads as a crash and not as a capacity
problem; it cost a round of generation before it was understood.

The fix rests on one torch guarantee: Module.to moves parameters in place, so
an optimiser holding references to them is still holding the right objects when
they come back. That guarantee is pinned here rather than assumed, because if
it ever stopped holding the failure would be silent -- training would continue
against stale parameters instead of raising.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from selfsight.backbones.showo2 import Showo2Adapter

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4_train", ROOT / "scripts" / "v4_train.py")
train = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(train)


class Recorder:
    """Stands in for the transformer and remembers where it was told to go."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def to(self, device) -> "Recorder":
        self.calls.append(str(device))
        return self


def adapter_on(device: str) -> Showo2Adapter:
    adapter = Showo2Adapter.__new__(Showo2Adapter)
    adapter.model = Recorder()
    adapter.device = torch.device(device)
    return adapter


def test_parking_moves_the_transformer_off_the_card_and_puts_it_back():
    adapter = adapter_on("cuda:0")
    with adapter.parked():
        assert adapter.model.calls == ["cpu"], "must be gone while the ladder loads"
    assert adapter.model.calls == ["cpu", "cuda:0"]


def test_the_transformer_comes_back_even_when_the_ladder_fails():
    """A failed adjudication must not leave the backbone stranded on the CPU."""
    adapter = adapter_on("cuda:0")
    with pytest.raises(RuntimeError):
        with adapter.parked():
            raise RuntimeError("detector died")
    assert adapter.model.calls == ["cpu", "cuda:0"]


def test_two_cards_park_nothing():
    adapter = adapter_on("cuda:0")
    with adapter.parked(enabled=False):
        pass
    assert adapter.model.calls == []


def test_moving_a_module_keeps_the_parameter_objects_an_optimiser_holds():
    """The guarantee the fix rests on, pinned so a torch change cannot hide."""
    module = torch.nn.Linear(4, 4)
    optimiser = torch.optim.Adam(module.parameters())
    before = [id(p) for p in module.parameters()]

    module.to("cpu")

    assert [id(p) for p in module.parameters()] == before
    held = [id(p) for group in optimiser.param_groups for p in group["params"]]
    assert held == before


@pytest.mark.parametrize("left,right,same", [
    ("cuda:0", "cuda:0", True),
    ("cuda", "cuda:0", True),
    ("cuda:0", "cuda:1", False),
    ("cuda:1", "cuda:1", True),
    ("cpu", "cpu", False),
    ("cpu", "cuda:0", False),
])
def test_the_two_spellings_of_one_card_count_as_one_card(left, right, same):
    assert train._same_cuda_device(left, right) is same
