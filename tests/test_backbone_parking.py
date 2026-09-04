"""The ladder needs its own card, and the check for that must be exact.

Parking the backbone on the CPU so one card could host both stages was tried
and abandoned: with LoRA attached and accelerate's hooks on the blocks, moving
the dispatched model segfaults the training process. It does so after the
round's images are drawn, so the run looks healthy for an hour first. What is
left is the refusal, and the refusal is only as good as the comparison behind
it -- "cuda" and "cuda:0" are one card spelled two ways, and torch.device does
not call them equal, so a naive == would wave through exactly the assignment
that cannot work.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4_train", ROOT / "scripts" / "v4_train.py")
train = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(train)


@pytest.mark.parametrize("left,right,same", [
    ("cuda:0", "cuda:0", True),
    ("cuda", "cuda:0", True),
    ("cuda:0", "cuda", True),
    ("cuda:0", "cuda:1", False),
    ("cuda:1", "cuda:1", True),
    ("cpu", "cpu", False),
    ("cpu", "cuda:0", False),
])
def test_the_two_spellings_of_one_card_count_as_one_card(left, right, same):
    assert train._same_cuda_device(left, right) is same


def test_the_backbone_no_longer_offers_to_park():
    """Removed rather than left callable: it segfaults under the real config."""
    from selfsight.backbones.showo2 import Showo2Adapter

    assert not hasattr(Showo2Adapter, "parked")
