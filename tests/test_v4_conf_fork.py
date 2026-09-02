"""Tests for the confirmatory fork test.

The whole value of the pre-registration is that the arithmetic was fixed before
the data arrived, so the two pieces that could silently drift -- the exact test
and the image-level clustering -- are pinned here rather than trusted.
"""

from __future__ import annotations

import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4_conf_fork", ROOT / "scripts" / "v4_conf_fork.py")
fork = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(fork)


def rational_fisher(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """The same test in exact arithmetic, as an independent second opinion."""
    row1, row2, col1, total = a + b, c + d, a + c, a + b + c + d

    def prob(x: int) -> Fraction:
        return Fraction(math.comb(row1, x) * math.comb(row2, col1 - x),
                        math.comb(total, col1))

    lo, hi = max(0, col1 - row2), min(row1, col1)
    observed = prob(a)
    return (float(sum(prob(x) for x in range(a, hi + 1))),
            float(sum(prob(x) for x in range(lo, hi + 1)
                      if prob(x) <= observed)))


@pytest.mark.parametrize("table", [
    (10, 33, 4, 50),   # the exploratory split, unclustered
    (3, 5, 7, 2),      # reversed direction
    (20, 10, 15, 25),
    (0, 10, 5, 5),     # an empty cell
    (1, 1, 1, 1),
])
def test_the_exact_test_agrees_with_exact_arithmetic(table):
    one, two = fork.fisher(*table)
    want_one, want_two = rational_fisher(*table)
    assert one == pytest.approx(want_one, abs=1e-10)
    assert two == pytest.approx(want_two, abs=1e-10)


def test_the_one_sided_p_is_for_the_first_cell_being_large():
    # A strong effect in the pre-registered direction is small one-sided; the
    # same table flipped is not. Getting this backwards would turn the
    # pre-declared direction check into a coin toss.
    strong, _ = fork.fisher(20, 5, 5, 20)
    reversed_, _ = fork.fisher(5, 20, 20, 5)
    assert strong < 0.001
    assert reversed_ > 0.999


def test_proneness_needs_only_one_seed_to_have_dropped_the_noun(tmp_path):
    spec = {"spec_id": "s1", "prompt": "one red mug and two blue plates",
            "objects": [{"object": "mug", "color": "red", "count": 1},
                        {"object": "plate", "color": "blue", "count": 2}],
            "relations": [], "surface": "white cloth", "metadata": {}}
    run = tmp_path / "run"
    run.mkdir()
    images = ["a.png", "b.png", "c.png"]
    (run / "manifest.jsonl").write_text("".join(
        f'{{"image_path": "{name}", "spec": {__import__("json").dumps(spec)}}}\n'
        for name in images), encoding="utf-8")
    rows = [
        # drew it, adjudicated
        '{"image_path": "a.png", "resolution": "agreed", '
        '"detections": [{"object": "mug"}, {"object": "plate"}]}',
        # dropped it, but never adjudicated: must not count
        '{"image_path": "b.png", "resolution": "pending_human", '
        '"detections": [{"object": "plate"}]}',
        # dropped it, adjudicated: this is what makes the cell prone
        '{"image_path": "c.png", "resolution": "human", '
        '"detections": [{"object": "plate"}]}',
    ]
    (run / "verified.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    labels = fork.proneness((str(run),))
    assert labels[("s1", "mug")] is True
    # The plate is requested twice, so it is not a singleton cell at all.
    assert ("s1", "plate") not in labels


def test_proneness_is_false_when_every_adjudicated_seed_drew_it(tmp_path):
    spec = {"spec_id": "s2", "prompt": "one red mug",
            "objects": [{"object": "mug", "color": "red", "count": 1}],
            "relations": [], "surface": "white cloth", "metadata": {}}
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.jsonl").write_text(
        f'{{"image_path": "a.png", "spec": {__import__("json").dumps(spec)}}}\n',
        encoding="utf-8")
    (run / "verified.jsonl").write_text(
        '{"image_path": "a.png", "resolution": "agreed", '
        '"detections": [{"object": "mug"}]}\n', encoding="utf-8")
    assert fork.proneness((str(run),)) == {("s2", "mug"): False}
