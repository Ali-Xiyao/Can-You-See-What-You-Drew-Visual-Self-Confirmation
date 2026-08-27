from __future__ import annotations

import json
from pathlib import Path

from selfsight.analysis.capability_figure import FAMILY_ORDER, render_capability_figure


def _report(path: Path, observer_id: str, values: list[float], bias: float) -> Path:
    family = dict(zip(FAMILY_ORDER, values))
    path.write_text(
        json.dumps(
            {
                "observer_id": observer_id,
                "observer_revision": "a" * 40,
                "images": 120,
                "family_open_accuracy": family,
                "macro_open_accuracy": sum(values) / len(values),
                "absolute_yes_bias": bias,
                "gate_minus_1_capability_pass": sum(value >= 0.8 for value in values) >= 4,
                "gate_minus_1_bias_pass": bias <= 0.1,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_capability_figure_exports_and_qa(tmp_path) -> None:
    first = _report(tmp_path / "first.json", "showlab/show-o-w-clip-vit-512x512", [0.7] * 6, 0.05)
    second = _report(
        tmp_path / "second.json",
        "Qwen/Qwen2-VL-2B-Instruct",
        [0.9, 0.8, 1.0, 0.7, 0.85, 0.95],
        0.02,
    )
    outputs = render_capability_figure([first, second], tmp_path / "figure", evidence_status="test")
    for path in outputs.values():
        assert Path(path).is_file()
    qa = json.loads(Path(outputs["qa"]).read_text(encoding="utf-8"))
    assert qa["observer_count"] == 2
    assert qa["family_count"] == 6
    assert qa["nonwhite_pixel_fraction"] > 0.05
