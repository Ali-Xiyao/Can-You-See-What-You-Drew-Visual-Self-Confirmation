from __future__ import annotations

from pathlib import Path

import pandas as pd

from selfsight.analysis.breakpoints import estimate_d_g, estimate_d_star
from selfsight.analysis.figure1 import render_figure1_from_csv


def _rows():
    output = []
    for index in range(11):
        output.append(
            {
                "seed": 1,
                "arm": "naive",
                "step": index * 25,
                "internal_score": 0.6 + 0.025 * index,
                "external_correctness": 0.65 + 0.015 * min(index, 4) - 0.02 * max(0, index - 4),
                "gda_free": 0.92 - 0.02 * min(index, 2) - 0.11 * max(0, index - 2),
                "gda_gold": 0.94 - 0.015 * min(index, 2) - 0.115 * max(0, index - 2),
                "noise_low": 0.82,
                "noise_high": 0.96,
                "scfr_competent": 0.05 + 0.02 * max(0, index - 3),
            }
        )
    return output


def test_breakpoints_and_figure_exports(tmp_path):
    frame = pd.DataFrame(_rows())
    d_star = estimate_d_star(frame.step, frame.internal_score, frame.external_correctness)
    d_g = estimate_d_g(frame.step, frame.gda_free, noise_low=0.82, noise_high=0.96)
    assert d_star.d_star is not None
    assert d_g.d_g is not None
    metrics = tmp_path / "metrics.csv"
    frame.to_csv(metrics, index=False)
    outputs = render_figure1_from_csv(
        metrics,
        tmp_path / "figure1",
        d_g=d_g.d_g,
        d_star=d_star.d_star,
        evidence_status="test",
    )
    for path in outputs.values():
        assert Path(path).is_file()
