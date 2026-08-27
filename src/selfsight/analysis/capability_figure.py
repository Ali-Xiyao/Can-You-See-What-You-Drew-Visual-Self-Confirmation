"""Publication Figure 2 for the preregistered observer capability-floor fallback."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from PIL import Image

from selfsight.analysis.figure1 import setup_publication_style
from selfsight.utils.jsonl import atomic_write_json

FAMILY_ORDER = ("existence", "count", "color", "size", "spatial", "binding")
FAMILY_LABELS = {
    "existence": "Existence",
    "count": "Count",
    "color": "Color",
    "size": "Size",
    "spatial": "Spatial",
    "binding": "Binding",
}


def _short_name(model_id: str) -> str:
    names = {
        "showlab/show-o-w-clip-vit-512x512": "Show-o (CLIP)",
        "showlab/show-o-512x512": "Show-o (discrete)",
        "HuggingFaceTB/SmolVLM-500M-Instruct": "SmolVLM 0.5B",
        "OpenGVLab/InternVL2-1B": "InternVL2 1B",
        "Qwen/Qwen2-VL-2B-Instruct": "Qwen2-VL 2B",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL 7B",
        "deepseek-ai/Janus-Pro-1B": "Janus-Pro 1B",
    }
    return names.get(model_id, model_id.rsplit("/", maxsplit=1)[-1])


def load_capability_reports(report_paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load completed observer audits into a tidy, validation-ready table."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for order, raw_path in enumerate(report_paths):
        path = Path(raw_path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise TypeError(f"Observer report must be a JSON object: {path}")
        observer_id = str(report["observer_id"])
        if observer_id in seen:
            raise ValueError(f"Duplicate observer report: {observer_id}")
        seen.add(observer_id)
        family = report.get("family_open_accuracy")
        if not isinstance(family, dict) or set(family) != set(FAMILY_ORDER):
            raise ValueError(f"Observer report has an incomplete family matrix: {path}")
        values = [float(family[name]) for name in FAMILY_ORDER]
        macro = float(report["macro_open_accuracy"])
        bias = float(report["absolute_yes_bias"])
        if any(not 0.0 <= value <= 1.0 for value in [*values, macro, bias]):
            raise ValueError(f"Observer report contains an out-of-range proportion: {path}")
        for family_name, value in zip(FAMILY_ORDER, values):
            rows.append(
                {
                    "order": order,
                    "observer_id": observer_id,
                    "observer_label": _short_name(observer_id),
                    "revision": str(report["observer_revision"]),
                    "family": family_name,
                    "accuracy": value,
                    "macro_accuracy": macro,
                    "absolute_yes_bias": bias,
                    "images": int(report["images"]),
                    "capability_pass": bool(report["gate_minus_1_capability_pass"]),
                    "bias_pass": bool(report["gate_minus_1_bias_pass"]),
                    "source_report": str(path),
                }
            )
    if not rows:
        raise ValueError("At least one observer audit report is required")
    return pd.DataFrame(rows)


def _render(frame: pd.DataFrame, output_dir: Path, *, evidence_status: str) -> dict[str, str]:
    setup_publication_style()
    observers = (
        frame[["order", "observer_id", "observer_label", "macro_accuracy", "absolute_yes_bias"]]
        .drop_duplicates()
        .sort_values("order")
        .reset_index(drop=True)
    )
    matrix = np.array(
        [
            [
                float(
                    frame.loc[
                        (frame["observer_id"] == observer_id) & (frame["family"] == family),
                        "accuracy",
                    ].iloc[0]
                )
                for family in FAMILY_ORDER
            ]
            for observer_id in observers["observer_id"]
        ]
    )
    row_labels = [
        f"{row.observer_label}\nmacro {row.macro_accuracy:.1%}"
        for row in observers.itertuples(index=False)
    ]
    figure_height = max(2.8, 0.55 * len(observers) + 1.55)
    fig = plt.figure(figsize=(7.15, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(5.6, 1.8))
    heat = fig.add_subplot(grid[0, 0])
    bias_ax = fig.add_subplot(grid[0, 1], sharey=heat)

    image = heat.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
    heat.set_xticks(np.arange(len(FAMILY_ORDER)), [FAMILY_LABELS[name] for name in FAMILY_ORDER])
    heat.set_yticks(np.arange(len(observers)), row_labels)
    heat.tick_params(axis="x", rotation=32, length=0)
    for label in heat.get_xticklabels():
        label.set_horizontalalignment("right")
    heat.tick_params(axis="y", length=0, pad=5)
    heat.set_title("a  Open-question capability floor", loc="left", fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            passing = value >= 0.80
            heat.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="black" if value >= 0.65 else "white",
                fontweight="bold" if passing else "normal",
                fontsize=7,
            )
            if passing:
                heat.add_patch(
                    patches.Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.25,
                    )
                )
    colorbar = fig.colorbar(image, ax=heat, orientation="horizontal", fraction=0.08, pad=0.16)
    colorbar.set_label("Accuracy", labelpad=2)
    colorbar.set_ticks([0.0, 0.5, 0.8, 1.0])
    colorbar.set_ticklabels(["0", "50", "80", "100%"])
    heat.text(
        1.0,
        -0.31,
        "Outlined + bold: preregistered pass (≥80%)",
        transform=heat.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )

    y = np.arange(len(observers))
    bias = observers["absolute_yes_bias"].to_numpy(dtype=float)
    for index, value in enumerate(bias):
        bias_ax.plot([0.0, value], [index, index], color="#999999", linewidth=1.0, zorder=1)
    bias_ax.scatter(bias, y, color="#0072B2", edgecolor="black", linewidth=0.45, s=24, zorder=2)
    for index, value in enumerate(bias):
        bias_ax.annotate(
            f"{value:.1%}",
            (value, index),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.5,
        )
    bias_ax.axvline(0.10, color="#D55E00", linestyle="--", linewidth=1.1, label="10% limit")
    upper = max(0.18, float(bias.max()) + 0.09)
    bias_ax.set_xlim(0.0, min(0.5, upper))
    bias_ax.set_xlabel("Absolute yes-bias")
    bias_ax.set_title("b  Bias control", loc="left", fontweight="bold")
    bias_ax.tick_params(axis="y", left=False, labelleft=False)
    bias_ax.xaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    bias_ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, zorder=0)
    bias_ax.legend(frameon=False, loc="lower right", handlelength=1.7)
    for side in ("top", "right", "left"):
        bias_ax.spines[side].set_visible(False)

    fig.suptitle(
        f"Observer capability audit on program-rendered references  |  {evidence_status}",
        fontsize=8,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure2_capability_floor"
    paths = {
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "grayscale_png": str(output_dir / "figure2_capability_floor_grayscale.png"),
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(paths["png"]) as opened:
        opened.convert("L").save(paths["grayscale_png"])
    return paths


def render_capability_figure(
    report_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    evidence_status: str = "local Gate -1 audit",
) -> dict[str, str]:
    """Validate observer reports, export tidy data, and render Figure 2 in four formats."""

    output = Path(output_dir).resolve()
    frame = load_capability_reports(report_paths)
    paths = _render(frame, output, evidence_status=evidence_status)
    tidy = output / "capability_floor_tidy.csv"
    frame.to_csv(tidy, index=False)
    with Image.open(paths["png"]) as image:
        width, height = image.size
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    report = {
        "schema_version": 1,
        "evidence_status": evidence_status,
        "observer_count": int(frame["observer_id"].nunique()),
        "family_count": int(frame["family"].nunique()),
        "threshold": 0.80,
        "bias_limit": 0.10,
        "figure_pixels": [width, height],
        "nonwhite_pixel_fraction": float(np.mean(grayscale < 250)),
        "reports": [str(Path(path).resolve()) for path in report_paths],
        "outputs": {**paths, "tidy_csv": str(tidy)},
    }
    atomic_write_json(output / "figure2_qa.json", report)
    return {**paths, "tidy_csv": str(tidy), "qa": str(output / "figure2_qa.json")}
