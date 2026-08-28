"""Publication-scale Gate -2 readiness matrix with exact, non-aggregated evidence."""

from __future__ import annotations

import json
import warnings
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

from selfsight.analysis.capability_figure import FAMILY_LABELS, FAMILY_ORDER
from selfsight.analysis.figure1 import setup_publication_style
from selfsight.utils.hashing import sha256_file
from selfsight.utils.jsonl import atomic_write_json

GATE_COLUMNS = (
    "minus_2a_unified_functionality",
    "minus_2b_reference_observation",
    "minus_2c_generated_measurability",
    "minus_2d_joint_families",
    "passed",
)
GATE_LABELS = ("2A\nUnified", "2B\nObserve", "2C\nMeasure", "2D\nJoint", "Final")
METRICS = ("observation", "coverage", "precision", "oracle_at_4")
METRIC_LABELS = ("Observe", "Coverage", "Precision", "Oracle@4")
PASS_BLUE = "#0072B2"
FAIL_ORANGE = "#E69F00"
NOT_TESTED_GRAY = "#BDBDBD"


def _short_name(model_id: str) -> str:
    return {
        "showlab/show-o2-1.5B": "Show-o2 1.5B",
        "showlab/show-o2-1.5B-HQ": "Show-o2 1.5B-HQ",
        "showlab/show-o2-7B": "Show-o2 7B",
    }.get(model_id, model_id.rsplit("/", maxsplit=1)[-1])


def load_readiness_decisions(
    decision_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Profile exact Gate/family values; never average across candidates or families."""

    gate_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    sources = []
    for order, raw_path in enumerate(decision_paths):
        path = Path(raw_path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("gate") != "minus_2_joint_readiness":
            raise RuntimeError(f"Not a Gate -2 decision: {path}")
        rank = int(report["candidate_rank"])
        if rank in seen_ranks:
            raise RuntimeError(f"Duplicate candidate rank in readiness figure: {rank}")
        seen_ranks.add(rank)
        model_id = str(report["model_id"])
        checks = report["checks"]
        decision_mode = report.get("decision_mode")
        upstream_stop = decision_mode == "upstream_stop_before_human_and_a4"
        human_stop = decision_mode == "stop_after_human_before_a4"
        if upstream_stop:
            unmeasured_gates = {
                "minus_2a_unified_functionality",
                "minus_2d_joint_families",
            }
        elif human_stop:
            unmeasured_gates = {"minus_2a_unified_functionality"}
        else:
            unmeasured_gates = set()
        for gate in GATE_COLUMNS[:-1]:
            if gate not in checks:
                raise RuntimeError(f"Decision is missing {gate}: {path}")
            measured = gate not in unmeasured_gates
            gate_rows.append(
                {
                    "order": order,
                    "candidate_rank": rank,
                    "model_id": model_id,
                    "model_label": _short_name(model_id),
                    "gate": gate,
                    "passed": measured and bool(checks[gate]),
                    "measured": measured,
                    "source_decision": str(path),
                }
            )
        gate_rows.append(
            {
                "order": order,
                "candidate_rank": rank,
                "model_id": model_id,
                "model_label": _short_name(model_id),
                "gate": "passed",
                "passed": bool(report["passed"]),
                "measured": True,
                "source_decision": str(path),
            }
        )
        thresholds = report["thresholds"]
        metric_thresholds = {
            "observation": float(thresholds["reference"]["family_open_accuracy_min"]),
            "coverage": float(thresholds["generated"]["family_coverage_min"]),
            "precision": float(thresholds["generated"]["verifier_precision_min"]),
            "oracle_at_4": float(thresholds["generated"]["oracle_at_4_min"]),
        }
        metric_values = {
            "observation": report["metrics"]["family_open_accuracy"],
            "coverage": report["metrics"]["family_coverage"],
            "precision": report["metrics"]["family_precision"],
            "oracle_at_4": report["metrics"]["family_oracle_at_4"],
        }
        for family in FAMILY_ORDER:
            joint_eligible = True
            for metric in METRICS:
                values = metric_values[metric]
                if values is None or family not in values:
                    joint_eligible = False
                    break
                joint_eligible = joint_eligible and (
                    float(values[family]) >= metric_thresholds[metric]
                )
            for metric in METRICS:
                values = metric_values[metric]
                measured = values is not None
                if not measured:
                    skipped = report.get("skipped_by_stop_rule", [])
                    if not (
                        upstream_stop
                        and metric == "precision"
                        and "blind_human_precision" in skipped
                    ):
                        raise RuntimeError(
                            f"Missing {metric}/{family} without a registered stop rule: {path}"
                        )
                    value = np.nan
                else:
                    if not isinstance(values, dict) or family not in values:
                        raise RuntimeError(f"Missing {metric}/{family} value in {path}")
                    value = float(values[family])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"Out-of-range {metric}/{family} value in {path}: {value}")
                threshold = metric_thresholds[metric]
                family_rows.append(
                    {
                        "order": order,
                        "candidate_rank": rank,
                        "model_id": model_id,
                        "model_label": _short_name(model_id),
                        "family": family,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "measured": measured,
                        "metric_pass": measured and value >= threshold,
                        "joint_eligible": joint_eligible,
                        "source_decision": str(path),
                    }
                )
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "candidate_rank": rank,
                "model_id": model_id,
            }
        )
    if not sources:
        raise ValueError("At least one Gate -2 decision is required")
    gate_frame = pd.DataFrame(gate_rows).sort_values(["candidate_rank", "gate"])
    family_frame = pd.DataFrame(family_rows).sort_values(["candidate_rank", "family", "metric"])
    profile = {
        "candidate_count": len(sources),
        "gate_rows": len(gate_frame),
        "family_metric_rows": len(family_frame),
        "missing_values": int(
            gate_frame.isna().sum().sum() + family_frame.drop(columns=["value"]).isna().sum().sum()
        ),
        "not_tested_cells": int((~family_frame["measured"]).sum()),
        "family_count": int(family_frame["family"].nunique()),
        "metric_count": int(family_frame["metric"].nunique()),
        "value_range": [
            float(family_frame["value"].min()),
            float(family_frame["value"].max()),
        ],
        "duplicate_gate_cells": int(gate_frame.duplicated(["candidate_rank", "gate"]).sum()),
        "duplicate_family_metric_cells": int(
            family_frame.duplicated(["candidate_rank", "family", "metric"]).sum()
        ),
        "sources": sources,
    }
    if (
        profile["missing_values"]
        or profile["duplicate_gate_cells"]
        or profile["duplicate_family_metric_cells"]
    ):
        raise RuntimeError(f"Readiness figure data profile failed: {profile}")
    return gate_frame, family_frame, profile


def _panel_label(ax: Any, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-30, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        annotation_clip=False,
    )


def _gate_panel(ax: Any, gate_frame: pd.DataFrame) -> None:
    candidates = (
        gate_frame[["candidate_rank", "model_label"]]
        .drop_duplicates()
        .sort_values("candidate_rank")
    )
    matrix = np.array(
        [
            [
                int(
                    gate_frame.loc[
                        (gate_frame["candidate_rank"] == row.candidate_rank)
                        & (gate_frame["gate"] == gate),
                        "passed",
                    ].iloc[0]
                )
                for gate in GATE_COLUMNS
            ]
            for row in candidates.itertuples(index=False)
        ],
        dtype=int,
    )
    ax.set_xlim(-0.5, len(GATE_COLUMNS) - 0.5)
    ax.set_ylim(len(candidates) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(GATE_COLUMNS)), GATE_LABELS)
    ax.set_yticks(
        np.arange(len(candidates)),
        [f"{row.model_label}\nrank {row.candidate_rank}" for row in candidates.itertuples()],
    )
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            record = gate_frame.loc[
                (gate_frame["candidate_rank"] == candidates.iloc[row]["candidate_rank"])
                & (gate_frame["gate"] == GATE_COLUMNS[column])
            ].iloc[0]
            measured = bool(record["measured"])
            passed = bool(matrix[row, column])
            ax.add_patch(
                patches.Rectangle(
                    (column - 0.49, row - 0.49),
                    0.98,
                    0.98,
                    facecolor=(
                        PASS_BLUE if passed else FAIL_ORANGE if measured else NOT_TESTED_GRAY
                    ),
                    edgecolor="black",
                    linewidth=0.65,
                    hatch=None if passed else "////" if measured else "..",
                )
            )
            ax.text(
                column,
                row,
                "PASS" if passed else "FAIL" if measured else "N/T",
                ha="center",
                va="center",
                color="white" if passed else "black",
                fontweight="bold",
                fontsize=7,
            )
    ax.set_title("Registered candidate route", loc="left", fontweight="bold")
    _panel_label(ax, "a")


def _family_panel(ax: Any, family_frame: pd.DataFrame) -> None:
    latest_rank = int(family_frame["candidate_rank"].max())
    latest = family_frame.loc[family_frame["candidate_rank"] == latest_rank]
    model_label = str(latest["model_label"].iloc[0])
    ax.set_xlim(-1.35, len(METRICS) + 1.15)
    ax.set_ylim(len(FAMILY_ORDER) - 0.5, -1.0)
    ax.axis("off")
    for column, label in enumerate((*METRIC_LABELS, "Joint")):
        ax.text(column, -0.62, label, ha="center", va="center", fontweight="bold", fontsize=7)
    for row, family in enumerate(FAMILY_ORDER):
        ax.text(-0.55, row, FAMILY_LABELS[family], ha="right", va="center", fontsize=7)
        family_values = latest.loc[latest["family"] == family]
        joint = bool(family_values["joint_eligible"].iloc[0])
        for column, metric in enumerate(METRICS):
            record = family_values.loc[family_values["metric"] == metric].iloc[0]
            measured = bool(record["measured"])
            passed = bool(record["metric_pass"])
            rectangle = patches.Rectangle(
                (column - 0.44, row - 0.39),
                0.88,
                0.78,
                facecolor="#F7F7F7" if measured else NOT_TESTED_GRAY,
                edgecolor=PASS_BLUE if passed else FAIL_ORANGE if measured else "#666666",
                linewidth=1.45 if passed else 1.0,
                hatch=None if passed else "////" if measured else "..",
            )
            ax.add_patch(rectangle)
            ax.text(
                column,
                row,
                f"{float(record['value']):.0%}" if measured else "N/T",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold" if passed else "normal",
            )
        joint_column = len(METRICS)
        joint_measured = bool(family_values["measured"].all())
        ax.add_patch(
            patches.Rectangle(
                (joint_column - 0.44, row - 0.39),
                0.88,
                0.78,
                facecolor=(
                    NOT_TESTED_GRAY if not joint_measured else PASS_BLUE if joint else FAIL_ORANGE
                ),
                edgecolor="#666666" if not joint_measured else "black",
                linewidth=0.7,
                hatch=None if joint else "////" if joint_measured else "..",
            )
        )
        ax.text(
            joint_column,
            row,
            "YES" if joint else "NO" if joint_measured else "N/T",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white" if joint else "black",
        )
    ax.text(
        0.0,
        1.015,
        f"Latest audited candidate: {model_label}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8,
    )
    ax.text(
        1.0,
        -0.06,
        "Thresholds: observe 80%; coverage 70%; precision 95%; Oracle@4 70%.\n"
        "Blue bold border = pass; orange hatch = fail; gray dots = not tested.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    _panel_label(ax, "b")


def build_readiness_figure(
    gate_frame: pd.DataFrame,
    family_frame: pd.DataFrame,
    *,
    evidence_status: str,
) -> plt.Figure:
    setup_publication_style()
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    figure_height = max(5.0, 0.42 * gate_frame["candidate_rank"].nunique() + 4.2)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.16, figure_height),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 2.6)},
    )
    _gate_panel(axes[0], gate_frame)
    _family_panel(axes[1], family_frame)
    fig.suptitle(
        f"Joint Generate-Observe Readiness  |  {evidence_status}",
        fontsize=9,
        fontweight="bold",
    )
    return fig


def render_readiness_matrix(
    decision_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    evidence_status: str = "local engineering evidence",
) -> dict[str, str]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite readiness figure directory: {output}")
    output.mkdir(parents=True)
    gate_frame, family_frame, profile = load_readiness_decisions(decision_paths)
    gate_csv = output / "readiness_gate_matrix.csv"
    family_csv = output / "readiness_family_metrics.csv"
    gate_frame.to_csv(gate_csv, index=False)
    family_frame.to_csv(family_csv, index=False)
    fig = build_readiness_figure(gate_frame, family_frame, evidence_status=evidence_status)
    paths = {
        "preview_png": str(output / "figure_readiness_preview.png"),
        "png": str(output / "figure_readiness.png"),
        "pdf": str(output / "figure_readiness.pdf"),
        "svg": str(output / "figure_readiness.svg"),
        "grayscale_png": str(output / "figure_readiness_grayscale.png"),
    }
    layout_warnings = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fig.savefig(paths["preview_png"], dpi=150, bbox_inches="tight", facecolor="white")
    layout_warnings.extend(str(item.message) for item in captured if "Glyph" in str(item.message))
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(paths["png"]) as opened:
        width, height = opened.size
        gray = opened.convert("L")
        gray.save(paths["grayscale_png"], dpi=(600, 600))
        nonwhite = float(np.mean(np.asarray(gray) < 250))
    qa = {
        "schema_version": 2,
        "evidence_status": evidence_status,
        "chart_selection": (
            "Two exact annotated matrices: categorical gate route plus family-by-metric threshold "
            "table; no means, error bars, dual axes, or continuous color inference."
        ),
        "profile": profile,
        "figure_pixels": [width, height],
        "nonwhite_pixel_fraction": nonwhite,
        "glyph_warnings": layout_warnings,
        "redundant_encoding": {
            "pass": "blue + bold text/border",
            "fail": "orange + hatch + FAIL/NO text",
            "not_tested": "gray + dotted hatch + N/T text",
        },
        "outputs": {
            **paths,
            "gate_csv": str(gate_csv),
            "family_csv": str(family_csv),
        },
    }
    qa["checks"] = {
        "complete_six_family_matrix": profile["family_count"] == 6,
        "complete_four_metric_matrix": profile["metric_count"] == 4,
        "no_missing_or_duplicate_cells": profile["missing_values"] == 0
        and profile["duplicate_gate_cells"] == 0
        and profile["duplicate_family_metric_cells"] == 0,
        "no_glyph_warnings": not layout_warnings,
        "nonempty_render": nonwhite > 0.03,
        "vector_and_grayscale_exports": all(Path(path).is_file() for path in paths.values()),
    }
    qa["passed"] = all(qa["checks"].values())
    qa_path = output / "figure_readiness_qa.json"
    atomic_write_json(qa_path, qa)
    return {
        **paths,
        "gate_csv": str(gate_csv),
        "family_csv": str(family_csv),
        "qa": str(qa_path),
    }
