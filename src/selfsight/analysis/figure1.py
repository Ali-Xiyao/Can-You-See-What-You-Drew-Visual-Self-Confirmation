"""Publication-scale Figure 1: divergence anchor, warning signal, and SCFR."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from selfsight.utils.jsonl import atomic_write_json

OKABE_ITO = {
    "internal": "#0072B2",
    "external": "#D55E00",
    "gda_free": "#CC79A7",
    "gda_gold": "#009E73",
    "scfr": "#000000",
    "noise": "#999999",
    "lead": "#F0E442",
}
REQUIRED_COLUMNS = {
    "seed",
    "arm",
    "step",
    "internal_score",
    "external_correctness",
    "gda_free",
    "gda_gold",
    "noise_low",
    "noise_high",
    "scfr_competent",
}


def setup_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "savefig.transparent": False,
        }
    )


def profile_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Figure metrics are missing columns: {missing}")
    numeric_columns = sorted(REQUIRED_COLUMNS.difference({"arm"}))
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "seeds": sorted(int(item) for item in frame["seed"].unique()),
        "arms": sorted(str(item) for item in frame["arm"].unique()),
        "steps": sorted(float(item) for item in frame["step"].unique()),
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
        "nonfinite_by_numeric_column": {
            key: int((~np.isfinite(numeric[key].to_numpy(dtype=float))).sum()) for key in numeric_columns
        },
        "ranges": {
            key: [float(numeric[key].min()), float(numeric[key].max())] for key in numeric_columns
        },
        "duplicate_seed_arm_step": int(frame.duplicated(["seed", "arm", "step"]).sum()),
    }


def _summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby("step", sort=True)[column]
    return grouped.agg(median="median", low="min", high="max", n="count").reset_index()


def _plot_series(
    ax: Any,
    summary: pd.DataFrame,
    *,
    label: str,
    color: str,
    linestyle: str,
    marker: str,
) -> None:
    x = summary["step"].to_numpy(dtype=float)
    median = summary["median"].to_numpy(dtype=float)
    if int(summary["n"].max()) > 1:
        ax.fill_between(
            x,
            summary["low"].to_numpy(dtype=float),
            summary["high"].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    ax.plot(
        x,
        median,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.7,
        markevery=1,
        label=label,
    )


def _plot_seed_traces(
    ax: Any,
    frame: pd.DataFrame,
    column: str,
    *,
    color: str,
    linestyle: str,
) -> None:
    """Show every formal seed faintly; the emphasized curve remains the median."""

    if frame["seed"].nunique() <= 1:
        return
    for _seed, seed_frame in frame.groupby("seed", sort=True):
        seed_frame = seed_frame.sort_values("step")
        ax.plot(
            seed_frame["step"].to_numpy(dtype=float),
            seed_frame[column].to_numpy(dtype=float),
            color=color,
            linestyle=linestyle,
            linewidth=0.65,
            alpha=0.30,
            zorder=1,
        )


def _shade_lead(axes: list[Any], d_g: float | None, d_star: float | None) -> None:
    for ax in axes:
        if d_g is not None:
            ax.axvline(d_g, color=OKABE_ITO["gda_free"], linestyle=":", linewidth=1.0)
        if d_star is not None:
            ax.axvline(d_star, color="#333333", linestyle="--", linewidth=1.0)
        if d_g is not None and d_star is not None and d_star > d_g:
            ax.axvspan(d_g, d_star, color=OKABE_ITO["lead"], alpha=0.18, linewidth=0)


def _panel_label(ax: Any, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-28, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        annotation_clip=False,
    )


def build_figure1(
    frame: pd.DataFrame,
    *,
    arm: str = "naive",
    d_g: float | None = None,
    d_star: float | None = None,
    evidence_status: str = "formal",
) -> plt.Figure:
    setup_publication_style()
    profile = profile_metrics(frame)
    if profile["duplicate_seed_arm_step"]:
        raise ValueError("Figure input has duplicate seed/arm/step rows")
    selected = frame.loc[frame["arm"] == arm].copy()
    if selected.empty:
        raise ValueError(f"No metrics found for arm={arm}")
    selected = selected.sort_values(["seed", "step"])
    fig, axes_array = plt.subplots(
        3,
        1,
        figsize=(7.16, 5.45),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.2, 1.0, 0.8)},
    )
    axes = list(axes_array)

    _plot_seed_traces(axes[0], selected, "internal_score", color=OKABE_ITO["internal"], linestyle="-")
    _plot_series(
        axes[0],
        _summary(selected, "internal_score"),
        label="Internal cycle score",
        color=OKABE_ITO["internal"],
        linestyle="-",
        marker="o",
    )
    _plot_seed_traces(
        axes[0], selected, "external_correctness", color=OKABE_ITO["external"], linestyle="--"
    )
    _plot_series(
        axes[0],
        _summary(selected, "external_correctness"),
        label="External correctness",
        color=OKABE_ITO["external"],
        linestyle="--",
        marker="s",
    )
    axes[0].set_ylabel("Score (proportion)")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(frameon=False, loc="lower right", ncol=2, handlelength=2.3)

    noise_low = _summary(selected, "noise_low")["median"].to_numpy(dtype=float)
    noise_high = _summary(selected, "noise_high")["median"].to_numpy(dtype=float)
    steps = _summary(selected, "noise_low")["step"].to_numpy(dtype=float)
    axes[1].fill_between(
        steps,
        noise_low,
        noise_high,
        color=OKABE_ITO["noise"],
        alpha=0.22,
        linewidth=0,
        label="Gate -1b noise floor (95%)",
    )
    _plot_seed_traces(axes[1], selected, "gda_free", color=OKABE_ITO["gda_free"], linestyle="-")
    _plot_series(
        axes[1],
        _summary(selected, "gda_free"),
        label="GDA-free",
        color=OKABE_ITO["gda_free"],
        linestyle="-",
        marker="^",
    )
    _plot_seed_traces(axes[1], selected, "gda_gold", color=OKABE_ITO["gda_gold"], linestyle="-.")
    _plot_series(
        axes[1],
        _summary(selected, "gda_gold"),
        label="GDA-gold",
        color=OKABE_ITO["gda_gold"],
        linestyle="-.",
        marker="D",
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.5)
    axes[1].set_ylabel("Gradient cosine")
    axes[1].set_ylim(-1.02, 1.02)
    axes[1].legend(frameon=False, loc="lower left", ncol=3, handlelength=2.3)

    _plot_seed_traces(
        axes[2], selected, "scfr_competent", color=OKABE_ITO["scfr"], linestyle="-."
    )
    _plot_series(
        axes[2],
        _summary(selected, "scfr_competent"),
        label="SCFR@competent",
        color=OKABE_ITO["scfr"],
        linestyle="-.",
        marker="v",
    )
    axes[2].set_ylabel("SCFR (proportion)")
    axes[2].set_xlabel("Optimizer step")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].legend(frameon=False, loc="upper left")

    _shade_lead(axes, d_g, d_star)
    for ax, label in zip(axes, ("a", "b", "c")):
        _panel_label(ax, label)
        ax.grid(axis="y", color="#D9D9D9", linestyle=":", linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if d_g is not None:
        axes[0].text(d_g, 1.01, r"$D_g$", ha="right", va="bottom", fontsize=7, color=OKABE_ITO["gda_free"])
    if d_star is not None:
        axes[0].text(d_star, 1.01, r"$D^{*}$", ha="left", va="bottom", fontsize=7, color="#333333")
    if d_g is not None and d_star is not None and d_star > d_g:
        midpoint = (d_g + d_star) / 2.0
        axes[0].text(midpoint, 0.04, "Lead", ha="center", va="bottom", fontsize=7)
    if evidence_status.lower() != "formal":
        fig.suptitle(
            f"{evidence_status.upper()} PIPELINE OUTPUT - NOT SCIENTIFIC EVIDENCE",
            color="#9C2F2F",
            fontsize=8,
            fontweight="bold",
        )
    return fig


def audit_layout(fig: plt.Figure) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    for warning in caught:
        message = str(warning.message)
        severity = "fail" if "glyph" in message.lower() or "font" in message.lower() else "warn"
        issues.append({"severity": severity, "message": message})
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    for text_item in fig.findobj(match=mpl.text.Text):
        if not text_item.get_visible() or not text_item.get_text().strip():
            continue
        box = text_item.get_window_extent(renderer=renderer)
        if box.x0 < figure_box.x0 - 2 or box.y0 < figure_box.y0 - 2 or box.x1 > figure_box.x1 + 2 or box.y1 > figure_box.y1 + 2:
            issues.append({"severity": "warn", "message": f"Text may be clipped: {text_item.get_text()}"})
    return issues


def export_figure1(
    fig: plt.Figure,
    output_basename: str | Path,
    *,
    dpi: int = 600,
) -> dict[str, str]:
    base = Path(output_basename)
    base.parent.mkdir(parents=True, exist_ok=True)
    preview = base.with_name(base.name + "_preview").with_suffix(".png")
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    svg = base.with_suffix(".svg")
    grayscale = base.with_name(base.name + "_grayscale").with_suffix(".png")
    fig.savefig(preview, dpi=150, facecolor="white")
    issues = audit_layout(fig)
    if any(issue["severity"] == "fail" for issue in issues):
        raise RuntimeError(f"Figure QA failed: {issues}")
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    with Image.open(png) as image:
        image.convert("L").save(grayscale, dpi=(dpi, dpi))
    qa_path = base.with_name(base.name + "_qa").with_suffix(".json")
    atomic_write_json(
        qa_path,
        {
            "schema_version": 1,
            "issues": issues,
            "passed": not any(issue["severity"] == "fail" for issue in issues),
            "figure_size_inches": list(fig.get_size_inches()),
            "dpi": dpi,
            "files": [str(path.resolve()) for path in (preview, png, pdf, svg, grayscale)],
        },
    )
    return {
        "preview": str(preview),
        "png": str(png),
        "pdf": str(pdf),
        "svg": str(svg),
        "grayscale": str(grayscale),
        "qa": str(qa_path),
    }


def render_figure1_from_csv(
    metrics_csv: str | Path,
    output_basename: str | Path,
    *,
    arm: str = "naive",
    d_g: float | None = None,
    d_star: float | None = None,
    evidence_status: str = "formal",
) -> dict[str, str]:
    frame = pd.read_csv(metrics_csv)
    profile_path = Path(output_basename).with_name(Path(output_basename).name + "_data_profile.json")
    atomic_write_json(profile_path, profile_metrics(frame))
    fig = build_figure1(frame, arm=arm, d_g=d_g, d_star=d_star, evidence_status=evidence_status)
    try:
        outputs = export_figure1(fig, output_basename)
    finally:
        plt.close(fig)
    outputs["data_profile"] = str(profile_path)
    return outputs
