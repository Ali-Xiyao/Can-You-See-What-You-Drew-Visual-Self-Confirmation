"""Plot an exploratory pilot without changing its data or inferring new events.

    python scripts/v4_decoupling_plot.py --outdir runs/v4/decoupling-pilot-20260906

Reads checkpoint_metrics.csv and gradient-probes/**/report.json (also probes/).
Only recorded paired differences supply gradient confidence intervals. Runtime
canaries are excluded. Fewer than two checkpoints yields a successful no-op.
PNG and provenance JSON use new timestamped names; existing files are preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def is_canary(path: Path | str, row: dict[str, Any] | None = None) -> bool:
    row = row or {}
    return ("canary" in str(path).lower() or bool(row.get("runtime_canary")) or bool(row.get("canary"))
            or "canary" in str(row.get("arm", "")).lower()
            or "canary" in str(row.get("checkpoint", {}).get("path", "")).lower()
            or "canary" in str(row.get("checkpoint", {}).get("arm", "")).lower())


def read_text(path: Path, sources: dict[str, str]) -> str:
    raw = path.read_bytes()
    sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8-sig")


def load_inputs(outdir: Path) -> dict[str, Any]:
    sources: dict[str, str] = {}
    metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gradients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = 0
    if is_canary(outdir):
        return {"metrics": {}, "gradients": {}, "summary": {}, "sources": {},
                "excluded_canaries": 1}
    path = outdir / "checkpoint_metrics.csv"
    if path.exists():
        seen: set[tuple[str, int]] = set()
        for raw in csv.DictReader(io.StringIO(read_text(path, sources))):
            if is_canary(path, raw):
                excluded += 1
                continue
            if not raw.get("arm") or number(raw.get("step")) is None:
                raise ValueError("Incomplete checkpoint_metrics.csv; retry after the writer finishes")
            row = {k: number(v) for k, v in raw.items() if k != "arm"}
            row["arm"], row["step"] = raw["arm"], int(raw["step"])
            key = (raw["arm"], row["step"])
            if key in seen:
                raise ValueError(f"Duplicate metric checkpoint: {key}")
            seen.add(key)
            metrics[raw["arm"]].append(row)
    for root in ("gradient-probes", "probes"):
        for path in sorted((outdir / root).glob("**/report.json")):
            if is_canary(path):
                excluded += 1
                continue
            row = json.loads(read_text(path, sources))
            if is_canary(path, row):
                excluded += 1
                continue
            identity = row.get("checkpoint", {})
            if "gda_free" not in row or identity.get("arm") is None:
                continue
            if number(identity.get("step")) is None:
                raise ValueError(f"Gradient report has no checkpoint step: {path}")
            gradients[str(identity["arm"])].append(row)
    shared = gradients.pop("base", [])
    if len(shared) > 1 or any(int(r["checkpoint"]["step"]) != 0 for r in shared):
        raise ValueError("Expected at most one shared step-0 gradient report")
    for arm in set(metrics) | set(gradients):
        if shared and not any(int(r["checkpoint"]["step"]) == 0 for r in gradients[arm]):
            gradients[arm] = [*shared, *gradients[arm]]
        steps = [int(r["checkpoint"]["step"]) for r in gradients[arm]]
        if len(steps) != len(set(steps)):
            raise ValueError(f"Duplicate gradient checkpoint for {arm}")
        gradients[arm].sort(key=lambda r: int(r["checkpoint"]["step"]))
    summary_path = outdir / "decoupling_report.json"
    summary = json.loads(read_text(summary_path, sources)) if summary_path.exists() else {}
    for rows in metrics.values():
        rows.sort(key=lambda r: r["step"])
    return {"metrics": dict(metrics), "gradients": dict(gradients), "summary": summary,
            "sources": sources, "excluded_canaries": excluded}


def gradient_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((number(r.get("gda_free", {}).get("cosine")) for r in rows
                     if int(r["checkpoint"]["step"]) == 0), None)
    points = []
    for row in rows:
        step = int(row["checkpoint"]["step"])
        cosine = number(row.get("gda_free", {}).get("cosine"))
        delta = row.get("delta_from_reference", {}).get("gda_free", {})
        point, low, high = (number(delta.get(k)) for k in ("point_difference", "ci_low", "ci_high"))
        paired = (delta.get("reference_step") == 0
                  and delta.get("method") == "paired_bootstrap_difference"
                  and (number(delta.get("n_common")) or 0) >= 4
                  and point is not None and low is not None and high is not None and low <= high)
        if not paired:
            point = cosine - baseline if cosine is not None and baseline is not None else None
            low = high = None
        points.append({"step": step, "delta": point, "low": low, "high": high,
                       "paired": paired, "n_common": delta.get("n_common"),
                       "n_retained": row.get("n_retained"), "n_bank": row.get("n_bank")})
    return points


def has_trajectory(data: dict[str, Any]) -> bool:
    for arm in set(data["metrics"]) | set(data["gradients"]):
        outcomes = {r["step"] for r in data["metrics"].get(arm, [])
                    if r.get("s_select") is not None or r.get("external_correct") is not None}
        gradients = {p["step"] for p in gradient_points(data["gradients"].get(arm, []))
                     if p["delta"] is not None}
        if len(outcomes) >= 2 or len(gradients) >= 2:
            return True
    return False


def fraction(row: dict[str, Any], numerator: str, denominator: str) -> float:
    n, d = row.get(numerator), row.get(denominator)
    return n / d if n is not None and d is not None and d > 0 else float("nan")


def verdict_coverage(row: dict[str, Any]) -> float:
    known, unknown = row.get("external_n"), row.get("external_unadjudicated")
    return (known / (known + unknown) if known is not None and unknown is not None
            and known + unknown > 0 else float("nan"))


def render(data: dict[str, Any], outdir: Path, output: Path | None = None) -> Path | None:
    if not has_trajectory(data):
        print("No trajectory yet: fewer than two measured checkpoints; no PNG written.")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (output or outdir / "plots" / f"decoupling-{stamp}.png").resolve()
    if not output.is_relative_to(outdir.resolve()) or output.suffix.lower() != ".png":
        raise ValueError("Output must be a PNG inside the pilot directory")
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    arms = sorted(set(data["metrics"]) | set(data["gradients"]))
    fig, axes = plt.subplots(3, len(arms), figsize=(6.3 * len(arms), 10.0),
                             squeeze=False, sharex="col")
    colours = {"score": "#176B9C", "external": "#C56A22", "known": "#17806F",
               "unknown": "#B84745", "answers": "#676E7A", "gradient": "#71529B"}
    for column, arm in enumerate(arms):
        top, cover, grad = axes[:, column]
        rows = data["metrics"].get(arm, [])
        xs = [r["step"] for r in rows]
        top.set_title(arm, fontsize=13, fontweight="bold")
        for key, label, marker, colour in (
            ("s_select", "Self-score on outcome images", "o", colours["score"]),
            ("external_correct", "External correctness (known verdicts)", "s", colours["external"]),
        ):
            values = [r.get(key) if r.get(key) is not None else float("nan") for r in rows]
            if any(math.isfinite(v) for v in values):
                top.plot(xs, values, marker=marker, color=colour, label=label, linewidth=1.6)
        top.set_ylim(-0.03, 1.03)
        top.set_ylabel("Score / correct fraction")
        if not rows:
            top.text(.5, .5, "No scored outcome checkpoints", ha="center", transform=top.transAxes)
        known = [verdict_coverage(r) for r in rows]
        unknown = [1 - v for v in known]
        answers = [fraction(r, "s_select_available", "s_select_total") for r in rows]
        for values, label, colour, style in (
            (known, "Known verdicts / images", colours["known"], "-"),
            (unknown, "Unknown verdicts / images", colours["unknown"], "--"),
            (answers, "Available answers / questions", colours["answers"], ":"),
        ):
            if any(math.isfinite(v) for v in values):
                cover.plot(xs, values, marker=".", linestyle=style, color=colour, label=label)
        cover.set_ylim(-0.04, 1.12)
        cover.set_ylabel("Coverage / unknown fraction")
        if rows:
            last = rows[-1]
            cover.text(.02, .07, f"Latest known verdicts: {last.get('external_n')}\n"
                       f"Unknown: {last.get('external_unadjudicated')}; "
                       f"answers: {last.get('s_select_available')}/{last.get('s_select_total')}",
                       fontsize=8, transform=cover.transAxes,
                       bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
        points = gradient_points(data["gradients"].get(arm, []))
        grad.axhline(0, color="#999999", linewidth=.8)
        paired_label = marginal_label = False
        for p in points:
            if p["delta"] is None:
                continue
            if p["paired"]:
                # Percentile intervals need not contain the point estimate.
                grad.vlines(p["step"], p["low"], p["high"], color=colours["gradient"], linewidth=1.5)
                grad.plot(p["step"], p["delta"], "o", color=colours["gradient"],
                          label="Paired delta + 95% CI" if not paired_label else None)
                grad.annotate(f"n={p['n_common']}", (p["step"], p["high"]),
                              xytext=(3, 4), textcoords="offset points", fontsize=7)
                paired_label = True
            else:
                grad.plot(p["step"], p["delta"], "o", markerfacecolor="none", color=colours["gradient"],
                          label="Point only; no paired CI" if not marginal_label else None)
                marginal_label = True
        if not any(p["delta"] is not None for p in points):
            grad.text(.5, .5, "No GDA-free comparison with base", ha="center", transform=grad.transAxes)
        grad.set_ylabel("GDA-free change from step 0")
        grad.set_xlabel("Optimizer updates")
        note = data["summary"].get("arms", {}).get(arm, {})
        if note.get("first_candidate_step") is not None:
            top.text(.02, .04, f"Report candidate: step {note['first_candidate_step']} (exploratory)",
                     fontsize=8, transform=top.transAxes)
        for axis in (top, cover, grad):
            axis.grid(axis="y", alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle("Exploratory training trajectories", fontsize=16, y=.985)
    fig.text(.5, .014, "Exploratory; no confirmed D* or Dg.\n"
             "Gradient CIs: paired prompts; no correction for repeated looks.\n"
             "Answer availability is not factual accuracy. Canary data excluded.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .085, 1, .965), h_pad=2.1)
    for name, expected in data["sources"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            plt.close(fig)
            raise ValueError(f"Input changed while plotting; retry: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "exploratory": True,
        "sources_sha256": data["sources"], "excluded_canaries": data["excluded_canaries"],
        "arms": arms, "plot": str(output),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}; exploratory, not a confirmed event.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.outdir.is_dir():
        parser.error("--outdir must be an existing pilot directory")
    try:
        render(load_inputs(args.outdir), args.outdir, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Cannot plot: {error}\n")


if __name__ == "__main__":
    main()
