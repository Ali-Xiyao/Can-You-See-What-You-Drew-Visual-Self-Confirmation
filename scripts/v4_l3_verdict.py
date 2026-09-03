"""Apply the pre-registered stopping rule in STATUS 34 to what the run produced.

Deliberately mechanical, and deliberately a separate file from the analysis it
reads. The rule was written before any curve existed; running it as code rather
than reading the numbers and deciding is what stops the threshold from moving
once the numbers are unwelcome.

Prints the verdict on line 1 and the reasoning below it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from selfsight.v4.evaluate import internal_noise_slope, read_metrics_csv
from selfsight.v4.train import ARMS


def observed_slope(rows, arm: str) -> float | None:
    """Least-squares slope of the internal curve, in score units per step."""

    usable = sorted((row for row in rows if row.arm == arm), key=lambda row: row.step)
    if len(usable) < 2:
        return None
    steps = [float(row.step) for row in usable]
    values = [float(row.internal_cycle) for row in usable]
    mean_step = sum(steps) / len(steps)
    mean_value = sum(values) / len(values)
    denominator = sum((step - mean_step) ** 2 for step in steps)
    if denominator == 0.0:
        return None
    return sum((step - mean_step) * (value - mean_value)
               for step, value in zip(steps, values)) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = read_metrics_csv(args.outdir / "checkpoint_metrics.csv")
    divergence = json.loads((args.outdir / "divergence.json").read_text(encoding="utf-8"))

    lines: list[str] = []
    moved: list[str] = []
    found: list[str] = []

    for arm in ARMS:
        report = divergence.get(arm)
        if report is None or "error" in (report or {}):
            lines.append(f"{arm}: no usable curve ({(report or {}).get('error', 'absent')})")
            continue
        slope = observed_slope(rows, arm)
        try:
            floor = internal_noise_slope(rows, arm)
        except ValueError as exc:
            lines.append(f"{arm}: no noise floor ({exc})")
            continue
        above = slope is not None and slope > floor
        if above:
            moved.append(arm)
        star = report.get("d_star")
        lines.append(
            f"{arm}: internal slope {slope:+.3e}/step vs noise floor {floor:.3e} "
            f"-> {'above' if above else 'BELOW'}; d_star={star}")
        if star is not None and above:
            found.append(arm)

    if found:
        verdict = ("A  D* found on " + ", ".join(found) +
                   " -- descriptive only: single seed, n=%d, must not be written as "
                   "significance (STATUS 34)" % int(config["data"]["local_outcome"]))
    elif not moved:
        steps = int(config["training"]["rounds"]) * int(
            config["training"]["optimizer_steps_per_round"])
        verdict = (f"B.1  no training happened -- both arms' internal slopes sit below "
                   f"their own noise floors after {steps} updates. This is a statement "
                   f"about the corpus, NOT about internal/external divergence "
                   f"(STATUS 33, 34)")
    else:
        verdict = ("B.1  internal moved on " + ", ".join(moved) +
                   " but no d_star is estimable; see the reasons below")

    print(verdict)
    for line in lines:
        print("  " + line)
    for arm in ARMS:
        report = divergence.get(arm) or {}
        if report.get("reason"):
            print(f"  {arm} reason: {report['reason']}")


if __name__ == "__main__":
    main()
