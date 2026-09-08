#!/usr/bin/env python
"""The co-evolution curve: what the description costs, checkpoint by checkpoint.

The cross-sectional result (`review-packets/context-ablation-20260908`) was
measured with the *untrained* backbone answering about a fixed corpus. It says
the description costs 31.8 points of perception on the trials where the
description and the pixels disagree. It cannot say whether training makes that
worse, which is the whole co-evolution question.

This walks a finished training run and asks each checkpoint about the images
that checkpoint drew, in both conditions, wearing its own adapter. Two stages,
because one needs a GPU for hours and the other is arithmetic:

    observe   for each (arm, checkpoint): blind pass + prompted pass
    report    endpoints 2 and 3 of .planning/2026-09-08-iclr-redesign/PREREG.md

`observe` is resumable and skips a checkpoint whose answers are already there,
so an interrupted pass costs the checkpoint it was on and nothing before it.

It requires `detect` / `crop` / `verify` to have run over each checkpoint's
evaluation directory first -- the questions are built from the detections, never
from the description, and `verified.jsonl` is where the detections settle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfsight.analysis.context import (
    context_effect,
    discrimination_gap,
    dose_response,
    pools,
    selection_gain,
)

PIPELINE = "scripts/v4_run_pipeline.py"
CONDITIONS = {"image_only": "answers.self.jsonl", "prompted": "answers.prompted.self.jsonl"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def checkpoints(outdir: Path, arm: str, steps_per_round: int) -> list[tuple[int, Path, Path]]:
    """(step, evaluation directory, checkpoint directory), in training order.

    Step 0 wears the base adapter, not the arm's: before round 0 the two arms
    are the same model, and pointing at `checkpoints/<arm>/round--01` would be
    pointing at a directory that never existed.
    """

    found = []
    for eval_dir in sorted((outdir / "evaluations" / arm).glob("step-*")):
        step = int(eval_dir.name.removeprefix("step-"))
        round_index = step // steps_per_round - 1
        checkpoint = (outdir / "checkpoints" / "base" / "round--01" if round_index < 0 else
                      outdir / "checkpoints" / arm / f"round-{round_index:03d}")
        found.append((step, eval_dir, checkpoint))
    return found


def stage_observe(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    steps = int(load_yaml(args.config)["training"]["optimizer_steps_per_round"])
    for arm in args.arms:
        for step, eval_dir, checkpoint in checkpoints(outdir, arm, steps):
            if not (eval_dir / "verified.jsonl").exists():
                print(f"{arm} step {step}: no verified.jsonl, skipping "
                      f"(run detect/crop/verify over it first)")
                continue
            if not checkpoint.is_dir():
                print(f"{arm} step {step}: no checkpoint at {checkpoint}, skipping")
                continue
            for condition, filename in CONDITIONS.items():
                if (eval_dir / filename).exists() and not args.overwrite:
                    print(f"{arm} step {step} {condition}: already answered")
                    continue
                command = [args.python, "-u", PIPELINE, "observe",
                           "--run", str(eval_dir), "--device", args.device,
                           "--condition", condition,
                           "--checkpoint", str(checkpoint), "--config", args.config]
                if args.overwrite:
                    command.append("--overwrite")
                print(f"{arm} step {step} {condition}: {' '.join(command)}", flush=True)
                subprocess.run(command, check=True)


def load_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def curve(outdir: Path, arms: list[str], steps_per_round: int) -> list[dict]:
    rows = []
    for arm in arms:
        for step, eval_dir, _ in checkpoints(outdir, arm, steps_per_round):
            blind_path = eval_dir / CONDITIONS["image_only"]
            told_path = eval_dir / CONDITIONS["prompted"]
            if not blind_path.exists() or not told_path.exists():
                continue
            blind, told = read_jsonl(blind_path), read_jsonl(told_path)
            grouped = pools(blind, told)
            rows.append({
                "arm": arm,
                "step": step,
                "images": len({(r["spec_id"], r["candidate_index"]) for r in blind}),
                "context_effect": context_effect(blind, told),
                "selection_gain": selection_gain(grouped),
                "gap_blind": discrimination_gap(grouped, "blind"),
                "gap_prompted": discrimination_gap(grouped, "prompted"),
            })
    return rows


def stage_report(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    steps = int(load_yaml(args.config)["training"]["optimizer_steps_per_round"])
    rows = curve(outdir, args.arms, steps)
    if not rows:
        raise SystemExit("no checkpoint has both conditions answered yet")

    print(f"{'arm':<12}{'step':>6}{'imgs':>6}{'ctx effect':>12}"
          f"{'sel gain':>10}{'gap blind':>11}{'gap told':>10}")
    for row in rows:
        print(f"{row['arm']:<12}{row['step']:>6}{row['images']:>6}"
              f"{row['context_effect']:>12.3f}{row['selection_gain']:>10.3f}"
              f"{row['gap_blind']:>11.3f}{row['gap_prompted']:>10.3f}")
    print()

    # Endpoint 2: the prompted gap is the one predicted to collapse.
    for arm in args.arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if len(arm_rows) < 2:
            continue
        for key in ("gap_prompted", "gap_blind"):
            trend = dose_response([(row["step"], row[key]) for row in arm_rows], draws=args.draws)
            print(f"{arm:<12}{key:<14} slope per 100 steps "
                  f"{trend['slope'] * 100:+.4f} [{trend['low'] * 100:+.4f},"
                  f"{trend['high'] * 100:+.4f}]  n={trend['n']}")
    print()

    # Endpoint 3: stronger confirmation bias, bigger selection gain.
    points = [(row["context_effect"], row["selection_gain"]) for row in rows]
    trend = dose_response(points, draws=args.draws)
    print(f"dose-response  slope {trend['slope']:+.3f} "
          f"[{trend['low']:+.3f},{trend['high']:+.3f}]  n={trend['n']}")

    output = outdir / "context_curve.json"
    output.write_text(json.dumps({"curve": rows, "dose_response": trend}, indent=2),
                      encoding="utf-8")
    print(f"\nwrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--outdir", required=True)
        target.add_argument("--config", required=True)
        target.add_argument("--arms", nargs="+", default=["naive", "rfo_gold"])

    o = sub.add_parser("observe", help="both conditions, every checkpoint, its own adapter")
    common(o)
    o.add_argument("--device", default="cuda:0")
    o.add_argument("--python", default="envs/showo2/python.exe")
    o.add_argument("--overwrite", action="store_true")
    o.set_defaults(func=stage_observe)

    r = sub.add_parser("report", help="endpoints 2 and 3")
    common(r)
    r.add_argument("--draws", type=int, default=20000)
    r.set_defaults(func=stage_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
