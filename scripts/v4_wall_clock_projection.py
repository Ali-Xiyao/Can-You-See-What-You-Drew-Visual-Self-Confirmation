"""Project the pilot's finish time without counting the remaining rounds by hand.

Sections 0.12 and 0.21 of the redesign log both enumerated the remaining
checkpoints in prose -- "(56 / 64 / 72 / 80)" -- and both dropped the last one.
Nothing in either projection could notice: a list typed by hand has no way to
be short.  So this derives the ladder from the same two config keys the
supervisor uses, in the same expression, and refuses to project unless the
reports that have already landed are a prefix of what it derived.

The supervisor's own loop (scripts/run_decoupling_pilot.py) is::

    for index in range(self.limit):          # limit = training.rounds
        ...
        if index == 0:
            self.report(0)                   # the untrained baseline
        ...
        self.report((index + 1) * self.config["training"]["optimizer_steps_per_round"])

so the ladder is [0] + [8, 16, ..., 8 * rounds] -- rounds + 1 entries, not
rounds, and the last one is 8 * rounds, not 8 * (rounds - 1).

Read-only.  Touches nothing inside the run directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def ladder(config: dict) -> list[int]:
    """Every step that will get a report, in order, derived not typed."""
    rounds = config["training"]["rounds"]
    per_round = config["training"]["optimizer_steps_per_round"]
    return [0] + [(index + 1) * per_round for index in range(rounds)]


def landed(outdir: Path) -> list[tuple[int, float]]:
    """(step, completion time) for every report marker, oldest first."""
    out = []
    for path in (outdir / "stage-completion").glob("step-*.report.json"):
        step = int(path.name.split("-")[1].split(".")[0])
        out.append((step, path.stat().st_mtime))
    return sorted(out)


def check_prefix(expected: list[int], seen: list[int]) -> None:
    """The self-check.  A ladder that disagrees with the run is not a ladder."""
    if seen != expected[: len(seen)]:
        raise SystemExit(
            "the reports that landed are not a prefix of the derived ladder; "
            f"derived {expected[: len(seen)]}, landed {seen}. "
            "Either the config changed mid-run or this derivation is wrong -- "
            "do not project until that is settled."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v4_decoupling_main_20260908.yaml")
    ap.add_argument("--outdir", default="runs/v4/decoupling-main-20260908")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="defaults to the config's pilot.max_wall_hours")
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    outdir = ROOT / args.outdir
    state = json.loads((outdir / "state.json").read_text(encoding="utf-8"))
    started = state["started_unix"]
    budget = args.budget_hours or config["pilot"]["max_wall_hours"]

    steps = ladder(config)
    reports = landed(outdir)
    check_prefix(steps, [s for s, _ in reports])

    remaining = len(steps) - len(reports)
    print(f"ladder      {steps[0]} .. {steps[-1]}  ({len(steps)} reports, derived from "
          f"rounds={config['training']['rounds']} x "
          f"{config['training']['optimizer_steps_per_round']} steps)")
    print(f"landed      {len(reports)}  -> last step-{reports[-1][0]:05d} at "
          f"{datetime.fromtimestamp(reports[-1][1]):%m-%d %H:%M:%S}")
    print(f"remaining   {remaining}  -> {', '.join(f'step-{s:05d}' for s in steps[len(reports):]) or '(none)'}")

    # Round lengths.  Index 0 carries two reports and one train, so step-0 ->
    # step-8 is not a full round; full rounds start at the step-8 -> step-16 gap.
    gaps = [(b[0], (b[1] - a[1]) / 3600) for a, b in zip(reports, reports[1:])]
    full = [g for step, g in gaps[1:]]
    print("\nround lengths (report -> report; the first gap is not a full round)")
    for step, hours in gaps:
        tag = "   <- index 0 carries two reports, no train between them" if step == 8 else ""
        print(f"  -> step-{step:05d}  {hours:5.2f} h{tag}")

    elapsed = (reports[-1][1] - started) / 3600
    left = budget - elapsed
    print(f"\nelapsed {elapsed:.2f} h of {budget:g}   left {left:.2f} h   "
          f"{remaining} rounds   budget {left / remaining:.2f} h/round" if remaining else "")
    if not remaining or not full:
        return
    rows = [("worst full round", max(full)),
            ("mean of the last four", statistics.mean(full[-4:])),
            ("mean of every full round", statistics.mean(full)),
            ("fastest full round", min(full))]
    print(f"{'pace':26s} {'h/round':>8} {'finish':>15} {'elapsed':>9} {'margin':>8}")
    for name, pace in rows:
        finish = reports[-1][1] + remaining * pace * 3600
        used = (finish - started) / 3600
        print(f"{name:26s} {pace:8.2f} {datetime.fromtimestamp(finish):%m-%d %H:%M} "
              f"{used:9.2f} {budget - used:+8.2f}")
    print(f"\nstop line  {datetime.fromtimestamp(started + budget * 3600):%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
