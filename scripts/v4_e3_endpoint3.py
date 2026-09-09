"""E3 endpoint 3, final analysis: is the selection gain a dose of the context effect?

One point per (seed, arm, checkpoint), x the context effect on conflict trials
and y the selection gain from scoring candidates blind. One fit over all of
them, with the cluster bootstrap resampling seeds and then checkpoints within
a seed (deviation 9.4 moved the cluster unit from checkpoint to (seed,
checkpoint); both arms of a drawn checkpoint travel together).

    envs/core/python.exe scripts/v4_e3_endpoint3.py --outdir review-packets/e3-endpoint3

Secondary, and the registered downgrade is as binding as the registered
prediction: if the 2.5th percentile of the slope draws is not above zero, this
is written up as an unexplained moderation and not as a dose-response. A large
positive point estimate with an interval spanning zero takes the downgrade
too -- "encouraging" is not a clause the pre-registration contains.

The input is `scripts/v4_e3_selection_observe.py`'s output, which is a
re-measurement rather than anything the run recorded. Deviation 13.2 point 7:
if that pass does not finish, endpoint 3 reads *not done*, and the corpus
replay is explicitly not allowed to stand in for per-checkpoint points. Unlike
deviation 13.1 point 6, which lets endpoint 2 fit a shorter series as long as
the gaps are named, 13.2 gives no partial rule -- so a missing arm-checkpoint
stops the endpoint here rather than shortening it. Exit 2 for the same reason
as endpoint 2: *not done* is a sentence for the paper, not a result to carry
on from, and not an error either.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfsight.analysis.endpoint1 import completed_steps
from selfsight.analysis.endpoint3 import (
    DOWNGRADE,
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    CheckpointPoint,
    available_steps,
    load_point,
    nested_bootstrap,
    verdict,
)

# Deviation 9, the same five as endpoints 1 and 2: 20260908 is spent on the
# bootstrap seed in section 3.
REGISTERED_SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)

NOT_DONE = 2


class NotDone(Exception):
    """Deviation 13.2 point 7's registered outcome, carrying its sentence."""


def points_for(run: Path, seed: int, arms: list[str]) -> list[CheckpointPoint]:
    """Every adjudicated arm-checkpoint of one replicate, or NotDone.

    The ladder is what the run adjudicated: a checkpoint with verdicts is a
    checkpoint endpoint 3 expects a point at. Missing ones are the pass not
    having finished, and 13.2 point 7 spends the whole endpoint on that rather
    than fitting whatever exists.
    """

    collected: list[CheckpointPoint] = []
    for arm in arms:
        expected = set(completed_steps(run, arm))
        have = set(available_steps(run, arm))
        stray = sorted(have - expected)
        if stray:
            raise NotDone(f"{run.name} {arm}: selection pass at step(s) {stray} with no "
                          f"adjudicated verdicts to score against")
        if not have:
            raise NotDone(f"{run.name} {arm}: no selection pass; run "
                          f"scripts/v4_e3_selection_observe.py")
        missing = sorted(expected - have)
        if missing:
            raise NotDone(f"{run.name} {arm}: no selection pass at step(s) {missing}; "
                          f"deviation 13.2 gives no rule for fitting the rest")
        for step in sorted(have):
            try:
                collected.append(load_point(run, arm, step, seed=seed))
            except ValueError as exc:
                raise NotDone(f"{run.name} {arm} step {step}: {exc}") from exc
    return collected


def report(points: list[CheckpointPoint], result, *, arms: list[str],
           resamples: int, bootstrap_seed: int, confirmatory: bool) -> str:
    lines = ["E3 endpoint 3: selection gain against the context effect", ""]
    lines.append("x = blind minus prompted `correct` on conflict trials "
                 "(deviation 13.2 point 3)")
    lines.append("y = pick(pool, blind) minus pick(pool, prompted), uniform tie-break "
                 "(point 4)")
    lines.append(f"cluster bootstrap {resamples} resamples, seed {bootstrap_seed}, "
                 f"seeds then checkpoints within a seed (point 5)")
    lines.append(f"arms {arms}, both inside each (seed, checkpoint) cluster")
    lines.append("")
    lines.append(f"{'seed':>9} {'arm':<11} {'step':>5} {'x':>9} {'y':>9} {'y first':>9} "
                 f"{'pools':>6} {'conflict':>9}")
    for point in sorted(points, key=lambda item: (item.seed, item.arm, item.step)):
        conflicts = (point.conflict_trials.get("blind_trials", 0)
                     + point.conflict_trials.get("prompted_trials", 0))
        lines.append(f"{point.seed:>9} {point.arm:<11} {point.step:>5} "
                     f"{point.context_effect:>+9.4f} {point.selection_gain:>+9.4f} "
                     f"{point.selection_gain_first_index:>+9.4f} {point.pools:>6} "
                     f"{conflicts:>9}")
    lines.append("")
    lines.append(f"slope {result.slope:+.5f}  95% CI "
                 f"[{result.interval[0]:+.5f}, {result.interval[1]:+.5f}]")
    lines.append(f"{result.points} points, {result.seeds} seeds, {result.clusters} clusters, "
                 f"{result.discarded_resamples} degenerate resamples discarded")
    lines.append("")
    if result.dose_response:
        lines.append("SUPPORTED: " + result.wording)
    else:
        lines.append("DOWNGRADED: " + result.wording)
        lines.append("The registered downgrade is not a hedge to be softened. A positive")
        lines.append("point estimate whose interval spans zero takes it as well.")
    if not confirmatory:
        lines.append("")
        lines.append(f"NOT THE REGISTERED ANALYSIS: it takes seeds "
                     f"{list(REGISTERED_SEEDS)}. Exploratory only.")
    return "\n".join(lines) + "\n"


def write_not_done(outdir: Path, reasons: list[str], *, arms: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    text = ("E3 endpoint 3: NOT DONE\n\n"
            "Deviation 13.2 point 7: if the selection pass does not finish, endpoint 3\n"
            "is judged not done and reported as such. The corpus runs' offline replay\n"
            "is not a substitute for per-checkpoint points and is not printed here.\n\n"
            + "".join(f"  {reason}\n" for reason in reasons))
    (outdir / "endpoint3.txt").write_text(text, encoding="utf-8")
    (outdir / "endpoint3.json").write_text(json.dumps({
        "endpoint": "E3 endpoint 3: selection gain vs context effect",
        "status": "not_done", "registered_by": "deviation 13.2 point 7",
        "arms": arms, "reasons": reasons,
        "substituted_corpus_replay": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+",
                        help="Replicate run directories; default runs/v4/e3-s<seed>")
    parser.add_argument("--arms", nargs="+", default=["naive", "blind_self"])
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=FINAL_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=FINAL_BOOTSTRAP_SEED)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Run with fewer than five seeds and label the result exploratory")
    args = parser.parse_args()

    if args.runs:
        pairs = [(int(run.name.rsplit("s", 1)[-1]) if run.name.rsplit("s", 1)[-1].isdigit()
                  else 0, run) for run in args.runs]
    else:
        pairs = [(seed, ROOT / f"runs/v4/e3-s{seed}") for seed in REGISTERED_SEEDS]
    present = [(seed, run) for seed, run in pairs if run.is_dir()]
    if not present:
        raise SystemExit(f"No replicate run directories: {[str(run) for _, run in pairs]}")
    confirmatory = sorted(seed for seed, _ in present) == sorted(REGISTERED_SEEDS)
    if not confirmatory and not args.allow_partial:
        raise SystemExit(
            f"Endpoint 3 is registered over seeds {list(REGISTERED_SEEDS)}; present are "
            f"{sorted(seed for seed, _ in present)}. Pass --allow-partial to look anyway.")

    points: list[CheckpointPoint] = []
    reasons: list[str] = []
    for seed, run in present:
        try:
            points.extend(points_for(run, seed, args.arms))
        except NotDone as exc:
            reasons.append(str(exc))
    if reasons:
        # One seed short is the whole endpoint short, as it is for endpoint 2:
        # the registered analysis is over the five, and reporting the seeds
        # that did finish would be inventing a rule 13.2 does not contain.
        write_not_done(args.outdir, reasons, arms=args.arms)
        return NOT_DONE

    draws = nested_bootstrap(points, resamples=args.resamples, seed=args.bootstrap_seed)
    result = verdict(points, draws)
    text = report(points, result, arms=args.arms, resamples=args.resamples,
                  bootstrap_seed=args.bootstrap_seed, confirmatory=confirmatory)
    print(text)

    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoint": "E3 endpoint 3: selection gain vs context effect",
        "status": "done",
        "confirmatory": confirmatory,
        "registered_seeds": list(REGISTERED_SEEDS),
        "arms": args.arms,
        "conditions": ["blind", "prompted"],
        "bootstrap": {"resamples": args.resamples, "seed": args.bootstrap_seed,
                      "unit": "(seed, checkpoint)", "nesting": "seeds, then checkpoints",
                      "registered_by": "deviation 13.2 point 5"},
        "downgrade_wording": DOWNGRADE,
        "verdict": {**asdict(result), "interval": list(result.interval)},
        "points": [{**asdict(point), "cluster": list(point.cluster)} for point in points],
    }
    (args.outdir / "endpoint3.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "endpoint3.txt").write_text(text, encoding="utf-8")
    print(f"wrote {args.outdir / 'endpoint3.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
