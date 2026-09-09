"""E3 endpoint 2, final analysis across the five registered replicates.

Does the blind discrimination gap flatten as training goes on, and does arm B
hold it open? Two layers, exactly as endpoint 1 has: within a seed the prompt
is the unit and the bootstrap pairs the two arms on one resampled prompt
multiset; across seeds the seed is the unit, and deviation 9.4 asks for A's
slope to be significantly negative in at least 4 of the 5 *and* a one-sided
exact sign test on the five slope differences. Deviation 9.3 forbids
bootstrapping those five.

    envs/core/python.exe scripts/v4_e3_endpoint2.py --outdir review-packets/e3-endpoint2

The input is the `image_only` pass written by `scripts/v4_e3_blind_observe.py`,
not anything the run recorded. Deviation 11.3 is explicit that the prompted
gap is not a substitute, so when that pass has not run this exits 2 with
`status: "not_done"` and prints no slope at all. Exit 2 rather than 0 because
*not done* is a sentence for the paper, not a result a wrapper should carry
on from; and rather than 1 because it is not an error either.
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
from selfsight.analysis.endpoint2 import (
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    MIN_CHECKPOINTS_PER_FIT,
    SeedVerdict,
    across_seeds,
    available_steps,
    load_series,
    paired_bootstrap,
    seed_verdict,
)

# Deviation 9: fixed in advance, 20260908 skipped because section 3 spent it.
REGISTERED_SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)

NOT_DONE = 2


class NotDone(Exception):
    """Deviation 11.3's registered outcome, carrying the sentence to print."""


def steps_for(run: Path, arm_a: str, arm_b: str) -> tuple[list[int], list[int]]:
    """(fit these, could not get these).

    The ladder endpoint 2 *should* have is the set of checkpoints both arms
    adjudicated, because that is what makes a labelled gap possible. What it
    has is where the blind pass also finished. Deviation 13.1 point 6 wants
    the difference named rather than quietly absorbed into a shorter fit.
    """

    expected = set(completed_steps(run, arm_a)) & set(completed_steps(run, arm_b))
    have = set(available_steps(run, arm_a)) & set(available_steps(run, arm_b))
    stray = sorted(have - expected)
    if stray:
        raise NotDone(f"{run.name}: blind pass at step(s) {stray} with no adjudicated "
                      f"verdicts to pair against")
    usable = sorted(have)
    if not usable:
        raise NotDone(f"{run.name}: no image_only pass; run scripts/v4_e3_blind_observe.py")
    if len(usable) < MIN_CHECKPOINTS_PER_FIT:
        raise NotDone(f"{run.name}: image_only pass covers {len(usable)} checkpoint(s), "
                      f"and a slope needs {MIN_CHECKPOINTS_PER_FIT}")
    return usable, sorted(expected - have)


def analyse_run(run: Path, seed: int, arm_a: str, arm_b: str,
                *, resamples: int, bootstrap_seed: int) -> SeedVerdict:
    steps, missing = steps_for(run, arm_a, arm_b)
    series_a = load_series(run, arm_a, steps)
    series_b = load_series(run, arm_b, steps)
    draws = paired_bootstrap(series_a, series_b, resamples=resamples, seed=bootstrap_seed)
    return seed_verdict(series_a, series_b, draws,
                        seed=seed, missing_steps=tuple(missing))


def report(verdicts: list[SeedVerdict], summary, *, arm_a: str, arm_b: str,
           resamples: int, bootstrap_seed: int) -> str:
    lines = ["E3 endpoint 2: the blind discrimination gap against training step", ""]
    lines.append("condition image_only (deviation 11.3); label from candidate 0 "
                 "(deviation 11.1)")
    lines.append(f"bootstrap {resamples} resamples, seed {bootstrap_seed}, one prompt "
                 f"multiset shared by both arms and every checkpoint (deviation 13.1)")
    lines.append("")
    lines.append(f"Within seed -- {arm_a} is A, {arm_b} is B, prompt is the bootstrap unit")
    lines.append(f"{'seed':>9} {'ckpts':>6} {'slope A':>9} {'95% CI A':>20} "
                 f"{'slope B':>9} {'95% CI B-A':>20}  clauses")
    for verdict in verdicts:
        clauses = ("A collapses" if verdict.a_collapses else "A does not collapse")
        clauses += ", " + ("B > A" if verdict.b_exceeds_a else "B not > A")
        lines.append(
            f"{verdict.seed:>9} {len(verdict.steps):>6} {verdict.slope_a:>+9.5f} "
            f"[{verdict.a_interval[0]:>+8.5f},{verdict.a_interval[1]:>+8.5f}] "
            f"{verdict.slope_b:>+9.5f} "
            f"[{verdict.difference_interval[0]:>+8.5f},"
            f"{verdict.difference_interval[1]:>+8.5f}]  {clauses}")
    lines.append("")
    lines.append("Significance is the interval, not the point estimate: A collapses when the")
    lines.append("97.5th percentile of its slope draws is below 0, and B exceeds A when the")
    lines.append("2.5th percentile of the paired difference is above 0 (deviation 13.1).")
    lines.append("")
    for verdict in verdicts:
        lines.append(f"  seed {verdict.seed}: fitted on {len(verdict.steps)} checkpoints "
                     f"{list(verdict.steps)}")
        if verdict.missing_steps:
            lines.append(f"    ^ no blind pass at {list(verdict.missing_steps)}; "
                         "deviation 13.1 point 6 requires listing them, not shortening "
                         "the series silently")
        if verdict.discarded_resamples or verdict.dropped_checkpoints:
            lines.append(f"    degenerate resamples discarded {verdict.discarded_resamples}, "
                         f"checkpoints dropped {verdict.dropped_checkpoints} "
                         "(deviation 13.1 point 4)")
    lines.append("")
    lines.append("Across seeds -- deviation 9.4's two halves, both required")
    lines.append(f"  A's slope significantly negative in {summary.collapsing}/{summary.total} "
                 f"seeds; the rule is 4 of 5 -> {'met' if summary.a_half else 'NOT met'}")
    lines.append(f"  one-sided exact sign test on slope_B - slope_A: "
                 f"{summary.sign_supporting}/{summary.total}, p = {summary.sign_p:.5f} "
                 f"-> {'met' if summary.b_half else 'NOT met'}")
    lines.append("  " + ", ".join(f"{verdict.slope_b - verdict.slope_a:+.5f}"
                                  for verdict in verdicts))
    lines.append("  the five differences are reported as themselves; deviation 9.3 bans "
                 "bootstrapping them")
    if not summary.confirmatory:
        lines.append(f"  NOT THE REGISTERED TEST: it takes {len(REGISTERED_SEEDS)} seeds, "
                     f"this is {summary.total}. Exploratory only.")
    elif summary.confirmed:
        lines.append("  CONFIRMED across seeds")
    else:
        lines.append("  NOT confirmed. Deviation 9.3 forbids a rescue: no two-sided test, "
                     "no other test, no seed removed as an outlier.")
    return "\n".join(lines) + "\n"


def write_not_done(outdir: Path, reasons: list[str], *, arm_a: str, arm_b: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    text = ("E3 endpoint 2: NOT DONE\n\n"
            "Deviation 11.3: if the image_only pass does not run, endpoint 2 is judged\n"
            "not done and reported as such. The prompted gap the run already holds is\n"
            "not a substitute for it and is not printed here.\n\n"
            + "".join(f"  {reason}\n" for reason in reasons))
    (outdir / "endpoint2.txt").write_text(text, encoding="utf-8")
    (outdir / "endpoint2.json").write_text(json.dumps({
        "endpoint": "E3 endpoint 2: blind discrimination gap vs step",
        "status": "not_done", "registered_by": "deviation 11.3",
        "arms": {"a": arm_a, "b": arm_b}, "reasons": reasons,
        "substituted_prompted_gap": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+",
                        help="Replicate run directories; default runs/v4/e3-s<seed>")
    parser.add_argument("--arm-a", default="naive")
    parser.add_argument("--arm-b", default="blind_self")
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
    # Confirmatory means these five seeds, not five of anything -- the same
    # check endpoint 1 makes, for the same reason.
    confirmatory = sorted(seed for seed, _ in present) == sorted(REGISTERED_SEEDS)
    if not confirmatory and not args.allow_partial:
        raise SystemExit(
            f"Deviation 9.4 registers endpoint 2 over seeds {list(REGISTERED_SEEDS)}; present "
            f"are {sorted(seed for seed, _ in present)}. Pass --allow-partial to look anyway.")

    verdicts, reasons = [], []
    for seed, run in present:
        try:
            verdicts.append(analyse_run(run, seed, args.arm_a, args.arm_b,
                                        resamples=args.resamples,
                                        bootstrap_seed=args.bootstrap_seed))
        except NotDone as exc:
            reasons.append(str(exc))
    if reasons:
        # One seed short is the whole endpoint short. Deviation 9.4's test is
        # over five seeds and 13.1 point 7 gives no rule for four, so reporting
        # the seeds that did finish would be inventing one.
        write_not_done(args.outdir, reasons, arm_a=args.arm_a, arm_b=args.arm_b)
        return NOT_DONE

    summary = across_seeds(verdicts)
    text = report(verdicts, summary, arm_a=args.arm_a, arm_b=args.arm_b,
                  resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)
    print(text)

    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoint": "E3 endpoint 2: blind discrimination gap vs step",
        "status": "done",
        "confirmatory": summary.confirmatory,
        "registered_seeds": list(REGISTERED_SEEDS),
        "arms": {"a": args.arm_a, "b": args.arm_b},
        "condition": "image_only",
        "bootstrap": {"resamples": args.resamples, "seed": args.bootstrap_seed,
                      "unit": "prompt", "pairing": "across arms, one multiset per resample",
                      "registered_by": "deviation 13.1"},
        "seeds": [asdict(verdict) for verdict in verdicts],
        "across_seeds": {**asdict(summary), "confirmed": summary.confirmed,
                         "registered_by": "deviation 9.4"},
    }
    (args.outdir / "endpoint2.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "endpoint2.txt").write_text(text, encoding="utf-8")
    print(f"wrote {args.outdir / 'endpoint2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
