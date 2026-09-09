"""E3 endpoint 1, final analysis across the five registered replicates.

Two layers, and the pre-registration is explicit that they answer different
questions. Within a seed, prompts are the unit and the answer is how large
the effect was in that run. Across seeds, the seed is the unit and the answer
is whether the direction held in all five. Deviation 9.3 forbids mixing them
-- in particular it forbids bootstrapping the five seed-level differences,
because a resample distribution over five atoms is a near relative of the
range, and this pipeline has already measured that estimator's type-I error
at 9.5% against a nominal 5%.

    envs/core/python.exe scripts/v4_e3_endpoint1.py --outdir review-packets/e3-endpoint1

Reads runs/v4/e3-s* by default. Writes endpoint1.json and a text report that
carries the coverage numbers deviation 10.2 requires beside every estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfsight.analysis.endpoint1 import (
    DETECTABLE_EFFECT,
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    SeedResult,
    between_seed_spread,
    bootstrap_interval,
    completed_steps,
    exact_mcnemar,
    load_checkpoint,
    paired_difference,
    sign_test,
    verdict,
)

# Deviation 9: the five seed values were fixed in advance, 20260908 skipped
# because section 3 had already spent it.
REGISTERED_SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)


def analyse_run(run: Path, seed: int, arm_a: str, arm_b: str,
                *, resamples: int, bootstrap_seed: int) -> SeedResult:
    steps = completed_steps(run, arm_a)
    if not steps:
        raise SystemExit(f"{run}: no adjudicated checkpoint for arm {arm_a}")
    trajectory = []
    for step in steps:
        checkpoint = load_checkpoint(run, step, arm_a, arm_b)
        trajectory.append((step, paired_difference(checkpoint)))
    final = load_checkpoint(run, steps[-1], arm_a, arm_b)
    point = paired_difference(final)
    low, high = bootstrap_interval(final, resamples=resamples, seed=bootstrap_seed)
    b_only, a_only, p_value = exact_mcnemar(final)
    return SeedResult(
        seed=seed, run=run, step=steps[-1], point=point, ci_low=low, ci_high=high,
        mcnemar_b_only=b_only, mcnemar_a_only=a_only, mcnemar_p=p_value,
        coverage=final.coverage, trajectory=tuple(trajectory))


def report(results: list[SeedResult], *, confirmatory: bool,
           resamples: int, bootstrap_seed: int) -> str:
    lines = ["E3 endpoint 1: final external correctness, B over A", ""]
    lines.append(f"bootstrap {resamples} resamples, seed {bootstrap_seed} "
                 f"(deviation 6.4; the per-checkpoint report runs 2000 at 20260906)")
    lines.append(f"detectable effect fixed in advance at +{DETECTABLE_EFFECT}")
    lines.append("")
    lines.append("Within seed -- prompt is the unit, prompts are the bootstrap cluster")
    lines.append(f"{'seed':>9} {'step':>5} {'B-A':>8} {'95% CI':>20} "
                 f"{'McNemar p':>10} {'pairs':>9} {'verdict':>14}")
    for result in results:
        coverage = result.coverage
        flag = " LOW" if coverage.low else ""
        decision = verdict(result.point, result.ci_low, result.ci_high)
        lines.append(
            f"{result.seed:>9} {result.step:>5} {result.point:>+8.4f} "
            f"[{result.ci_low:>+7.4f},{result.ci_high:>+7.4f}] "
            f"{result.mcnemar_p:>10.4g} {coverage.pairs_kept:>4}/{coverage.pairs_total:<4}"
            f"{flag} {'detected' if decision.detected else 'NOT DETECTED':>14}")
    lines.append("")
    lines.append("McNemar treats each (prompt, draw) as independent and they are not;")
    lines.append("the bootstrap interval is the clustered one. Both were registered.")
    lines.append("")
    for result in results:
        coverage = result.coverage
        lines.append(
            f"  seed {result.seed}: {coverage.pairs_kept}/{coverage.pairs_total} pairs kept, "
            f"{coverage.prompts_kept}/{coverage.prompts_total} prompts, "
            f"unadjudicated {coverage.unadjudicated_a} in A / {coverage.unadjudicated_b} in B, "
            f"skipped {coverage.skipped_a} in A / {coverage.skipped_b} in B")
        if coverage.skipped_a or coverage.skipped_b:
            # Deviation 12.2 point 5: a skip is the detector finding nothing in
            # the picture, which is not missing at random and need not be
            # symmetric between the arms.
            lines.append("    ^ skipped rows are not missing at random; "
                         "deviation 12.2 point 5 requires listing them in the appendix")
    lines.append("")

    differences = [result.point for result in results]
    supporting, total, p_value = sign_test(differences)
    spread, ratio = between_seed_spread(results)
    lines.append("Across seeds -- seed is the unit, one-sided exact sign test (deviation 9.3)")
    lines.append(f"  {supporting}/{total} seeds in the registered direction, p = {p_value:.5f}")
    if not confirmatory:
        lines.append(f"  NOT THE REGISTERED TEST: it takes {len(REGISTERED_SEEDS)} seeds, "
                     f"this is {total}. Exploratory only.")
    elif supporting == total:
        lines.append("  confirmed across seeds")
    else:
        lines.append("  NOT confirmed. Deviation 9.3 forbids a rescue: no two-sided test, "
                     "no other test, no seed removed as an outlier.")
    lines.append(f"  between-seed SD (df={max(total - 1, 0)}) = {spread:.4f}"
                 + (f", SD/effect = {ratio:.2f}" if ratio is not None else ", no ratio (mean 0)"))
    lines.append("  the five differences are reported as themselves; deviation 9.3 bans "
                 "bootstrapping them")
    lines.append("  " + ", ".join(f"{value:+.4f}" for value in differences))
    lines.append("")
    lines.append("Per-seed trajectories (deviation 9.3: five curves, no mean band)")
    for result in results:
        drawn = " ".join(f"{step}:{value:+.3f}" for step, value in result.trajectory)
        lines.append(f"  {result.seed}: {drawn}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+",
                        help="Replicate run directories; default runs/v4/e3-s<seed> "
                             "for the five registered seeds")
    parser.add_argument("--arm-a", default="naive")
    parser.add_argument("--arm-b", default="blind_self")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=FINAL_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=FINAL_BOOTSTRAP_SEED)
    # The sign test registered in deviation 9.3 is a five-seed test. Running
    # it on fewer and printing a p-value beside the registered wording is the
    # mistake this flag exists to make deliberate: it stays runnable, because
    # watching the differences accumulate is useful, but the output says
    # exploratory and the JSON records confirmatory=false.
    parser.add_argument("--allow-partial", action="store_true",
                        help="Run with fewer than five seeds and label the result exploratory")
    args = parser.parse_args()

    if args.runs:
        pairs = [(int(run.name.rsplit("s", 1)[-1]) if run.name.rsplit("s", 1)[-1].isdigit()
                  else 0, run) for run in args.runs]
    else:
        pairs = [(seed, ROOT / f"runs/v4/e3-s{seed}") for seed in REGISTERED_SEEDS]
    present = [(seed, run) for seed, run in pairs if run.is_dir()]
    missing = [run for _, run in pairs if not run.is_dir()]
    # Confirmatory means these five seeds, not five of anything. Counting
    # would call a rerun of one seed under five directory names a five-seed
    # replication, which is the failure v4_verify_replicates.py exists to
    # catch on the launch side and this catches on the analysis side.
    confirmatory = sorted(seed for seed, _ in present) == sorted(REGISTERED_SEEDS)
    if not present:
        raise SystemExit(f"No replicate run directories found: {[str(p) for p in missing]}")
    if not confirmatory and not args.allow_partial:
        found = sorted(seed for seed, _ in present)
        raise SystemExit(
            f"Deviation 9.3 registers the sign test over seeds {list(REGISTERED_SEEDS)}; "
            f"present are {found}. Pass --allow-partial to look anyway, and the output "
            f"will be labelled exploratory."
            + (f" Missing directories: {[str(p) for p in missing]}" if missing else ""))

    results = [analyse_run(run, seed, args.arm_a, args.arm_b,
                           resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)
               for seed, run in present]
    text = report(results, confirmatory=confirmatory,
                  resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)
    print(text)

    args.outdir.mkdir(parents=True, exist_ok=True)
    supporting, total, p_value = sign_test([result.point for result in results])
    spread, ratio = between_seed_spread(results)
    payload = {
        "endpoint": "E3 endpoint 1: final external correctness, B > A",
        "confirmatory": confirmatory,
        "registered_seeds": list(REGISTERED_SEEDS),
        "arms": {"a": args.arm_a, "b": args.arm_b},
        "bootstrap": {"resamples": args.resamples, "seed": args.bootstrap_seed,
                      "registered_by": "deviation 6.4"},
        "detectable_effect": DETECTABLE_EFFECT,
        "seeds": [
            {**{key: value for key, value in asdict(result).items()
                if key not in {"run", "coverage", "trajectory"}},
             "run": str(result.run),
             "coverage": asdict(result.coverage),
             "trajectory": [list(point) for point in result.trajectory],
             "verdict": asdict(verdict(result.point, result.ci_low, result.ci_high))}
            for result in results],
        "sign_test": {"supporting": supporting, "total": total, "p_value": p_value,
                      "confirmed": bool(confirmatory and supporting == total),
                      "registered_by": "deviation 9.3"},
        "between_seed": {"sd": spread, "sd_over_effect": ratio, "reported_not_tested": True},
    }
    (args.outdir / "endpoint1.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "endpoint1.txt").write_text(text, encoding="utf-8")
    print(f"wrote {args.outdir / 'endpoint1.json'}")


if __name__ == "__main__":
    main()
