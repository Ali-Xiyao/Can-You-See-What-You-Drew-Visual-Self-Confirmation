"""Deviation 7.2's drift bound and deviation 7.3's round bootstrap, per replicate.

    envs/core/python.exe scripts/v4_e3_drift.py \
        --endpoint1 review-packets/e3-endpoint1/endpoint1.json \
        --outdir review-packets/e3-drift

`naive` is trained once in the main run and once in every replicate, so the
gap between those columns bounds how much of an A-B difference could be
run-to-run drift. Deviation 7.2 registers the comparison and both of its
consequences; deviation 14 pins the parts it left open.

Two gates, and neither is this script's opinion. Deviation 14.6 reads
`endpoint1.json` rather than recomputing anything: if the five-seed sign test
did not confirm endpoint 1, the whole analysis is out of force and this exits
2; a replicate whose own endpoint 1 was not detected gets a row saying failure
condition 1 has taken over, and no verdict.

Deviation 14.3 forbids the obvious next step. Five replicates give five
verdicts and no aggregate rule is registered, so this prints all five and the
count, and does not decide anything from the count. Borrowing deviation 9.4's
4-of-5 would be inventing a test after the data existed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfsight.analysis.drift import (
    FINAL_BOOTSTRAP_SEED,
    FINAL_RESAMPLES,
    NOT_SEPARABLE,
    SEPARABLE,
    load_columns,
    round_bootstrap,
)
from selfsight.analysis.endpoint1 import completed_steps, load_checkpoint, paired_difference

REGISTERED_SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)

OUT_OF_FORCE = 2

NOT_APPLICABLE = ("not applicable: endpoint 1 was not detected here, so failure "
                  "condition 1 has taken over (deviation 14.6)")


class OutOfForce(Exception):
    """Deviation 14.6's study-level gate."""


def endpoint1_gate(path: Path) -> tuple[bool, dict[int, bool]]:
    """(the sign test confirmed, per-seed detected). Read, never recomputed."""

    if not path.exists():
        raise OutOfForce(f"No endpoint 1 verdict at {path}; deviation 7.2 only applies "
                         f"once endpoint 1 is judged, and it is not judged here")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # All three endpoint drivers write an "endpoint" line, and their outputs sit
    # side by side under review-packets/ with names one character apart. Being
    # handed the wrong one must not come out as out of force: that exits 2 and
    # records a legitimate-looking non-result for a rule nobody evaluated.
    label = str(payload.get("endpoint", ""))
    if "endpoint 1" not in label:
        raise SystemExit(f"{path} is not endpoint 1: it says {label!r}. Deviation 7.2 "
                         f"reads endpoint 1, and reading another endpoint here would "
                         f"be a different quantity under the same name")
    # Endpoint 1 has no not-done path of its own -- its early exits happen
    # before anything is written, so a file that exists is a complete one.
    # This stays for a truncated or foreign file that got past the line above.
    if payload.get("status") == "not_done":
        raise OutOfForce("endpoint 1 reads not done")
    confirmed = bool(payload["sign_test"]["confirmed"])
    detected = {int(seed["seed"]): bool(seed["verdict"]["detected"])
                for seed in payload["seeds"]}
    return confirmed, detected


def round_differences(run: Path, arm_a: str, arm_b: str) -> dict[int, float]:
    """Endpoint 1's paired difference at every checkpoint both arms adjudicated."""

    steps = sorted(set(completed_steps(run, arm_a)) & set(completed_steps(run, arm_b)))
    return {step: paired_difference(load_checkpoint(run, step, arm_a, arm_b))
            for step in steps}


def analyse(main: Path, replicate: Path, seed: int, *, arm_a: str, arm_b: str,
            detected: bool, resamples: int, bootstrap_seed: int) -> dict:
    columns = load_columns(main, replicate, arm_a=arm_a, arm_b=arm_b)
    rounds = round_bootstrap(round_differences(replicate, arm_a, arm_b),
                             resamples=resamples, seed=bootstrap_seed)
    row = {"seed": seed, "run": str(replicate), "endpoint1_detected": detected,
           **asdict(columns), "keys": len(columns.keys),
           "drift": columns.drift, "effect": columns.effect,
           "rounds": {**asdict(rounds), "interval": list(rounds.interval),
                      "spans_zero": rounds.spans_zero,
                      "excluded_step": 0, "registered_by": "deviation 14.5"}}
    if detected:
        row["separable"] = columns.separable
        row["wording"] = columns.wording
    else:
        row["separable"] = None
        row["wording"] = NOT_APPLICABLE
    return row


def report(rows: list[dict], *, confirmatory: bool, confirmed: bool) -> str:
    lines = ["Deviation 7.2: is the A-B difference bigger than run-to-run drift?", ""]
    lines.append("theta_A is the main run's naive column; theta_A' and theta_B are the")
    lines.append("replicate's. All three are rates on the keys adjudicated in all three")
    lines.append("columns (deviation 14.2), at the last step all three reached (14.4).")
    lines.append("")
    head = ("seed", "step", "theta_A", "theta_A'", "theta_B", "drift", "effect", "keys")
    lines.append(f"{head[0]:>9} {head[1]:>5} {head[2]:>9} {head[3]:>9} {head[4]:>9} "
                 f"{head[5]:>8} {head[6]:>8} {head[7]:>5}  verdict")
    for row in rows:
        if row["separable"] is None:
            mark = "n/a"
        else:
            mark = "separable" if row["separable"] else "NOT separable"
        lines.append(f"{row['seed']:>9} {row['step']:>5} {row['theta_a']:>9.4f} "
                     f"{row['theta_a_prime']:>9.4f} {row['theta_b']:>9.4f} "
                     f"{row['drift']:>8.4f} {row['effect']:>+8.4f} {row['keys']:>5}  {mark}")
    lines.append("")
    lines.append("`drift` is |theta_A - theta_A'| and `effect` is theta_B - theta_A',")
    lines.append("signed: a B below A' is not an effect this bound could clear.")
    lines.append("")
    for row in rows:
        lines.append(f"  seed {row['seed']}: {row['wording']}")
        lines.append(f"    keys dropped reaching the shared set "
                     f"{row['dropped']}; marginal rates "
                     + ", ".join(f"{name} {value:.4f}"
                                 for name, value in sorted(row["marginal"].items())))
    lines.append("")
    applicable = [row for row in rows if row["separable"] is not None]
    separable = [row for row in applicable if row["separable"]]
    lines.append(f"{len(separable)} of {len(applicable)} applicable replicates clear the "
                 f"drift bound.")
    lines.append("No aggregate rule is registered for this count (deviation 14.3). It is")
    lines.append("reported and not tested; a majority rule borrowed from deviation 9.4")
    lines.append("would be a test invented after the data existed.")
    if applicable and len(separable) not in (0, len(applicable)):
        lines.append("")
        lines.append("The replicates disagree, so the paper must say that the relative size")
        lines.append("of the drift bound and the A-B difference is inconsistent across")
        lines.append("replicates, and must not pick one as representative (deviation 14.3).")
    lines.append("")
    lines.append("Deviation 7.3, the round axis -- descriptive, no failure condition")
    for row in rows:
        rounds = row["rounds"]
        lines.append(f"  seed {row['seed']}: mean {rounds['mean']:+.4f} over "
                     f"{len(rounds['steps'])} rounds, 95% CI "
                     f"[{rounds['interval'][0]:+.4f}, {rounds['interval'][1]:+.4f}]"
                     + ("  (spans 0)" if rounds["spans_zero"] else ""))
    lines.append("  Step 0 is excluded: both arms are the same model on the same latents")
    lines.append("  there and their verdicts are bit-identical (deviation 14.5).")
    if not confirmatory:
        lines.append("")
        lines.append(f"NOT THE REGISTERED SET: it takes seeds {list(REGISTERED_SEEDS)}. "
                     f"Exploratory only (deviation 14.6b).")
    if not confirmed:
        lines.append("")
        lines.append("Endpoint 1's five-seed sign test did not confirm, so none of the")
        lines.append("above supports the training-period claim (deviation 14.6).")
    return "\n".join(lines) + "\n"


def write_out_of_force(outdir: Path, reason: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    text = ("Deviation 7.2: OUT OF FORCE\n\n"
            "7.2 applies only once endpoint 1 is judged detected. Deviation 14.6 reads\n"
            "that judgement out of endpoint1.json rather than recomputing it.\n\n"
            f"  {reason}\n")
    (outdir / "drift.txt").write_text(text, encoding="utf-8")
    (outdir / "drift.json").write_text(json.dumps({
        "analysis": "deviation 7.2 run-to-run drift bound",
        "status": "out_of_force", "registered_by": "deviation 14.6",
        "reason": reason,
    }, indent=2) + "\n", encoding="utf-8")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main", type=Path,
                        default=ROOT / "runs/v4/decoupling-main-20260908")
    parser.add_argument("--runs", type=Path, nargs="+",
                        help="Replicate run directories; default runs/v4/e3-s<seed>")
    parser.add_argument("--endpoint1", type=Path, required=True,
                        help="endpoint1.json, whose verdict decides whether 7.2 applies")
    parser.add_argument("--arm-a", default="naive")
    parser.add_argument("--arm-b", default="blind_self")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=FINAL_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=FINAL_BOOTSTRAP_SEED)
    args = parser.parse_args()

    try:
        confirmed, detected = endpoint1_gate(args.endpoint1)
    except OutOfForce as exc:
        write_out_of_force(args.outdir, str(exc))
        return OUT_OF_FORCE
    if not confirmed:
        write_out_of_force(args.outdir,
                           "endpoint 1's five-seed sign test did not confirm; the "
                           "training-period claim is already out of force")
        return OUT_OF_FORCE

    if args.runs:
        pairs = [(int(run.name.rsplit("s", 1)[-1]) if run.name.rsplit("s", 1)[-1].isdigit()
                  else 0, run) for run in args.runs]
    else:
        pairs = [(seed, ROOT / f"runs/v4/e3-s{seed}") for seed in REGISTERED_SEEDS]
    present = [(seed, run) for seed, run in pairs if run.is_dir()]
    if not present:
        raise SystemExit(f"No replicate run directories: {[str(run) for _, run in pairs]}")
    confirmatory = sorted(seed for seed, _ in present) == sorted(REGISTERED_SEEDS)

    rows = [analyse(args.main, run, seed, arm_a=args.arm_a, arm_b=args.arm_b,
                    detected=detected.get(seed, False), resamples=args.resamples,
                    bootstrap_seed=args.bootstrap_seed)
            for seed, run in present]
    text = report(rows, confirmatory=confirmatory, confirmed=confirmed)
    print(text)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "drift.json").write_text(json.dumps({
        "analysis": "deviation 7.2 run-to-run drift bound, with deviation 7.3's rounds",
        "status": "done",
        "confirmatory": confirmatory,
        "registered_seeds": list(REGISTERED_SEEDS),
        "main_run": str(args.main),
        "arms": {"a": args.arm_a, "b": args.arm_b},
        "wordings": {"separable": SEPARABLE, "not_separable": NOT_SEPARABLE,
                     "not_applicable": NOT_APPLICABLE},
        "aggregate_rule": "none registered (deviation 14.3); the count is reported, "
                          "not tested",
        "replicates": rows,
    }, indent=2) + "\n", encoding="utf-8")
    (args.outdir / "drift.txt").write_text(text, encoding="utf-8")
    print(f"wrote {args.outdir / 'drift.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
