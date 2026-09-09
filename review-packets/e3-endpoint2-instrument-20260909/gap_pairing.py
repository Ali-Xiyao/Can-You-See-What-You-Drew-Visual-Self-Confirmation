"""What endpoint 2 measures, and three things wrong with measuring it that way.

Endpoint 2 (PREREG, E3) registers the blind discrimination gap: the amount by
which `s_select` is higher on images that turned out externally correct than
on images that turned out wrong, with the generating prompt removed from the
question. Its 口径 pointer is `review-packets/dg-power-20260908/prompt_recital.py`.

Running that pointer against the runs it names turns up three separate
problems. This script is the evidence for all three; it reads only saved
artifacts, touches no GPU, and does not write into any run directory.

    envs/core/python.exe review-packets/e3-endpoint2-instrument-20260909/gap_pairing.py

D1  the label is taken from the wrong draw
    `s_select` is recorded once per prompt, on candidate 0. prompt_recital.py
    builds `correct[spec_id]` by iterating verified.jsonl into a dict, so with
    R=4 the surviving value is candidate 3's verdict. The pilot ran R=1 and
    never noticed. The main run runs R=4.

D2  the pilot trajectory quoted in the endpoint 2 hypothesis is not what the
    quoted script prints
    PREREG endpoint 2 says "pilot 已观测:+0.047 → 0.000,32 步内,两臂皆然",
    citing prompt_recital.py through
    `review-packets/context-ablation-20260908/RESULTS.md`. The script prints a
    gap that starts at +0.070 and ends at +0.047, for both arms. It never
    reaches 0.000. The pilot's verified.jsonl files were last written
    2026-09-06 and RESULTS.md 2026-09-08, so the data did not move underneath
    it.

D4  a (prompt, draw) can be absent rather than unadjudicated
    `verified.summary.json` counts `skipped_no_detection`: an image the
    detector returned nothing for produces no verified.jsonl row at all.
    Deviation 10 registered paired deletion for rows that are *present and*
    *unadjudicated*, and `endpoint1.load_checkpoint` raises on a key one arm
    has and the other does not. It has already happened once, at rfo_gold
    step 16 of the main run.

D3  the run does not record the blind condition at all
    `src/selfsight/v4/evaluate.py` calls `observe_naive` for every arm, which
    wraps each question in PROMPTED_PREAMBLE. So the per-checkpoint s_select
    is the prompted score for naive, for rfo_gold, and for blind_self when it
    runs. Endpoint 2 wants the blind one. It is not in the artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "runs/v4/decoupling-pilot-20260906"
MAIN = ROOT / "runs/v4/decoupling-main-20260908"


def read(run: Path, arm: str, step: int):
    directory = run / "evaluations" / arm / f"step-{step:05d}"
    if not (directory / "s_select.jsonl").exists():
        return None
    if not (directory / "verified.jsonl").exists():
        return None
    select = {}
    for line in (directory / "s_select.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            select[row["prompt_id"]] = float(row["s_select"])
    verified = [json.loads(line) for line
                in (directory / "verified.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return select, verified


def label(verified, mode: str) -> dict[str, bool]:
    """The two ways of turning 4 draws into one verdict per prompt."""

    if mode == "candidate0":
        return {row["spec_id"]: bool(row["image_correct"])
                for row in verified if int(row["candidate_index"]) == 0}
    # What a dict comprehension over every row leaves behind: the last one.
    out = {}
    for row in verified:
        out[row["spec_id"]] = bool(row["image_correct"])
    return out


def gap(select, labels) -> tuple[float, int, int]:
    shared = sorted(set(select) & set(labels))
    right = np.array([select[key] for key in shared if labels[key]])
    wrong = np.array([select[key] for key in shared if not labels[key]])
    if not len(right) or not len(wrong):
        return float("nan"), len(right), len(wrong)
    return float(right.mean() - wrong.mean()), len(right), len(wrong)


def table(run: Path, steps) -> None:
    print(f"{'arm':10s}{'step':>5}{'gap cand0':>12}{'gap last row':>14}"
          f"{'difference':>12}{'n right':>9}{'n wrong':>9}{'draws':>7}")
    for arm in ("naive", "rfo_gold"):
        for step in steps:
            got = read(run, arm, step)
            if got is None:
                continue
            select, verified = got
            counts = {}
            for row in verified:
                counts[row["spec_id"]] = counts.get(row["spec_id"], 0) + 1
            draws = (f"{min(counts.values())}-{max(counts.values())}"
                     if min(counts.values()) != max(counts.values())
                     else str(max(counts.values())))
            first, right_n, wrong_n = gap(select, label(verified, "candidate0"))
            last, _, _ = gap(select, label(verified, "lastrow"))
            print(f"{arm:10s}{step:>5}{first:>+12.4f}{last:>+14.4f}"
                  f"{first - last:>+12.4f}{right_n:>9}{wrong_n:>9}{draws:>7}")


def main() -> None:
    print(__doc__.split("    envs/core")[0].strip())
    print()

    print("=" * 78)
    print("D1/D2  pilot, R=1 -- the two pairings cannot differ, and none of it")
    print("       collapses to zero (PREREG endpoint 2 says +0.047 -> 0.000)")
    print("=" * 78)
    table(PILOT, (0, 8, 16, 24, 32))

    print()
    print("=" * 78)
    print("D1  main run, R=4 -- the two pairings differ, by more than the effect")
    print("    endpoint 1 was powered to detect (+0.042)")
    print("=" * 78)
    table(MAIN, (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88))

    print()
    print("=" * 78)
    print("D3  which observation condition the artifacts hold")
    print("=" * 78)
    evaluate = (ROOT / "src/selfsight/v4/evaluate.py").read_text(encoding="utf-8")
    print(f"  evaluate.py calls observe_naive : {evaluate.count('observe_naive(')} time(s)")
    print(f"  evaluate.py calls observe_rfo   : {evaluate.count('observe_rfo(')} time(s)")
    print("  observe_naive wraps every question in PROMPTED_PREAMBLE, so the")
    print("  per-checkpoint s_select is the prompted score for every arm.")
    print("  The blind gap endpoint 2 registers is not in any run directory.")
    print()
    print("  Recoverable offline: the images and the per-round adapters are both")
    print("  saved, so the blind pass is a re-observation and needs no retraining.")
    print()
    print("=" * 78)
    print("D4  how many rows are absent, and how many are present but unadjudicated")
    print("=" * 78)
    print(f"{'run':26s}{'arm':10s}{'step':>5}{'rows':>6}{'skipped':>9}"
          f"{'pending':>9}{'unnameable':>12}")
    for run in (PILOT, MAIN):
        for arm in ("naive", "rfo_gold"):
            for step in range(0, 96, 8):
                path = (run / "evaluations" / arm / f"step-{step:05d}"
                        / "verified.summary.json")
                if not path.exists():
                    continue
                summary = json.loads(path.read_text(encoding="utf-8"))
                print(f"{run.name:26s}{arm:10s}{step:>5}{summary['n']:>6}"
                      f"{summary.get('skipped_no_detection', 0):>9}"
                      f"{summary.get('pending_human', 0):>9}"
                      f"{summary.get('unnameable', 0):>12}")
    print()
    print("  A skipped image is not missing at random: the detector found nothing")
    print("  in it. Deleting it silently removes a likely-failed generation from")
    print("  the denominator, and the arms need not skip the same ones.")

    print()
    for run in (PILOT, MAIN):
        checkpoints = sorted((run / "checkpoints/naive").glob("round-*"))
        images = sorted((run / "evaluations/naive").glob("step-*/images"))
        print(f"    {run.name}: {len(checkpoints)} adapter(s), "
              f"{len(images)} image directory(ies)")


if __name__ == "__main__":
    main()
