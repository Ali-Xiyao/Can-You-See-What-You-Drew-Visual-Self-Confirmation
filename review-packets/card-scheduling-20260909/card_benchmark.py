#!/usr/bin/env python
"""Which card should run detection, measured instead of extrapolated.

`run_decoupling_pilot.py` pins one arm's whole chain to one card, so the
detect stages run on cuda:0 and cuda:1 at the same time and a checkpoint's
wall clock is the slower of the two. STATUS 32's rule says the opposite:
transfer-bound work (detect, observe) belongs on cuda:0, because cuda:1 is
on PCIe gen3 x4. EXECUTION 0.2 extrapolated off single points that the
serial arrangement might be about 28% faster despite doing the work twice
on one card. That extrapolation is not good enough to change a 52 h run on.

This measures it. Four numbers per detector, N images each:

    A   cuda:0 alone
    B   cuda:1 alone
    C   cuda:0 and cuda:1 at the same time  ->  wall_P = max(C0, C1)
                                                wall_S = 2 * A

wall_P is what arm B costs per detect stage today. wall_S is what it would
cost with both arms' detection serialised onto cuda:0.

REGISTERED BEFORE THE MEASUREMENT
Adopt the serial arrangement for arm B if and only if `wall_S <= 0.90 *
wall_P` for BOTH detectors. If the two detectors disagree about the
direction, keep the per-arm pinning: a split verdict is not evidence for
changing a design, and the current one is the one the pilot ran.

The 10% margin is there because N is smaller than the run's 256 and the
benchmark cannot reproduce every source of contention. The comparison is
conservative in the other direction too -- it gives the serial arrangement
no credit for leaving cuda:1 free to generate, which is the larger prize
and the harder thing to measure.

Timing is wall clock of the whole `v4_run_pipeline.py detect` subprocess,
model load included, because that is what the stage costs. Two arms means
two invocations either way, so both regimes pay the load twice.

ALSO CHECKED, AND NOT A TIMING QUESTION
A and B detect the same images on different cards. If their outputs
differ, moving detection between cards changes the measurements and not
just the schedule, and the timing question is moot until that is
understood. Same-card determinism is already established: 48 images
re-detected on cuda:0 on 2026-09-09 reproduced the run's own detections
exactly.

Run it with both cards empty. It refuses otherwise -- a no-load number
measured next to someone else's job is the thing STATUS 32 already has
two of.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DETECTORS = ("internvl", "qwen3vl")
CARDS = ("cuda:0", "cuda:1")
IDLE_MIB = 512
MARGIN = 0.90


def used_mib(index: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True)
    return int(out.strip())


def require_idle() -> None:
    """Both cards empty, or say which one is not and stop.

    Checked once at the start rather than before each measurement: a job
    that appears halfway through invalidates the run either way, and the
    per-image rates in the output are enough to notice it happened.
    """

    busy = {index: used_mib(index) for index in range(len(CARDS))}
    busy = {index: mib for index, mib in busy.items() if mib > IDLE_MIB}
    if busy:
        raise SystemExit(
            "This is a no-load benchmark and "
            + ", ".join(f"cuda:{index} is holding {mib} MiB" for index, mib in busy.items())
            + ". Wait for the card, or measure something else.")


def detect(manifest: Path, detector: str, device: str, out: Path) -> dict:
    """One `v4_run_pipeline.py detect`, timed end to end.

    The environment is the one the supervisor gives its own detect stages
    (run_decoupling_pilot.Pilot.__init__), because a detector that picks a
    different cache or a different thread count is not the detector the run
    uses.
    """

    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [str(ROOT / "envs" / "observer" / "python.exe"), "-u",
               str(ROOT / "scripts" / "v4_run_pipeline.py"), "detect",
               "--manifest", str(manifest), "--detector", detector,
               "--device", device, "--output", str(out), "--overwrite"]
    started = time.time()
    finished = subprocess.run(command, env=env, cwd=ROOT, capture_output=True,
                              text=True, check=False)
    elapsed = time.time() - started
    if finished.returncode != 0:
        raise RuntimeError(f"{detector} on {device} failed:\n{finished.stderr[-2000:]}")
    return {"seconds": elapsed, "device": device, "output": str(out),
            "log": finished.stdout[-4000:]}


def read_detections(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["image_path"]: row.get("detections") for row in rows}


def agree(first: Path, second: Path) -> dict:
    """Do the two cards see the same things in the same pictures?

    Reported per image rather than as a count, because one disagreement is
    already the answer: it would mean the schedule change is a change to the
    measurements.
    """

    left, right = read_detections(first), read_detections(second)
    shared = sorted(set(left) & set(right))
    differing = [image for image in shared
                 if json.dumps(left[image], sort_keys=True)
                 != json.dumps(right[image], sort_keys=True)]
    return {"compared": len(shared), "differing": len(differing),
            "examples": [Path(image).name for image in differing[:5]]}


def measure(manifest: Path, detector: str, outdir: Path) -> dict:
    solo = {}
    for index, device in enumerate(CARDS):
        solo[device] = detect(manifest, detector, device,
                              outdir / f"{detector}.solo.cuda{index}.jsonl")
        print(f"  {detector:9s} solo   {device}  {solo[device]['seconds']:7.1f} s", flush=True)

    with ThreadPoolExecutor(max_workers=len(CARDS)) as pool:
        futures = {device: pool.submit(detect, manifest, detector, device,
                                       outdir / f"{detector}.parallel.cuda{index}.jsonl")
                   for index, device in enumerate(CARDS)}
        together = {device: future.result() for device, future in futures.items()}
    for device, value in together.items():
        print(f"  {detector:9s} shared {device}  {value['seconds']:7.1f} s", flush=True)

    wall_parallel = max(value["seconds"] for value in together.values())
    fastest = min(CARDS, key=lambda device: solo[device]["seconds"])
    wall_serial = 2 * solo[fastest]["seconds"]
    return {
        "detector": detector,
        "solo": {device: value["seconds"] for device, value in solo.items()},
        "concurrent": {device: value["seconds"] for device, value in together.items()},
        "fastest_card_alone": fastest,
        "wall_parallel_one_arm_per_card": wall_parallel,
        "wall_serial_both_on_fastest": wall_serial,
        "serial_is_faster_by": 1 - wall_serial / wall_parallel,
        "cards_agree": agree(Path(solo[CARDS[0]]["output"]), Path(solo[CARDS[1]]["output"])),
    }


def verdict(results: list[dict]) -> dict:
    """The rule in the docstring, applied without looking at anything else."""

    disagreeing = [row["detector"] for row in results if row["cards_agree"]["differing"]]
    if disagreeing:
        return {"adopt_serial": False, "reason":
                f"the two cards do not detect the same things ({', '.join(disagreeing)}); "
                "this is a correctness question before it is a scheduling one"}
    passing = [row["detector"] for row in results
               if row["wall_serial_both_on_fastest"]
               <= MARGIN * row["wall_parallel_one_arm_per_card"]]
    if len(passing) == len(results):
        return {"adopt_serial": True, "reason":
                f"serial clears the {1 - MARGIN:.0%} margin on every detector"}
    return {"adopt_serial": False, "reason":
            f"serial clears the margin on {passing or 'no detector'}, not on all of "
            f"{[row['detector'] for row in results]}; the registered tie-break keeps "
            "the per-arm pinning"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="a finished evaluate manifest; its images are only read")
    parser.add_argument("--images", type=int, default=64,
                        help="prefix of the manifest to time; the per-image rate is "
                             "flat after about 25, so this need not be the run's 256")
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--allow-busy-cards", action="store_true",
                        help="record the result as not a no-load measurement")
    args = parser.parse_args()

    if not args.allow_busy_cards:
        require_idle()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = list(itertools.islice(
        (line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line),
        args.images))
    if len(rows) < args.images:
        raise SystemExit(f"{args.manifest} has {len(rows)} rows, fewer than --images")
    prefix = args.outdir / "manifest.jsonl"
    prefix.write_text("\n".join(rows) + "\n", encoding="utf-8")

    results = [measure(prefix, detector, args.outdir) for detector in args.detectors]
    payload = {"images": args.images, "manifest": str(args.manifest),
               "no_load": not args.allow_busy_cards, "margin": MARGIN,
               "results": results, "verdict": verdict(results)}
    (args.outdir / "card_benchmark.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'detector':10s}{'cuda:0':>9s}{'cuda:1':>9s}{'parallel':>10s}"
          f"{'serial':>9s}{'serial gain':>13s}")
    for row in results:
        print(f"{row['detector']:10s}{row['solo']['cuda:0']:9.1f}{row['solo']['cuda:1']:9.1f}"
              f"{row['wall_parallel_one_arm_per_card']:10.1f}"
              f"{row['wall_serial_both_on_fastest']:9.1f}"
              f"{row['serial_is_faster_by']:+12.1%}")
    print(f"\nadopt serial: {payload['verdict']['adopt_serial']} "
          f"-- {payload['verdict']['reason']}")


if __name__ == "__main__":
    main()
