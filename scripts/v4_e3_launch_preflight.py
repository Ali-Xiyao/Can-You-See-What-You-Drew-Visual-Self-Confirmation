"""Every gate that has to hold before replicate 1 starts, as code.

EXECUTION section 1 writes these as prose. Prose is the wrong medium for a
check that runs unattended at four in the morning when the main run ends, so
this is the same list executable. Every gate is a function that returns a
(passed, message) pair, nothing here touches a GPU, and a failure exits
non-zero with the reason rather than warning and continuing.

    envs/core/python.exe scripts/v4_e3_launch_preflight.py

The gate that matters most is the quiet one. Against a v4_train.py from
before the arm B merge, `training.seed` is not an error -- it is silently
ignored, and five replicates all train on 20260906 while the directory
names, the configs and the logs agree that they are five seeds. That is the
hardest failure in this plan to notice after the fact, and it costs 400
GPU-hours to make.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Deviation 9, fixed in advance. 20260908 is skipped: section 3 spent it.
REGISTERED_SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)
REGISTERED_ARMS = ["naive", "blind_self"]
MAIN_RUN = ROOT / "runs/v4/decoupling-main-20260908"
IDLE_MIB = 500

Gate = tuple[bool, str]


def _yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def gate_main_run_finished(state_path: Path = MAIN_RUN / "state.json") -> Gate:
    """Finished, not stopped.

    The 96 h wall clock and the projected finish are within hours of each
    other (EXECUTION 0.5), so `pilot_complete` and `canary_complete` are both
    live outcomes and they mean opposite things. Launching replicates off a
    truncated main run would spend 400 GPU-hours replicating an experiment
    whose own arm C never finished.
    """

    if not state_path.exists():
        return False, f"no {state_path}"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    status, rounds = state.get("status"), state.get("completed_rounds")
    if status != "pilot_complete":
        return False, (f"status is {status!r}, not 'pilot_complete'"
                       + (" -- the 96 h stop line fired, this is not a finished run"
                          if status == "canary_complete" else ""))
    if rounds != 11:
        return False, f"completed_rounds is {rounds}, not 11"
    return True, f"pilot_complete, 11 rounds, {state.get('elapsed_hours', 0):.1f} h"


def gate_merge_landed(root: Path = ROOT) -> Gate:
    """The arm B merge is on disk, checked by what it has to provide."""

    train = (root / "scripts/v4_train.py").read_text(encoding="utf-8")
    supervisor = (root / "scripts/run_decoupling_pilot.py").read_text(encoding="utf-8")
    missing = []
    if "def training_seed(" not in train:
        missing.append("scripts/v4_train.py has no training_seed: every replicate "
                       "would silently train on the top-level seed")
    if "def partition_seed(" not in train:
        missing.append("scripts/v4_train.py has no partition_seed")
    if '"--arms"' not in supervisor and "'--arms'" not in supervisor:
        missing.append("run_decoupling_pilot.py has no --arms: the arm set is still hardcoded")
    if "self.arm_device" not in supervisor:
        missing.append("run_decoupling_pilot.py still uses a module-level ARM_DEVICE")
    if not (root / "scripts/v4_verify_replicates.py").exists():
        missing.append("scripts/v4_verify_replicates.py is absent")
    # training.arms is in all five configs and the supervisor reads the arm
    # set from --arms alone. Forgetting the flag trains the registered
    # pairing under a replicate's name, and the frozen manifest only notices
    # on resume. The merge carries a guard that refuses the contradiction.
    if "registered_arms" not in supervisor:
        missing.append("run_decoupling_pilot.py does not check training.arms against --arms: "
                       "a launch without the flag would train the registered pairing")
    return (not missing), ("merged" if not missing else "; ".join(missing))


def gate_configs(root: Path = ROOT) -> Gate:
    """Five configs, five training seeds, one partition."""

    problems = []
    digests, partition_seeds = set(), set()
    for seed in REGISTERED_SEEDS:
        path = root / f"configs/v4_e3_replicate_s{seed}.yaml"
        if not path.exists():
            problems.append(f"{path.name} is missing")
            continue
        config = _yaml(path)
        training = config.get("training", {})
        if training.get("seed") != seed:
            problems.append(f"{path.name}: training.seed is {training.get('seed')!r}, not {seed}")
        # The plural key is read by the v2.x formal pipeline and never by v4,
        # so setting it is a request that gets ignored without a word.
        if "seeds" in training:
            problems.append(f"{path.name}: training.seeds (plural) is dead config in v4")
        # Nested under training, and read by nothing -- see gate_merge_landed.
        if list(training.get("arms") or []) != REGISTERED_ARMS:
            problems.append(
                f"{path.name}: training.arms is {training.get('arms')!r}, not {REGISTERED_ARMS}")
        partition_seeds.add(config.get("seed"))
        digests.add(json.dumps({
            "seed": config.get("seed"),
            "outcome": config.get("data", {}).get("local_outcome"),
            "probe": config.get("data", {}).get("local_probe"),
        }, sort_keys=True))
    if len(partition_seeds) > 1:
        problems.append(f"top-level seed differs across configs: {sorted(partition_seeds)} -- "
                        "the five would land on different splits and nothing would pair")
    if len(digests) > 1:
        problems.append(f"split inputs differ across configs: {sorted(digests)}")
    if problems:
        return False, "; ".join(problems)
    return True, (f"5 configs, training seeds {list(REGISTERED_SEEDS)}, "
                  f"one partition seed {partition_seeds.pop()}, one split")


def gate_cards_free(idle_mib: int = IDLE_MIB) -> Gate:
    """Both cards, because one replicate needs both.

    ARM_CARDS is two cards and __init__ refuses any other arity, and train
    puts the backbone on one card and the adjudication ladder on the other
    and refuses to share (STATUS 37). There is no one-card replicate.
    """

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as exc:  # noqa: BLE001 - any failure here is "cannot tell"
        return False, f"could not read nvidia-smi: {exc}"
    busy = []
    seen = 0
    for line in out.strip().splitlines():
        index, used = (part.strip() for part in line.split(","))
        seen += 1
        if int(used) > idle_mib:
            busy.append(f"cuda:{index} holds {used} MiB")
    if seen < 2:
        return False, f"only {seen} card visible; a replicate needs two"
    return (not busy), ("both cards idle" if not busy else
                        "; ".join(busy) + " -- do not preempt, it may not be ours")


def gate_card_schedule_decided(root: Path = ROOT) -> Gate:
    """Section 0.3's benchmark has run and its registered criterion applied.

    The window for it is the one this preflight sits in: both cards free
    between the main run ending and replicate 1 starting. Once replicate 1
    starts the window is gone until all five finish, and the criterion was
    registered before the measurement precisely so it could not be skipped
    once the cards were needed.
    """

    verdict = root / "review-packets/card-scheduling-20260909/verdict.json"
    if not verdict.exists():
        return False, (f"no {verdict.relative_to(root)}; run card_benchmark.py while both "
                       "cards are free and apply the registered criterion "
                       "(wall_S <= 0.90 x wall_P on both detectors)")
    payload = json.loads(verdict.read_text(encoding="utf-8"))
    if "serialise_detect" not in payload:
        return False, f"{verdict.name} has no serialise_detect decision"
    return True, f"decided: serialise_detect={payload['serialise_detect']}"


def gate_model_root() -> Gate:
    import os

    root = os.environ.get("SELFSIGHT_MODEL_ROOT")
    if not root:
        return False, "SELFSIGHT_MODEL_ROOT is unset"
    if not Path(root).is_dir():
        return False, f"SELFSIGHT_MODEL_ROOT={root} is not a directory"
    return True, root


GATES = {
    "main run finished": gate_main_run_finished,
    "arm B merge landed": gate_merge_landed,
    "five replicate configs": gate_configs,
    "both cards free": gate_cards_free,
    "card schedule decided": gate_card_schedule_decided,
    "model root": gate_model_root,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip", nargs="*", default=[], metavar="GATE",
                        help="Gate names to report but not enforce. Every skip is printed "
                             "in the summary so a green run cannot hide one.")
    args = parser.parse_args()
    unknown = [name for name in args.skip if name not in GATES]
    if unknown:
        raise SystemExit(f"unknown gate(s) {unknown}; known: {sorted(GATES)}")

    failures, skipped = [], []
    width = max(len(name) for name in GATES)
    for name, check in GATES.items():
        passed, message = check()
        if passed:
            mark = "ok  "
        elif name in args.skip:
            mark = "SKIP"
            skipped.append(name)
        else:
            mark = "FAIL"
            failures.append(name)
        print(f"  {mark}  {name:<{width}}  {message}")

    print()
    if skipped:
        print(f"skipped (not enforced): {skipped}")
    if failures:
        print(f"BLOCKED by {len(failures)} gate(s): {failures}")
        return 1
    print("all gates pass; replicate 1 may start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
