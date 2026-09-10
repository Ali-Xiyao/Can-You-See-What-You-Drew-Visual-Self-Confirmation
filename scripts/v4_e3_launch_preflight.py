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
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Deviation 9, fixed in advance. 20260908 is skipped: section 3 spent it.
REGISTERED_SEEDS = (20260906, 20260907, 20260909, 20260910, 20260911)
REGISTERED_ARMS = ["naive", "blind_self"]
MAIN_RUN = ROOT / "runs/v4/decoupling-main-20260908"
IDLE_MIB = 500
# Section 0.3, registered before the measurement.
CARD_MARGIN = 0.90

Gate = tuple[bool, str]


def _yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _supervisor_alive(pid: object) -> bool | None:
    """True, False, or None when the platform will not answer.

    Deliberately not `os.kill(pid, 0)`. On Windows CPython implements os.kill
    by opening the process and calling TerminateProcess for any signal that
    is not a console-control event, so the portable liveness idiom would kill
    the run this gate exists to protect.
    """

    if not isinstance(pid, int) or os.name != "nt":
        return None
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:  # noqa: BLE001 - any failure here is "cannot tell"
        return None
    # Windows reuses pids, so this can be wrong. It can only be wrong about
    # the wording: every branch below returns False either way, and the gate
    # never passes on the strength of a live pid.
    return str(pid) in out


def gate_main_run_finished(state_path: Path = MAIN_RUN / "state.json") -> Gate:
    """Finished, not stopped, and not still going.

    Four outcomes have to be told apart and only the first is a finished run.

    - `pilot_complete` with completed_rounds == 11 -- the loop reached its end.
    - `canary_complete` -- the supervisor was launched with --through-round
      below training.rounds and stopped where it was told to. STATUS 43.15
      calls this 有界运行结束. It is *not* the 96 h line; an earlier draft of
      this gate and of EXECUTION section 1 both said it was, and both were
      wrong about the code.
    - `running` and not moving -- that is what the 96 h line looks like.
      `run()` raises at the next stage boundary, the exception leaves the
      round loop above the terminal `state()` call, and nothing ever
      rewrites state.json. It says running for as long as the disk survives.
      A traceback in any stage, a killed supervisor and a reboot all leave
      the same file.
    - `running` and moving -- the ordinary case while this is being read.

    The verdict is identical for the last three: refuse. The message is not,
    and the message is what this gate is for.
    """

    if not state_path.exists():
        return False, f"no {state_path}"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    status, rounds = state.get("status"), state.get("completed_rounds")
    if status == "pilot_complete" and rounds == 11:
        return True, f"pilot_complete, 11 rounds, {state.get('elapsed_hours', 0):.1f} h"
    if status == "pilot_complete":
        # completed_rounds is written only by the terminal state() call, so
        # its absence beside a terminal status means a hand-edited file.
        return False, f"status is 'pilot_complete' but completed_rounds is {rounds!r}, not 11"
    if status == "canary_complete":
        return False, (f"status is 'canary_complete' (completed_rounds={rounds!r}): launched "
                       f"with --through-round below training.rounds and stopped where it was "
                       f"told to. A bounded run, not a finished one")
    if status == "running":
        age_min = (time.time() - float(state.get("updated_unix") or 0)) / 60
        pid = state.get("supervisor_pid")
        if _supervisor_alive(pid) is False:
            return False, (f"status is 'running' but supervisor pid {pid} is gone and "
                           f"state.json has not moved for {age_min:.0f} min -- the run "
                           f"stopped without writing a terminal status. That is what the "
                           f"96 h line, a stage traceback and a reboot all look like. Read "
                           f"the supervisor log before deciding anything")
        return False, (f"still running: {state.get('stage')}, {age_min:.0f} min since the "
                       f"last state write, {state.get('elapsed_hours', 0):.1f} h elapsed")
    return False, f"status is {status!r}, not 'pilot_complete'"


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


SPLIT_CHECK = "scripts/v4_verify_replicates.py"
SPLIT_PATCH = "review-packets/replicate-split-identity-20260910/split_check.patch"


def gate_split_check_landed(root: Path = ROOT) -> Gate:
    """The split check is a patch on top of the merge, not part of it.

    `staging/arm-b-merged` supplies the verifier at version 1, which compares
    the five replicates only with each other and compares them by the recipe
    `digest`. Measured 2026-09-10: that digest is one number for the main
    config and all five replicate configs, computable from config text before
    a split stage has run, so the condition cannot fail -- and it stays true
    for a replicate whose corpus grew and whose held-out set therefore
    differs. Version 2 compares identity and requires the main run.

    The test suite says all this too, by going red. This gate exists because
    the preflight is what actually runs at four in the morning, and a patch
    with no gate is a patch someone forgets.
    """

    path = root / SPLIT_CHECK
    if not path.exists():
        return False, f"{SPLIT_CHECK} is absent; see the arm B merge gate first"
    source = path.read_text(encoding="utf-8")
    missing = []
    if 'VERSION = "v4_verify_replicates 2"' not in source:
        missing.append(f"{SPLIT_CHECK} is not version 2: the split it compares is the "
                       f"recipe digest, which is one number for all six configs and "
                       f"cannot fail. Apply {SPLIT_PATCH}")
    if "split_matches_main" not in source:
        missing.append("nothing compares the replicates' split to the main run's")
    if '"--main"' not in source or "required=True" not in source:
        missing.append("--main is not required on the command line, so the comparison "
                       "can be skipped by leaving a flag off")
    return (not missing), ("version 2, --main required" if not missing
                           else "; ".join(missing))


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
    """Section 0.3's benchmark has run, at no load, and decided.

    The window for it is the one this preflight sits in: both cards free
    between the main run ending and replicate 1 starting. Once replicate 1
    starts the window is gone until all five finish, and the criterion was
    registered before the measurement precisely so it could not be skipped
    once the cards were needed.

    This reads what `card_benchmark.py` writes and does not re-apply the
    criterion. An earlier version of this gate looked for a `verdict.json`
    with a top-level `serialise_detect`, a file and a key that exist nowhere
    else -- the benchmark writes `card_benchmark.json` and puts the decision
    under `verdict.adopt_serial`. The two scripts were written on two
    branches and had never been in one tree, so nothing could catch it, and
    the cost would have been paid as a hand-written decision file at the one
    moment the registered criterion is easiest to replace with a judgement
    call.

    What is checked besides the decision is that the recorded run was the
    registered one: no load, and the registered 10% margin. Neither is a new
    condition -- section 0.3 is titled 先测空载 and the benchmark refuses
    busy cards without `--allow-busy-cards` -- but an unchecked flag in a
    file is not a check.
    """

    packet = root / "review-packets/card-scheduling-20260909"
    report = packet / "card_benchmark.json"
    if not report.exists():
        return False, (f"no {report.relative_to(root)}; run card_benchmark.py with "
                       f"--outdir {packet.relative_to(root)} while both cards are free. "
                       "It applies the registered criterion itself "
                       "(wall_S <= 0.90 x wall_P on both detectors)")
    payload = json.loads(report.read_text(encoding="utf-8"))
    decision = payload.get("verdict") or {}
    if "adopt_serial" not in decision:
        return False, f"{report.name} has no verdict.adopt_serial decision"
    if not payload.get("no_load"):
        return False, (f"{report.name} records no_load=false; section 0.3 registered a "
                       "no-load benchmark, and a measurement taken beside another job "
                       "is not the one the criterion was written for")
    if payload.get("margin") != CARD_MARGIN:
        return False, (f"{report.name} was produced with margin {payload.get('margin')!r}, "
                       f"not the registered {CARD_MARGIN}")
    return True, (f"decided: adopt_serial={decision['adopt_serial']} -- "
                  f"{decision.get('reason', 'no reason recorded')}")

def gate_model_root() -> Gate:
    root = os.environ.get("SELFSIGHT_MODEL_ROOT")
    if not root:
        return False, "SELFSIGHT_MODEL_ROOT is unset"
    if not Path(root).is_dir():
        return False, f"SELFSIGHT_MODEL_ROOT={root} is not a directory"
    return True, root


GATES = {
    "main run finished": gate_main_run_finished,
    "arm B merge landed": gate_merge_landed,
    "replicate split check": gate_split_check_landed,
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
