"""Run the frozen exploratory pilot, interleaving each update with measurement.

Use the core environment. Each GPU stage is a separate process in its existing
environment; state.json and stage logs make unattended work inspectable. A
failed stage pauses the run for diagnosis instead of silently changing scale.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["runs/v4/main-2plus1", "runs/v4/main-1plus1plus1"]
ARMS = ["naive", "rfo_gold"]

# One card per arm for the whole evaluate-and-probe chain. The two arms share
# nothing there -- separate checkpoints, image directories, manifests, logs and
# completion markers -- so running them one after the other left a 3090 idle
# through the entire generate-and-adjudicate pass of the other. That pass is
# most of the wall clock: 256 images at 38.8 s each, twice per checkpoint.
#
# Deliberately NOT parallelised: `train`, which already spans both cards
# (backbone on cuda:1, adjudication ladder on cuda:0), and `report`, which is
# CPU and reads the one metrics table both arms write into.
ARM_DEVICE = {"naive": "cuda:0", "rfo_gold": "cuda:1"}

# Roughly a third of a second of retries in total, which is far longer than any
# reader holds state.json and far shorter than any stage.
REPLACE_ATTEMPTS = 5
SOURCES = [
    "scripts/v4_train.py", "src/selfsight/v4/train.py",
    "src/selfsight/v4/evaluate.py", "src/selfsight/v4/probe.py",
    "scripts/v4_checkpoint_probe.py", "src/selfsight/v4/checkpoint_probe.py",
    "scripts/v4_decoupling_report.py", "src/selfsight/v4/factual_truth.py",
    "src/selfsight/training/checkpoint.py", "scripts/run_decoupling_pilot.py",
    "scripts/v4_decoupling_plot.py", "scripts/v4_gradient_sensitivity.py",
]


def write_json(path: Path, value: dict) -> None:
    """Atomic publish, safe against a concurrent writer and a concurrent reader.

    The tmp name carries the thread id because the two arm chains write from two
    threads and a single shared `.tmp` would let one truncate the other's
    half-written bytes before either replace ran.

    The retry is for Windows: os.replace refuses with WinError 5 while anything
    holds a handle on the destination, and the destination here is the file a
    human checks to see how the run is doing. Losing the run because someone
    read its status is not a trade worth making, so the replace is attempted a
    few times before the error is allowed out.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (attempt + 1))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Pilot:
    # Class level, not per instance, because the supervisor tests build a Pilot
    # with object.__new__ to exercise resume logic without a config or a lock
    # file, and state() must not depend on __init__ having run. One Pilot per
    # process is guaranteed by the supervisor.lock flock anyway, so there is no
    # instance for a shared lock to be wrong about.
    state_lock = threading.Lock()

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = args.outdir.resolve()
        if not self.out.is_relative_to(ROOT / "runs" / "v4"):
            raise ValueError("Pilot output must remain inside this checkout's runs/v4")
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / "supervisor.lock").open("a+b")
        if self.lock.tell() == 0:
            self.lock.write(b" ")
            self.lock.flush()
        self.lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.config_path = args.config.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.limit = min(args.through_round or self.config["training"]["rounds"],
                         self.config["training"]["rounds"])
        if args.through_round is not None and args.through_round < 1:
            raise ValueError("--through-round must be positive")
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1",
                        PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                        TOKENIZERS_PARALLELISM="false")
        self.env.pop("CUDA_VISIBLE_DEVICES", None)
        self.manifest_path = self.out / "run_manifest.json"
        fingerprint = {name: digest(ROOT / name) for name in SOURCES}
        self.protocol_path = args.protocol.resolve()
        manifest = {"config_sha256": digest(self.config_path),
                    "protocol_sha256": digest(self.protocol_path),
                    "protocol_path": str(self.protocol_path.relative_to(ROOT)),
                    "source_sha256": fingerprint, "runs": RUNS,
                    "started_unix": time.time(), "config": self.config}
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for key in ("config_sha256", "protocol_sha256", "runs"):
                if old[key] != manifest[key]:
                    raise ValueError(f"Frozen run {key} changed; use a new run version")
            if old["source_sha256"] != fingerprint:
                if not args.accept_code_update:
                    raise ValueError("Source changed; review and record the repair before resume")
                repairs = old.get("code_repairs", [])
                repairs.append({"at": time.time(), "previous": old["source_sha256"],
                                "git_head": subprocess.check_output(
                                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
                old.update(source_sha256=fingerprint, code_repairs=repairs)
                write_json(self.manifest_path, old)
            manifest = old
        else:
            manifest["git_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            write_json(self.manifest_path, manifest)
        self.started = manifest["started_unix"]
        self.frozen = manifest
        self.deadline = self.started + self.config["pilot"]["max_wall_hours"] * 3600
        self.common = ["--outdir", str(self.out), "--config", str(self.config_path),
                       "--runs", *RUNS]
        self.done_stages = self.out / "stage-completion"
        self.done_stages.mkdir(exist_ok=True)
        self.state("ready", "preflight")

    def state(self, status: str, stage: str, **extra) -> None:
        """Serialised because the two arm chains write this from two threads.

        write_json goes through one fixed tmp path before os.replace, so
        concurrent callers would interleave into it and could publish a
        truncated status file -- the one artifact used to check on an
        unattended multi-day run from outside.
        """

        with self.state_lock:
            try:
                self._write_state(status, stage, **extra)
            except OSError as exc:
                # Telemetry, not evidence. Every artifact the results rest on is
                # written by the stage subprocesses and by write_json calls that
                # still raise; this one file only reports progress, and killing
                # a multi-day run because a status line could not be published
                # would be the more expensive failure by a wide margin.
                print(f"warning: could not publish state.json ({exc})", flush=True)

    def _write_state(self, status: str, stage: str, **extra) -> None:
        write_json(self.out / "state.json", {
            "status": status, "stage": stage, "supervisor_pid": os.getpid(),
            "updated_unix": time.time(), "started_unix": self.started,
            "elapsed_hours": (time.time() - self.started) / 3600,
            "through_round": self.limit, **extra})

    def run(self, stage: str, environment: str, script: str, args: list[str]) -> None:
        if digest(self.config_path) != self.frozen["config_sha256"]:
            raise RuntimeError("Configuration changed during execution")
        if any(digest(ROOT / name) != expected
               for name, expected in self.frozen["source_sha256"].items()):
            raise RuntimeError("Experiment source changed during execution; review before resume")
        complete = self.done_stages / f"{stage}.json"
        if complete.exists():
            return
        if time.time() >= self.deadline:
            raise RuntimeError("Pilot wall budget reached; review without extending silently")
        free = shutil.disk_usage(self.out).free / 1024**3
        if free < self.config["pilot"]["min_free_gib"]:
            raise RuntimeError(f"Only {free:.1f} GiB free; preserve artifacts and review storage")
        command = [str(ROOT / "envs" / environment / "python.exe"), "-u",
                   str(ROOT / script), *map(str, args)]
        log = self.out / "logs" / f"{stage}.log"
        log.parent.mkdir(exist_ok=True)
        self.state("running", stage, command=command, log=str(log), free_gib=free)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stage}", flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\nCOMMAND " + subprocess.list2cmdline(command) + "\n")
            handle.flush()
            # The stage name rides in the child's environment rather than being
            # looked up in state.json. Two arm chains publish that file from two
            # threads, so a reader can only learn which stage is "current", not
            # which stage it is itself -- and with concurrency those differ.
            process = subprocess.Popen(command, cwd=ROOT,
                                       env={**self.env, "SELFSIGHT_STAGE": stage},
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.state("running", stage, child_pid=process.pid, log=str(log), command=command)
            try:
                result = process.wait(timeout=max(1, self.deadline - time.time()))
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=handle, stderr=subprocess.STDOUT, check=False)
                else:
                    process.terminate()
                raise RuntimeError("Wall budget expired during stage; owned child stopped")
        if result:
            raise RuntimeError(f"{stage} failed with exit {result}; see {log}")
        write_json(complete, {"stage": stage, "completed_unix": time.time(), "command": command})

    def chains(self, bodies: dict[str, Any]) -> None:
        """Run one stage chain per arm at the same time, each pinned to a card.

        Every stage inside a chain still runs in order and still writes its own
        completion marker, so a crash resumes exactly where the sequential
        version would have. What changes is only that the other arm has stopped
        waiting for this one.

        A failure in one chain does not kill the other mid-stage. The survivor
        finishes what it started -- a half-written manifest costs more than the
        few minutes saved -- and then the first error is re-raised, so the
        supervisor still pauses for diagnosis instead of continuing.
        """

        errors: dict[str, BaseException] = {}

        def guard(name: str, body: Any) -> None:
            try:
                body()
            except BaseException as exc:  # noqa: BLE001 - re-raised after the join
                errors[name] = exc

        threads = [threading.Thread(target=guard, args=(name, body), name=name)
                   for name, body in sorted(bodies.items())]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            first = min(errors)
            raise RuntimeError(f"{first} chain failed: {errors[first]}") from errors[first]

    def measure_arm(self, arm: str, round_index: int) -> None:
        """One arm's whole per-checkpoint chain, on that arm's card.

        This is the body `chains` runs in a thread. It exists as a method rather
        than a lambda so the arm and round are bound as arguments, not captured
        from the enclosing loop.
        """

        self.evaluate(arm, round_index, ARM_DEVICE[arm])
        self.probe(arm, round_index, ARM_DEVICE[arm])

    def adjudicate(self, directory: Path, label: str, device: str) -> None:
        for detector in ("qwen3vl", "internvl"):
            self.run(f"{label}.detect.{detector}", "observer", "scripts/v4_run_pipeline.py",
                     ["detect", "--manifest", str(directory / "manifest.jsonl"),
                      "--detector", detector, "--device", device])
        self.run(f"{label}.crop", "observer", "scripts/v4_run_pipeline.py",
                 ["crop", "--run", str(directory), "--device", device])
        self.run(f"{label}.verify", "core", "scripts/v4_run_pipeline.py",
                 ["verify", "--run", str(directory)])

    def evaluate(self, arm: str, round_index: int, device: str) -> None:
        step = (round_index + 1) * self.config["training"]["optimizer_steps_per_round"]
        label = f"{arm}.step-{step:05d}"
        self.run(f"{label}.generate", "showo2", "scripts/v4_train.py",
                 ["generate", *self.common, "--arm", arm, "--round", str(round_index),
                  "--device", device])
        self.adjudicate(self.out / "evaluations" / arm / f"step-{step:05d}", label, device)

    def probe(self, arm: str, round_index: int, device: str) -> None:
        step = (round_index + 1) * self.config["training"]["optimizer_steps_per_round"]
        target = self.out / "gradient-probes" / arm / f"step-{step:05d}"
        args = ["run", "--config", str(self.config_path), "--bank", str(self.out / "probe-bank"),
                "--outdir", str(target), "--device", device, "--arm", arm,
                "--resamples", str(self.config["gradient_probe"]["resamples"])]
        if round_index == -1:
            args += ["--checkpoint", str(self.out / "checkpoints/base/round--01")]
        else:
            args += ["--checkpoint", str(self.out / "checkpoints" / arm / f"round-{round_index:03d}"),
                     "--reference", str(self.out / "gradient-probes/base/step-00000")]
        self.run(f"{arm}.step-{step:05d}.gradient", "showo2", "scripts/v4_checkpoint_probe.py", args)

    def report(self, step: int) -> None:
        self.run(f"step-{step:05d}.score", "core", "scripts/v4_train.py",
                 ["score", "--outdir", str(self.out)])
        # self.protocol_path, not a literal. The manifest records the protocol the
        # run was launched under and refuses to start if it changes; hardcoding a
        # different one here put the pilot's document into every report's frozen
        # provenance block while run_manifest.json named the main-run protocol.
        # The two disagreed for the whole of 2026-09-08, and because
        # v4_decoupling_report.py freezes provenance across steps, the first
        # report to land would have locked the wrong one in for all twelve.
        self.run(f"step-{step:05d}.report", "core", "scripts/v4_decoupling_report.py",
                 ["--outdir", str(self.out), "--config", str(self.config_path),
                  "--protocol", str(self.protocol_path)])
        self.run(f"step-{step:05d}.gradient-sensitivity", "core", "scripts/v4_gradient_sensitivity.py",
                 ["--outdir", str(self.out)])
        self.run(f"step-{step:05d}.plot", "core", "scripts/v4_decoupling_plot.py",
                 ["--outdir", str(self.out)])

    def validate_round(self, index: int) -> None:
        path = self.out / "rounds" / f"round-{index:03d}" / "DONE.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["paired"] < self.config["pilot"]["min_paired_prompts"]:
            raise RuntimeError(f"Round {index} has too few paired prompts")
        reports = row.get("arms", [])
        if sorted(report.get("arm", "") for report in reports) != sorted(ARMS):
            raise RuntimeError(f"Round {index}: missing or duplicate arm reports")
        for report in row["arms"]:
            if report.get("round") != index or report.get("optimizer_steps") != self.config["training"]["optimizer_steps_per_round"]:
                raise RuntimeError(f"Round {index}: report does not match scheduled optimizer updates")
            for key in ("mean_t2i_loss", "mean_gradient_norm_before_clip", "parameter_delta_l2"):
                if not math.isfinite(report[key]):
                    raise RuntimeError(f"Round {index}: nonfinite {key}")
            if report["mean_gradient_norm_before_clip"] <= 0:
                raise RuntimeError(f"Round {index}: no measured objective gradient")
            if report["parameter_delta_l2"] <= 0:
                raise RuntimeError(f"Round {index}: no measured parameter update")

    def execute(self) -> None:
        if not (self.out / "split.json").exists():
            self.run("split", "core", "scripts/v4_train.py", ["split", *self.common])
        self.run("freeze-probe", "core", "scripts/v4_checkpoint_probe.py",
                 ["freeze", "--config", str(self.config_path), "--split", str(self.out / "split.json"),
                  "--source", self.config["gradient_probe"]["source"],
                  "--outdir", str(self.out / "probe-bank"),
                  "--max-prompts", str(self.config["gradient_probe"]["size"])])
        for index in range(self.limit):
            # Round 0 also persists the exact untrained adapter. Measuring that
            # saved base afterwards cannot change its weights or feed outcomes
            # into training, and reaches the first optimizer update sooner.
            self.run(f"round-{index:03d}.train", "showo2", "scripts/v4_train.py",
                     ["train", *self.common, "--device", "cuda:1", "--ladder-device", "cuda:0",
                      "--max-epochs", "1", "--round-index", str(index)])
            self.validate_round(index)
            if index == 0:
                # Both arms measure the same untrained adapter here, so the two
                # chains draw identical images. They are still kept apart so the
                # per-arm directory layout is uniform across every checkpoint.
                self.chains({arm: partial(self.evaluate, arm, -1, ARM_DEVICE[arm])
                             for arm in ARMS})
                self.probe("base", -1, "cuda:1")
                self.report(0)
            self.chains({arm: partial(self.measure_arm, arm, index) for arm in ARMS})
            self.report((index + 1) * self.config["training"]["optimizer_steps_per_round"])
        self.state("pilot_complete" if self.limit == self.config["training"]["rounds"] else "canary_complete",
                   "review_results", completed_rounds=self.limit,
                   research_goal_complete=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "runs/v4/decoupling-pilot-20260906")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v4_decoupling_pilot.yaml")
    # The protocol a run records itself against. Defaulted rather than derived
    # so a new config cannot quietly inherit the pilot's provenance: the hash
    # of whatever is passed here is frozen into run_manifest.json and a later
    # mismatch stops the resume.
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "docs/prereg/2026-09-06-decoupling-pilot.md")
    parser.add_argument("--through-round", type=int)
    parser.add_argument("--accept-code-update", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    pilot = Pilot(args)
    try:
        pilot.execute()
    except Exception as exc:
        pilot.state("needs_diagnosis", "paused", error=str(exc))
        raise


if __name__ == "__main__":
    main()
