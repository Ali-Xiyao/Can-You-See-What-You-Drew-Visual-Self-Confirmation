"""Run the frozen exploratory pilot, interleaving each update with measurement.

Use the core environment. Each GPU stage is a separate process in its existing
environment; state.json and stage logs make unattended work inspectable. A
failed stage pauses the run for diagnosis instead of silently changing scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["runs/v4/main-2plus1", "runs/v4/main-1plus1plus1"]
ARMS = ["naive", "rfo_gold"]
SOURCES = [
    "scripts/v4_train.py", "src/selfsight/v4/train.py",
    "src/selfsight/v4/evaluate.py", "src/selfsight/v4/probe.py",
    "scripts/v4_checkpoint_probe.py", "src/selfsight/v4/checkpoint_probe.py",
    "scripts/v4_decoupling_report.py", "src/selfsight/v4/factual_truth.py",
    "src/selfsight/training/checkpoint.py", "scripts/run_decoupling_pilot.py",
    "scripts/v4_decoupling_plot.py", "scripts/v4_gradient_sensitivity.py",
]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Pilot:
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
        manifest = {"config_sha256": digest(self.config_path),
                    "protocol_sha256": digest(ROOT / "docs/prereg/2026-09-06-decoupling-pilot.md"),
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
            process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=handle,
                                       stderr=subprocess.STDOUT,
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

    def adjudicate(self, directory: Path, label: str) -> None:
        for detector in ("qwen3vl", "internvl"):
            self.run(f"{label}.detect.{detector}", "observer", "scripts/v4_run_pipeline.py",
                     ["detect", "--manifest", str(directory / "manifest.jsonl"),
                      "--detector", detector, "--device", "cuda:0"])
        self.run(f"{label}.crop", "observer", "scripts/v4_run_pipeline.py",
                 ["crop", "--run", str(directory), "--device", "cuda:0"])
        self.run(f"{label}.verify", "core", "scripts/v4_run_pipeline.py",
                 ["verify", "--run", str(directory)])

    def evaluate(self, arm: str, round_index: int) -> None:
        step = (round_index + 1) * self.config["training"]["optimizer_steps_per_round"]
        label = f"{arm}.step-{step:05d}"
        self.run(f"{label}.generate", "showo2", "scripts/v4_train.py",
                 ["generate", *self.common, "--arm", arm, "--round", str(round_index),
                  "--device", "cuda:1"])
        self.adjudicate(self.out / "evaluations" / arm / f"step-{step:05d}", label)

    def probe(self, arm: str, round_index: int) -> None:
        step = (round_index + 1) * self.config["training"]["optimizer_steps_per_round"]
        target = self.out / "gradient-probes" / arm / f"step-{step:05d}"
        args = ["run", "--config", str(self.config_path), "--bank", str(self.out / "probe-bank"),
                "--outdir", str(target), "--device", "cuda:1", "--arm", arm,
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
        self.run(f"step-{step:05d}.report", "core", "scripts/v4_decoupling_report.py",
                 ["--outdir", str(self.out), "--config", str(self.config_path),
                  "--protocol", str(ROOT / "docs/prereg/2026-09-06-decoupling-pilot.md")])
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
                      "--max-epochs", "1", "--max-rounds", str(index + 1)])
            self.validate_round(index)
            if index == 0:
                for arm in ARMS:
                    self.evaluate(arm, -1)
                self.probe("base", -1)
                self.report(0)
            for arm in ARMS:
                self.evaluate(arm, index)
                self.probe(arm, index)
            self.report((index + 1) * self.config["training"]["optimizer_steps_per_round"])
        self.state("pilot_complete" if self.limit == self.config["training"]["rounds"] else "canary_complete",
                   "review_results", completed_rounds=self.limit,
                   research_goal_complete=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "runs/v4/decoupling-pilot-20260906")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v4_decoupling_pilot.yaml")
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
