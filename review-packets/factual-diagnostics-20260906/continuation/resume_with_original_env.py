"""Recover the failed environment-only handoff; never stop any process.

--check-only validates frozen inputs, no active owners and the actual Show-o2
Python environment using CPU-only model-path/revision checks. --execute starts
the original supervisor through round 10, with the original 60-hour deadline.
No hold-complete record or successful measurement is invented.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "runs/v4/decoupling-pilot-20260906"
EXPECTED_SHA256 = "738bb78114ec88057a0202b4470f3543011d92c77d16963e6f2822b3867be9d2"
NO_WINDOW = 0x08000000
PREFLIGHT = """
import json, os, sys
from pathlib import Path
import yaml
from selfsight.models import repository_path, snapshot_path
cfg = yaml.safe_load(Path('configs/backbones/showo2_1p5b.yaml').read_text(encoding='utf-8'))
assert os.environ['SELFSIGHT_MODEL_ROOT'] == r'H:\\selfsight-models'
assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
source = repository_path(cfg['source']['repository_id'])
assert (source / cfg['source']['subtree'] / 'models/modeling_showo2_qwen2_5.py').is_file()
wan_id = 'Wan-AI/Wan2.1-T2V-14B'
paths = {model: str(snapshot_path(model, require_complete=model != wan_id))
         for model in cfg['dependencies']}
assert (Path(paths[wan_id]) / 'Wan2.1_VAE.pth').is_file()
assert 'torch' not in sys.modules
print(json.dumps({'cpu_only': True, 'model_root': os.environ['SELFSIGHT_MODEL_ROOT'],
                  'source_repository': str(source), 'snapshots': paths,
                  'offline': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}}))
"""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(name, value):
    with (HERE / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def check_owners():
    query = ("$OutputEncoding=[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
             "@(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
             "Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", query],
                            capture_output=True, text=True, encoding="utf-8", check=True,
                            timeout=30, creationflags=NO_WINDOW)
    records = json.loads(result.stdout)
    records = records if isinstance(records, list) else [records]
    scripts = ("run_decoupling_pilot.py", "finish_step24.py", "resume_after_step24.py",
               "resume_with_original_env.py", "v4_train.py", "v4_checkpoint_probe.py", "v4_run_pipeline.py")
    owners = [p["ProcessId"] for p in records if p["ProcessId"] != os.getpid()
              and any(name in (p.get("CommandLine") or "") for name in scripts)]
    require(not owners, f"Active experiment/controller PIDs remain: {owners}")
    return {"active_owner_pids": owners}


def frozen_check(expected):
    require(time.time() < expected["original_deadline_unix"], "Original deadline expired; no extension")
    for group in ("frozen_sha256", "failure_sha256"):
        for name, value in expected[group].items():
            require(sha(ROOT / name) == value, f"Frozen or failed input changed: {name}")
    for name, value in expected["existing_stage_sha256"].items():
        require(sha(RUN / "stage-completion" / name) == value, f"Existing stage changed: {name}")
    require(not (RUN / "stage-completion/naive.step-00024.gradient.json").exists(),
            "Failed gradient was already completed by another owner")
    require(not (RUN / "rounds/round-003").exists(), "Another owner already started round3")
    state = read(RUN / "state.json")
    require(state["status"] == "needs_diagnosis" and state["supervisor_pid"] == 26896 and
            "naive.step-00024.gradient failed with exit 1" in state["error"], "Failure state differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(os.name == "nt", "Windows-only recovery launcher")
    require(sha(HERE / "recovery-expected.json") == EXPECTED_SHA256, "Recovery expectation changed")
    expected = read(HERE / "recovery-expected.json")
    require(Path(expected["root"]) == ROOT and Path(expected["run"]) == RUN, "Workspace differs")
    config = expected["manifest"]["config"]
    require(config["training"]["rounds"] == 10 and config["pilot"]["max_wall_hours"] == 60 and
            expected["original_deadline_unix"] == expected["original_started_unix"] + 60 * 3600,
            "Original schedule or deadline differs")
    lock, attempted = None, False
    try:
        if args.execute:
            import msvcrt
            lock = (HERE / "recovery.lock").open("a+b")
            if lock.tell() == 0:
                lock.write(b" ")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        require(not (HERE / "recovery-started.json").exists(), "One-time recovery already attempted")
        frozen_check(expected)
        owners = check_owners()
        env = os.environ.copy()
        env.update(expected["environment_allowlist"])
        for name in expected["environment_remove"]:
            env.pop(name, None)
        require(Path(env["SELFSIGHT_MODEL_ROOT"]).is_dir() and Path(env["SELFSIGHT_TMP_ROOT"]).is_dir(),
                "Original model/temp roots are missing")
        # Exercise the precise environment where the KeyError happened, without
        # importing torch, loading models, generating images or allocating CUDA.
        checked = subprocess.run([str(ROOT / "envs/showo2/python.exe"), "-c", PREFLIGHT], cwd=ROOT,
                                 env=env | {"PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True,
                                 text=True, encoding="utf-8", timeout=60, creationflags=NO_WINDOW)
        require(checked.returncode == 0, f"Show-o2 CPU environment preflight failed: {checked.stderr}")
        preflight = json.loads(checked.stdout)
        frozen_check(expected)
        check_owners()
        if args.check_only:
            print(json.dumps({"status": "check_passed", "owners": owners, "preflight": preflight,
                              "environment_allowlist": expected["environment_allowlist"],
                              "original_deadline_unix": expected["original_deadline_unix"],
                              "no_gpu_or_experiment_started": True, "no_files_written": True}, indent=2))
            return
        save_new("recovery-started.json", {"at_unix": time.time(), "launcher_pid": os.getpid(),
                 "launcher_sha256": sha(__file__), "expected_sha256": EXPECTED_SHA256,
                 "reason": expected["reason"], "preflight": preflight,
                 "environment_allowlist": expected["environment_allowlist"],
                 "failure_evidence_sha256": expected["failure_sha256"],
                 "original_deadline_unix": expected["original_deadline_unix"]})
        attempted = True
        command = [str(ROOT / "envs/core/python.exe"), "-u", str(ROOT / "scripts/run_decoupling_pilot.py"),
                   "--outdir", str(RUN), "--config", str(ROOT / "configs/v4_decoupling_pilot.yaml"),
                   "--through-round", "10"]
        frozen_check(expected)
        check_owners()
        with (HERE / "recovered-supervisor.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            save_new("recovered-supervisor.json", {"pid": process.pid, "command": command,
                     "at_unix": time.time(), "original_deadline_unix": expected["original_deadline_unix"]})
            deadline = min(time.time() + 60, expected["original_deadline_unix"])
            while True:
                state = read(RUN / "state.json")
                if state.get("supervisor_pid") == process.pid:
                    require(state["through_round"] == 10 and state["started_unix"] == expected["original_started_unix"],
                            "Resumed supervisor changed limit or time origin")
                    require(state["status"] not in ("needs_diagnosis",), "Resumed supervisor immediately failed")
                    save_new("recovery-launch-confirmed.json", {"at_unix": time.time(), "state": state,
                             "launch_only": True, "pilot_complete": False,
                             "note": "Root continues monitoring; launch is not completion of the pilot."})
                    print(json.dumps({"status": "launch_confirmed", "supervisor_pid": process.pid,
                                      "through_round": 10, "original_deadline_unix": expected["original_deadline_unix"]}))
                    return
                require(process.poll() is None, f"Resumed supervisor exited early: {process.returncode}")
                require(time.time() < deadline, "New supervisor ownership not confirmed; no automatic retry")
                time.sleep(0.25)
    except Exception as exc:
        if attempted:
            save_new("recovery-needs-diagnosis.json", {"at_unix": time.time(), "error": str(exc),
                     "automatic_retry": False, "processes_stopped": False})
        print(f"REFUSED / NEEDS DIAGNOSIS: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
