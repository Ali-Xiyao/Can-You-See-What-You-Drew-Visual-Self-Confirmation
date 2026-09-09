"""Run the existing frozen detector ladder on the 44 fixed Naive images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SIDE = Path(__file__).resolve().parent
OUT = SIDE / "independent-labels"
MAIN = ROOT / "runs/v4/decoupling-pilot-20260906"
SOURCES = ["scripts/v4_run_pipeline.py", "src/selfsight/v4/detectors.py",
           "src/selfsight/v4/verifier.py", "src/selfsight/v4/spec.py"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def state(status, **details):
    path = OUT / "state.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"status": status, "at_unix": time.time(), "controller_pid": os.getpid(), **details}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def check_hold():
    hold = read(SIDE / "hold-step32/hold-complete.json")
    current = read(MAIN / "state.json")
    assert hold["supervisor_exit_code"] == 0 and hold["state"] == current
    assert current["through_round"] == current["completed_rounds"] == 4
    assert current["status"] == "canary_complete"
    assert not (MAIN / "rounds/round-004").exists()
    return hold


def prepare():
    hold = check_hold()
    check = subprocess.run([str(ROOT / "envs/core/python.exe"), "-B", str(SIDE / "validate_followup.py")], cwd=ROOT, capture_output=True, text=True, check=True)
    public = SIDE / "followup-label-request.public.jsonl"
    rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 44 and len({r["image_path"] for r in rows}) == 44
    OUT.mkdir(exist_ok=False)
    with (OUT / "manifest.jsonl").open("xb") as f:
        f.write(public.read_bytes())
    models = {}
    for name, path in {"qwen3vl": Path(r"H:\Xiyao_Wang\001_models\Qwen3-VL-8B-Instruct"), "internvl": Path(r"H:\Xiyao_Wang\001_models\InternVL3_5-8B")}.items():
        assert path.is_dir()
        models[name] = {"path": str(path), "resolved_path": str(path.resolve()), "metadata_sha256": {f.name: sha(f) for f in (path / "config.json", path / "tokenizer_config.json", path / "model.safetensors.index.json") if f.is_file()}, "weights_rehashed": False}
    expected = {"prepared_unix": time.time(), "population": "Fixed independent selected/random-control label request; 24 Naive natural pools, 44 distinct files needing labels.",
                "main_hold": hold, "deadline": read(SIDE / "hold-step32/expected.json")["deadline"],
                "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
                "request_sha256": sha(public), "pairing_sha256": sha(SIDE / "followup-label-pairs.private.json"),
                "image_sha256": {r["image_path"]: r["image_file_sha256"] for r in rows}, "models": models,
                "runner_sha256": sha(Path(__file__)), "device": "cuda:0",
                "rules": ["No new training or image generation", "Original detector instructions and crop/verify ladder", "Pending human remains unknown", "No Gold-arm image label transfer", "Do not use verified.summary.p as a known-only or completed-label estimate"]}
    save_new(OUT / "expected.json", expected)
    print(json.dumps({"status": "prepared", "images": 44, "new_training": False}))


def run():
    import msvcrt
    expected = read(OUT / "expected.json")
    assert sha(Path(__file__)) == expected["runner_sha256"]
    assert not (OUT / "controller-started.json").exists(), "No automatic duplicate controller or retry"
    check_hold()
    # Hold the existing scheduler lock while diagnostics use the GPUs. This
    # prevents an accidental concurrent main supervisor from acquiring ownership.
    with (MAIN / "supervisor.lock").open("r+b") as ownership:
        ownership.seek(0)
        msvcrt.locking(ownership.fileno(), msvcrt.LK_NBLCK, 1)
        save_new(OUT / "controller-started.json", {"pid": os.getpid(), "started_unix": time.time(), "expected_sha256": sha(OUT / "expected.json")})
        recovery = read(ROOT / "review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json")
        env = os.environ.copy()
        env.update(recovery["environment_allowlist"])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in recovery["environment_remove"]:
            env.pop(name, None)
        pipeline = ROOT / "scripts/v4_run_pipeline.py"
        stages = [("detect.qwen3vl", "observer", ["detect", "--manifest", str(OUT / "manifest.jsonl"), "--detector", "qwen3vl", "--device", "cuda:0"]),
                  ("detect.internvl", "observer", ["detect", "--manifest", str(OUT / "manifest.jsonl"), "--detector", "internvl", "--device", "cuda:0"]),
                  ("crop", "observer", ["crop", "--run", str(OUT), "--device", "cuda:0"]),
                  ("verify", "core", ["verify", "--run", str(OUT)])]
        try:
            for stage, environment, args in stages:
                check_hold()
                assert time.time() < expected["deadline"], "Original budget exhausted"
                for path, value in expected["source_sha256"].items():
                    assert sha(ROOT / path) == value, f"Detector source changed: {path}"
                assert sha(OUT / "manifest.jsonl") == expected["request_sha256"]
                for path, value in expected["image_sha256"].items():
                    assert sha(Path(path)) == value, f"Requested image changed: {path}"
                command = [str(ROOT / "envs" / environment / "python.exe"), "-B", "-u", str(pipeline), *args]
                with (OUT / f"{stage}.log").open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                    state("running", stage=stage, child_pid=proc.pid, command=command)
                    code = proc.wait(timeout=max(1, expected["deadline"] - time.time()))
                assert code == 0, f"{stage} failed with exit {code}; diagnose, do not auto-retry"
                save_new(OUT / f"{stage}.complete.json", {"completed_unix": time.time(), "command": command, "exit_code": code})
            verdicts = [json.loads(line) for line in (OUT / "verified.jsonl").read_text(encoding="utf-8").splitlines() if line]
            requested = {r["image_path"] for r in (json.loads(line) for line in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line)}
            assert len({r["image_path"] for r in verdicts}) == len(verdicts)
            assert {r["image_path"] for r in verdicts} <= requested
            state("detector_ladder_complete", n_requested=44, n_verdicts=len(verdicts), n_missing=len(requested) - len(verdicts), n_pending_human=sum(r["resolution"] == "pending_human" for r in verdicts), verified_sha256=sha(OUT / "verified.jsonl"), research_goal_complete=False)
        except Exception as exc:
            state("needs_diagnosis", error=str(exc), automatic_retry=False)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else run()
