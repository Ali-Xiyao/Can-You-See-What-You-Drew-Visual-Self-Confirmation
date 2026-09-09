"""One-time parent-only handoff to the unchanged four-round supervisor.

Keep the active measurement child alive and wait for its real exit. The resumed
supervisor re-enters that resumable measurement normally; this controller never
manufactures a successful stage sentinel. No training stage may be missing.
"""
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / "hold-step32"
RUN = ROOT / "runs/v4/decoupling-pilot-20260906"
PARENT_PID = 18576
PARENT_CREATED = 134331427797587987
STOP_CODE = 0xE0320004
NO_WINDOW = 0x08000000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    HERE.mkdir(exist_ok=True)
    with (HERE / name).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def event(kind, **details):
    with (HERE / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at_unix": time.time(), "event": kind, **details}) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Process:
    def __init__(self, pid, terminate=False):
        self.pid = pid
        self.api = ct.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "OpenProcess": ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
            "CloseHandle": ([wt.HANDLE], wt.BOOL),
            "WaitForSingleObject": ([wt.HANDLE, wt.DWORD], wt.DWORD),
            "GetExitCodeProcess": ([wt.HANDLE, ct.POINTER(wt.DWORD)], wt.BOOL),
            "GetProcessTimes": ([wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4, wt.BOOL),
            "QueryFullProcessImageNameW": ([wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)], wt.BOOL),
            "TerminateProcess": ([wt.HANDLE, wt.UINT], wt.BOOL),
        }
        for name, (args, result) in signatures.items():
            getattr(self.api, name).argtypes = args
            getattr(self.api, name).restype = result
        self.handle = self.api.OpenProcess(0x1000 | 0x100000 | int(terminate), False, pid)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())

    def identity(self):
        values = [wt.FILETIME() for _ in range(4)]
        require(self.api.GetProcessTimes(self.handle, *(ct.byref(x) for x in values)), "Cannot read process time")
        length = wt.DWORD(32768)
        buf = ct.create_unicode_buffer(length.value)
        require(self.api.QueryFullProcessImageNameW(self.handle, 0, buf, ct.byref(length)), "Cannot read process executable")
        return {"pid": self.pid, "created": (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime, "exe": buf.value}

    def alive(self):
        value = self.api.WaitForSingleObject(self.handle, 0)
        require(value in (0, 258), "Process wait failed")
        return value == 258

    def wait(self, deadline):
        while self.alive():
            require(time.time() < deadline, "Original deadline expired; no automatic retry or child termination")
            self.api.WaitForSingleObject(self.handle, 1000)
        code = wt.DWORD()
        require(self.api.GetExitCodeProcess(self.handle, ct.byref(code)), "Cannot obtain real process exit code")
        return code.value

    def close(self):
        self.api.CloseHandle(self.handle)


def parents():
    class Entry(ct.Structure):
        _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("pid", wt.DWORD),
                    ("heap", ct.c_size_t), ("module", wt.DWORD), ("threads", wt.DWORD),
                    ("parent", wt.DWORD), ("priority", wt.LONG), ("flags", wt.DWORD),
                    ("exe", wt.WCHAR * 260)]
    api = ct.WinDLL("kernel32", use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    api.CreateToolhelp32Snapshot.restype = wt.HANDLE
    api.CloseHandle.argtypes = [wt.HANDLE]
    for name in ("Process32FirstW", "Process32NextW"):
        getattr(api, name).argtypes = [wt.HANDLE, ct.POINTER(Entry)]
        getattr(api, name).restype = wt.BOOL
    h = api.CreateToolhelp32Snapshot(2, 0)
    require(h != ct.c_void_p(-1).value, "Process inventory unavailable")
    try:
        row = Entry()
        row.dwSize = ct.sizeof(row)
        ok = api.Process32FirstW(h, ct.byref(row))
        result = {}
        while ok:
            result[row.pid] = row.parent
            ok = api.Process32NextW(h, ct.byref(row))
        return result
    finally:
        api.CloseHandle(h)


def frozen(expected):
    require(time.time() < expected["deadline"], "Original deadline expired")
    require(not (RUN / "rounds/round-004").exists(), "Another training round already started")
    for path, value in expected["frozen_sha256"].items():
        require(sha(ROOT / path) == value, f"Frozen input changed: {path}")
    for path, value in expected["completed_training_sha256"].items():
        require(sha(RUN / path) == value, f"Completed training evidence changed: {path}")


def check_bound(expected, parent, child):
    frozen(expected)
    require(parent.identity() == expected["parent"], "Parent identity changed")
    require(child.identity() == expected["child"], "Child identity changed")
    require(parent.alive() and child.alive(), "Recorded stage already ended")
    require(parents().get(child.pid) == parent.pid, "Child-parent relationship differs")
    require(read(RUN / "state.json") == expected["state"], "Stage changed; prepare a fresh binding before any action")


def prepare():
    state = read(RUN / "state.json")
    # Only the supervisor can be stopped. The measurement child, including any
    # gradient export, remains alive and must exit successfully before resume.
    require(state["supervisor_pid"] == PARENT_PID and state["through_round"] == 10, "Unexpected owner or limit")
    require(state["stage"] in {"rfo_gold.step-00032.detect.internvl", "rfo_gold.step-00032.crop", "rfo_gold.step-00032.gradient"}, "Wait for reviewed measurement stage; no generic PID stop")
    manifest = read(RUN / "run_manifest.json")
    parent = Process(PARENT_PID)
    child = Process(state["child_pid"])
    try:
        require(parent.identity()["created"] == PARENT_CREATED, "Supervisor PID reused")
        require(Path(parent.identity()["exe"]).resolve() == ROOT / "envs/core/python.exe", "Wrong supervisor executable")
        require(Path(child.identity()["exe"]).resolve() == Path(state["command"][0]).resolve(), "Wrong child executable")
        sources = dict(manifest["source_sha256"])
        sources["configs/v4_decoupling_pilot.yaml"] = manifest["config_sha256"]
        sources["docs/prereg/2026-09-06-decoupling-pilot.md"] = manifest["protocol_sha256"]
        training = {}
        for index in range(4):
            for name in (f"stage-completion/round-{index:03d}.train.json", f"rounds/round-{index:03d}/DONE.json"):
                training[name] = sha(RUN / name)
        expected = {"state": state, "parent": parent.identity(), "child": child.identity(),
                    "frozen_sha256": sources, "completed_training_sha256": training,
                    "deadline": manifest["started_unix"] + manifest["config"]["pilot"]["max_wall_hours"] * 3600,
                    "started_unix": manifest["started_unix"], "through_round": 4,
                    "controller_sha256": sha(Path(__file__))}
        check_bound(expected, parent, child)
        save("expected.json", expected)
        print(json.dumps({"status": "prepared", "stage": state["stage"], "child": child.pid, "through_round": 4}))
    finally:
        parent.close()
        child.close()


def execute(check_only=False):
    expected = read(HERE / "expected.json")
    require(sha(Path(__file__)) == expected["controller_sha256"], "Reviewed controller changed")
    require(not (HERE / "control-started.json").exists(), "One-time handoff already attempted")
    child = Process(expected["child"]["pid"])
    parent = None
    started = False
    try:
        parent = Process(PARENT_PID, terminate=not check_only)
        check_bound(expected, parent, child)
        if check_only:
            print(json.dumps({"status": "check_passed", "no_process_signaled": True, "through_round": 4}))
            return
        save("control-started.json", {"at_unix": time.time(), "helper_pid": os.getpid(), "expected_sha256": sha(HERE / "expected.json"), "reason": "Explicit user request: finish step32 measurement, stop new training, start selection-benefit diagnosis."})
        started = True
        check_bound(expected, parent, child)
        require(parent.api.TerminateProcess(parent.handle, STOP_CODE), "Unable to stop bound supervisor")
        require(parent.wait(min(time.time() + 10, expected["deadline"])) == STOP_CODE, "Unexpected parent exit")
        event("parent_only_stopped", pid=PARENT_PID, child_pid=child.pid, child_terminated=False, normal_parent_exit=False)
        require(read(RUN / "state.json") == expected["state"], "Stage transition raced the handoff; no restart")
        require(not [pid for pid, ppid in parents().items() if ppid == PARENT_PID and pid != child.pid], "Unexpected extra child; no restart")
        code = child.wait(expected["deadline"])
        event("original_measurement_child_exited", pid=child.pid, exit_code=code)
        require(code == 0, "Original measurement failed; no automatic retry")
        frozen(expected)
        # Re-enter cached measurement normally. The supervisor owns its new
        # success sentinel after a real successful subprocess exit.
        recovery = read(ROOT / "review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json")
        env = os.environ.copy()
        env.update(recovery["environment_allowlist"])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for name in recovery["environment_remove"]:
            env.pop(name, None)
        require(env["SELFSIGHT_MODEL_ROOT"] == "H:\\selfsight-models", "Original model root missing")
        command = [str(ROOT / "envs/core/python.exe"), "-u", str(ROOT / "scripts/run_decoupling_pilot.py"), "--outdir", str(RUN), "--config", str(ROOT / "configs/v4_decoupling_pilot.yaml"), "--through-round", "4"]
        with (HERE / "bounded-supervisor.log").open("x", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            save("bounded-supervisor.json", {"pid": proc.pid, "command": command, "deadline": expected["deadline"]})
            event("bounded_supervisor_started", pid=proc.pid, through_round=4)
            result = proc.wait(timeout=max(1, expected["deadline"] - time.time()))
        require(result == 0, f"Bounded supervisor failed: {result}")
        frozen(expected)
        state = read(RUN / "state.json")
        require(state["supervisor_pid"] == proc.pid and state["status"] == "canary_complete" and state["completed_rounds"] == 4 and state["through_round"] == 4 and state["research_goal_complete"] is False, "Final bounded state differs")
        save("hold-complete.json", {"at_unix": time.time(), "state": state, "supervisor_exit_code": result, "new_training_started": False})
        event("step32_measurement_complete_training_paused")
    except Exception as exc:
        if started:
            event("needs_diagnosis", error=str(exc), automatic_retry=False)
        raise
    finally:
        if parent is not None:
            parent.close()
        child.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute(check_only=args.check_only)
