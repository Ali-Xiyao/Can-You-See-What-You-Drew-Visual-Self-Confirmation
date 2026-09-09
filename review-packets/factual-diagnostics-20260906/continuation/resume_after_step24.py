"""Wait for the bound step24 owners, then resume the original ten-round pilot.

This one-time Windows controller never signals or kills a process. --check-only
is read-only; --execute holds real handles for owners 18492/26896, waits for
their actual successful exits and validated hold evidence, then starts the
unchanged supervisor with --through-round 10. The original deadline is retained.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from ctypes import wintypes as wt
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
OLD = RUN / "runtime-check/method-review-hold-20260906"
OLD_HELPER, OLD_SUPERVISOR = 18492, 26896
EXPECTED_SHA256 = "ccb8943b1a801c03d0888d7d52b9b24aacf2fb2ee1f8243c81c7b52293bf7daa"
NO_WINDOW = 0x08000000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(name, data):
    with (HERE / name).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def event(kind, **data):
    with (HERE / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at_unix": time.time(), "event": kind, **data}, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class BoundProcess:
    """Query/synchronize rights only. No process-control permissions or APIs."""
    def __init__(self, pid):
        self.pid = pid
        self.api = ct.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "OpenProcess": ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
            "CloseHandle": ([wt.HANDLE], wt.BOOL),
            "WaitForSingleObject": ([wt.HANDLE, wt.DWORD], wt.DWORD),
            "GetExitCodeProcess": ([wt.HANDLE, ct.POINTER(wt.DWORD)], wt.BOOL),
            "GetProcessTimes": ([wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4, wt.BOOL),
            "QueryFullProcessImageNameW": ([wt.HANDLE, wt.DWORD, wt.LPWSTR,
                                           ct.POINTER(wt.DWORD)], wt.BOOL),
        }
        for name, (args, result) in signatures.items():
            getattr(self.api, name).argtypes = args
            getattr(self.api, name).restype = result
        self.handle = self.api.OpenProcess(0x1000 | 0x100000, False, pid)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())

    def alive(self):
        result = self.api.WaitForSingleObject(self.handle, 0)
        require(result in (0, 258), f"Wait failed for PID {self.pid}: {result}")
        return result == 258

    def identity(self):
        times = [wt.FILETIME() for _ in range(4)]
        require(self.api.GetProcessTimes(self.handle, *(ct.byref(x) for x in times)),
                f"Cannot read creation time for {self.pid}")
        length = wt.DWORD(32768)
        text = ct.create_unicode_buffer(length.value)
        require(self.api.QueryFullProcessImageNameW(self.handle, 0, text, ct.byref(length)),
                f"Cannot read executable for {self.pid}")
        return {"creation_filetime": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime,
                "ExecutablePath": text.value}

    def exit_code(self):
        require(not self.alive(), f"PID {self.pid} has not exited")
        value = wt.DWORD()
        require(self.api.GetExitCodeProcess(self.handle, ct.byref(value)), "Real exit-code query failed")
        return value.value

    def wait(self, deadline):
        while self.alive():
            remaining = deadline - time.time()
            require(remaining > 0, "Original deadline reached; leave existing processes untouched")
            result = self.api.WaitForSingleObject(self.handle, min(5000, max(1, int(remaining * 1000))))
            require(result in (0, 258), f"Wait failed for {self.pid}")
        return self.exit_code()

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def inventory():
    query = ("$OutputEncoding=[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
             "@(Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
             "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", query],
                            check=True, capture_output=True, text=True, encoding="utf-8", timeout=30,
                            creationflags=NO_WINDOW)
    rows = json.loads(result.stdout)
    return rows if isinstance(rows, list) else [rows]


def check_other_owners(records, allow_old):
    supervisors = {p["ProcessId"] for p in records
                   if "run_decoupling_pilot.py" in (p.get("CommandLine") or "")}
    require(supervisors.issubset({OLD_SUPERVISOR} if allow_old else set()),
            f"Unexpected active supervisor PIDs: {sorted(supervisors)}")
    controllers = {p["ProcessId"] for p in records if p["ProcessId"] != os.getpid()
                   and Path(__file__).name in (p.get("CommandLine") or "")}
    require(not controllers, f"Another continuation controller is active: {sorted(controllers)}")


def frozen_check(expected, before_resume=True):
    require(time.time() < expected["original_deadline_unix"], "Original 60-hour deadline expired")
    for name, value in expected["frozen_sha256"].items():
        require(sha(ROOT / name) == value, f"Frozen input changed: {name}")
    for name, value in expected["existing_stage_sha256"].items():
        require(sha(RUN / "stage-completion" / name) == value, f"Existing stage changed: {name}")
    if before_resume:
        require(not (RUN / "rounds/round-003").exists(), "Round 3 already started under another owner")
        require(all(not (RUN / "checkpoints" / a / "round-003").exists() for a in ("naive", "rfo_gold")),
                "Round 3 checkpoint already exists")


def bind_check(expected, owners):
    frozen_check(expected)
    records = inventory()
    check_other_owners(records, allow_old=True)
    by_pid = {p["ProcessId"]: p for p in records}
    result = {}
    for owner in owners:
        actual, wanted = owner.identity(), expected["processes"][str(owner.pid)]
        require(actual["creation_filetime"] == wanted["creation_filetime"], f"PID reused: {owner.pid}")
        require(Path(actual["ExecutablePath"]).resolve() == Path(wanted["ExecutablePath"]).resolve(),
                f"Executable changed: {owner.pid}")
        active = owner.alive()
        if active:
            require(owner.pid in by_pid, f"Active bound PID absent from inventory: {owner.pid}")
            for key in ("ExecutablePath", "CommandLine", "ParentProcessId"):
                require(by_pid[owner.pid][key] == wanted[key], f"Owner {owner.pid} {key} changed")
        else:
            require(owner.exit_code() == 0, f"Owner {owner.pid} already failed")
        result[str(owner.pid)] = {**wanted, "alive_at_binding": active,
                                  "real_exit_code": None if active else owner.exit_code()}
    state = read(RUN / "state.json")
    require(state["supervisor_pid"] == OLD_SUPERVISOR and state["through_round"] == 3 and
            state["started_unix"] == expected["original_started_unix"], "Step24 owner/state changed")
    return result


def stage_names(step):
    return [f"{arm}.step-{step:05d}.{suffix}" for arm in ("naive", "rfo_gold")
            for suffix in ("generate", "detect.qwen3vl", "detect.internvl", "crop", "verify", "gradient")] + [
                f"step-{step:05d}.{suffix}" for suffix in ("score", "report", "gradient-sensitivity", "plot")]


def stage_evidence(step):
    evidence = {}
    for stage in stage_names(step):
        path = RUN / "stage-completion" / f"{stage}.json"
        row = read(path)
        require(row["stage"] == stage and isinstance(row["command"], list) and row["command"],
                f"Invalid completion sentinel: {stage}")
        evidence[str(path)] = sha(path)
    return evidence


def hold_evidence(expected):
    frozen_check(expected)
    hold_path = OLD / "hold-complete.json"
    hold = read(hold_path)
    state = read(RUN / "state.json")
    require(hold["supervisor_exit_code"] == 0 and hold["next_training_round_not_started"] is True,
            "Old helper did not record successful bounded completion")
    require(hold["state"] == state and state["supervisor_pid"] == OLD_SUPERVISOR and
            state["status"] == "canary_complete" and state["through_round"] == 3 and
            state["completed_rounds"] == 3 and state["research_goal_complete"] is False and
            state["started_unix"] == expected["original_started_unix"], "Hold state does not match original owners/budget")
    events = [json.loads(x) for x in (OLD / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    require(events and events[-1]["event"] == "step24_hold_complete", "Old helper has no final successful event")
    return {"hold": hold, "hold_sha256": sha(hold_path), "old_events_sha256": sha(OLD / "events.jsonl"),
            "step24_stage_sha256": stage_evidence(24), "state_sha256": sha(RUN / "state.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(os.name == "nt", "Controller is bound to Windows processes")
    require(sha(HERE / "expected.json") == EXPECTED_SHA256, "Expected bindings changed")
    expected = read(HERE / "expected.json")
    require(Path(expected["root"]) == ROOT and Path(expected["run"]) == RUN, "Workspace binding differs")
    require(expected["original_max_wall_hours"] == 60 and expected["resume_through_round"] == 10 and
            expected["manifest"]["config"]["training"]["rounds"] == 10 and
            expected["original_deadline_unix"] == expected["original_started_unix"] + 60 * 3600,
            "Original scientific schedule/deadline differs")
    owners, lock, armed = [], None, False
    try:
        if args.execute:
            import msvcrt
            lock = (HERE / "controller.lock").open("a+b")
            if lock.tell() == 0:
                lock.write(b" ")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        require(not (HERE / "controller-started.json").exists(), "One-time continuation already attempted")
        for pid in (OLD_HELPER, OLD_SUPERVISOR):
            owners.append(BoundProcess(pid))
        identities = bind_check(expected, owners)
        if args.check_only:
            print(json.dumps({"status": "check_passed", "owners": identities,
                              "would_resume_through_round": 10, "original_deadline_unix": expected["original_deadline_unix"],
                              "no_process_signaled_or_started": True, "no_files_written": True}, indent=2))
            return
        save_new("controller-started.json", {"at_unix": time.time(), "controller_pid": os.getpid(),
                 "controller_sha256": sha(__file__), "expected_sha256": EXPECTED_SHA256,
                 "owners": identities, "reason": expected["reason"],
                 "original_deadline_unix": expected["original_deadline_unix"]})
        armed = True
        # Both real handles remain open. The helper must finish its final hold
        # verification as well as the bounded supervisor itself exiting cleanly.
        for owner in reversed(owners):
            event("waiting_for_bound_owner", pid=owner.pid)
            code = owner.wait(expected["original_deadline_unix"])
            event("bound_owner_exited", pid=owner.pid, exit_code=code)
            require(code == 0, f"Original owner {owner.pid} failed; leave run untouched")
        evidence = hold_evidence(expected)
        check_other_owners(inventory(), allow_old=False)
        save_new("accepted-step24-hold.json", evidence)
        require(hold_evidence(expected) == evidence, "Hold evidence changed before launch")
        check_other_owners(inventory(), allow_old=False)
        frozen_check(expected)
        command = [str(ROOT / "envs/core/python.exe"), "-u", str(ROOT / "scripts/run_decoupling_pilot.py"),
                   "--outdir", str(RUN), "--config", str(ROOT / "configs/v4_decoupling_pilot.yaml"),
                   "--through-round", "10"]
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", PYTHONNOUSERSITE="1")
        with (HERE / "resumed-supervisor.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            save_new("resumed-supervisor.json", {"pid": process.pid, "command": command,
                     "at_unix": time.time(), "original_started_unix": expected["original_started_unix"],
                     "original_deadline_unix": expected["original_deadline_unix"], "through_round": 10})
            event("original_pilot_resumed", pid=process.pid, through_round=10)
            startup_deadline = min(time.time() + 60, expected["original_deadline_unix"])
            while True:
                state = read(RUN / "state.json")
                if state.get("supervisor_pid") == process.pid:
                    require(state["through_round"] == 10 and state["started_unix"] == expected["original_started_unix"],
                            "New supervisor changed limit or original start time")
                    save_new("resumed-state.json", state)
                    break
                require(process.poll() is None, "New supervisor exited before establishing ownership")
                require(time.time() < startup_deadline, "New supervisor ownership not confirmed; no retry")
                time.sleep(0.25)
            result = process.wait(timeout=max(1, expected["original_deadline_unix"] - time.time()))
        event("resumed_supervisor_exited", pid=process.pid, exit_code=result)
        require(result == 0, f"Resumed supervisor failed with {result}; no retry")
        frozen_check(expected, before_resume=False)
        state = read(RUN / "state.json")
        require(state["supervisor_pid"] == process.pid and state["status"] == "pilot_complete" and
                state["through_round"] == state["completed_rounds"] == 10 and
                state["started_unix"] == expected["original_started_unix"] and
                state["research_goal_complete"] is False, "Final execution status does not match original pilot")
        done = read(RUN / "rounds/round-009/DONE.json")
        require(done["round"] == 9 and {a["arm"] for a in done["arms"]} == {"naive", "rfo_gold"},
                "Final training completion missing")
        for arm in ("naive", "rfo_gold"):
            cp = read(RUN / "checkpoints" / arm / "round-009/manifest.json")
            require(cp["step"] == 80 and cp["round_index"] == 9 and cp["metadata"]["arm"] == arm,
                    "Final checkpoint identity differs")
        require(not (RUN / "rounds/round-010").exists(), "Unexpected extra training round")
        save_new("continuation-complete.json", {"at_unix": time.time(), "state": state,
                 "real_supervisor_exit_code": result, "step80_stage_sha256": stage_evidence(80),
                 "research_goal_complete": False, "note": "Execution finished; this does not assert D* or D_g."})
    except Exception as exc:
        if armed:
            event("needs_diagnosis", error=str(exc), automatic_retry=False, processes_signaled=False)
        print(f"REFUSED / NEEDS DIAGNOSIS: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        for owner in owners:
            owner.close()
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
