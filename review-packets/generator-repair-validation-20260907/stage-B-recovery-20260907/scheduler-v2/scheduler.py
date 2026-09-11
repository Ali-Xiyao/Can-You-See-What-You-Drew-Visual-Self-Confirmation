"""Scheduling-only handoff: fixed observer shards and a shared whole-batch queue.

The frozen recovery workers are imported/called unchanged.  An adopted Windows
process handle preserves its real exit status even after its old parent exits.
No adopted worker is terminated until commit, old-parent exit and lock ownership.
"""
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import time

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S.parent))
import runtime_common as runtime

R = runtime.R


def identity_matches(actual, expected):
    return (actual['pid'] == expected['pid']
            and abs(actual['create_time'] - expected['create_time']) < 0.001
            and actual['cmdline'] == expected['cmdline'])


def process_identity(pid):
    import psutil
    proc = psutil.Process(pid)
    return {'pid': proc.pid, 'create_time': proc.create_time(), 'cmdline': proc.cmdline()}


class WindowsHandle:
    """Read-only handle until ownership; WAIT distinguishes active from exit 259."""
    def __init__(self, expected, api=None, identity=process_identity):
        if api is None:
            if os.name != 'nt':
                raise RuntimeError('Live handoff requires Windows process handles')
            api = ctypes.WinDLL('kernel32', use_last_error=True)
            api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            api.OpenProcess.restype = wintypes.HANDLE
            api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            api.WaitForSingleObject.restype = wintypes.DWORD
            api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            api.GetExitCodeProcess.restype = wintypes.BOOL
            api.CloseHandle.argtypes = (wintypes.HANDLE,)
            api.CloseHandle.restype = wintypes.BOOL
        self.api = api
        self.expected = expected
        self.pid = expected['pid']
        self.handle = None
        assert identity_matches(identity(self.pid), expected), 'Process identity changed before adoption'
        self.handle = api.OpenProcess(0x1000 | 0x00100000, False, self.pid)
        if not self.handle:
            raise OSError('Cannot open query/synchronize handle for PID ' + str(self.pid))
        try:
            assert identity_matches(identity(self.pid), expected), 'Process identity changed during adoption'
        except BaseException:
            self.close()
            raise

    def poll(self):
        result = self.api.WaitForSingleObject(self.handle, 0)
        if result == 0x102:
            return None
        if result != 0:
            raise OSError('WaitForSingleObject failed: ' + str(result))
        code = wintypes.DWORD()
        if not self.api.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise OSError('GetExitCodeProcess failed')
        return int(code.value)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def dispatch_jobs(batch_ids, completed_batches, completed_shards, active,
                  deadline, current_time, failed=False):
    """Pure scheduler: shard i stays on GPU i, observers precede remaining batches."""
    if failed or current_time >= deadline:
        return []
    jobs = list(active)
    occupied = [job['device'] for job in jobs]
    assert len(occupied) == len(set(occupied)), 'Two jobs occupy one GPU'
    assert set(occupied) <= {'cuda:0', 'cuda:1'}
    active_batches = {job['batch'] for job in jobs if job['kind'] == 'detection'}
    assert len(active_batches) == sum(job['kind'] == 'detection' for job in jobs)
    for job in jobs:
        if job['kind'] == 'observation':
            assert job['device'] == f"cuda:{job['shard']}", 'Frozen observer device changed'
    pending = sorted(set(batch_ids) - set(completed_batches) - active_batches)
    out = []
    for shard in (0, 1):
        device = f'cuda:{shard}'
        if device in occupied:
            continue
        if shard not in completed_shards:
            out.append({'name': f'observe-gpu{shard}', 'kind': 'observation',
                        'shard': shard, 'device': device})
        elif pending:
            batch = pending.pop(0)
            out.append({'name': 'detect-' + batch, 'kind': 'detection',
                        'batch': batch, 'device': device})
    return out


def worker_command(job, deadline):
    if job['kind'] == 'observation':
        assert job['device'] == f"cuda:{job['shard']}"
        env, script = 'showo2', 'observation_resume.py'
        args = ['--shard', str(job['shard']), '--deadline', str(deadline)]
    elif job['kind'] == 'detection':
        env, script = 'observer', 'detector_recovery.py'
        args = ['detect-batch', '--batch', job['batch'], '--deadline', str(deadline),
                '--device', job['device']]
    else:
        assert job['kind'] == 'analysis'
        env, script, args = 'core', 'analyze_recovery.py', []
    return [str(runtime.ROOT / f'envs/{env}/python.exe'), '-B', '-u', str(R / script), *args]


def validate_artifact(job, plan):
    if job['kind'] == 'detection':
        batch = next(b for b in plan['batches'] if b['id'] == job['batch'])
        folder = R / 'generation' / job['batch']
        done = runtime.read(folder / 'verification-complete.json')
        assert done['n_images'] == len(batch['scene_ids']) * 6
        assert done['device'] == job['device'], 'Detection completion device mismatch'
        assert done['detector_snapshot_digest'] == plan['detector_snapshot_digest']
        assert {'qwen3vl', 'internvl', 'verified'} <= set(done['sha256'])
        for name, expected in done['sha256'].items():
            path = folder / ('verified.jsonl' if name == 'verified' else f'detections.{name}.jsonl')
            assert runtime.sha(path) == expected, path
        assert len(runtime.lines(folder / 'verified.jsonl')) == done['n_images']
        return {'path': str(folder / 'verification-complete.json'),
                'sha256': runtime.sha(folder / 'verification-complete.json')}
    if job['kind'] == 'observation':
        shard = job['shard']
        assert job['device'] == f'cuda:{shard}'
        folder = R / 'observations' / f'gpu{shard}'
        done = runtime.read(folder / 'complete.json')
        assert done['n_scenes'] == len(plan['observation_shards'][str(shard)])
        assert done['models'] == len(plan['models'])
        model_hashes = {}
        for model in plan['models']:
            item = folder / model['id']
            complete = runtime.read(item / 'complete.json')
            assert complete['n_images'] == 6 * done['n_scenes']
            assert complete['parameter_digest_before'] == complete['parameter_digest_after'] == model['parameter_digest']
            assert complete['answers_sha256'] == runtime.sha(item / 'answers.jsonl')
            answers = runtime.lines(item / 'answers.jsonl')
            assert len(answers) == complete['n_images'] * len(plan['conditions'])
            assert sum(len(row['observation']['answers']) for row in answers) == complete['n_answers']
            assert all(row['device'] == job['device'] for row in answers)
            model_hashes[model['id']] = runtime.sha(item / 'complete.json')
        return {'path': str(folder / 'complete.json'), 'sha256': runtime.sha(folder / 'complete.json'),
                'model_completion_sha256': model_hashes}
    result = R / 'analysis-recovery/results.json'
    runtime.read(result)
    return {'path': str(result), 'sha256': runtime.sha(result)}


def validate_handoff(handoff, plan, recovery):
    assert handoff['schema_version'] == 1
    assert handoff['deadline'] == recovery['deadline'] == runtime.approved_deadline()
    assert handoff['recovery_plan_sha256'] == runtime.sha(R / 'recovery-plan.json')
    known = {b['id'] for b in plan['batches']}
    complete = handoff['existing_complete_batches']
    shards = handoff.get('completed_observer_shards', [])
    assert len(complete) == len(set(complete)) and set(complete) <= known
    assert len(shards) == len(set(shards)) and set(shards) <= {0, 1}
    active = handoff['active_jobs']
    assert len({j['name'] for j in active}) == len(active)
    assert len({j['pid'] for j in active}) == len(active)
    assert handoff['old_controller']['pid'] not in {j['pid'] for j in active}
    for job in active:
        assert job['kind'] in ('detection', 'observation')
        assert job['cmdline'] == worker_command(job, handoff['deadline']), 'Unexpected worker command'
        if job['kind'] == 'detection':
            assert job['batch'] in known - set(complete)
        else:
            assert job['shard'] not in shards
    dispatch_jobs(known, complete, shards, active, handoff['deadline'], 0)
    for batch in complete:
        marker = runtime.read(R / 'generation' / batch / 'verification-complete.json')
        validate_artifact({'kind': 'detection', 'batch': batch, 'device': marker['device']}, plan)
    for shard in shards:
        validate_artifact({'kind': 'observation', 'shard': shard, 'device': f'cuda:{shard}'}, plan)


def terminate_identity(expected):
    """Terminate only the still-matching process and its captured descendants."""
    import psutil
    try:
        if not identity_matches(process_identity(expected['pid']), expected):
            return
        parent = psutil.Process(expected['pid'])
        targets = parent.children(recursive=True) + [parent]
        snapshots = [(p, p.create_time()) for p in targets]
        for proc, created in snapshots:
            try:
                if proc.create_time() == created:
                    proc.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs([p for p, _ in snapshots], timeout=5)
        by_pid = {p.pid: created for p, created in snapshots}
        for proc in alive:
            try:
                if proc.create_time() == by_pid[proc.pid]:
                    proc.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def write_once(path, value):
    runtime.save_new(path, value)


def event(kind, **value):
    with (S / 'events.jsonl').open('a', encoding='utf-8') as handle:
        runtime.append(handle, {'at_utc': runtime.now(), 'event': kind, **value})


def state(active, complete_batches, complete_shards, deadline, status):
    path = S / 'state.json'
    temporary = S / 'state.json.tmp'
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump({'at_utc': runtime.now(), 'status': status, 'pid': os.getpid(),
                   'deadline': deadline, 'active_jobs': list(active.values()),
                   'completed_batches': sorted(complete_batches),
                   'completed_observer_shards': sorted(complete_shards)}, handle, indent=2)
        handle.flush()
    os.replace(temporary, path)


def main():
    import msvcrt
    import psutil
    ownership = False
    handles, processes, jobs, active, logs = {}, {}, {}, {}, []
    parent_handle = None
    lock = None
    try:
        assert not (S / 'adoption-ready.json').exists(), 'No automatic duplicate handoff'
        assert not (S / 'controller-started.json').exists(), 'No automatic duplicate scheduler'
        recovery = runtime.verify_recovery()
        plan = runtime.original_plan()
        handoff_path = S / 'handoff.json'
        handoff = runtime.read(handoff_path)
        handoff_sha = runtime.sha(handoff_path)
        validate_handoff(handoff, plan, recovery)
        deadline = handoff['deadline']
        runtime.check_time(deadline)
        parent_handle = WindowsHandle(handoff['old_controller'])
        for job in handoff['active_jobs']:
            handles[job['name']] = WindowsHandle(job)
            jobs[job['name']] = dict(job)
            active[job['name']] = dict(job)
        write_once(S / 'adoption-ready.json', {
            'at_utc': runtime.now(), 'scheduler_pid': os.getpid(),
            'scheduler_identity': process_identity(os.getpid()),
            'handoff_sha256': handoff_sha, 'adopted_pids': [j['pid'] for j in jobs.values()],
            'read_only_process_handles_open': True, 'ownership_committed': False})
        event('adoption_ready', adopted_pids=[j['pid'] for j in jobs.values()])
        commit_timeout = min(deadline, time.time() + 300)
        while not (S / 'handoff-commit.json').exists():
            assert time.time() < commit_timeout, 'No handoff commit within 300 seconds; workers left alone'
            runtime.check_time(deadline)
            time.sleep(1)
        commit = runtime.read(S / 'handoff-commit.json')
        assert commit['handoff_sha256'] == handoff_sha == runtime.sha(handoff_path)
        assert commit['scheduler_pid'] == os.getpid()
        assert commit['old_controller_terminated'] is True
        assert parent_handle.poll() is not None, 'Old controller is still running'
        lock = (runtime.MAIN / 'supervisor.lock').open('r+b')
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        ownership = True
        runtime.check_time(deadline)
        complete_batches = set(handoff['existing_complete_batches'])
        complete_shards = set(handoff.get('completed_observer_shards', []))
        write_once(S / 'controller-started.json', {
            'at_utc': runtime.now(), 'pid': os.getpid(), 'deadline': deadline,
            'identity': process_identity(os.getpid()), 'handoff_sha256': handoff_sha,
            'recovery_plan_sha256': runtime.sha(R / 'recovery-plan.json'),
            'scheduler_sha256': runtime.sha(Path(__file__)), 'new_optimizer_steps': 0,
            'observer_devices_unchanged': True, 'whole_batch_shared_detection_queue': True})
        event('ownership_committed')
        for name, job in jobs.items():
            write_once(S / f'{name}-started.json', {**job, 'at_utc': runtime.now(),
                       'adopted': True, 'original_log_path': job.get('log_path')})

        def start(job):
            runtime.check_time(deadline)
            name = job['name']
            assert name not in jobs, 'Refuse repeated job name'
            cmd = worker_command(job, deadline)
            log = (S / f'{name}.log').open('x', encoding='utf-8')
            logs.append(log)
            proc = subprocess.Popen(cmd, cwd=runtime.ROOT, env=runtime.environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            # Register Popen before further checks so a failed bookkeeping step
            # cannot leave an unowned worker running.
            processes[name] = proc
            identity = process_identity(proc.pid)
            record = {**job, **identity, 'log_path': str(S / f'{name}.log')}
            jobs[name] = record
            active[name] = record
            write_once(S / f'{name}-started.json', {**record, 'at_utc': runtime.now(), 'adopted': False})
            event('job_started', **record)

        def poll(name):
            return handles[name].poll() if name in handles else processes[name].poll()

        def collect_finished():
            for name, job in list(active.items()):
                code = poll(name)
                if code is None:
                    continue
                if code != 0:
                    write_once(S / f'{name}-failed.json', {'at_utc': runtime.now(), 'exit_code': code, **job})
                    raise RuntimeError(f'{name}: real process exit {code}')
                try:
                    artifact = validate_artifact(job, plan)
                except BaseException as exc:
                    write_once(S / f'{name}-failed.json', {'at_utc': runtime.now(), 'exit_code': code,
                               'artifact_error': str(exc), **job})
                    raise
                write_once(S / f'{name}-completed.json', {'at_utc': runtime.now(), 'exit_code': code,
                           'artifact': artifact, **job})
                event('job_completed', name=name, exit_code=code, artifact=artifact)
                if job['kind'] == 'detection':
                    complete_batches.add(job['batch'])
                elif job['kind'] == 'observation':
                    complete_shards.add(job['shard'])
                del active[name]

        batch_ids = [b['id'] for b in plan['batches']]
        while True:
            runtime.check_time(deadline)
            collect_finished()
            if complete_batches == set(batch_ids) and complete_shards == {0, 1}:
                assert not active
                break
            for job in dispatch_jobs(batch_ids, complete_batches, complete_shards,
                                     active.values(), deadline, time.time()):
                start(job)
            state(active, complete_batches, complete_shards, deadline, 'running')
            time.sleep(5)
        write_once(R / 'gpu-complete.json', {'at_utc': runtime.now(), 'n_images': 900,
                   'expected_answers': 108000, 'original_failed_attempt_preserved': True,
                   'scheduler_revision': str(S)})
        start({'name': 'analysis', 'kind': 'analysis'})
        while active:
            runtime.check_time(deadline)
            collect_finished()
            state(active, complete_batches, complete_shards, deadline, 'analysis')
            if active:
                time.sleep(5)
        result = runtime.read(R / 'analysis-recovery/results.json')
        completion = {'at_utc': runtime.now(), 'exit_code': 0,
                      'results_sha256': runtime.sha(R / 'analysis-recovery/results.json'),
                      'pending_review_images': result['gold']['review_images'],
                      'research_goal_complete': False, 'new_optimizer_steps': 0,
                      'scheduler_revision': str(S)}
        write_once(R / 'controller-complete.json', completion)
        write_once(S / 'controller-complete.json', completion)
        state(active, complete_batches, complete_shards, deadline, 'complete')
        event('controller_completed', **completion)
    except BaseException as exc:
        failure = {'at_utc': runtime.now(), 'error': f'{type(exc).__name__}: {exc}',
                   'automatic_retry': False, 'ownership_committed': ownership,
                   'scheduler_revision': str(S)}
        if ownership:
            cleanup_errors = []
            for name, job in jobs.items():
                try:
                    handle = handles.get(name)
                    if handle is None or handle.poll() is None:
                        terminate_identity(job)
                except BaseException as cleanup:
                    cleanup_errors.append(f'{name}: {cleanup}')
            # Covers a process started immediately before identity capture fails.
            for name, proc in processes.items():
                if name not in jobs and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=10)
                    except BaseException as cleanup:
                        cleanup_errors.append(f'{name}: {cleanup}')
            failure['cleanup_errors'] = cleanup_errors
            for path in (R / 'controller-failed.json', S / 'controller-failed.json'):
                if not path.exists():
                    write_once(path, failure)
        elif not (S / 'adoption-failed.json').exists():
            write_once(S / 'adoption-failed.json', failure)
        event('controller_failed' if ownership else 'adoption_failed', **failure)
        raise
    finally:
        for handle in handles.values():
            handle.close()
        if parent_handle is not None:
            parent_handle.close()
        if lock is not None:
            lock.close()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
