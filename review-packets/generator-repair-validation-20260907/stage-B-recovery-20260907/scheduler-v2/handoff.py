"""One-time transactional transfer of the dispatcher, retaining active workers."""
from pathlib import Path
import sys, os, time, json, subprocess, hashlib
import psutil

S = Path(__file__).resolve().parent
R = S.parent
sys.path.insert(0, str(R))
from runtime_common import (ROOT, MAIN, read, save_new, now, sha, environment,
                            verify_recovery, approved_deadline)
sys.path.insert(0, str(S))
from scheduler import WindowsHandle


def prepare_commit(path, payload):
    """Make commit bytes durable while the original dispatcher is still alive."""
    with Path(path).open('xb') as handle:
        handle.write((json.dumps(payload, indent=2) + '\n').encode('utf-8'))
        handle.flush()
        os.fsync(handle.fileno())


def publish_commit(candidate, destination):
    """Publish a fully written commit atomically, never replace another commit."""
    candidate, destination = Path(candidate), Path(destination)
    if destination.exists():
        if candidate.exists():
            assert destination.read_bytes() == candidate.read_bytes(), 'Conflicting handoff commit'
        return
    os.rename(candidate, destination)  # Same-directory atomic rename; no Windows overwrite.


def recover_transfer(old, successor, old_handle, candidate, destination,
                     publish=publish_commit, termination_attempted=False):
    """Choose rollback or finish from the retained process handle, not a flag."""
    try:
        exited = old_handle.poll() is not None
    except BaseException as exc:
        return {'action': 'takeover_required', 'reason': 'old_process_status_unknown',
                'detail': str(exc), 'old_controller_resumed': False,
                'successor_terminated': False}
    if exited:
        # Killing the successor here would orphan workers permanently. Even an
        # uncertain wait()/filesystem error must take the forward-only branch.
        try:
            publish(candidate, destination)
        except BaseException as exc:
            return {'action': 'takeover_required', 'reason': 'commit_publication_failed',
                    'detail': str(exc), 'old_controller_exited': True,
                    'old_controller_resumed': False, 'successor_terminated': False}
        if successor is None or successor.poll() is not None:
            return {'action': 'takeover_required', 'reason': 'successor_unavailable_after_commit',
                    'old_controller_exited': True, 'commit_written': True,
                    'old_controller_resumed': False, 'successor_terminated': False}
        return {'action': 'commit_recovered', 'old_controller_exited': True,
                'commit_written': True, 'old_controller_resumed': False,
                'successor_terminated': False}
    if termination_attempted:
        # TerminateProcess is asynchronous. A currently unsignaled handle does
        # not mean the old controller can safely be resumed after a kill request.
        return {'action': 'takeover_required', 'reason': 'old_termination_pending_or_uncertain',
                'termination_attempted': True, 'old_controller_resumed': False,
                'successor_terminated': False}
    if Path(destination).exists():
        return {'action': 'takeover_required', 'reason': 'commit_exists_with_live_old_controller',
                'old_controller_resumed': False, 'successor_terminated': False}
    try:
        if successor is not None and successor.poll() is None:
            successor.terminate()
            successor.wait(timeout=10)
        assert successor is None or successor.poll() is not None, 'Successor still running'
        # The old parent can exit between the first poll and rollback. Do not
        # resume a reused/dead PID; this state needs an explicit new successor.
        if old_handle.poll() is not None:
            return {'action': 'takeover_required', 'reason': 'old_exited_during_rollback',
                    'old_controller_resumed': False, 'successor_terminated': successor is not None}
        old.resume()
        return {'action': 'rolled_back', 'commit_written': False,
                'old_controller_resumed': True, 'successor_terminated': successor is not None}
    except BaseException as exc:
        return {'action': 'takeover_required', 'reason': 'rollback_incomplete', 'detail': str(exc),
                'old_controller_resumed': False}


def identity(p):
    return {'pid': p.pid, 'create_time': p.create_time(), 'cmdline': p.cmdline()}


def normalized(path):
    return os.path.normcase(str(Path(path).resolve()))


def classify(p):
    item = identity(p)
    cmd = item['cmdline']
    if any(normalized(a) == normalized(R / 'observation_resume.py') for a in cmd if a.endswith('.py')):
        shard = int(cmd[cmd.index('--shard') + 1])
        return {**item, 'name': f'observe-gpu{shard}', 'kind': 'observation', 'shard': shard,
                'device': f'cuda:{shard}', 'log_path': str(R / f'observe-gpu{shard}.log')}
    if any(normalized(a) == normalized(R / 'detector_recovery.py') for a in cmd if a.endswith('.py')):
        batch = cmd[cmd.index('--batch') + 1]; device = cmd[cmd.index('--device') + 1]
        return {**item, 'name': f'detect-{batch}', 'kind': 'detection', 'batch': batch,
                'device': device, 'log_path': str(R / f'detect-{batch}.log')}
    raise AssertionError(f'Unexpected direct child: {item}')


def main():
    assert not (S / 'handoff.json').exists(), 'One handoff only'
    assert read(S / 'validation.json')['passed']
    verify_recovery()
    started = read(R / 'controller-started.json')
    old = psutil.Process(started['pid'])
    before = identity(old)
    assert normalized(before['cmdline'][-1]) == normalized(R / 'launch_recovery.py')
    assert old.pid == read(R / 'controller-process.json')['pid']
    save_new(S / 'schedule-plan.json', {'at_utc': now(), 'deadline': approved_deadline(),
        'recovery_plan_sha256': sha(R / 'recovery-plan.json'),
        'source_sha256': {str(f): sha(f) for f in S.glob('*.py')},
        'addendum_sha256': sha(S / 'SCHEDULE_ADDENDUM.md'),
        'observation_devices_unchanged': True, 'reallocate_only_unstarted_detection_batches': True})
    old_handle = WindowsHandle(before)
    successor = None; committed = False; termination_attempted = False
    candidate, destination = S / 'handoff-commit.prepared.json', S / 'handoff-commit.json'
    try:
        old.suspend()
        assert identity(old) == before
        jobs = []
        for child in old.children(recursive=False):
            if child.is_running():
                jobs.append(classify(child))
        assert jobs, 'No active workers to transfer'
        assert len({j['device'] for j in jobs}) == len(jobs), 'Overlapping active devices'
        for job in jobs:
            assert float(job['cmdline'][job['cmdline'].index('--deadline') + 1]) == approved_deadline()
        completed = [d.name for d in sorted((R / 'generation').glob('batch-*'))
                     if (d / 'verification-complete.json').exists()]
        complete_shards = [s for s in (0, 1) if (R / 'observations' / f'gpu{s}' / 'complete.json').exists()]
        save_new(S / 'handoff.json', {'schema_version': 1, 'created_utc': now(),
            'deadline': approved_deadline(), 'recovery_plan_sha256': sha(R / 'recovery-plan.json'),
            'old_controller': before, 'active_jobs': jobs,
            'existing_complete_batches': completed, 'completed_observer_shards': complete_shards})
        with (S / 'controller.log').open('x', encoding='utf-8') as log:
            cmd = [str(ROOT / 'envs/core/python.exe'), '-B', '-u', str(S / 'scheduler.py')]
            successor = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        save_new(S / 'controller-process.json', {'at_utc': now(), 'pid': successor.pid, 'command': cmd})
        stop = min(time.time() + 90, approved_deadline())
        while not (S / 'adoption-ready.json').exists():
            assert successor.poll() is None, f'New scheduler exited {successor.returncode}'
            assert time.time() < stop, 'Adoption readiness timeout'
            time.sleep(.5)
        ready = read(S / 'adoption-ready.json')
        assert ready['scheduler_pid'] == successor.pid
        assert ready['handoff_sha256'] == sha(S / 'handoff.json')
        assert identity(old) == before
        prepare_commit(candidate, {'created_utc': now(), 'scheduler_pid': successor.pid,
            'handoff_sha256': sha(S / 'handoff.json'), 'old_controller_terminated': True,
            'old_controller': before, 'active_workers_not_interrupted': True})
        termination_attempted = True
        old.terminate()
        old.wait(timeout=10)
        assert old_handle.poll() is not None, 'Old controller did not exit'
        publish_commit(candidate, destination)
        committed = True
        save_new(R / 'controller-handoff.json', {'at_utc': now(), 'successor_pid': successor.pid,
            'handoff_directory': str(S), 'reason': 'scheduler_optimization',
            'old_controller_pid': before['pid'], 'old_workers_preserved': True})
        print(json.dumps({'new_scheduler_pid': successor.pid, 'preserved_workers': [j['pid'] for j in jobs],
                          'deadline': approved_deadline()}), flush=True)
    except BaseException as exc:
        if not committed:
            outcome = recover_transfer(old, successor, old_handle, candidate, destination,
                                       termination_attempted=termination_attempted)
            marker = ('handoff-recovered.json' if outcome['action'] == 'commit_recovered' else
                      'takeover-required.json' if outcome['action'] == 'takeover_required' else
                      'handoff-failed.json')
            record = {'at_utc': now(), 'error': str(exc), **outcome}
            print(json.dumps(record), flush=True)
            if not (S / marker).exists():
                save_new(S / marker, record)
            if outcome['action'] == 'commit_recovered':
                if not (R / 'controller-handoff.json').exists():
                    save_new(R / 'controller-handoff.json', {'at_utc': now(), 'successor_pid': successor.pid,
                        'handoff_directory': str(S), 'reason': 'scheduler_optimization',
                        'old_controller_pid': before['pid'], 'old_workers_preserved': True,
                        'commit_recovered_after_error': str(exc)})
                return
        raise
    finally:
        old_handle.close()


if __name__ == '__main__':
    main()
