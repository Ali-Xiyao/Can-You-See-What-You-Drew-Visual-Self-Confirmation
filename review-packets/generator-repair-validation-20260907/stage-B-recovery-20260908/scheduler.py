"""Cold recovery dispatcher: fixed observation devices, shared whole-batch queue.

Only this controller's children are owned.  Completed data are validated before
reuse; observation workers validate/replay their copied prefixes themselves.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_common as runtime

R = HERE


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def process_identity(pid):
    import psutil
    process = psutil.Process(pid)
    return {'pid': pid, 'create_time': process.create_time(), 'cmdline': process.cmdline()}


def worker_command(job, deadline):
    if job['kind'] == 'observation':
        assert job['device'] == f"cuda:{job['shard']}"
        env, script = 'showo2', 'observation_resume.py'
        args = ['--shard', str(job['shard']), '--deadline', str(deadline)]
    elif job['kind'] == 'detection':
        env, script = 'observer', 'detector_recovery.py'
        args = ['detect-batch', '--batch', job['batch'], '--deadline', str(deadline), '--device', job['device']]
    else:
        assert job['kind'] == 'analysis'
        env, script, args = 'core', 'analyze_recovery.py', []
    return [str(runtime.ROOT / f'envs/{env}/python.exe'), '-B', '-u', str(R / script), *args]


def dispatch_jobs(batch_ids, completed_batches, completed_shards, active, deadline, current_time, failed=False):
    if failed or current_time >= deadline:
        return []
    active = list(active)
    occupied = [job['device'] for job in active]
    assert len(occupied) == len(set(occupied)) and set(occupied) <= {'cuda:0', 'cuda:1'}
    active_batches = [job['batch'] for job in active if job['kind'] == 'detection']
    assert len(active_batches) == len(set(active_batches))
    for job in active:
        if job['kind'] == 'observation':
            assert job['device'] == f"cuda:{job['shard']}"
    pending = sorted(set(batch_ids) - set(completed_batches) - set(active_batches))
    jobs = []
    for shard in (0, 1):
        device = f'cuda:{shard}'
        if device in occupied:
            continue
        if shard not in completed_shards:
            jobs.append({'name': f'observe-gpu{shard}', 'kind': 'observation', 'shard': shard, 'device': device})
        elif pending:
            batch = pending.pop(0)
            jobs.append({'name': 'detect-' + batch, 'kind': 'detection', 'batch': batch, 'device': device})
    return jobs


def validate_artifact(job, plan):
    if job['kind'] == 'detection':
        batch = next(item for item in plan['batches'] if item['id'] == job['batch'])
        folder = R / 'generation' / job['batch']
        generation = runtime.read(folder / 'complete.json')
        done = runtime.read(folder / 'verification-complete.json')
        assert done['n_images'] == generation['n_images'] == 6 * len(batch['scene_ids'])
        assert done['device'] == job['device']
        assert done['detector_snapshot_digest'] == plan['detector_snapshot_digest']
        assert generation['manifest_sha256'] == runtime.sha(folder / 'manifest.jsonl')
        manifest = runtime.lines(folder / 'manifest.jsonl')
        assert len(manifest) == done['n_images']
        assert {(row['spec_id'], row['candidate_index']) for row in manifest} == {
            (sid, i) for sid in batch['scene_ids'] for i in range(6)}
        assert {'qwen3vl', 'internvl', 'verified'} <= set(done['sha256'])
        for name, expected in done['sha256'].items():
            path = folder / ('verified.jsonl' if name == 'verified' else f'detections.{name}.jsonl')
            assert runtime.sha(path) == expected, path
        verified = runtime.lines(folder / 'verified.jsonl')
        assert len(verified) == done['n_images']
        assert {row['image_path'] for row in verified} == {row['image_path'] for row in manifest}
        return {'path': str(folder / 'verification-complete.json'), 'sha256': runtime.sha(folder / 'verification-complete.json')}
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
    assert job['kind'] == 'analysis'
    result = R / 'analysis-recovery/results.json'
    audit = runtime.read(R / 'analysis-recovery/recovery-integrity.json')
    integrity = runtime.read(R / 'analysis-recovery/integrity.json')
    assert audit['results_sha256'] == integrity['results_sha256'] == runtime.sha(result)
    data = runtime.read(result)
    assert data['n_scenes'] == 150 and data['n_images'] == 900
    return {'path': str(result), 'sha256': runtime.sha(result)}


def initial_completed(plan):
    batches, shards = set(), set()
    for batch in plan['batches']:
        path = R / 'generation' / batch['id'] / 'verification-complete.json'
        if path.exists():
            done = runtime.read(path)
            validate_artifact({'kind': 'detection', 'batch': batch['id'], 'device': done['device']}, plan)
            batches.add(batch['id'])
    for shard in (0, 1):
        if (R / 'observations' / f'gpu{shard}' / 'complete.json').exists():
            validate_artifact({'kind': 'observation', 'shard': shard, 'device': f'cuda:{shard}'}, plan)
            shards.add(shard)
    return batches, shards


def terminate_owned(proc, expected):
    """Popen retains its native handle; descendants require matching identity."""
    import psutil
    if proc.poll() is not None:
        return
    descendants = []
    if expected is not None:
        try:
            current = process_identity(proc.pid)
            assert current == expected, 'Worker PID identity changed; refuse unrelated tree'
            descendants = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
    snapshots = [(p, p.create_time()) for p in descendants]
    for child, created in snapshots:
        try:
            if child.create_time() == created:
                child.terminate()
        except psutil.NoSuchProcess:
            pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=10)
    _, alive = psutil.wait_procs(descendants, timeout=5)
    creations = {p.pid: created for p, created in snapshots}
    for child in alive:
        try:
            if child.create_time() == creations[child.pid]:
                child.kill()
        except psutil.NoSuchProcess:
            pass


class Controller:
    def __init__(self, plan, deadline, completed_batches=(), completed_shards=(), *,
                 folder=R, popen_factory=subprocess.Popen, identity_fn=process_identity,
                 artifact_fn=validate_artifact, check_fn=runtime.check_time,
                 sleep_fn=time.sleep, clock_fn=time.time, environment_fn=runtime.environment,
                 terminate_fn=terminate_owned, results_fn=None):
        self.plan, self.deadline, self.folder = plan, deadline, Path(folder)
        self.completed_batches, self.completed_shards = set(completed_batches), set(completed_shards)
        self.processes, self.identities, self.jobs, self.active = {}, {}, {}, {}
        self.popen_factory, self.identity_fn, self.artifact_fn = popen_factory, identity_fn, artifact_fn
        self.check_fn, self.sleep_fn, self.clock_fn = check_fn, sleep_fn, clock_fn
        self.environment_fn, self.terminate_fn = environment_fn, terminate_fn
        self.results_fn = results_fn or (lambda: runtime.read(R / 'analysis-recovery/results.json'))
        self.logs = []

    def event(self, event_type, **value):
        # The worker record legitimately contains kind=observation/detection.
        # event_type must not collide with that payload field (R1 incident).
        assert 'event' not in value, 'Reserved event name'
        with (self.folder / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'at_utc': runtime.now(), 'event': event_type, **value}) + '\n')

    def start(self, job):
        self.check_fn(self.deadline)
        assert self.clock_fn() < self.deadline, 'Deadline reached before dispatch'
        name = job['name']
        assert name not in self.processes, 'No duplicate job launch'
        command = worker_command(job, self.deadline)
        log = (self.folder / f'{name}.log').open('x', encoding='utf-8')
        self.logs.append(log)
        proc = self.popen_factory(command, cwd=runtime.ROOT, env=self.environment_fn(),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        # Register the native Popen owner before any fallible identity/log work.
        self.processes[name] = proc
        self.jobs[name] = dict(job)
        self.active[name] = dict(job)
        identity = self.identity_fn(proc.pid)
        self.identities[name] = identity
        assert identity['pid'] == proc.pid and identity['cmdline'] == command
        record = {**job, **identity, 'command': command, 'log_path': str(self.folder / f'{name}.log')}
        self.jobs[name] = record
        self.active[name] = record
        write_new(self.folder / f'{name}-started.json', {'at_utc': runtime.now(), **record})
        self.event('job_started', **record)

    def collect_finished(self):
        for name, job in list(self.active.items()):
            code = self.processes[name].poll()
            if code is None:
                continue
            if code != 0:
                write_new(self.folder / f'{name}-failed.json', {'at_utc': runtime.now(), 'exit_code': code, **job})
                raise RuntimeError(f'{name}: real process exit {code}')
            try:
                artifact = self.artifact_fn(job, self.plan)
            except BaseException as exc:
                write_new(self.folder / f'{name}-failed.json', {'at_utc': runtime.now(), 'exit_code': code,
                           'artifact_error': str(exc), **job})
                raise
            write_new(self.folder / f'{name}-completed.json', {'at_utc': runtime.now(), 'exit_code': code,
                       'artifact': artifact, **job})
            self.event('job_completed', name=name, kind=job['kind'], exit_code=code, artifact=artifact)
            if job['kind'] == 'detection':
                self.completed_batches.add(job['batch'])
            elif job['kind'] == 'observation':
                self.completed_shards.add(job['shard'])
            del self.active[name]

    def write_state(self, status):
        temporary = self.folder / 'state.json.tmp'
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump({'at_utc': runtime.now(), 'status': status, 'pid': os.getpid(),
                       'deadline': self.deadline, 'active_jobs': list(self.active.values()),
                       'completed_batches': sorted(self.completed_batches),
                       'completed_observer_shards': sorted(self.completed_shards)}, handle, indent=2)
        os.replace(temporary, self.folder / 'state.json')

    def stop_all(self):
        errors = []
        for name, proc in self.processes.items():
            if proc.poll() is None:
                try:
                    self.terminate_fn(proc, self.identities.get(name))
                except BaseException as exc:
                    errors.append(f'{name}: {exc}')
        return errors

    def run(self):
        batch_ids = [batch['id'] for batch in self.plan['batches']]
        try:
            while True:
                self.check_fn(self.deadline)
                self.collect_finished()
                if self.completed_batches == set(batch_ids) and self.completed_shards == {0, 1}:
                    assert not self.active
                    break
                for job in dispatch_jobs(batch_ids, self.completed_batches, self.completed_shards,
                                         self.active.values(), self.deadline, self.clock_fn()):
                    self.start(job)
                self.write_state('running')
                self.sleep_fn(5)
            write_new(self.folder / 'gpu-complete.json', {'at_utc': runtime.now(), 'n_images': 900,
                'expected_answers': 108000, 'original_failed_attempts_preserved': True})
            self.start({'name': 'analysis', 'kind': 'analysis'})
            while self.active:
                self.check_fn(self.deadline)
                self.collect_finished()
                self.write_state('analysis')
                if self.active:
                    self.sleep_fn(5)
            result = self.results_fn()
            analysis = runtime.read(self.folder / 'analysis-completed.json')
            complete = {'at_utc': runtime.now(), 'exit_code': 0,
                        'results_sha256': analysis['artifact']['sha256'],
                        'pending_review_images': result['gold']['review_images'],
                        'research_goal_complete': False, 'new_optimizer_steps': 0}
            self.write_state('complete')
            self.event('controller_completed', **complete)
            # This is the final commit marker. A prior audit/state write error
            # must not leave both controller-complete and controller-failed.
            write_new(self.folder / 'controller-complete.json', complete)
            return complete
        except BaseException as exc:
            errors = self.stop_all()
            failure = {'at_utc': runtime.now(), 'error': f'{type(exc).__name__}: {exc}',
                       'automatic_retry': False, 'cleanup_errors': errors}
            if not (self.folder / 'controller-failed.json').exists():
                write_new(self.folder / 'controller-failed.json', failure)
            self.write_state('failed')
            raise
        finally:
            for log in self.logs:
                log.close()


def main():
    import msvcrt
    assert runtime.R == R, 'Recovery common module bound to a different directory'
    assert not (R / 'controller-started.json').exists(), 'No automatic duplicate launch'
    assert not (R / 'controller-failed.json').exists(), 'Failed attempt requires new explicit recovery'
    recovery = runtime.verify_recovery()
    plan = runtime.original_plan()
    deadline = recovery['deadline']
    assert deadline == runtime.approved_deadline()
    with (runtime.MAIN / 'supervisor.lock').open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            runtime.check_time(deadline)
            batches, shards = initial_completed(plan)
            write_new(R / 'controller-started.json', {'at_utc': runtime.now(), 'pid': os.getpid(),
                'identity': process_identity(os.getpid()), 'deadline': deadline,
                'recovery_plan_sha256': runtime.sha(R / 'recovery-plan.json'),
                'scheduler_sha256': runtime.sha(Path(__file__)), 'new_optimizer_steps': 0,
                'completed_batches_at_start': sorted(batches), 'completed_shards_at_start': sorted(shards),
                'observer_devices_unchanged': True, 'original_failed_attempts_preserved': True})
            Controller(plan, deadline, batches, shards).run()
        except BaseException as exc:
            # Only a lock-owning controller can mark this attempt failed. An
            # accidental concurrent/duplicate launcher must not stop its workers.
            if not (R / 'controller-failed.json').exists():
                write_new(R / 'controller-failed.json', {'at_utc': runtime.now(),
                    'error': f'{type(exc).__name__}: {exc}', 'automatic_retry': False})
            raise


if __name__ == '__main__':
    main()
