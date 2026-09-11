"""Real controller/start/finish/analysis paths with fake CPU-only child processes."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cold_recovery_scheduler', Path(__file__).with_name('scheduler.py'))
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)

PLAN = {'batches': [{'id': f'batch-{i:02}', 'scene_ids': []} for i in range(6)]}
OBS0 = {'name': 'observe-gpu0', 'kind': 'observation', 'shard': 0, 'device': 'cuda:0'}
OBS1 = {'name': 'observe-gpu1', 'kind': 'observation', 'shard': 1, 'device': 'cuda:1'}


class FakeProcess:
    def __init__(self, pid, command):
        self.pid, self.command, self.returncode = pid, command, None
        self.terminated = False

    def poll(self):
        return self.returncode


class Rig:
    def __init__(self, folder):
        self.spawned, self.killed, self.artifacts = [], [], []
        self.controller = scheduler.Controller(PLAN, 100, ('batch-00', 'batch-01'), folder=folder,
            popen_factory=self.popen, identity_fn=self.identity, artifact_fn=self.artifact,
            check_fn=lambda deadline: None, sleep_fn=self.complete_tick, clock_fn=lambda: 1,
            environment_fn=lambda: {}, terminate_fn=self.terminate,
            results_fn=lambda: {'gold': {'review_images': 17}})

    def popen(self, command, **kwargs):
        assert kwargs['creationflags'] == scheduler.subprocess.CREATE_NO_WINDOW
        assert kwargs['stdin'] == scheduler.subprocess.DEVNULL
        process = FakeProcess(1000 + len(self.spawned), command)
        self.spawned.append(process)
        return process

    def identity(self, pid):
        process = next(p for p in self.spawned if p.pid == pid)
        return {'pid': pid, 'create_time': float(pid), 'cmdline': process.command}

    def artifact(self, job, plan):
        self.artifacts.append(job['name'])
        return {'path': 'synthetic/' + job['name'], 'sha256': 'sha-' + job['name']}

    def terminate(self, process, identity):
        self.killed.append((process.pid, identity))
        process.terminated = True
        process.returncode = 1

    def complete_tick(self, seconds):
        for process in self.spawned:
            if process.poll() is None:
                process.returncode = 0


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close_logs(controller):
    for log in controller.logs:
        log.close()


def test_actual_start_records_kind_without_event_argument_collision(tmp_path):
    rig = Rig(tmp_path)
    rig.controller.start(OBS1)
    command = rig.spawned[0].command
    assert command[3] == str(scheduler.R / 'observation_resume.py')
    assert command[4:] == ['--shard', '1', '--deadline', '100']
    record = read(tmp_path / 'observe-gpu1-started.json')
    event = json.loads((tmp_path/'events.jsonl').read_text(encoding='utf-8'))
    assert record['command'] == record['cmdline'] == command
    assert record['kind'] == event['kind'] == 'observation'
    assert event['event'] == 'job_started' and event['device'] == 'cuda:1'
    assert record['pid'] == 1000 and rig.controller.identities['observe-gpu1']['pid'] == 1000
    close_logs(rig.controller)


def test_actual_detection_start_uses_complete_frozen_worker_command(tmp_path):
    rig = Rig(tmp_path)
    job = {'name': 'detect-batch-02', 'kind': 'detection', 'batch': 'batch-02', 'device': 'cuda:0'}
    rig.controller.start(job)
    assert rig.spawned[0].command[3:] == [str(scheduler.R/'detector_recovery.py'),
        'detect-batch', '--batch', 'batch-02', '--deadline', '100', '--device', 'cuda:0']
    assert read(tmp_path/'detect-batch-02-started.json')['kind'] == 'detection'
    close_logs(rig.controller)


def test_full_controller_path_reuses_completed_batches_and_runs_analysis(tmp_path):
    rig = Rig(tmp_path)
    completed = rig.controller.run()
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    starts = [row for row in events if row['event'] == 'job_started']
    assert [row['name'] for row in starts] == ['observe-gpu0', 'observe-gpu1',
        'detect-batch-02', 'detect-batch-03', 'detect-batch-04', 'detect-batch-05', 'analysis']
    assert [(row['batch'], row['device']) for row in starts if row['kind'] == 'detection'] == [
        ('batch-02', 'cuda:0'), ('batch-03', 'cuda:1'), ('batch-04', 'cuda:0'), ('batch-05', 'cuda:1')]
    assert rig.artifacts == [row['name'] for row in starts]
    assert completed['exit_code'] == 0 and completed['results_sha256'] == 'sha-analysis'
    assert completed['pending_review_images'] == 17
    assert read(tmp_path/'gpu-complete.json')['expected_answers'] == 108000
    assert read(tmp_path/'controller-complete.json')['research_goal_complete'] is False
    assert read(tmp_path/'state.json')['status'] == 'complete' and not rig.killed
    assert all(log.closed for log in rig.controller.logs)


def test_finish_then_dispatch_releases_only_the_finished_card(tmp_path):
    rig = Rig(tmp_path)
    rig.controller.start(OBS0); rig.controller.start(OBS1)
    rig.spawned[0].returncode = 0
    rig.controller.collect_finished()
    jobs = scheduler.dispatch_jobs([b['id'] for b in PLAN['batches']], rig.controller.completed_batches,
        rig.controller.completed_shards, rig.controller.active.values(), 100, 1)
    assert jobs == [{'name': 'detect-batch-02', 'kind': 'detection', 'batch': 'batch-02', 'device': 'cuda:0'}]
    assert read(tmp_path/'observe-gpu0-completed.json')['exit_code'] == 0
    rig.controller.start(jobs[0])
    assert {j['device'] for j in rig.controller.active.values()} == {'cuda:0', 'cuda:1'}
    close_logs(rig.controller)


def test_start_event_failure_cleans_both_owned_workers_only(tmp_path):
    rig = Rig(tmp_path)
    untouched = FakeProcess(99999, ['unrelated.exe'])
    original = rig.controller.event
    def fail_second(event_type, **record):
        if event_type == 'job_started' and record['name'] == 'observe-gpu1':
            raise OSError('injected event write failure')
        return original(event_type, **record)
    rig.controller.event = fail_second
    with pytest.raises(OSError, match='event write failure'):
        rig.controller.run()
    assert [pid for pid, _ in rig.killed] == [1000, 1001]
    assert untouched.poll() is None and untouched.terminated is False
    assert len(rig.spawned) == 2 and read(tmp_path/'controller-failed.json')['cleanup_errors'] == []
    assert read(tmp_path/'state.json')['status'] == 'failed'


def test_identity_failure_after_popen_cannot_leak_unregistered_process(tmp_path):
    rig = Rig(tmp_path)
    def fail(pid):
        raise OSError('identity unavailable')
    rig.controller.identity_fn = fail
    with pytest.raises(OSError, match='identity unavailable'):
        rig.controller.run()
    assert rig.killed == [(1000, None)] and len(rig.spawned) == 1
    assert read(tmp_path/'controller-failed.json')['automatic_retry'] is False


def test_nonzero_exit_stops_sibling_and_does_not_dispatch_more(tmp_path):
    rig = Rig(tmp_path)
    rig.controller.sleep_fn = lambda seconds: setattr(rig.spawned[0], 'returncode', 7)
    with pytest.raises(RuntimeError, match='real process exit 7'):
        rig.controller.run()
    assert len(rig.spawned) == 2 and [pid for pid, _ in rig.killed] == [1001]
    assert read(tmp_path/'observe-gpu0-failed.json')['exit_code'] == 7
    assert rig.artifacts == []


def test_zero_exit_without_valid_artifact_is_not_completion(tmp_path):
    rig = Rig(tmp_path)
    def invalid(job, plan):
        raise AssertionError('completion hash mismatch')
    rig.controller.artifact_fn = invalid
    with pytest.raises(AssertionError, match='completion hash mismatch'):
        rig.controller.run()
    assert len(rig.spawned) == 2
    assert read(tmp_path/'observe-gpu0-failed.json')['exit_code'] == 0
    assert rig.controller.completed_shards == set()


def test_analysis_failure_prevents_controller_complete(tmp_path):
    rig = Rig(tmp_path)
    original_tick = rig.complete_tick
    def tick(seconds):
        original_tick(seconds)
        for process in rig.spawned:
            if process.command[3].endswith('analyze_recovery.py'):
                process.returncode = 9
    rig.controller.sleep_fn = tick
    with pytest.raises(RuntimeError, match='analysis: real process exit 9'):
        rig.controller.run()
    assert (tmp_path/'gpu-complete.json').exists()
    assert not (tmp_path/'controller-complete.json').exists()
    assert read(tmp_path/'analysis-failed.json')['exit_code'] == 9


def test_deadline_before_start_never_spawns_a_worker(tmp_path):
    rig = Rig(tmp_path)
    def exhausted(deadline):
        raise AssertionError('budget exhausted')
    rig.controller.check_fn = exhausted
    with pytest.raises(AssertionError, match='budget exhausted'):
        rig.controller.run()
    assert rig.spawned == [] and rig.killed == []
    assert not (tmp_path/'gpu-complete.json').exists()


def test_finished_data_goes_directly_to_analysis_without_repeating_gpu(tmp_path):
    rig = Rig(tmp_path)
    rig.controller.completed_batches = {b['id'] for b in PLAN['batches']}
    rig.controller.completed_shards = {0, 1}
    rig.controller.run()
    assert len(rig.spawned) == 1 and rig.spawned[0].command[3].endswith('analyze_recovery.py')


@pytest.mark.parametrize('now,failed', [(100, False), (101, False), (1, True)])
def test_pure_deadline_and_failure_block_dispatch(now, failed):
    assert scheduler.dispatch_jobs(['batch-02'], set(), set(), [], 100, now, failed) == []


def test_wrong_observer_device_is_rejected_before_popen(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(AssertionError):
        rig.controller.start({**OBS1, 'device': 'cuda:0'})
    assert rig.spawned == []


def test_final_audit_failure_never_leaves_a_false_completion_marker(tmp_path):
    rig = Rig(tmp_path)
    original = rig.controller.event
    def fail_completion(event_type, **record):
        if event_type == 'controller_completed':
            raise OSError('final event write failed')
        return original(event_type, **record)
    rig.controller.event = fail_completion
    with pytest.raises(OSError, match='final event write failed'):
        rig.controller.run()
    assert (tmp_path/'gpu-complete.json').exists()
    assert (tmp_path/'controller-failed.json').exists()
    assert not (tmp_path/'controller-complete.json').exists()
