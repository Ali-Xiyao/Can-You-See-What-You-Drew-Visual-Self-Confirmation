"""CPU checks for queue allocation, frozen devices and adopted-process status."""
import ctypes
from ctypes import wintypes
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('scheduler_v2', Path(__file__).with_name('scheduler.py'))
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)

BATCHES = [f'batch-{i:02}' for i in range(6)]
OBS0 = {'name': 'observe-gpu0', 'kind': 'observation', 'shard': 0, 'device': 'cuda:0'}
OBS1 = {'name': 'observe-gpu1', 'kind': 'observation', 'shard': 1, 'device': 'cuda:1'}
DETECT1 = {'name': 'detect-batch-01', 'kind': 'detection', 'batch': 'batch-01', 'device': 'cuda:1'}


def dispatch(done=('batch-00',), shards=(), active=(), now=10, failed=False):
    return scheduler.dispatch_jobs(BATCHES, done, shards, active, 100, now, failed)


def test_adoption_does_not_dispatch_over_existing_workers():
    assert dispatch(active=(OBS0, DETECT1)) == []


def test_gpu1_observer_starts_after_current_batch_before_remaining_detection():
    assert dispatch(done=('batch-00', 'batch-01'), active=(OBS0,)) == [OBS1]


def test_gpu0_takes_remaining_whole_batch_as_soon_as_observer_finishes():
    jobs = dispatch(shards=(0,), active=(DETECT1,))
    assert jobs == [{'name': 'detect-batch-02', 'kind': 'detection',
                     'batch': 'batch-02', 'device': 'cuda:0'}]


def test_both_idle_gpus_take_distinct_ordered_batches():
    jobs = dispatch(done=('batch-00', 'batch-01'), shards=(0, 1))
    assert [(j['batch'], j['device']) for j in jobs] == [
        ('batch-02', 'cuda:0'), ('batch-03', 'cuda:1')]


def test_both_observer_shards_keep_registered_devices():
    assert dispatch() == [OBS0, OBS1]
    for job in (OBS0, OBS1):
        command = scheduler.worker_command(job, 100)
        assert command[-4:] == ['--shard', str(job['shard']), '--deadline', '100']
        assert command[3].endswith('observation_resume.py')


@pytest.mark.parametrize('now,failed', [(100, False), (101, False), (10, True)])
def test_deadline_and_failure_never_dispatch(now, failed):
    assert dispatch(now=now, failed=failed) == []


def test_no_duplicate_batch_while_other_gpu_busy():
    busy = {'name': 'detect-batch-02', 'kind': 'detection', 'batch': 'batch-02', 'device': 'cuda:0'}
    jobs = dispatch(done=('batch-00', 'batch-01'), shards=(0, 1), active=(busy,))
    assert [j['batch'] for j in jobs] == ['batch-03']


@pytest.mark.parametrize('active', [
    (OBS0, {**DETECT1, 'device': 'cuda:0'}),
    ({**OBS1, 'device': 'cuda:0'},),
    (DETECT1, {**DETECT1, 'device': 'cuda:0'}),
])
def test_invalid_gpu_or_batch_assignment_is_rejected(active):
    with pytest.raises(AssertionError):
        dispatch(active=active)


def test_all_work_finished_yields_no_jobs():
    assert dispatch(done=BATCHES, shards=(0, 1)) == []


def test_detector_command_changes_only_explicit_batch_and_device():
    command = scheduler.worker_command(DETECT1, 100)
    assert command[3] == str(scheduler.R / 'detector_recovery.py')
    assert command[4:] == ['detect-batch', '--batch', 'batch-01', '--deadline', '100', '--device', 'cuda:1']


IDENTITY = {'pid': 123, 'create_time': 42.0, 'cmdline': ['python.exe', '-u', 'worker.py']}


class FakeAPI:
    def __init__(self, wait=0x102, exit_code=0):
        self.wait, self.exit_code = wait, exit_code
        self.open_args = None
        self.closed = []

    def OpenProcess(self, access, inherit, pid):
        self.open_args = (access, inherit, pid)
        return 987

    def WaitForSingleObject(self, handle, timeout):
        assert handle == 987 and timeout == 0
        return self.wait

    def GetExitCodeProcess(self, handle, output):
        ctypes.cast(output, ctypes.POINTER(wintypes.DWORD)).contents.value = self.exit_code
        return True

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return True


def test_adopted_handle_has_no_terminate_access_and_preserves_real_exit():
    api = FakeAPI()
    handle = scheduler.WindowsHandle(IDENTITY, api, lambda pid: IDENTITY)
    assert api.open_args == (0x1000 | 0x00100000, False, 123)
    assert api.open_args[0] & 1 == 0  # PROCESS_TERMINATE is never requested.
    assert handle.poll() is None
    api.wait, api.exit_code = 0, 7
    assert handle.poll() == 7
    handle.close()
    handle.close()
    assert api.closed == [987]


def test_exited_process_code_259_is_not_mistaken_for_still_active():
    handle = scheduler.WindowsHandle(IDENTITY, FakeAPI(wait=0, exit_code=259), lambda pid: IDENTITY)
    assert handle.poll() == 259
    handle.close()


def test_pid_reuse_before_open_rejected():
    api = FakeAPI()
    with pytest.raises(AssertionError):
        scheduler.WindowsHandle(IDENTITY, api, lambda pid: {**IDENTITY, 'create_time': 43})
    assert api.open_args is None


def test_identity_race_closes_handle_without_touching_worker():
    api = FakeAPI()
    calls = iter([IDENTITY, {**IDENTITY, 'cmdline': ['other.exe']}])
    with pytest.raises(AssertionError):
        scheduler.WindowsHandle(IDENTITY, api, lambda pid: next(calls))
    assert api.closed == [987]


def test_wait_error_is_not_a_successful_exit():
    handle = scheduler.WindowsHandle(IDENTITY, FakeAPI(wait=0xFFFFFFFF), lambda pid: IDENTITY)
    with pytest.raises(OSError):
        handle.poll()
    handle.close()


def test_full_schedule_finishes_each_batch_once_and_shards_once():
    complete, shards, active = {'batch-00'}, set(), [OBS0, DETECT1]
    launched = [OBS0['name'], DETECT1['name']]
    # Finish GPU0 first: it immediately starts batch02; then GPU1's batch01
    # finishes and shard1 must precede any additional batch on that card.
    for finish_device in ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0']:
        matching = [j for j in active if j['device'] == finish_device]
        if matching:
            job = matching[0]
            active.remove(job)
            if job['kind'] == 'observation':
                shards.add(job['shard'])
            else:
                complete.add(job['batch'])
        new = scheduler.dispatch_jobs(BATCHES, complete, shards, active, 100, 10)
        active.extend(new)
        launched.extend(j['name'] for j in new)
    assert complete == set(BATCHES) and shards == {0, 1} and active == []
    assert len(launched) == len(set(launched)) == 7
