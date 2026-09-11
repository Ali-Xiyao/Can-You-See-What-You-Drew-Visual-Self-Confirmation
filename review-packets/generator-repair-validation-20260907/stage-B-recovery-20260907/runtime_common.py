"""Reviewed continuation policy; original diagnostic module remains unchanged."""
from rcommon import *


def approved_deadline():
    budget = read(R / 'budget-approval.json')
    assert budget['user_authorized'] is True
    assert budget['original_deadline'] == read(OLD / 'controller-started.json')['deadline']
    assert budget['deadline_utc'] == '2026-09-08T19:00:00+00:00'
    return budget['deadline']


def verify_recovery():
    original_plan()
    p = read(R / 'recovery-plan.json')
    assert p['original_plan_sha256'] == sha(OLD / 'plan.json')
    assert p['deadline'] == approved_deadline()
    for key in ('frozen_files', 'original_artifacts'):
        for path, expected in p[key].items():
            assert stream_sha(path) == expected, path
    return p


def check_time(deadline):
    main_paused()
    assert deadline == approved_deadline(), 'Unapproved runtime deadline'
    assert time.time() < deadline, 'User-approved recovery budget exhausted'
    assert not (R / 'controller-failed.json').exists(), 'Recovery controller failed'


def wait_room(device, deadline, need=18000):
    next_log = 0
    while True:
        check_time(deadline)
        free = original.free_mib(device)
        if free >= need:
            print(f'{device}: {free} MiB free; starting', flush=True)
            return
        if time.time() >= next_log:
            print(f'{device}: waiting for {need} MiB, currently {free}', flush=True)
            next_log = time.time() + 300
        time.sleep(15)
