"""Failure injection for the irreversible parent-exit / commit-publication gap."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('handoff_v2', Path(__file__).with_name('handoff.py'))
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class Parent:
    def __init__(self):
        self.resumed = 0

    def resume(self):
        self.resumed += 1


class Child:
    def __init__(self, code=None):
        self.code, self.terminated = code, 0

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated += 1
        self.code = 1

    def wait(self, timeout):
        return self.code


class Handle:
    def __init__(self, code):
        self.code = code

    def poll(self):
        if isinstance(self.code, BaseException):
            raise self.code
        return self.code


def files(tmp_path):
    candidate, commit = tmp_path / 'prepared.json', tmp_path / 'commit.json'
    handoff.prepare_commit(candidate, {'scheduler_pid': 123, 'old_controller_terminated': True})
    return candidate, commit


def test_prepared_commit_fsynced_before_atomic_publication(tmp_path, monkeypatch):
    calls = []
    original = handoff.os.fsync
    monkeypatch.setattr(handoff.os, 'fsync', lambda fd: (calls.append(fd), original(fd))[-1])
    candidate, commit = files(tmp_path)
    assert len(calls) == 1 and not commit.exists()
    payload = candidate.read_bytes()
    handoff.publish_commit(candidate, commit)
    assert commit.read_bytes() == payload and not candidate.exists()


def test_publish_failure_after_parent_exit_keeps_successor_and_retries(tmp_path):
    candidate, commit = files(tmp_path)
    old, successor = Parent(), Child()
    # This represents the exception from the first attempted rename.
    result = handoff.recover_transfer(old, successor, Handle(1), candidate, commit)
    assert result['action'] == 'commit_recovered'
    assert commit.exists() and not candidate.exists()
    assert successor.terminated == 0 and old.resumed == 0


def test_repeated_publish_failure_after_exit_requests_takeover(tmp_path):
    candidate, commit = files(tmp_path)
    old, successor = Parent(), Child()
    def fail(*args):
        raise OSError('disk refused atomic rename')
    result = handoff.recover_transfer(old, successor, Handle(1), candidate, commit, publish=fail)
    assert result['action'] == 'takeover_required'
    assert result['reason'] == 'commit_publication_failed'
    assert successor.terminated == 0 and old.resumed == 0 and candidate.exists()


def test_uncertain_wait_error_uses_true_exited_status(tmp_path):
    candidate, commit = files(tmp_path)
    old, successor = Parent(), Child()
    # old.wait() may have raised despite a completed Windows termination.
    result = handoff.recover_transfer(old, successor, Handle(0), candidate, commit)
    assert result['action'] == 'commit_recovered'
    assert successor.terminated == 0 and old.resumed == 0


def test_precommit_failure_rolls_back_only_while_old_is_alive(tmp_path):
    old, successor = Parent(), Child()
    result = handoff.recover_transfer(old, successor, Handle(None),
        tmp_path / 'absent-prepared.json', tmp_path / 'absent-commit.json')
    assert result['action'] == 'rolled_back'
    assert successor.terminated == 1 and old.resumed == 1


def test_unknown_process_status_neither_kills_successor_nor_resumes_old(tmp_path):
    candidate, commit = files(tmp_path)
    old, successor = Parent(), Child()
    result = handoff.recover_transfer(old, successor, Handle(OSError('query failed')), candidate, commit)
    assert result['action'] == 'takeover_required'
    assert result['reason'] == 'old_process_status_unknown'
    assert successor.terminated == 0 and old.resumed == 0 and not commit.exists()


def test_published_commit_never_rolls_back(tmp_path):
    candidate, commit = files(tmp_path)
    handoff.publish_commit(candidate, commit)
    old, successor = Parent(), Child()
    result = handoff.recover_transfer(old, successor, Handle(1), candidate, commit)
    assert result['action'] == 'commit_recovered'
    assert successor.terminated == 0 and old.resumed == 0


def test_conflicting_commit_rejected(tmp_path):
    candidate, commit = files(tmp_path)
    commit.write_text('other commit')
    with pytest.raises(AssertionError):
        handoff.publish_commit(candidate, commit)


def test_parent_alive_with_commit_is_never_resumed(tmp_path):
    candidate, commit = files(tmp_path)
    handoff.publish_commit(candidate, commit)
    old, successor = Parent(), Child()
    result = handoff.recover_transfer(old, successor, Handle(None), candidate, commit)
    assert result['action'] == 'takeover_required'
    assert successor.terminated == 0 and old.resumed == 0


def test_dead_successor_after_old_exit_requests_takeover(tmp_path):
    candidate, commit = files(tmp_path)
    old, successor = Parent(), Child(code=1)
    result = handoff.recover_transfer(old, successor, Handle(1), candidate, commit)
    assert result['action'] == 'takeover_required'
    assert result['reason'] == 'successor_unavailable_after_commit'
    assert successor.terminated == 0 and old.resumed == 0 and commit.exists()


def test_asynchronous_old_termination_never_rolls_back(tmp_path):
    candidate, commit = files(tmp_path)
    old, successor = Parent(), Child()
    result = handoff.recover_transfer(old, successor, Handle(None), candidate, commit,
                                     termination_attempted=True)
    assert result['action'] == 'takeover_required'
    assert result['reason'] == 'old_termination_pending_or_uncertain'
    assert successor.terminated == 0 and old.resumed == 0
    assert candidate.exists() and not commit.exists()
