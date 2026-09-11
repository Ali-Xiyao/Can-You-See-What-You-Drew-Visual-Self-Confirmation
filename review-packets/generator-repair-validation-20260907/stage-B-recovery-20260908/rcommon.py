"""Isolated recovery: original scientific inputs and deadline stay frozen."""
from pathlib import Path
import sys, time, json, hashlib, subprocess

R = Path(__file__).resolve().parent
OLD = R.parent / 'stage-B'
sys.path.insert(0, str(OLD))
import bcommon as original
from common import (ROOT, SIDE, MAIN, read, lines, sha, save_new, append, now,
                    seed, digest, main_paused, make_backbone, load_model)


def original_plan(full=False):
    return original.verify_plan(full=full)


def stream_sha(path):
    return original.stream_sha(path)


def verify_recovery():
    original_plan()
    path = R / 'recovery-plan.json'
    if not path.exists():
        path = R / 'diagnostic-plan.json'
    p = read(path)
    assert p['original_plan_sha256'] == sha(OLD / 'plan.json')
    assert p['deadline'] == read(OLD / 'controller-started.json')['deadline']
    for f, expected in p.get('frozen_files', {}).items():
        assert stream_sha(f) == expected, f
    for f, expected in p.get('original_artifacts', {}).items():
        assert stream_sha(f) == expected, f
    return p


def check_time(deadline):
    main_paused()
    original_deadline = read(OLD / 'controller-started.json')['deadline']
    assert deadline <= original_deadline, 'Recovery cannot extend original budget'
    assert time.time() < deadline, 'Original Stage B budget exhausted'
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


def scene_rows():
    return original.scene_rows()


def all_images(p):
    out = []
    for batch in p['batches']:
        folder = R / 'generation' / batch['id']
        done = read(folder / 'complete.json')
        assert sha(folder / 'manifest.jsonl') == done['manifest_sha256']
        rows = lines(folder / 'manifest.jsonl')
        assert len(rows) == done['n_images'] == len(batch['scene_ids']) * 6
        out.extend(rows)
    assert len(out) == 900
    assert len({(r['spec_id'], r['candidate_index']) for r in out}) == 900
    return out


def pipeline_module():
    return original.pipeline_module()


def environment():
    import os
    prior = read(ROOT / 'review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json')
    env = os.environ.copy()
    env.update(prior['environment_allowlist'])
    for key in prior['environment_remove']:
        env.pop(key, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env
