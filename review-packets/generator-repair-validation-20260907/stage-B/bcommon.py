"""Isolated Stage B runtime and frozen-question truth adapter."""
from pathlib import Path
import sys, os, time, json, subprocess, hashlib, re
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent))
from common import (ROOT, SIDE, MAIN, read, lines, sha, digest, seed, now,
                    save_new, append, main_paused, make_backbone, load_model)

def stream_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def verify_plan(full=False):
    main_paused()
    p = read(B / 'plan.json')
    for path, value in p['input_sha256'].items():
        assert stream_sha(path) == value, path
    if full:
        for path, value in p['detector_sha256'].items():
            assert stream_sha(path) == value, path
    else:
        for path, stat in p['detector_stat'].items():
            s = Path(path).stat()
            assert [s.st_size, s.st_mtime_ns] == stat, path
    return p

def check_time(deadline):
    main_paused()
    assert time.time() < deadline, 'Stage B 24-hour budget exhausted'
    assert not (B / 'controller-failed.json').exists(), 'Stage B controller failed'

def wait_path(path, deadline):
    while not Path(path).exists():
        check_time(deadline); time.sleep(5)

def free_mib(device):
    s = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        timeout=20, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return int(s.stdout.splitlines()[int(device.split(':')[1])].strip())

def wait_room(device, deadline, need=18000):
    # Never start anyway after a timeout and never terminate another project.
    next_log = 0
    while True:
        check_time(deadline)
        free = free_mib(device)
        if free >= need:
            print(f'{device}: {free} MiB free; starting', flush=True); return
        if time.time() >= next_log:
            print(f'{device}: waiting for {need} MiB, currently {free}', flush=True)
            next_log = time.time() + 300
        time.sleep(15)

def scene_rows():
    return [r for r in read(SIDE / 'corpus.json')['rows'] if r['split'] == 'B']

def parser_question(q):
    """Only normalize a known new template for the legacy factual parser.

    The observer still receives the original, frozen text. A never used factual
    gold, so its answers and scores are unaffected by this adapter.
    """
    from selfsight.v4.spec import canonical_noun
    if ':diagnostic-absence:' not in q['atom_id'] or q['family'] != 'count':
        return q
    noun = q['atom_id'].split(':diagnostic-absence:')[1]
    expected = f'How many {noun} objects are in this picture? Answer with a single whole number only.'
    assert q['text'] == expected and canonical_noun(noun) == noun and ' ' not in noun
    # Singular canonical noun is sufficient for scope parsing, including glass.
    # This internal string is never sent to the observer.
    return {**q, 'text': f'How many {noun} are in this picture? Answer with a single whole number only.'}

def fact(q, verification):
    from selfsight.v4.factual_truth import factual_answer
    return factual_answer(parser_question(q), verification)

def image_gold(v):
    if v is None or v.get('resolution') == 'pending_human': return None
    assert v.get('resolution') in {'agreed', 'agreed_verdict', 'resolved_by_crop', 'human'}
    assert isinstance(v['image_correct'], bool)
    return int(v['image_correct'])

def pipeline_module():
    import importlib.util
    path = ROOT / 'scripts/v4_run_pipeline.py'
    spec = importlib.util.spec_from_file_location('frozen_v4_pipeline', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def all_images(p):
    out = []
    for batch in p['batches']:
        folder = B / 'generation' / batch['id']
        done = read(folder / 'complete.json')
        assert sha(folder / 'manifest.jsonl') == done['manifest_sha256']
        rs = lines(folder / 'manifest.jsonl')
        assert len(rs) == done['n_images'] == len(batch['scene_ids']) * 6
        out.extend(rs)
    assert len(out) == 900 and len({(r['spec_id'],r['candidate_index']) for r in out}) == 900
    return out
