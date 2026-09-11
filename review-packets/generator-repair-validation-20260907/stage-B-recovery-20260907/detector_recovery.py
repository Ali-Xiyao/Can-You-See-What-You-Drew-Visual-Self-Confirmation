"""Frozen Stage B detectors with complete raw logs and explicit JSON quarantine.

No parser, instruction, model, token budget or sampling changes. Existing output
may only be reused when every row is successful or has a proven JSON parse
failure. Parse failures stay unknown; other errors stop. No model retry.
"""
import argparse
import gc
import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from runtime_common import (R, append, check_time, lines, now, original_plan,
                     pipeline_module, read, save_new, seed, sha,
                     verify_recovery, wait_room)


class DetectorStopped(BaseException):
    """Bypass the legacy pipeline's per-row continue-on-error handlers."""


def validate_detections(path, data):
    rows = lines(path)
    expected = [row['image_path'] for row in data]
    actual = [row.get('image_path') for row in rows]
    assert len(expected) == len(set(expected)), 'Duplicate manifest image'
    assert len(actual) == len(expected) and set(actual) == set(expected), \
        f'Incomplete or mismatched detector file: {path}'
    assert len(actual) == len(set(actual)), f'Duplicate detector image: {path}'
    for row in rows:
        validate_row(row)
    return rows


def validate_row(row):
    if 'error' not in row:
        assert isinstance(row.get('detections'), list), 'Missing successful detections'
        return
    from selfsight.v4.detectors import parse_reply
    assert 'detections' not in row, 'A quarantined failure cannot invent detections'
    assert row.get('quarantined') is True
    assert row.get('failure_class') == 'json_parse_error'
    assert isinstance(row.get('reply'), str), 'Full failed reply must be retained'
    _, error = parse_reply(row['reply'], 1, 1)
    assert error and error.startswith('json_error:'), 'Only original JSON parse errors may be quarantined'


def annotate_quarantines(path, raw_path):
    """Retain original pipeline bytes, then add exact raw-call provenance."""
    if not path.exists():
        return
    rows = lines(path)
    failures = [r for r in rows if 'error' in r]
    if not failures:
        return
    events = [r for r in lines(raw_path) if r.get('event') == 'model_call']
    by_input = {}
    for event in events:
        key = (event['image_path'], tuple(event.get('bbox', ())))
        by_input.setdefault(key, []).append(event)
    for row in rows:
        matching = by_input.get((row['image_path'], tuple(row.get('bbox', ()))), [])
        event = matching.pop(0) if matching else None
        if 'error' not in row:
            continue
        assert event is not None, 'Failed output has no raw call'
        assert event['status'] == 'parse_error' and event['error'].startswith('json_error:')
        row.update(quarantined=True, failure_class='json_parse_error', reply=event['reply'],
            raw_capture={'path': str(raw_path), 'call_index': event['call_index'],
                         'sha256': sha(raw_path)})
        validate_row(row)
    archive = path.with_name(path.name + '.pipeline-original')
    with archive.open('xb') as handle:
        handle.write(path.read_bytes())
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            append(handle, row)


class AuditedDetector:
    """Wrap the frozen detector; append/flush the exact reply of every _run."""

    def __init__(self, detector, handle, deadline, detector_name, phase):
        from selfsight.v4.detectors import parse_reply
        self.detector = detector
        self.detector_id = detector.detector_id
        self.handle = handle
        self.deadline = deadline
        self.detector_name = detector_name
        self.phase = phase
        self.context = None
        self.index = 0
        self.current_parse_error = None
        self.parse_reply = parse_reply
        self.original_run = detector._run
        detector._run = self._run

    @property
    def last_reply(self):
        return self.detector.last_reply

    def _run(self, image, instruction):
        check_time(self.deadline)
        self.index += 1
        self.detector.last_reply = ''
        self.current_parse_error = None
        event = {
            'event': 'model_call', 'at_utc': now(), 'call_index': self.index,
            'detector': self.detector_name, 'detector_id': self.detector_id,
            'phase': self.phase, **self.context,
            'instruction': instruction,
            'input': {'mode': image.mode, 'width': image.width,
                      'height': image.height,
                      'pixel_bytes_sha256': hashlib.sha256(image.tobytes()).hexdigest()},
            'reply': None, 'error': None,
        }
        try:
            reply = self.original_run(image, instruction)
            event['reply'] = reply
            # The same pure parser annotates the log; the original detector
            # still performs its own parse and determines the returned result.
            _, error = self.parse_reply(reply, image.width, image.height)
            event['error'] = error
            self.current_parse_error = error
            event['status'] = 'parse_error' if error else 'ok'
            return reply
        except BaseException as exc:
            event['status'] = 'model_error'
            event['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            event['finished_at_utc'] = now()
            append(self.handle, event)

    def _operation(self, method, image_path, bbox=None):
        check_time(self.deadline)
        self.detector.last_reply = ''
        self.current_parse_error = None
        self.context = {'image_path': str(image_path), 'operation': method}
        if bbox is not None:
            self.context['bbox'] = list(bbox)
        try:
            args = (image_path,) if bbox is None else (image_path, bbox)
            return getattr(self.detector, method)(*args)
        except BaseException as exc:
            append(self.handle, {'event': 'operation_error', 'at_utc': now(),
                'detector': self.detector_name, 'phase': self.phase,
                **self.context, 'error': f'{type(exc).__name__}: {exc}',
                'last_reply': self.detector.last_reply})
            if (isinstance(exc, ValueError) and self.current_parse_error
                    and self.current_parse_error.startswith('json_error:')):
                # The frozen pipeline records this row without detections and
                # continues. The complete reply is attached before verification.
                raise
            raise DetectorStopped(f'{self.detector_name} {method} failed; raw audit '
                                  'saved, no retry or fallback') from exc

    def detect(self, image_path):
        return self._operation('detect', image_path)

    def detect_crop(self, image_path, bbox):
        return self._operation('detect_crop', image_path, bbox)


@contextmanager
def audited_loader(folder, phase, deadline):
    """Use the unchanged model loader, intercepting only runtime logging."""
    import selfsight.v4.detectors as module
    original_load = module.load
    handles = []

    def load(name, device='cuda:0'):
        check_time(deadline)
        path = folder / f'raw-calls.{phase}.{name}.jsonl'
        handle = path.open('x', encoding='utf-8')
        handles.append(handle)
        try:
            detector = original_load(name, device=device)
        except BaseException as exc:
            append(handle, {'event': 'load_error', 'at_utc': now(),
                'detector': name, 'device': device,
                'error': f'{type(exc).__name__}: {exc}'})
            raise DetectorStopped(f'{name} load failed; saved without retry') from exc
        return AuditedDetector(detector, handle, deadline, name, phase)

    module.load = load
    try:
        yield
    finally:
        module.load = original_load
        for handle in handles:
            handle.close()


def detect_batch(p, deadline, batch_id, device='cuda:1'):
    import torch
    from selfsight.v4.train import seed_training
    from selfsight.utils.hashing import rgb_sha256
    check_time(deadline)
    batch = next(b for b in p['batches'] if b['id'] == batch_id)
    folder = R / 'generation' / batch_id
    data = lines(folder / 'manifest.jsonl')
    complete = read(folder / 'complete.json')
    assert len(data) == complete['n_images'] == len(batch['scene_ids']) * 6
    assert sha(folder / 'manifest.jsonl') == complete['manifest_sha256']
    assert {r['spec_id'] for r in data} == set(batch['scene_ids'])
    assert {(r['spec_id'], r['candidate_index']) for r in data} == \
        {(sid, candidate) for sid in batch['scene_ids'] for candidate in range(6)}
    for row in data:
        assert sha(row['image_path']) == row['file_sha256']
        assert rgb_sha256(row['image_path']) == row['rgb_sha256']
    hashes = {}
    pipeline = pipeline_module()
    pipeline.await_room = lambda dev, *args, **kwargs: wait_room(dev, deadline)
    for name in ('qwen3vl', 'internvl'):
        check_time(deadline)
        out = folder / f'detections.{name}.jsonl'
        if out.exists():
            validate_detections(out, data)
            print(f'{batch_id} {name}: validated complete cached output', flush=True)
        else:
            seed_training(seed('B-detector', batch_id, name))
            with audited_loader(folder, 'detect', deadline):
                pipeline.stage_detect(SimpleNamespace(manifest=folder / 'manifest.jsonl',
                    detector=name, device=device, output=out, overwrite=False))
            annotate_quarantines(out, folder / f'raw-calls.detect.{name}.jsonl')
            validate_detections(out, data)
        hashes[name] = sha(out)
        gc.collect()
        torch.cuda.empty_cache()
    marker = folder / 'verification-complete.json'
    if marker.exists():
        previous = read(marker)
        assert previous['n_images'] == len(data)
        assert previous['detector_snapshot_digest'] == p['detector_snapshot_digest']
        for name, value in previous['sha256'].items():
            path = folder / ('verified.jsonl' if name == 'verified' else f'detections.{name}.jsonl')
            assert sha(path) == value
        print(f'{batch_id}: validated completed verification', flush=True)
        return
    # A partial earlier crop attempt is never treated as an implicit retry.
    assert not (folder / 'raw-calls.crop.qwen3vl.jsonl').exists(), \
        'Crop attempt already exists without completion; explicit review required'
    crops = folder / 'detections.crops.jsonl'
    if crops.exists():
        for row in lines(crops):
            validate_row(row)
    check_time(deadline)
    with audited_loader(folder, 'crop', deadline):
        pipeline.stage_crop(SimpleNamespace(run=folder, primary='qwen3vl',
            secondary='internvl', device=device, overwrite=False))
    if crops.exists():
        annotate_quarantines(crops, folder / 'raw-calls.crop.qwen3vl.jsonl')
        for row in lines(crops):
            validate_row(row)
        hashes['crops'] = sha(crops)
    check_time(deadline)
    assert not (folder / 'verified.jsonl').exists(), 'Unmarked verification needs explicit review'
    rows = verify_rows(data, lines(folder / 'detections.qwen3vl.jsonl'),
        lines(folder / 'detections.internvl.jsonl'), pipeline.cached_crops(crops), pipeline)
    with (folder / 'verified.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            append(handle, row)
    save_new(folder / 'verified.summary.json', {
        'n_images': len(rows), 'resolution_counts': dict(Counter(r['resolution'] for r in rows)),
        'unknown_image_labels': sum(r['resolution'] == 'pending_human' for r in rows),
        'quarantined_missing_detector_images': sum(bool(r.get('missing_detectors')) for r in rows),
        'single_detector_verdicts': 0, 'skipped_no_detection': 0})
    assert len(rows) == len(data)
    assert {r['image_path'] for r in rows} == {r['image_path'] for r in data}
    hashes['verified'] = sha(folder / 'verified.jsonl')
    save_new(marker, {'at_utc': now(), 'n_images': len(rows), 'sha256': hashes,
        'detector_snapshot_digest': p['detector_snapshot_digest'], 'device': device,
        'recovery': True, 'frozen_parameters_unchanged': True})


def verify_rows(data, primary_rows, secondary_rows, crops, pipeline):
    """Use original two-detector ladder, never its single-detector fallback."""
    from selfsight.v4.spec import SceneSpec
    from selfsight.v4.verifier import verify
    tables = {}
    for name, rows in [('qwen3vl', primary_rows), ('internvl', secondary_rows)]:
        assert len({r['image_path'] for r in rows}) == len(rows)
        for row in rows:
            validate_row(row)
        tables[name] = {r['image_path']: r['detections'] for r in rows if 'detections' in r}
    out = []
    for image in data:
        path = image['image_path']
        missing = [name for name, table in tables.items() if path not in table]
        if missing:
            record = {'image_path': path, 'spec_id': image['spec_id'], 'detections': [],
                'image_correct': None, 'resolution': 'pending_human',
                'verifier_agreement': None, 'disputed': [],
                'report': {'reason': 'detector_json_parse_error', 'parse_failure_quarantine': True,
                           'single_detector_fallback': False},
                'missing_detectors': missing, 'factual_truth_unavailable': True}
        else:
            record = verify(path, SceneSpec.from_dict(image['spec']),
                pipeline._Replay('qwen3vl', tables['qwen3vl'], crops),
                pipeline._Replay('internvl', tables['internvl'], crops)).to_dict()
        record.update(candidate_index=image['candidate_index'], seed=image['seed'])
        out.append(record)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['detect-batch'])
    parser.add_argument('--batch', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--device', default='cuda:1', choices=['cuda:0', 'cuda:1'])
    args = parser.parse_args()
    verify_recovery()
    detect_batch(original_plan(), args.deadline, args.batch, args.device)
    print(f'{args.batch} detection and verification complete', flush=True)


if __name__ == '__main__':
    main()
