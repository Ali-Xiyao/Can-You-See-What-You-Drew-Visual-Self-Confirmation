"""One authorized repeat, identical detector settings, full reply preserved."""
from rcommon import *


def main():
    p = verify_recovery()
    deadline = p['deadline']
    assert not (R / 'diagnostic-started.json').exists(), 'Only one diagnostic attempt'
    bad = [r for r in lines(OLD / 'generation/batch-00/detections.internvl.jsonl') if 'error' in r]
    assert len(bad) == 1
    target = bad[0]['image_path']
    images = {r['image_path']: r for r in lines(OLD / 'generation/batch-00/manifest.jsonl')}
    assert sha(target) == images[target]['file_sha256']
    save_new(R / 'diagnostic-started.json', {'at_utc': now(), 'image_path': target,
        'old_error': bad[0]['error'], 'deadline': deadline, 'new_optimizer_steps': 0})
    wait_room('cuda:1', deadline)
    from selfsight.v4.detectors import load
    from selfsight.v4.train import seed_training
    seed_training(seed('B-detector', 'batch-00', 'internvl'))
    detector = load('internvl', device='cuda:1')
    record = {'at_utc': now(), 'image_path': target, 'model': 'internvl',
              'same_prompt_and_decode_parameters': True, 'file_sha256': sha(target)}
    try:
        check_time(deadline)
        record['detections'] = detector.detect(target)
        record['success'] = True
    except Exception as exc:
        record.update(success=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        record['reply'] = detector.last_reply
        record['finished_at_utc'] = now()
        save_new(R / 'diagnostic-result.json', record)
    if not record['success']:
        raise RuntimeError('Same-configuration diagnostic failed; full reply preserved')
    replacement = {'image_path': target, 'detections': record['detections'], 'reply': record['reply']}
    old = lines(OLD / 'generation/batch-00/detections.internvl.jsonl')
    rows = [replacement if r['image_path'] == target else r for r in old]
    assert len(rows) == 150 and all('detections' in r and 'error' not in r for r in rows)
    dest = R / 'generation/batch-00/detections.internvl.jsonl'
    with dest.open('x', encoding='utf-8') as f:
        for row in rows:
            append(f, row)
    save_new(R / 'diagnostic-complete.json', {'at_utc': now(), 'success': True,
        'reused_successful_rows': 149, 'new_same_configuration_rows': 1,
        'new_file_sha256': sha(dest), 'original_file_sha256': sha(OLD / 'generation/batch-00/detections.internvl.jsonl'),
        'diagnostic_result_sha256': sha(R / 'diagnostic-result.json')})
    print('Single-image diagnostic succeeded; 149 original successful detections retained.', flush=True)


if __name__ == '__main__':
    main()
