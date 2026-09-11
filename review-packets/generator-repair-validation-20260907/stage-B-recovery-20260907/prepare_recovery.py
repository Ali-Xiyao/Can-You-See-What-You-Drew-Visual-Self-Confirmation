"""Copy exact completed artifacts/prefixes; never mutate the failed attempt."""
from rcommon import *
import shutil


def main():
    assert not (R / 'reuse-inventory.json').exists(), 'Preparation is single-use'
    p = original_plan()
    failure = read(OLD / 'controller-failed.json')
    assert failure['error'] == 'detect-batch-00: exit 1'
    assert read(OLD / 'generation/complete.json')['n_images'] == 900
    assert shutil.disk_usage(ROOT).free > 8 * 1024**3
    copied = []; hashes = {}

    def exact(src, dest):
        assert not dest.exists(), dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        h = stream_sha(src)
        assert stream_sha(dest) == h
        copied.append({'source': str(src), 'destination': str(dest), 'sha256': h,
                       'bytes': src.stat().st_size})
        hashes[str(src)] = h

    for f in sorted(OLD.rglob('*')):
        if f.is_file() and f.suffix != '.png':
            hashes[str(f)] = stream_sha(f)
    for batch in p['batches']:
        for name in ('manifest.jsonl', 'complete.json'):
            exact(OLD / 'generation' / batch['id'] / name, R / 'generation' / batch['id'] / name)
    exact(OLD / 'generation/complete.json', R / 'generation/complete.json')
    exact(OLD / 'generation/batch-00/detections.qwen3vl.jsonl',
          R / 'generation/batch-00/detections.qwen3vl.jsonl')
    bad = lines(OLD / 'generation/batch-00/detections.internvl.jsonl')
    assert len(bad) == 150 and sum('error' in row for row in bad) == 1
    for f in sorted((OLD / 'observations').rglob('*')):
        if f.is_file():
            assert f.name in {'answers.jsonl', 'complete.json'}, f
            exact(f, R / f.relative_to(OLD))
    counts = []
    for f in sorted((R / 'observations').rglob('answers.jsonl')):
        rows = lines(f)
        answers = [a for row in rows for a in row['observation']['answers']]
        assert not any(a.get('error') for a in answers)
        counts.append({'path': str(f), 'rows': len(rows), 'answers': len(answers),
                       'original_prefix_bytes': f.stat().st_size, 'sha256': sha(f),
                       'complete': (f.parent / 'complete.json').exists()})
    assert sum(c['answers'] for c in counts) == 23640
    images = all_images(p)
    for image in images:
        assert stream_sha(image['image_path']) == image['file_sha256'], image['image_path']
        hashes[image['image_path']] = image['file_sha256']
    save_new(R / 'reuse-inventory.json', {'at_utc': now(), 'user_authorized_recovery': True,
             'original_failure': failure, 'original_artifacts': hashes, 'copied': copied,
             'observations': counts, 'n_images': len(images), 'new_optimizer_steps': 0,
             'deadline': read(OLD / 'controller-started.json')['deadline']})
    print('Prepared 900 existing images, exact observation prefixes (23640 answers), and frozen evidence.', flush=True)


if __name__ == '__main__':
    main()
