"""Run the frozen analysis on the recovery logical directory, with provenance."""
from runtime_common import *
import importlib.util
import shutil


def recovery_verifications(p, human=None):
    pipeline = pipeline_module()
    from selfsight.v4.spec import SceneSpec, image_correct
    from selfsight.v4.verifier import verify
    out = {}
    for batch in p['batches']:
        folder = R / 'generation' / batch['id']
        complete = read(folder / 'verification-complete.json')
        for key, expected in complete['sha256'].items():
            f = folder / ('verified.jsonl' if key == 'verified' else f'detections.{key}.jsonl')
            assert sha(f) == expected, f
        first = {r['image_path']: r['detections'] for r in lines(folder / 'detections.qwen3vl.jsonl') if 'detections' in r and 'error' not in r}
        second = {r['image_path']: r['detections'] for r in lines(folder / 'detections.internvl.jsonl') if 'detections' in r and 'error' not in r}
        saved = {r['image_path']: r for r in lines(folder / 'verified.jsonl')}
        crops = pipeline.cached_crops(folder / 'detections.crops.jsonl')
        for row in lines(folder / 'manifest.jsonl'):
            path = row['image_path']; spec = SceneSpec.from_dict(row['spec'])
            assert path in saved and path not in out
            if path not in first or path not in second:
                v = saved[path]
                assert v['resolution'] == 'pending_human' and v['image_correct'] is None
                assert v['factual_truth_unavailable'] is True
                if human is not None and path in human:
                    v = {**v, 'resolution': 'human', 'detections': human[path],
                         'image_correct': image_correct(spec, human[path]), 'factual_truth_unavailable': False}
                out[path] = v
            elif human is not None and path in human:
                out[path] = verify(path, spec, pipeline._Replay('qwen3vl', first, crops),
                                   pipeline._Replay('internvl', second, crops), human_labels=human).to_dict()
            else:
                out[path] = saved[path]
    assert len(out) == 900
    return out


def main():
    verify_recovery()
    inventory = read(R / 'reuse-inventory.json')
    for item in inventory['observations']:
        path = Path(item['path'])
        with path.open('rb') as f:
            prefix = f.read(item['original_prefix_bytes'])
        assert hashlib.sha256(prefix).hexdigest() == item['sha256'], path
    assert sha(R / 'plan.json') == sha(OLD / 'plan.json')
    spec = importlib.util.spec_from_file_location('original_stage_b_analysis', OLD / 'analyze.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    # These are the only runtime substitutions. Frozen scientific functions stay unchanged.
    analysis.B = R
    analysis.all_images = all_images
    analysis.verify_plan = original_plan
    analysis.verifications = recovery_verifications
    sys.argv = [str(OLD / 'analyze.py'), '--tag', 'recovery']
    analysis.main()
    folder = R / 'analysis-recovery'
    quarantined = [r for b in original_plan()['batches'] for r in lines(R / 'generation' / b['id'] / 'verified.jsonl')
                   if r.get('factual_truth_unavailable')]
    for name in ('results.json', 'integrity.json'):
        original_report = read(folder / name)
        shutil.copyfile(folder / name, folder / (name + '.pre-annotation'))
        original_report['parse_failure_quarantine_images'] = len(quarantined)
        original_report['original_all_detectors_valid_requirement_met'] = not quarantined
        original_report['recovery_policy_addendum_sha256'] = sha(R / 'PARSE_FAILURE_ADDENDUM.md')
        if name == 'integrity.json':
            original_report['status'] = 'PASS_data_integrity_with_explicit_quarantine' if quarantined else 'PASS_data_integrity_recovery'
            original_report['results_sha256'] = sha(folder / 'results.json')
        (folder / name).write_text(json.dumps(original_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (folder / 'RESULTS.md').open('a', encoding='utf-8') as f:
        f.write(f'\n恢复政策偏离：{len(quarantined)}张图因检测JSON解析失败保留全未知并送人工；原要求的全图双检测有效条件未被自动补为通过。原始失败和本次回复都保留。\n')
    verify_recovery()
    save_new(R / 'analysis-recovery/recovery-integrity.json', {
        'at_utc': now(), 'old_artifacts_unchanged': True, 'reused_observation_prefixes_unchanged': True,
        'original_analysis_sha256': sha(OLD / 'analyze.py'),
        'recovery_plan_sha256': sha(R / 'recovery-plan.json'),
        'results_sha256': sha(R / 'analysis-recovery/results.json'),
        'original_failed_attempt_preserved': True})


if __name__ == '__main__':
    main()
