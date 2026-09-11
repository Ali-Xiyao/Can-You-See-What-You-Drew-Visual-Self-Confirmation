"""Post-completion descriptive checks; does not change frozen endpoints or rules."""
import json
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import numpy as np

SIDE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def interval(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(20260906).choice(values, (5000, len(values)), replace=True).mean(axis=1)
    return {'point': float(values.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}


def main():
    result = read(SIDE / 'results.json')
    validation = read(SIDE / 'results-validation.json')
    assert validation['results_sha256'] == sha(SIDE / 'results.json')
    paths = {key: Path(value) for key, value in read(SIDE / 'extension/analysis-input-resolution.json')['folders'].items()}
    out = {'created_utc': datetime.now(timezone.utc).isoformat(), 'results_sha256': sha(SIDE / 'results.json'),
           'scope': 'Post-completion descriptive supplement and integrity checks; frozen classifications unchanged.',
           'tasks': {}, 'cross_arm_agreement': {}, 'paired_condition_differences_32': {}}
    for name, folder in paths.items():
        start, done = read(folder / 'started.json'), read(folder / 'complete.json')
        assert start['base_parameter_digest'] != done['parameter_digest']
        probes = [read(folder / 'probes' / f'step-{step:05d}.json') for step in range(33)]
        for probe in probes:
            for condition, groups in probe['summary'].items():
                for group, recorded in groups.items():
                    group_rows = [r for r in probe['answers'] if r['condition'] == condition and r['group'] == group]
                    assert len(group_rows) == recorded['n'] == 25
                    assert dict(Counter(r['answer']['normalized_answer'] for r in group_rows)) == recorded['outputs']
                    accuracy = sum(r['answer']['normalized_answer'] == r['truth'] and not r['answer']['abstain'] and not r['answer'].get('error') for r in group_rows) / len(group_rows)
                    assert accuracy == recorded['accuracy']
        alarms = {}
        for condition in ('prompt_on', 'prompt_off'):
            for group in ('existence_no', 'count_zero'):
                baseline = probes[0]['summary'][condition][group]['accuracy']
                for step in range(2, 33):
                    if all(probes[s]['summary'][condition][group]['accuracy'] <= baseline - .2 + 1e-12 for s in (step - 1, step)):
                        alarms[f'{condition}:{group}'] = {'start_step': step - 1, 'first_persistent_step': step}
                        break
        full_final = read(folder / 'measurements/step-00032-complete.json')
        assert full_final['parameter_digest_before'] == full_final['parameter_digest_after'] == done['parameter_digest']
        out['tasks'][name] = {'weights_changed': True, 'base_parameter_digest': start['base_parameter_digest'],
                             'final_parameter_digest': done['parameter_digest'], 'probe_summaries_recomputed': 33,
                             'absence_alarms': alarms}
    for variant in ('original', 'alternate_ids'):
        out['cross_arm_agreement'][variant] = {}
        for step in (0, 16, 32):
            answers = []
            for arm in ('naive', 'rfo_gold'):
                obs = rows(paths[f'{arm}-{variant}'] / 'measurements' / f'step-{step:05d}.jsonl')
                answers.append({(r['condition'], r['prompt_id'], r['candidate_id'], a['question_id']): a['normalized_answer'] for r in obs for a in r['observation']['answers']})
            a, b = answers
            assert a.keys() == b.keys() and len(a) == 1818
            out['cross_arm_agreement'][variant][str(step)] = {'same': sum(a[k] == b[k] for k in a), 'total': len(a)}
    for arm, variant in [('naive', 'zero_weight'), ('naive', 'balanced_absence'), ('naive', 'alternate_ids'), ('rfo_gold', 'alternate_ids')]:
        differences = {}
        for condition in ('prompt_blank', 'prompt_off'):
            def lookup(task):
                return {p['prompt_id']: p['original']['top_correct_rate'] for p in result['per_pool'][task]['32'][condition] if p['primary_scene_disjoint']}
            old, new = lookup(f'{arm}-original'), lookup(f'{arm}-{variant}')
            assert old.keys() == new.keys() and len(old) == 14
            differences[condition] = interval([new[key] - old[key] for key in old])
        out['paired_condition_differences_32'][f'{arm}:{variant}-original'] = differences
    source = SIDE / 'runs/naive-alternate_ids'
    resumed = paths['naive-alternate_ids']
    assert (source / 'training.jsonl').read_bytes() == (resumed / 'training.jsonl').read_bytes()
    restore = read(SIDE / 'extension/resume-verification/complete.json')
    assert restore['matches'] == restore['total'] == 200
    assert restore['optimizer_step'] == restore['scheduler_epoch'] == 32
    assert read(resumed / 'complete.json')['new_optimizer_steps'] == 0
    assert read(SIDE / 'extension/overlap-reproduction.json')['rows'] == []
    for relative, metadata in read(SIDE / 'extension/prefix-provenance.json')['copied_files'].items():
        assert sha(Path(metadata['source'])) == metadata['sha256'] == sha(resumed / relative)
    assert read(SIDE / 'extension/controller-complete.json')['exit_code'] == 0
    out['continuation'] = {'resumed_from_step': 32, 'extra_optimizer_steps': 0, 'training_bytes_identical': True,
                           'all_prefix_files_hash_verified': True, 'restored_probe_match': '200/200', 'replacement_training_tail': False}
    out['probe_answers_rechecked'] = 39600
    with (SIDE / 'final-review.json').open('x', encoding='utf-8') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: out[key] for key in ('cross_arm_agreement', 'paired_condition_differences_32', 'continuation')}, indent=2))
    print('Absence alarms:', json.dumps({k: v['absence_alarms'] for k, v in out['tasks'].items()}))


if __name__ == '__main__':
    main()
