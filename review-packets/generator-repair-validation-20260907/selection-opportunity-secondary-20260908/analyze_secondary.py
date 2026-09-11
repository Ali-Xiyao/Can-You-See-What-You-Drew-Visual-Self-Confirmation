"""Registered explanatory pool strata; never changes the all-150 primary result."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import sys
import time

S = Path(__file__).resolve().parent
SIDE = S.parent
B = SIDE / 'stage-B'
R = SIDE / 'stage-B-recovery-20260907'
sys.path.insert(0, str(B))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = load_module('secondary_frozen_bcommon', B / 'bcommon.py')
stats = load_module('secondary_frozen_stats', B / 'stats.py')
STRUCTURES = ('all_wrong', 'all_correct', 'confirmed_mixed', 'unresolved_membership')
AVAILABILITY = ('atleast1knowntrue', 'certifiedallwrong', 'unresolvednoneknowntrue')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    data = Path(path).read_bytes()
    assert not data or data.endswith(b'\n'), 'Incomplete JSONL record'
    return [json.loads(line) for line in data.splitlines()]


def classify(labels):
    assert len(labels) == 6 and all(y is None or type(y) in (int, bool) and y in (0, 1) for y in labels)
    c, f, u = sum(y == 1 for y in labels), sum(y == 0 for y in labels), sum(y is None for y in labels)
    availability = 'atleast1knowntrue' if c else 'certifiedallwrong' if u == 0 else 'unresolvednoneknowntrue'
    structure = ('all_wrong' if c == u == 0 else 'all_correct' if f == u == 0
                 else 'confirmed_mixed' if c and f else 'unresolved_membership')
    return {'c': c, 'f': f, 'u': u, 'availability': availability, 'structure': structure}


def describe(pools):
    assignments = {sid: classify(labels) for sid, labels in sorted(pools.items())}
    n = len(pools)
    c = sum(item['c'] for item in assignments.values())
    f = sum(item['f'] for item in assignments.values())
    u = sum(item['u'] for item in assignments.values())
    return {'n_scenes': n, 'n_images': 6 * n, 'known_correct_images': c,
            'known_wrong_images': f, 'unknown_images': u,
            'image_correct_rate_bounds': [c / (6*n), (c+u) / (6*n)] if n else None,
            'availability': {group: {'n_scenes': sum(a['availability'] == group for a in assignments.values()),
                'scene_ids': [sid for sid, a in assignments.items() if a['availability'] == group]}
                for group in AVAILABILITY},
            'structure': {group: {'n_scenes': sum(a['structure'] == group for a in assignments.values()),
                'scene_ids': [sid for sid, a in assignments.items() if a['structure'] == group]}
                for group in STRUCTURES}, 'per_pool': assignments}


def inventory(plan, recovery_dir=R):
    batches, combined = {}, {}
    for batch in plan['batches']:
        folder = recovery_dir / 'generation' / batch['id']
        marker = folder / 'verification-complete.json'
        if not marker.exists():
            continue  # No reads of partial gold, raw calls, or model answers.
        verification = read(marker)
        generation = read(folder / 'complete.json')
        assert generation['manifest_sha256'] == sha(folder / 'manifest.jsonl')
        assert verification['sha256']['verified'] == sha(folder / 'verified.jsonl')
        assert verification['detector_snapshot_digest'] == plan['detector_snapshot_digest']
        manifest, verified = rows(folder / 'manifest.jsonl'), rows(folder / 'verified.jsonl')
        assert len(manifest) == len(verified) == generation['n_images'] == verification['n_images'] == 150
        assert len(batch['scene_ids']) == 25
        by_path = {row['image_path']: row for row in verified}
        assert len(by_path) == 150 and set(by_path) == {r['image_path'] for r in manifest}
        assert {(r['spec_id'], r['candidate_index']) for r in manifest} == {
            (sid, index) for sid in batch['scene_ids'] for index in range(6)}
        pools = {sid: [None] * 6 for sid in batch['scene_ids']}
        for image in manifest:
            value = by_path[image['image_path']]
            assert value['spec_id'] == image['spec_id']
            assert value['candidate_index'] == image['candidate_index']
            pools[image['spec_id']][image['candidate_index']] = frozen.image_gold(value)
        assert not set(pools) & set(combined), 'Scene repeated between batches'
        combined.update(pools)
        batches[batch['id']] = {**describe(pools), 'sources_sha256': {
            str(folder / name): sha(folder / name) for name in
            ('manifest.jsonl', 'complete.json', 'verified.jsonl', 'verification-complete.json')}}
    return {'completed_batches': list(batches), 'batches': batches, 'combined': describe(combined),
            'partial_batches_not_read': [b['id'] for b in plan['batches'] if b['id'] not in batches]}


def validate_per_pool(records, models, conditions, expected_scene_ids):
    expected_scene_ids = set(expected_scene_ids)
    assert len(models) == len(set(models)) == 4 and 'base' in models
    assert len(conditions) == len(set(conditions)) == 3
    expected = {(model, condition, sid) for model in models for condition in conditions for sid in expected_scene_ids}
    table, labels = {}, {}
    for row in records:
        key = row['model'], row['condition'], row['spec_id']
        assert key in expected and key not in table, 'Unexpected or repeated pool identity'
        y = row['labels']
        classify(y)
        assert len(row['original_scores']) == 6
        assert all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in row['original_scores'])
        if row['spec_id'] in labels:
            assert labels[row['spec_id']] == y, 'Models/conditions disagree on shared image gold'
        else:
            labels[row['spec_id']] = list(y)
        table[key] = row
    assert set(table) == expected, 'Missing model/condition/scene pool'
    return table, labels


def summarize(records, models, conditions, expected_scene_ids):
    table, labels = validate_per_pool(records, models, conditions, expected_scene_ids)
    population = describe(labels)
    output = {}
    for group in STRUCTURES:
        ids = population['structure'][group]['scene_ids']
        entry = {'n_scenes': len(ids), 'n_images': 6*len(ids), 'scene_ids': ids,
                 'estimated': bool(ids), 'summaries': {}, 'paired_changes_vs_base': {}}
        if not ids:
            entry['reason'] = 'Empty stratum; no estimate or interval'
            output[group] = entry
            continue
        weights = {(model, condition, sid): stats.top_weights(table[model, condition, sid]['original_scores'])
                   for model in models for condition in conditions for sid in ids}
        random = [1/6] * 6
        entry['random_correct_rate'] = stats.bounded_summary([stats.weighted_bounds(random, labels[sid]) for sid in ids])
        for model in models:
            for condition in conditions:
                key = model + ':' + condition
                entry['summaries'][key] = {
                    'top_correct_rate': stats.bounded_summary([
                        stats.weighted_bounds(weights[model, condition, sid], labels[sid]) for sid in ids]),
                    'selection_gain_vs_uniform_random': stats.bounded_summary([
                        stats.paired_selection(weights[model, condition, sid], random, labels[sid]) for sid in ids])}
                if model != 'base':
                    entry['paired_changes_vs_base'][key] = stats.bounded_summary([
                        stats.paired_selection(weights[model, condition, sid], weights['base', condition, sid], labels[sid])
                        for sid in ids])
        output[group] = entry
    return {'population': population, 'strata': output,
            'endpoint': 'uniform expectation within original-score highest-score tie set',
            'primary_all_150_result_unchanged': True, 'new_pass_fail_decisions': False,
            'limitations': [
                'Explanatory secondary analysis registered after seeing batch-00 gold availability.',
                'Confirmed mixed pools are a known subset, not the population of all truly mixed pools.',
                'Unknown memberships remain separately reported; unknown labels use paired worst completion.',
                'All four models and all three conditions use the same scene membership and labels.',
                'No alteration of the original precision, retention or stopping rules.']}


def registration(require_source_freeze=False):
    path = S / 'REGISTRATION.json'
    reg = read(path)
    timestamp = datetime.fromisoformat(reg['timestamp'].replace('Z', '+00:00'))
    assert timestamp.tzinfo is not None and timestamp <= datetime.now(timezone.utc)
    assert reg['prior_data_seen']['completed_batches'] == ['batch-00']
    assert reg['prior_data_seen']['model_selection_comparisons_seen'] is False
    for source, expected in reg.get('source_sha256', {}).items():
        assert sha(source) == expected, 'Original registered source changed: ' + str(source)
    provenance = {'path': str(path), 'sha256': sha(path), 'timestamp': reg['timestamp']}
    if require_source_freeze:
        freeze_path = S / 'CODE_FREEZE.json'
        freeze = read(freeze_path)
        freeze_time = datetime.fromisoformat(freeze['timestamp'].replace('Z', '+00:00'))
        assert freeze_time.tzinfo is not None and timestamp <= freeze_time <= datetime.now(timezone.utc)
        assert freeze['registration_sha256'] == sha(path), 'Code freeze is bound to a different registration'
        expected = freeze['source_sha256']
        for source in (S / 'analyze_secondary.py', S / 'test_secondary.py', B / 'stats.py', B / 'bcommon.py'):
            assert expected[str(source)] == sha(source), 'Secondary source freeze mismatch: ' + str(source)
        provenance['code_freeze'] = {'path': str(freeze_path), 'sha256': sha(freeze_path), 'timestamp': freeze['timestamp']}
    return provenance


def completed_results(path, plan):
    path = Path(path).resolve()
    assert path == (R / 'analysis-recovery/results.json').resolve(), 'Only the original completed recovery analysis is accepted'
    # Check completion artifacts before opening the potentially partial results.
    done = read(R / 'controller-complete.json')
    audit = read(R / 'analysis-recovery/recovery-integrity.json')
    integrity = read(R / 'analysis-recovery/integrity.json')
    digest = sha(path)
    assert done['exit_code'] == 0
    assert done['results_sha256'] == audit['results_sha256'] == integrity['results_sha256'] == digest
    assert integrity['images'] == 900 and integrity['pool_scores'] == 1800
    result = read(path)
    assert result['stage'] == 'B' and result['n_scenes'] == 150 and result['n_images'] == 900
    assert result['plan_sha256'] == sha(B / 'plan.json')
    models = [model['id'] for model in plan['models']]
    ids = [sid for batch in plan['batches'] for sid in batch['scene_ids']]
    assert len(ids) == len(set(ids)) == 150
    report = summarize(result['per_pool'], models, plan['conditions'], ids)
    return {**report, 'primary_results': {'path': str(path), 'sha256': digest},
            'completion_sha256': {str(p): sha(p) for p in
                (R/'controller-complete.json', R/'analysis-recovery/recovery-integrity.json', R/'analysis-recovery/integrity.json')}}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inventory', action='store_true')
    mode.add_argument('--results', type=Path)
    args = parser.parse_args()
    provenance = registration(require_source_freeze=bool(args.results))
    plan = read(B / 'plan.json')
    report = inventory(plan) if args.inventory else completed_results(args.results, plan)
    mode_name = 'inventory' if args.inventory else 'secondary'
    folder = S / (mode_name + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(time.time_ns()))
    folder.mkdir(exist_ok=False)
    report.update(created_utc=datetime.now(timezone.utc).isoformat(), registration=provenance,
                  code_sha256={str(p): sha(p) for p in (Path(__file__), B/'stats.py', B/'bcommon.py')})
    with (folder / 'results.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'output': str(folder / 'results.json'), 'sha256': sha(folder / 'results.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
