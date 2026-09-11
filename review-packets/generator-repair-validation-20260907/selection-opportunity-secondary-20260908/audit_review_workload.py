"""CPU-only review workload from completed gold and frozen build_facts rules."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import inspect
import json
import sys
import time

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
import analyze_secondary as secondary

sys.path.insert(0, str(secondary.B))
original_analysis = secondary.load_module('workload_frozen_analysis', secondary.B / 'analyze.py')


def summarize(images, verified, facts, unknown, review):
    paths = [row['image_path'] for row in review]
    assert len(paths) == len(set(paths)), 'Duplicate review image'
    whole = sum(original_analysis.image_gold(verified[row['image_path']]) is None for row in images.values())
    known_image_question_unknown = sum(original_analysis.image_gold(row['verification']) is not None for row in review)
    assert len(review) == whole + known_image_question_unknown
    return {'n_images': len(images), 'n_scenes': len({sid for sid, _ in images}),
            'review_images': len(review), 'review_scenes': len({row['spec_id'] for row in review}),
            'whole_image_unknown': whole,
            'known_image_but_question_unknown': known_image_question_unknown,
            'known_question_facts': sum(f.known for f in facts.values()),
            'unknown_question_facts': sum(not f.known for f in facts.values()),
            'n_question_facts': len(facts),
            'unknown_fact_reason_counts': dict(sorted(unknown.items())),
            'review_image_reason_counts': dict(sorted(Counter(reason for row in review for reason in row['reasons']).items())),
            'review': [{'image_path': row['image_path'], 'spec_id': row['spec_id'],
                        'candidate_index': row['candidate_index'],
                        'whole_image_gold': original_analysis.image_gold(row['verification']),
                        'reasons': row['reasons']} for row in review]}


def audit():
    registration = secondary.registration(require_source_freeze=True)
    plan = secondary.read(secondary.B / 'plan.json')
    checked = secondary.inventory(plan)
    scenes = {row['spec']['spec_id']: row for row in secondary.frozen.scene_rows()}
    assert len(scenes) == 150
    batches, all_images, all_verified = {}, {}, {}
    for batch_id in checked['completed_batches']:
        folder = secondary.R / 'generation' / batch_id
        # inventory() validated completion, manifest and gold hashes before any
        # semantic reads. Recheck the same snapshot after computing its facts.
        images = {(row['spec_id'], row['candidate_index']): row for row in secondary.rows(folder/'manifest.jsonl')}
        verified = {row['image_path']: row for row in secondary.rows(folder/'verified.jsonl')}
        assert not set(images) & set(all_images) and not set(verified) & set(all_verified)
        facts, unknown, review = original_analysis.build_facts(scenes, images, verified)
        batches[batch_id] = summarize(images, verified, facts, unknown, review)
        batches[batch_id]['sources_sha256'] = checked['batches'][batch_id]['sources_sha256']
        for path, expected in batches[batch_id]['sources_sha256'].items():
            assert secondary.sha(path) == expected, 'Completed inventory snapshot changed'
        all_images.update(images); all_verified.update(verified)
    facts, unknown, review = original_analysis.build_facts(scenes, all_images, all_verified)
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'completed_batches': checked['completed_batches'],
            'partial_batches_not_read': checked['partial_batches_not_read'],
            'batches': batches, 'combined': summarize(all_images, all_verified, facts, unknown, review),
            'registration': registration,
            'source_sha256': {str(path): secondary.sha(path) for path in
                (Path(__file__), S/'analyze_secondary.py', secondary.B/'analyze.py',
                 secondary.B/'bcommon.py', secondary.B/'plan.json', secondary.SIDE/'corpus.json')},
            'build_facts_source_sha256': hashlib.sha256(inspect.getsource(original_analysis.build_facts).encode('utf-8')).hexdigest(),
            'automatic_human_labels_added': 0, 'images_copied': 0,
            'limitations': ['Workload inventory, not a promise of total review time or completion within budget.',
                           'Whole-image known verdicts can still have unresolved question facts requiring review.',
                           'Only completed batches enter this snapshot; no model answers or partial gold were read.',
                           'No new scientific thresholds, human labels, or changes to the primary analysis.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect-batch00', action='store_true')
    args = parser.parse_args()
    report = audit()
    if args.expect_batch00:
        actual = report['batches']['batch-00']
        expected = {'review_images': 56, 'review_scenes': 23, 'whole_image_unknown': 11,
                    'known_image_but_question_unknown': 45, 'n_question_facts': 1536,
                    'known_question_facts': 1300, 'unknown_question_facts': 236,
                    'unknown_fact_reason_counts': {'disputed_scope': 112, 'unresolved_verification': 124}}
        for key, value in expected.items():
            assert actual[key] == value, (key, actual[key], value)
        report['batch00_registered_check'] = {'passed': True, 'expected': expected}
    folder = S / ('workload-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(time.time_ns()))
    folder.mkdir(exist_ok=False)
    path = folder / 'results.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'output': str(path), 'sha256': secondary.sha(path),
                      'completed_batches': report['completed_batches'],
                      'combined': {key: value for key, value in report['combined'].items() if key != 'review'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
