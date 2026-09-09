"""Apply provenance-bound human observations to a new side report only."""
import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path

from analyze_independent_labels import joint_gain

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
INPUTS = {}


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def enumerated_bounds(terms, n):
    known = {}
    for t in terms:
        if t['label'] is not None:
            assert type(t['label']) is bool
            assert t['identity'] not in known or known[t['identity']] == t['label']
            known[t['identity']] = t['label']
    unknown = sorted({t['identity'] for t in terms} - set(known))
    values = []
    for bits in itertools.product((False, True), repeat=len(unknown)):
        labels = {**known, **dict(zip(unknown, bits))}
        values.append(sum((F(t['coefficient']) * int(labels[t['identity']]) for t in terms), F(0)) / n)
    return str(min(values)), str(max(values)), len(values)


def summarize(rows):
    result = {}
    for index in (1, 2, None):
        for trained in (False, True):
            subset = [r for r in rows if (index is None or r['round'] == index) and (not trained or r['entered_training'])]
            name = (f'round_{index}' if index else 'pooled_two_rounds') + ('_entered_training' if trained else '_all_pools')
            terms = [t for r in subset for t in r['terms']]
            bounds = joint_gain(terms, len(subset))
            low, high, count = enumerated_bounds(terms, len(subset))
            assert (low, high) == (bounds['lower_exact'], bounds['upper_exact'])
            points = [joint_gain(r['terms'], 1)['point'] for r in subset]
            result[name] = {'n_pools': len(subset), 'gain_completion_bounds': bounds,
                            'wins': sum(x is not None and x > 0 for x in points),
                            'losses': sum(x is not None and x < 0 for x in points),
                            'ties': sum(x == 0 for x in points),
                            'unknown_gain_pools': sum(x is None for x in points),
                            'independent_enumeration_assignments': count}
    return result


def overlay_rows(original, cases, labels):
    rows = copy.deepcopy(original)
    lookup = {c['review_id']: c for c in cases}
    used = set()
    for row in rows:
        for term, role in zip(row['terms'], ('selected', 'uniform_control')):
            detail = row['labels'][role]
            rid = detail['review_id']
            if rid not in lookup:
                continue
            case = lookup[rid]
            assert detail['label'] is None and detail['unknown_reason'] == 'pending_human'
            assert term['label'] is None
            assert term['identity'].endswith(':' + case['recorded_rgb_sha256'])
            assert case['spec_id'] == row['prompt_id']
            label = labels[case['review_key']]
            assert label is None or type(label) is bool
            term['label'] = detail['label'] = label
            detail['unknown_reason'] = 'awaiting_rubric_clarification' if label is None else None
            detail['human_review_key'] = case['review_key']
            used.add(case['review_key'])
        row['paired_gain'] = joint_gain(row['terms'], 1)
    assert used == {'B', 'C', 'D', 'E'}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjudications', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = SIDE / args.output
    assert output.parent == SIDE and not output.exists()
    ledger = read(SIDE / args.adjudications)
    cases = read(SIDE / 'human-review-batch1.json')
    initial = read(SIDE / 'paired-gain-initial.json')
    for filename, digest in initial['input_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
        INPUTS[filename] = digest
    by_key = {c['review_key']: c for c in cases}
    assert set(by_key) == set(ledger['cases']) == set('ABCDE')
    for key, case in by_key.items():
        entry = ledger['cases'][key]
        assert entry['review_id'] == case['review_id']
        assert entry['image_path'] == case['image_path']
        assert entry['spec'] == case['spec']
        raw = Path(case['image_path']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == entry['image_file_sha256'] == case['image_file_sha256']
        INPUTS[case['image_path']] = digest
    labels = {k: e['spec_correct_label'] for k, e in ledger['cases'].items()}
    primary_rows = overlay_rows(initial['per_pool'], cases, labels)
    # This scenario records the assistant's interpretation of "fusion" as
    # incorrect, separately from explicitly confirmed original-rubric labels.
    fusion_labels = {**labels, 'C': False, 'D': False}
    fusion_rows = overlay_rows(initial['per_pool'], cases, fusion_labels)

    round0 = read(ROOT / 'review-packets/selector-gain-20260906/round000.json')
    priority = read(SIDE / 'priority-review.json')[0]
    assert labels['A'] is True
    terms_by_scope = {'all_prompts': [], 'trained_prompts': []}
    n_by_scope = {'all_prompts': 0, 'trained_prompts': 0}
    applied_a = 0
    for pool in round0['pools']:
        terms = []
        for c in pool['candidates']:
            label = c['label']
            if c['naive_image_path'] == by_key['A']['image_path']:
                assert label is None and c['naive_file_sha256'] == by_key['A']['image_file_sha256']
                assert c['candidate_id'] == pool['selected_candidate_id']
                label = True
                applied_a += 1
            coefficient = F(c['candidate_id'] == pool['selected_candidate_id']) - F(1, pool['n_candidates'])
            terms.append({'identity': pool['canonical_scene_sha256'] + ':' + c['rgb_sha256'],
                          'coefficient': coefficient, 'label': label})
        for scope in ('all_prompts', 'trained_prompts'):
            if scope == 'trained_prompts' and not pool['entered_training']:
                continue
            terms_by_scope[scope].extend(terms)
            n_by_scope[scope] += 1
    assert applied_a == 1
    first_round = {}
    for scope, terms in terms_by_scope.items():
        bounds = joint_gain(terms, n_by_scope[scope])
        low, high, count = enumerated_bounds(terms, n_by_scope[scope])
        expected = priority['conditional_gain_bounds']['True'][scope]
        assert low == bounds['lower_exact'] == expected['low']['exact']
        assert high == bounds['upper_exact'] == expected['high']['exact']
        assert count == (64 if scope == 'all_prompts' else 32)
        first_round[scope] = {'n_pools': n_by_scope[scope], 'gain_completion_bounds': bounds,
                              'independent_enumeration_assignments': count}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'human_adjudication_source': args.adjudications,
              'primary_confirmed_original_rubric': summarize(primary_rows),
              'scenario_C_D_fusion_counted_wrong': summarize(fusion_rows),
              'first_round_after_A_correct': first_round,
              'primary_per_pool': primary_rows,
              'remaining_original_rubric_clarifications': [k for k, v in labels.items() if v is None],
              'limitations': ['Unknown-label completion ranges are not confidence intervals.',
                              'Rounds 1 and 2 compare one frozen sampled control per pool; round 0 uses the full-pool expectation.',
                              'Different rounds use different prompts; these values are not a paired training-time trend.',
                              'Nesting alone is not excluded by the original object/color/count rubric. C/D defect observations do not by themselves settle that rubric.',
                              'No detector objects, historical verdicts, selection policy, thresholds or training state were changed.'],
              'input_sha256': INPUTS,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in INPUTS.items())
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('primary_confirmed_original_rubric', 'scenario_C_D_fusion_counted_wrong', 'first_round_after_A_correct', 'remaining_original_rubric_clarifications')}, indent=2))


if __name__ == '__main__':
    main()
