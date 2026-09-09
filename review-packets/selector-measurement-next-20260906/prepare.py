"""Freeze questions/facts/checkpoints and audit the 232 already seen pools."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import json
from pathlib import Path

from common import *
from selfsight.v4.factual_truth import load_verifications


def main():
    assert_main_paused()
    assert not (SIDE / 'plan.json').exists()
    source = ROOT / 'runs/v4/gate-b-openct2'
    inputs = {}
    def track(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
        return path
    pools = lines(track(source / 'pools.jsonl'))
    selected = {r['prompt_id']: r for r in lines(track(source / 'selection.jsonl'))}
    runs = read(track(source / 'runs.json'))['runs']
    specs, verifications = {}, {}
    for run in runs:
        for row in lines(track(ROOT / run / 'manifest.jsonl')):
            key = str((ROOT / row['image_path']).resolve())
            specs[key] = row['spec']
        for row in lines(track(ROOT / run / 'verified.jsonl')):
            verifications[str((ROOT / row['image_path']).resolve())] = row
    vocabulary = sorted({canonical_noun(o['object']) for spec in specs.values() for o in spec['objects']})
    assert len(pools) == len(selected) == 232
    tie_rows = {arm: [] for arm in ('naive', 'rfo')}
    ideal_old, ideal_new, eligible_ids = [], [], []
    coverage = Counter()
    prepared = {}
    for pool in pools:
        assert not selected[pool['prompt_id']]['dropped']
        spec = specs[str(Path(pool['candidates'][0]['image_path']).resolve())]
        assert spec['spec_id'] == pool['spec_id']
        old = pool['questions']
        n_positive = sum(q['family'] == 'existence' for q in old)
        extra = [as_serializable(q) for q in neg_questions(spec, vocabulary, n_positive)]
        assert all(q['expected_answer'] == 'yes' for q in old if q['family'] == 'existence')
        questions = old + extra
        expected = [q['expected_answer'] for q in questions]
        for arm in tie_rows:
            scores = selected[pool['prompt_id']]['scores'][arm]
            metrics = pool_metrics(pool['candidates'], scores)
            assert metrics['selected_candidate_id'] == selected[pool['prompt_id']]['selected'][arm]
            tie_rows[arm].append({'prompt_id': pool['prompt_id'], **metrics})
        candidates, old_scores, new_scores = [], {}, {}
        eligible = True
        for c in pool['candidates']:
            path = str(Path(c['image_path']).resolve())
            assert specs[path] == spec
            verdict = verifications[path]
            assert verdict['resolution'] != 'pending_human'
            assert type(verdict['image_correct']) is bool and verdict['image_correct'] == c['correct']
            facts = [factual_answer(q, verdict) for q in questions]
            for q, fact in zip(questions, facts):
                label = family(q)
                coverage[label + '_total'] += 1
                coverage[label + '_known'] += fact.known
                coverage[label + '_unknown_' + fact.reason] += not fact.known
                if label == 'negative_existence' and fact.known:
                    coverage['negative_known_present'] += fact.answer == 'yes'
            if all(f.known for f in facts):
                old_hits = sum(f.answer == e for f, e in zip(facts[:len(old)], expected[:len(old)]))
                new_hits = sum(f.answer == e for f, e in zip(facts, expected))
                old_scores[c['candidate_id']] = F(old_hits, len(old))
                new_scores[c['candidate_id']] = F(new_hits, len(questions))
                coverage['wrong_images_fully_fact_known'] += not c['correct']
                coverage['wrong_images_old_perfect'] += not c['correct'] and old_hits == len(old)
                coverage['wrong_images_old_perfect_new_detects'] += not c['correct'] and old_hits == len(old) and new_hits < len(questions)
            else:
                eligible = False
            candidates.append({**c, 'facts': [{'known': f.known, 'answer': f.answer, 'reason': f.reason} for f in facts]})
        if eligible:
            eligible_ids.append(pool['prompt_id'])
            ideal_old.append(pool_metrics(candidates, old_scores))
            ideal_new.append(pool_metrics(candidates, new_scores))
        prepared[pool['prompt_id']] = {**pool, 'spec': spec, 'n_original_questions': len(old), 'questions': questions, 'candidates': candidates}
    ties = {}
    for arm, rows in tie_rows.items():
        ties[arm] = {'all_pools': aggregate(rows), 'tied_pools_only': aggregate([r for r in rows if r['n_top'] > 1]),
                     'by_k': {str(k): aggregate([r for r in rows if r['n_candidates'] == k]) for k in (4, 6)}, 'per_pool': rows}
    cpu = {'created_utc': datetime.now(timezone.utc).isoformat(), 'n_pools': 232, 'vocabulary': vocabulary,
           'ties': ties, 'coverage': dict(coverage), 'ideal_same_eligible_pools': {'n': len(eligible_ids), 'ids': eligible_ids,
           'old': aggregate(ideal_old), 'with_negatives': aggregate(ideal_new)},
           'limitations': ['Known-fact oracle columns are a same-subset coverage diagnostic, not expected observer accuracy.',
                           'The spec-only negative sampling rule was fixed before this report; no tuning to these labels.',
                           'Top-set accuracy is uniform tie-break expectation, not a ceiling on another tie-break.',
                           'Previously observed balanced pools are development data, not an independent validation population.']}
    save_new(SIDE / 'cpu-232-audit.json', cpu)

    bank = read(track(MAIN / 'probe-bank/bank.json'))
    audit = read(track(MAIN / 'audit-splits/scene_overlap.json'))
    excluded = {r['spec_id'] for r in audit['overlap']['actual_probe_bank']}
    scene = {sid: c['scene_sha256'] for c in audit['within_probe_bank_clusters'] for sid in c['spec_ids']}
    frozen = []
    for p in bank['pools']:
        dev = prepared[p['prompt_id']]
        assert p['questions'] == dev['questions'][:dev['n_original_questions']]
        assert [c['candidate_id'] for c in p['candidates']] == [c['candidate_id'] for c in dev['candidates']]
        for orig, cand in zip(p['candidates'], dev['candidates']):
            assert (orig['image_path'], orig['correct'], orig['sampling_seed']) == (cand['image_path'], cand['correct'], cand['sampling_seed'])
            cand['image_file_sha256'] = sha(track(cand['image_path']))
            cand['rgb_sha256'] = rgb_sha256(Path(cand['image_path']))
            assert cand['rgb_sha256'] == orig['rgb_sha256']
        frozen.append({**dev, 'scene_sha256': scene[p['spec_id']], 'primary_scene_disjoint': p['spec_id'] not in excluded})
    assert len(frozen) == len({p['scene_sha256'] for p in frozen}) == 16
    assert sum(p['primary_scene_disjoint'] for p in frozen) == 14
    save_new(SIDE / 'fixed-bank.json', {'original_bank_fingerprint': bank['fingerprint'], 'vocabulary': vocabulary, 'pools': frozen})
    task_specs = [('base', 0), ('naive', 16), ('naive', 32), ('rfo_gold', 16), ('rfo_gold', 32)]
    tasks = []
    for arm, step in task_specs:
        folder = MAIN / f'gradient-probes/{arm}/step-{step:05d}'
        report = read(track(folder / 'report.json'))
        obs = track(folder / 'observations.naive.jsonl')
        assert report['checkpoint']['arm'] == arm and report['checkpoint']['step'] == step
        tasks.append({'id': f'{arm}-{step:05d}', 'arm': arm, 'step': step, 'checkpoint': report['checkpoint'],
                      'adapter_parameter_digest': report['adapter_parameter_digest'], 'old_observations': str(obs),
                      'device': 'cuda:1' if arm == 'rfo_gold' else 'cuda:0'})
    for filename in ('PROTOCOL.md', 'common.py', 'prepare.py', 'worker.py', 'launch.py', 'fixed-bank.json'):
        track(SIDE / filename)
    for filename in ('configs/v4_decoupling_pilot.yaml', 'configs/backbones/showo2_1p5b.yaml', 'configs/models.lock.yaml',
                     'runs/readiness/showo2-1p5b/a4-lora-targets-r1.json', 'src/selfsight/backbones/showo2.py',
                     'src/selfsight/training/checkpoint.py', 'src/selfsight/v4/train.py', 'src/selfsight/data/questions.py',
                     'src/selfsight/v4/factual_truth.py', 'src/selfsight/v4/spec.py', 'src/selfsight/v4/observe.py'):
        track(ROOT / filename)
    assert all(sha(p) == h for p, h in inputs.items())
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'primary_arm': 'naive', 'control_arm': 'rfo_gold',
            'conditions': ['prompt_on', 'prompt_off'], 'tasks': tasks,
            'n_pools': 16, 'n_images': sum(len(p['candidates']) for p in frozen),
            'original_answers_per_condition_checkpoint': sum(len(p['candidates']) * p['n_original_questions'] for p in frozen),
            'negative_answers_per_condition_checkpoint': sum(len(p['candidates']) * (len(p['questions']) - p['n_original_questions']) for p in frozen),
            'input_sha256': inputs, 'deadline': read(ROOT / 'review-packets/selection-benefit-side-20260906/hold-step32/expected.json')['deadline'],
            'config_path': str(ROOT / 'configs/v4_decoupling_pilot.yaml'),
            'targets_path': str(ROOT / 'runs/readiness/showo2-1p5b/a4-lora-targets-r1.json'),
            'research_goal_complete': False}
    save_new(SIDE / 'plan.json', plan)
    print(json.dumps({'ties': {a: {k:v for k,v in x.items() if k!='per_pool'} for a,x in ties.items()},
                      'coverage': dict(coverage), 'ideal': cpu['ideal_same_eligible_pools'],
                      'plan_counts': {k:plan[k] for k in ('n_pools','n_images','original_answers_per_condition_checkpoint','negative_answers_per_condition_checkpoint')}}, indent=2))


if __name__ == '__main__':
    main()
