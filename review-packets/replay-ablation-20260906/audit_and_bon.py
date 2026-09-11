"""Independent CPU audit and exact finite-pool BoN, without replacement."""
from pathlib import Path
import sys, json, itertools
from collections import Counter
from fractions import Fraction
SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
OLD = ROOT / 'review-packets/selector-measurement-next-20260906'
sys.path.insert(0, str(OLD))
from common import read, lines, save_new, sha, pool_metrics


def answers(task):
    records = lines(OLD/'observations'/task/'answers.jsonl')
    records += lines(OLD/'matched-frame'/task/'answers.jsonl')
    return {(r['condition'], r['prompt_id'], r['candidate_id'], a['question_id']): a
            for r in records for a in r['observation']['answers']}


def exact_bon(candidates, scores, n):
    """Average over all unordered subsets, uniform among each subset's top ties."""
    if not 1 <= n <= len(candidates):
        raise ValueError('N cannot exceed the number of distinct candidates')
    sums = Counter(); count = 0
    for subset in itertools.combinations(candidates, n):
        top = max(scores[c['candidate_id']] for c in subset)
        ties = [c for c in subset if scores[c['candidate_id']] == top]
        sums['uniform_top'] += Fraction(sum(c['correct'] for c in ties), len(ties))
        picked = max(ties, key=lambda c: (-c['sampling_seed'], c['candidate_id']))
        sums['deterministic'] += int(picked['correct'])
        sums['oracle_any_correct'] += any(c['correct'] for c in subset)
        sums['top_size'] += len(ties)
        count += 1
    return {k: float(v/count) for k, v in sums.items()} | {'subsets': count}


def main():
    bank = read(OLD/'fixed-bank.json'); previous = read(OLD/'results.json')
    tasks = list(previous['per_pool']); all_answers = {t: answers(t) for t in tasks}
    distributions = {}; concordance = {}
    for task, rows in all_answers.items():
        counter = {}
        for (condition, pid, cid, qid), a in rows.items():
            family = 'negative_existence' if ':negative-exists-v1:' in qid else ('count' if ':count:' in qid else 'original_existence')
            counter.setdefault(condition, {}).setdefault(family, Counter())[a['normalized_answer']] += 1
        distributions[task] = counter
    for step in (16,32):
        a=all_answers[f'naive-{step:05d}']; b=all_answers[f'rfo_gold-{step:05d}']
        assert a.keys()==b.keys()
        concordance[str(step)]={'n':len(a), 'same_normalized':sum(a[k]['normalized_answer']==b[k]['normalized_answer'] for k in a),
            'same_raw':sum(a[k]['raw_answer']==b[k]['raw_answer'] for k in a)}
    changed = {}; top_trajectories = {}
    for task, conditions in previous['per_pool'].items():
        changed[task] = {cond:sum(r['original']['selected_candidate_id']!=r['expanded']['selected_candidate_id'] for r in rows)
                         for cond, rows in conditions.items()}
        top_trajectories[task] = {cond:previous['summary']['full_16'][task][cond]['selection']['original'] for cond in conditions}
    available = all_answers['base-00000']; bon=[]
    for pool in bank['pools']:
        qs=pool['questions'][:pool['n_original_questions']]
        scores={c['candidate_id']: Fraction(sum(available['prompt_off',pool['prompt_id'],c['candidate_id'],q['question_id']]['normalized_answer']==q['expected_answer'] for q in qs),len(qs)) for c in pool['candidates']}
        bon.append({'prompt_id':pool['prompt_id'],'primary':pool['primary_scene_disjoint'], 'k':len(pool['candidates']),
                    'baseline':sum(c['correct'] for c in pool['candidates'])/len(pool['candidates']),
                    'by_n':{str(n):exact_bon(pool['candidates'],scores,n) for n in (1,2,4) }})
    summary={}
    for cohort in ('full_16','primary_14'):
        rows=[r for r in bon if cohort=='full_16' or r['primary']]
        summary[cohort]={'n_pools':len(rows),'baseline':sum(r['baseline'] for r in rows)/len(rows),
                        'by_n':{str(n):{key:sum(r['by_n'][str(n)][key] for r in rows)/len(rows) for key in ('uniform_top','deterministic','oracle_any_correct','top_size')} for n in (1,2,4)}}
    source=ROOT/'runs/v4/gate-b-openct2'
    inventory={'pools':len(lines(source/'pools.jsonl')),
       'k_distribution':dict(Counter(len(p['candidates']) for p in lines(source/'pools.jsonl'))),
       'matching_base_prompt_off_pools':len(bank['pools']),
       'matching_base_prompt_off_images':sum(len(p['candidates']) for p in bank['pools']),
       'missing_matching_images':1116-sum(len(p['candidates']) for p in bank['pools']),
       'old_naive_condition':'prompt_on', 'old_rfo_observer':'Qwen2-VL, not the base Show-o2 selector',
       'n8_without_replacement':'unavailable: no pool has eight distinct candidates'}
    result={'previous_results_sha256':sha(OLD/'results.json'),'distributions':distributions,
        'cross_arm_concordance':concordance,'expanded_score_changed_choices':changed,
        'full16_top_metrics':top_trajectories,'bon_data_inventory':inventory,
        'exact_development_bon':summary,'bon_per_pool':bon,
        'limitations':['Previously seen balanced 16-pool bank is descriptive; subset enumeration does not create new independent samples.',
            'Uniform-top expectation is not a universal selection upper bound.',
            'N=8 and 232-pool base/prompt_off BoN cannot be inferred from these response files.']}
    save_new(SIDE/'audit-and-bon.json',result)
    print(json.dumps({'cross_arm':concordance,'changed_choices':changed,'bon':summary,'inventory':inventory},indent=2))


if __name__=='__main__': main()
