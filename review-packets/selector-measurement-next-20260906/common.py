"""Helpers for a separately versioned diagnostic; never alter the live selector."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import random
import sys

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
MAIN = ROOT / 'runs/v4/decoupling-pilot-20260906'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from selfsight.schemas import AtomicQuestion, QuestionFamily, QuestionFormat, as_serializable
from selfsight.v4.spec import canonical_noun, SceneSpec
from selfsight.v4.questions import _place
from selfsight.v4.factual_truth import factual_answer
from selfsight.utils.hashing import sha256_json, rgb_sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(z) for z in Path(path).read_text(encoding='utf-8').splitlines() if z.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def neg_questions(spec, vocabulary, n):
    present = {canonical_noun(o['object']) for o in spec['objects']}
    eligible = sorted(set(map(canonical_noun, vocabulary)) - present)
    if len(eligible) < n:
        raise ValueError('Not enough unique absent categories')
    rng = random.Random('negative-existence-v1:' + spec['spec_id'])
    picked = rng.sample(eligible, n)
    out = []
    for noun in picked:
        placement = random.Random('negative-placement-v1:' + spec['spec_id'] + ':' + noun)
        a, b, gold = _place(placement, 'no', 'yes')
        article = 'an' if noun[0] in 'aeiou' else 'a'
        out.append(AtomicQuestion(
            question_id=f"{spec['spec_id']}:negative-exists-v1:{noun}",
            atom_id=f"{spec['spec_id']}:negative-exists-v1:{noun}",
            family=QuestionFamily.EXISTENCE,
            text=f'Is there {article} {noun} in this picture? Answer A or B only.\nA. {a}\nB. {b}',
            expected_answer='no', question_format=QuestionFormat.FORCED_CHOICE,
            choices=(a, b), choice_order_seed=ord(gold)))
    return out


def family(question):
    if ':negative-exists-v1:' in question['question_id']:
        return 'negative_existence'
    return question['family']


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def pool_metrics(candidates, scores):
    top_score = max(scores.values())
    top = [c for c in candidates if scores[c['candidate_id']] == top_score]
    selected = max(candidates, key=lambda c: (scores[c['candidate_id']], -c['sampling_seed'], c['candidate_id']))
    baseline = F(sum(c['correct'] for c in candidates), len(candidates))
    top_rate = F(sum(c['correct'] for c in top), len(top))
    good = [scores[c['candidate_id']] for c in candidates if c['correct']]
    bad = [scores[c['candidate_id']] for c in candidates if not c['correct']]
    gap = sum(good) / len(good) - sum(bad) / len(bad) if good and bad else None
    return {'n_candidates': len(candidates), 'n_top': len(top),
            'n_top_correct': sum(c['correct'] for c in top),
            'top_correct_rate': float(top_rate), 'baseline': float(baseline),
            'top_uniform_gain': float(top_rate - baseline),
            'selected_candidate_id': selected['candidate_id'],
            'selected_correct': bool(selected['correct']),
            'actual_gain': float(F(selected['correct']) - baseline),
            'actual_gain_exact': str(F(selected['correct']) - baseline),
            'all_tied': len(top) == len(candidates),
            'ceiling_share': sum(s == 1 for s in scores.values()) / len(scores),
            'correct_minus_wrong_mean_score': float(gap) if gap is not None else None}


def aggregate(rows):
    return {'n_pools': len(rows), 'n_with_top_ties': sum(r['n_top'] > 1 for r in rows),
            'n_top_without_correct': sum(r['n_top_correct'] == 0 for r in rows),
            'top_size_mean': mean([r['n_top'] for r in rows]),
            'top_correct_rate_pool_mean': mean([r['top_correct_rate'] for r in rows]),
            'baseline_pool_mean': mean([r['baseline'] for r in rows]),
            'top_uniform_gain': mean([r['top_uniform_gain'] for r in rows]),
            'actual_gain': mean([r['actual_gain'] for r in rows]),
            'actual_correct_rate': mean([int(r['selected_correct']) for r in rows]),
            'top_correct_rate_micro': (sum(r['n_top_correct'] for r in rows) / sum(r['n_top'] for r in rows)) if rows else None,
            'n_all_tied': sum(r['all_tied'] for r in rows),
            'n_correct_score_higher': sum(r['correct_minus_wrong_mean_score'] is not None and r['correct_minus_wrong_mean_score'] > 0 for r in rows),
            'n_correct_score_equal': sum(r['correct_minus_wrong_mean_score'] == 0 for r in rows),
            'n_correct_score_lower': sum(r['correct_minus_wrong_mean_score'] is not None and r['correct_minus_wrong_mean_score'] < 0 for r in rows)}


def assert_main_paused():
    state = read(MAIN / 'state.json')
    assert state['status'] == 'canary_complete'
    assert state['completed_rounds'] == state['through_round'] == 4
    assert not state['research_goal_complete'] and not (MAIN / 'rounds/round-004').exists()
