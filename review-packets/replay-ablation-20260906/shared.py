"""Isolated replay-ablation helpers; the original experiment remains frozen."""
from pathlib import Path
import sys, json, hashlib
from datetime import datetime, timezone
SIDE=Path(__file__).resolve().parent
ROOT=SIDE.parents[1]
OLD=ROOT/'review-packets/selector-measurement-next-20260906'
MAIN=ROOT/'runs/v4/decoupling-pilot-20260906'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(OLD))
from common import read, lines, sha, save_new, assert_main_paused, pool_metrics, aggregate
from selfsight.schemas import AtomicQuestion, as_serializable
from selfsight.utils.hashing import sha256_json

def now(): return datetime.now(timezone.utc).isoformat()

def stable_order(rows, salt, key):
    return sorted(rows,key=lambda r:sha256_json([salt,key(r)]))

def verify_inputs(plan):
    assert_main_paused()
    for path, digest in plan['input_sha256'].items():
        assert sha(path)==digest,path

def answer_group(question, truth):
    return ('existence_' + truth) if question['family']=='existence' else ('count_zero' if truth=='0' else 'count_positive')

def summarize_probe(records):
    from collections import Counter
    out={}
    for condition in ('prompt_on','prompt_off'):
        rows=[r for r in records if r['condition']==condition]
        groups={}
        for g in ('existence_yes','existence_no','count_positive','count_zero'):
            subset=[r for r in rows if r['group']==g]
            groups[g]={'n':len(subset),'accuracy':sum(r['answer']['normalized_answer']==r['truth'] and not r['answer']['abstain'] and not r['answer'].get('error') for r in subset)/len(subset),
                       'outputs':dict(Counter(r['answer']['normalized_answer'] for r in subset))}
        out[condition]=groups
    return out
