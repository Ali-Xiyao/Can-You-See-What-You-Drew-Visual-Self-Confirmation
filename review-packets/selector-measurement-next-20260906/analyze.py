"""Paired fixed-bank diagnostics; no change to the registered D* or D_g."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
import numpy as np

from common import *


def interval(values):
    values=[v for v in values if v is not None]
    if not values:
        return {'point':None,'ci_low':None,'ci_high':None,'n_scenes':0}
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20260906)
    indices=rng.integers(0,len(x),size=(5000,len(x)))
    draws=x[indices].mean(axis=1)
    return {'point':float(x.mean()),'ci_low':float(np.quantile(draws,.025)),
            'ci_high':float(np.quantile(draws,.975)), 'n_scenes':len(x),
            'scope':'Exploratory scene bootstrap conditional on this already-seen fixed bank; no training-seed uncertainty or multiplicity correction.'}


def factual_metrics(pool, observations):
    result={}
    for group in ('existence','count','negative_existence','original_all','expanded_all'):
        image_values=defaultdict(list)
        n_known=n_conflict=n_all=0
        status=Counter()
        for c in pool['candidates']:
            values=defaultdict(list)
            answers=observations[c['candidate_id']]['observation']['answers']
            assert len(answers)==len(pool['questions'])
            for q,f,a in zip(pool['questions'],c['facts'],answers):
                fam=family(q)
                if not (group=='expanded_all' or group=='original_all' and fam!='negative_existence' or group==fam):
                    continue
                assert a['question_id']==q['question_id']
                valid=not a.get('error') and not a['abstain'] and a['normalized_answer'] is not None
                response=a['normalized_answer'] if valid else None
                status['valid' if valid else 'invalid']+=1
                n_all+=1
                values['response_request_all'].append(int(valid and response==q['expected_answer']))
                if f['known']:
                    n_known+=1
                    values['response_fact'].append(int(valid and response==f['answer']))
                    values['response_request_known'].append(int(valid and response==q['expected_answer']))
                    values['fact_request'].append(int(f['answer']==q['expected_answer']))
                    if f['answer']!=q['expected_answer']:
                        n_conflict+=1
                        values['conflicting_fact_accuracy'].append(int(valid and response==f['answer']))
                    if fam in ('existence','negative_existence'):
                        values['yes_rate_known'].append(int(valid and response=='yes'))
            for metric,x in values.items():
                if x: image_values[metric].append(float(F(sum(x),len(x))))
        result[group]={'n_requested_atoms':n_all,'n_known_atoms':n_known,'n_conflicting_atoms':n_conflict,
                       'response_status':dict(status),
                       **{k:mean(image_values[k]) for k in ('response_request_all','response_fact','response_request_known','fact_request','conflicting_fact_accuracy','yes_rate_known')}}
    return result


def score_pool(pool, observations, expanded):
    n=len(pool['questions']) if expanded else pool['n_original_questions']
    scores={}
    for c in pool['candidates']:
        answers=observations[c['candidate_id']]['observation']['answers'][:n]
        assert len(answers)==n
        hits=sum(not a.get('error') and not a['abstain'] and a['normalized_answer']==q['expected_answer'] for a,q in zip(answers,pool['questions'][:n]))
        scores[c['candidate_id']]=F(hits,n)
    return pool_metrics(pool['candidates'],scores)


def self_test():
    p={'n_original_questions':1,'questions':[{'question_id':'q','family':'count','expected_answer':'1'}],
       'candidates':[{'candidate_id':'a','sampling_seed':1,'correct':True,'facts':[{'known':True,'answer':'1'}]},
                     {'candidate_id':'b','sampling_seed':2,'correct':False,'facts':[{'known':True,'answer':'2'}]}]}
    obs={c['candidate_id']:{'observation':{'answers':[{'question_id':'q','normalized_answer':'1','abstain':False,'error':None}]}} for c in p['candidates']}
    f=factual_metrics(p,obs)['count']
    assert f['response_request_all']==1 and f['response_fact']==.5 and f['conflicting_fact_accuracy']==0
    assert score_pool(p,obs,False)['actual_gain']==.5
    obs['b']['observation']['answers'][0]['normalized_answer']='2'
    assert factual_metrics(p,obs)['count']['response_fact']==1
    p['candidates'][1]['facts'][0]['known']=False
    assert factual_metrics(p,obs)['count']['n_known_atoms']==1
    assert factual_metrics(p,obs)['count']['conflicting_fact_accuracy'] is None
    obs['a']['observation']['answers'][0].update(normalized_answer=None,abstain=True)
    assert factual_metrics(p,obs)['count']['response_fact']==0
    assert interval([])['point'] is None
    print('8 analysis checks passed: request/fact distinction, unknowns, abstentions, tie policy and empty intervals.')


def main():
    assert_main_paused()
    plan=read(SIDE/'plan.json')
    controller=read(SIDE/'controller-complete.json')
    assert controller['exit_code']==0
    assert read(SIDE/'frame-controller-complete.json')['exit_code']==0
    for path,h in read(SIDE/'frame-source-freeze.json').items(): assert sha(path)==h,path
    conditions_all=['prompt_on','prompt_off','prompt_blank']
    for path,h in plan['input_sha256'].items(): assert sha(path)==h,path
    bank=read(SIDE/'fixed-bank.json')
    inputs={str(SIDE/'plan.json'):sha(SIDE/'plan.json'), str(SIDE/'fixed-bank.json'):sha(SIDE/'fixed-bank.json')}
    per_pool={}; completions={}
    for task in plan['tasks']:
        folder=SIDE/'observations'/task['id']
        completion=read(folder/'complete.json')
        assert completion['answers_sha256']==sha(folder/'answers.jsonl')
        assert completion['adapter_parameter_digest_before']==completion['adapter_parameter_digest_after']==task['adapter_parameter_digest']
        assert completion['plan_sha256']==sha(SIDE/'plan.json')
        rows=lines(folder/'answers.jsonl')
        assert len(rows)==completion['n_observations']==160
        inputs[str(folder/'complete.json')]=sha(folder/'complete.json')
        inputs[str(folder/'answers.jsonl')]=sha(folder/'answers.jsonl')
        extra_folder=SIDE/'matched-frame'/task['id']
        extra_complete=read(extra_folder/'complete.json')
        assert extra_complete['answers_sha256']==sha(extra_folder/'answers.jsonl')
        assert extra_complete['adapter_parameter_digest_before']==extra_complete['adapter_parameter_digest_after']==task['adapter_parameter_digest']
        assert extra_complete['plan_sha256']==sha(SIDE/'plan.json')
        extra_rows=lines(extra_folder/'answers.jsonl')
        assert len(extra_rows)==extra_complete['n_observations']==80
        assert all(r['condition']=='prompt_blank' for r in extra_rows)
        rows+=extra_rows
        lookup={(r['condition'],r['prompt_id'],r['candidate_id']):r for r in rows}
        assert len(lookup)==len(rows)==len({r['measurement_id'] for r in rows})==240
        inputs[str(extra_folder/'complete.json')]=sha(extra_folder/'complete.json')
        inputs[str(extra_folder/'answers.jsonl')]=sha(extra_folder/'answers.jsonl')
        completions[task['id']]={'standard':completion,'matched_frame':extra_complete}
        per_pool[task['id']]={}
        for condition in conditions_all:
            result=[]
            for p in bank['pools']:
                obs={c['candidate_id']:lookup[condition,p['prompt_id'],c['candidate_id']] for c in p['candidates']}
                for c in p['candidates']:
                    r=obs[c['candidate_id']]
                    assert r['image_rgb_sha256']==c['rgb_sha256']==r['observation']['rgb_sha256']
                    assert r['checkpoint_adapter_sha256']==task['checkpoint']['adapter_sha256']
                result.append({'prompt_id':p['prompt_id'],'scene_sha256':p['scene_sha256'],
                               'primary_scene_disjoint':p['primary_scene_disjoint'],
                               'original':score_pool(p,obs,False),'expanded':score_pool(p,obs,True),
                               'factual':factual_metrics(p,obs)})
            per_pool[task['id']][condition]=result
    summary={}; contrasts={}
    for cohort in ('primary_14','full_16'):
        keep=lambda r:cohort=='full_16' or r['primary_scene_disjoint']
        summary[cohort]={};contrasts[cohort]={}
        for task_id,conditions in per_pool.items():
            summary[cohort][task_id]={}
            for condition,rows in conditions.items():
                rows=[r for r in rows if keep(r)]
                scores={}
                for variant in ('original','expanded'):
                    scores[variant]={**aggregate([r[variant] for r in rows]),
                                     'actual_gain_interval':interval([r[variant]['actual_gain'] for r in rows]),
                                     'top_uniform_gain_interval':interval([r[variant]['top_uniform_gain'] for r in rows])}
                factual={g:{'n_known_atoms':sum(r['factual'][g]['n_known_atoms'] for r in rows),
                            'n_conflicting_atoms':sum(r['factual'][g]['n_conflicting_atoms'] for r in rows),
                            **{m:interval([r['factual'][g][m] for r in rows]) for m in ('response_request_all','response_fact','response_request_known','fact_request','conflicting_fact_accuracy','yes_rate_known')}}
                         for g in ('existence','count','negative_existence','original_all','expanded_all')}
                summary[cohort][task_id][condition]={'selection':scores,'factual':factual}
            on=[r for r in conditions['prompt_on'] if keep(r)]
            contrasts[cohort][task_id]={}
            for control_condition in ('prompt_blank','prompt_off'):
                off=[r for r in conditions[control_condition] if keep(r)]
                assert [r['prompt_id'] for r in on]==[r['prompt_id'] for r in off]
                diffs={}
                for variant in ('original','expanded'):
                    diffs[variant+'_actual_gain']=interval([b[variant]['actual_gain']-a[variant]['actual_gain'] for a,b in zip(on,off)])
                for g in ('existence','count','negative_existence','original_all','expanded_all'):
                    for metric in ('response_fact','conflicting_fact_accuracy','response_request_all'):
                        x=[(b['factual'][g][metric]-a['factual'][g][metric]) if b['factual'][g][metric] is not None and a['factual'][g][metric] is not None else None for a,b in zip(on,off)]
                        diffs[g+'_'+metric]=interval(x)
                contrasts[cohort][task_id][control_condition+'_minus_on']=diffs
        for arm in ('naive','rfo_gold'):
            for step in (16,32):
                task_id=f'{arm}-{step:05d}'
                base_on=[r for r in per_pool['base-00000']['prompt_on'] if keep(r)]
                now_on=[r for r in per_pool[task_id]['prompt_on'] if keep(r)]
                for control_condition in ('prompt_blank','prompt_off'):
                    base_off=[r for r in per_pool['base-00000'][control_condition] if keep(r)]
                    now_off=[r for r in per_pool[task_id][control_condition] if keep(r)]
                    interactions=[]
                    for a,b,c,d in zip(base_on,base_off,now_on,now_off):
                        vals=[r['factual']['original_all']['conflicting_fact_accuracy'] for r in (a,b,c,d)]
                        interactions.append((vals[3]-vals[2])-(vals[1]-vals[0]) if all(v is not None for v in vals) else None)
                    contrasts[cohort][task_id][control_condition+'_effect_change_from_base_conflicting_original_fact_accuracy']=interval(interactions)
                for condition in conditions_all:
                    old=[r for r in per_pool['base-00000'][condition] if keep(r)]
                    new=[r for r in per_pool[task_id][condition] if keep(r)]
                    contrasts[cohort][task_id][condition+'_original_selection_change_from_base']=interval([c['original']['actual_gain']-a['original']['actual_gain'] for a,c in zip(old,new)])
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'summary':summary,'contrasts':contrasts,'per_pool':per_pool,
            'completion_audits':completions,'input_sha256':inputs,'script_sha256':sha(__file__),
            'limitations':['Development diagnosis on an already-seen balanced bank; not a new population validation.',
                           'Factual summaries weight images equally within each pool, then pools equally; do not conflate with historical image-only weighting.',
                           'prompt_blank removes description content with framing fixed; prompt_off removes the entire generation/self-drawing context. Spec-derived questions remain in all conditions.',
                           'Expanded score is diagnostic only; historical training used the original score.',
                           'Fixed pixels test observation/selection drift, not generative D*. No gradient signal evaluated.']}
    assert all(sha(p)==h for p,h in inputs.items())
    save_new(SIDE/'results.json',result)
    print(json.dumps({'primary_selection':{k:{c:v['selection'] for c,v in conditions.items()} for k,conditions in summary['primary_14'].items()},'primary_contrasts':contrasts['primary_14']},indent=2))


if __name__=='__main__':
    import sys
    self_test() if '--self-test' in sys.argv else main()
