"""Frozen paired endpoints, reproduction checks and non-promotional decision rules."""
import argparse
from collections import defaultdict
import numpy as np
from shared import *
from analyze import score_pool, factual_metrics


def ci(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(20260906)
    draws=x[rng.integers(len(x),size=(5000,len(x)))].mean(axis=1)
    return {'point':float(x.mean()),'n_scenes':len(x),'ci95':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
            'ci90':[float(np.quantile(draws,.05)),float(np.quantile(draws,.95))]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    plan=read(SIDE/'plan.json');bank=read(OLD/'fixed-bank.json');summary={};per_pool={};probes={};reproduction={}
    from audit_and_bon import answers as old_answers
    for task in plan['tasks']:
        folder=SIDE/'runs'/task['id']
        if not folder.exists():continue
        if not args.partial:assert (folder/'complete.json').exists()
        steps=[s for s in (0,16,32) if (folder/'measurements'/f'step-{s:05d}-complete.json').exists()]
        summary[task['id']]={};per_pool[task['id']]={};reproduction[task['id']]={}
        probes[task['id']]=[{'step':r['step'],'summary':r['summary']} for p in sorted((folder/'probes').glob('step-*.json')) for r in [read(p)]]
        for step in steps:
            path=folder/'measurements'/f'step-{step:05d}.jsonl';done=read(path.with_name(path.stem+'-complete.json'))
            assert sha(path)==done['answers_sha256'];rows=lines(path)
            lookup={(r['condition'],r['prompt_id'],r['candidate_id']):r for r in rows};assert len(lookup)==len(rows)==240
            per_pool[task['id']][str(step)]={};summary[task['id']][str(step)]={}
            for cond in ('prompt_on','prompt_blank','prompt_off'):
                pools=[]
                for p in bank['pools']:
                    obs={c['candidate_id']:lookup[cond,p['prompt_id'],c['candidate_id']] for c in p['candidates']}
                    pools.append({'prompt_id':p['prompt_id'],'primary_scene_disjoint':p['primary_scene_disjoint'],
                                  'original':score_pool(p,obs,False),'expanded':score_pool(p,obs,True),'factual':factual_metrics(p,obs)})
                per_pool[task['id']][str(step)][cond]=pools
                summary[task['id']][str(step)][cond]={cohort:{'original':aggregate([r['original'] for r in pools if cohort=='full_16' or r['primary_scene_disjoint']]),
                    'expanded':aggregate([r['expanded'] for r in pools if cohort=='full_16' or r['primary_scene_disjoint']]),
                    'original_spec_match':float(np.mean([r['factual']['original_all']['response_request_all'] for r in pools if cohort=='full_16' or r['primary_scene_disjoint']]))}
                    for cohort in ('primary_14','full_16')}
            if step==0 or task['variant']=='original':
                old=old_answers('base-00000' if step==0 else f"{task['arm']}-{step:05d}")
                new={(r['condition'],r['prompt_id'],r['candidate_id'],a['question_id']):a for r in rows for a in r['observation']['answers']}
                assert old.keys()==new.keys()
                reproduction[task['id']][str(step)]={'n':len(old),'same_normalized':sum(old[k]['normalized_answer']==new[k]['normalized_answer'] for k in old)}
        if 0 in steps and 32 in steps:
            changes={}
            for cond in ('prompt_on','prompt_blank','prompt_off'):
                before=[r for r in per_pool[task['id']]['0'][cond] if r['primary_scene_disjoint']]
                after=[r for r in per_pool[task['id']]['32'][cond] if r['primary_scene_disjoint']]
                changes[cond]=ci([b['original']['top_correct_rate']-a['original']['top_correct_rate'] for a,b in zip(before,after)])
            summary[task['id']]['top_rate_change_32_minus_base']=changes
            pp={r['step']:r['summary'] for r in probes[task['id']]}
            repair={cond:all(pp[32][cond][group]['accuracy']>=pp[0][cond][group]['accuracy']-.1-1e-12 for group in pp[0][cond]) for cond in pp[0]}
            change=changes['prompt_blank'];repaired=all(repair.values())
            verdict='inconclusive_or_repair_failed'
            if repaired:
                if change['ci95'][1]<-.05:verdict='residual_decline_on_development_bank'
                elif change['ci90'][0]>=-.05 and change['ci90'][1]<=.05:verdict='development_equivalence_within_5pp'
                elif change['point']<0:verdict='negative_point_estimate_uncertain_or_attenuated'
            summary[task['id']]['provisional_interpretation']={'absence_and_positive_retention':repair,'classification':verdict,
                  'requires_original_reproduction_check':True,'independent_validation':False}
    result={'created_utc':now(),'partial':args.partial,'summary':summary,'per_pool':per_pool,'probes':probes,'reproduction':reproduction,
            'plan_sha256':sha(SIDE/'plan.json'),'limitations':'Single-seed, previously viewed fixed scene bank; no generative D* or D_g conclusion.'}
    if args.partial:print(json.dumps({k:result[k] for k in ('created_utc','summary','reproduction')},indent=2))
    else:save_new(SIDE/'results.json',result);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
