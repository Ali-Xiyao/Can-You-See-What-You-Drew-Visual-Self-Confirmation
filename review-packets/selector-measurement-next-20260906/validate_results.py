"""Recompute selection and prompt-content contrasts directly from saved raw answers."""
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction as F
from common import *
from selfsight.data.questions import normalize_answer
from selfsight.v4.observe import PROMPTED_PREAMBLE


def main():
    r=read(SIDE/'results.json')
    plan=read(SIDE/'plan.json')
    frozen_bank=read(SIDE/'fixed-bank.json')
    pools=frozen_bank['pools']
    plan_hash=sha(SIDE/'plan.json')
    condition_names=('prompt_on','prompt_blank','prompt_off')
    metrics={}; integrity=Counter(); repro={}
    for task in plan['tasks']:
        task_rows=[]
        for folder in (SIDE/'observations'/task['id'],SIDE/'matched-frame'/task['id']):
            done=read(folder/'complete.json')
            assert done['plan_sha256']==plan_hash
            assert done['answers_sha256']==sha(folder/'answers.jsonl')
            assert done['adapter_parameter_digest_before']==done['adapter_parameter_digest_after']==task['adapter_parameter_digest']
            task_rows+=lines(folder/'answers.jsonl')
        lookup={(x['condition'],x['prompt_id'],x['candidate_id']):x for x in task_rows}
        assert len(lookup)==len(task_rows)==240
        metrics[task['id']]={}
        for condition in condition_names:
            by_pool={}
            for p in pools:
                base_q=[AtomicQuestion.from_dict(q) for q in p['questions']]
                if condition=='prompt_off':
                    qs=base_q
                else:
                    text=p['prompt'] if condition=='prompt_on' else ''
                    qs=[replace(q,text=PROMPTED_PREAMBLE.format(prompt=text,question=q.text)) for q in base_q]
                scores={'original':{},'expanded':{}}
                image_conflict=[]
                for c in p['candidates']:
                    row=lookup[condition,p['prompt_id'],c['candidate_id']]
                    assert row['question_sha256']==sha256_json([as_serializable(q) for q in qs])
                    assert row['observation']['rgb_sha256']==row['image_rgb_sha256']==c['rgb_sha256']
                    assert row['seed']==int(sha256_json([frozen_bank['original_bank_fingerprint'],p['prompt_id'],c['candidate_id'],'observe'])[:8],16)
                    request={k:v for k,v in row.items() if k not in ('measurement_id','observation')}
                    assert row['measurement_id']==sha256_json(request)
                    ans=row['observation']['answers']
                    assert len(ans)==len(qs)
                    hits=[];conflicting=[]
                    for i,(q,a,fact) in enumerate(zip(qs,ans,c['facts'])):
                        assert q.question_id==a['question_id']
                        if not a.get('error'):
                            parsed=normalize_answer(a['raw_answer'],q)
                            assert parsed==a['normalized_answer'] and (parsed is None)==a['abstain']
                        good=not a.get('error') and not a['abstain'] and a['normalized_answer'] is not None
                        hits.append(int(bool(good and a['normalized_answer']==q.expected_answer)))
                        if i<p['n_original_questions'] and fact['known'] and fact['answer']!=q.expected_answer:
                            conflicting.append(int(bool(good and a['normalized_answer']==fact['answer'])))
                        integrity['answers_checked']+=1
                    if conflicting:image_conflict.append(F(sum(conflicting),len(conflicting)))
                    scores['original'][c['candidate_id']]=F(sum(hits[:p['n_original_questions']]),p['n_original_questions'])
                    scores['expanded'][c['candidate_id']]=F(sum(hits),len(hits))
                verified={}
                baseline=F(sum(c['correct'] for c in p['candidates']),len(p['candidates']))
                for variant,values in scores.items():
                    top=[c for c in p['candidates'] if values[c['candidate_id']]==max(values.values())]
                    pick=sorted(p['candidates'],key=lambda c:(values[c['candidate_id']],-c['sampling_seed'],c['candidate_id']))[-1]
                    saved=next(x for x in r['per_pool'][task['id']][condition] if x['prompt_id']==p['prompt_id'])[variant]
                    gain=F(pick['correct'])-baseline
                    assert saved['selected_candidate_id']==pick['candidate_id']
                    assert saved['actual_gain_exact']==str(gain)
                    assert saved['n_top']==len(top)
                    assert abs(saved['top_correct_rate']-float(F(sum(c['correct'] for c in top),len(top))))<1e-12
                    verified[variant+'_gain']=gain
                    integrity['pool_selections_checked']+=1
                verified['conflict']=sum(image_conflict)/len(image_conflict) if image_conflict else None
                by_pool[p['prompt_id']]=verified
            metrics[task['id']][condition]=by_pool
        done=read(SIDE/'observations'/task['id']/'complete.json')
        rr=done['old_prompted_answer_reproduction']
        repro[task['id']]={'matches':rr['matches'],'total':rr['total']}
    for cohort in ('primary_14','full_16'):
        ids=[p['prompt_id'] for p in pools if cohort=='full_16' or p['primary_scene_disjoint']]
        for task in plan['tasks']:
            tid=task['id']
            for control in ('prompt_blank','prompt_off'):
                changes=[metrics[tid][control][i]['conflict']-metrics[tid]['prompt_on'][i]['conflict'] for i in ids if metrics[tid][control][i]['conflict'] is not None]
                wanted=r['contrasts'][cohort][tid][control+'_minus_on']['original_all_conflicting_fact_accuracy']['point']
                got=float(sum(changes)/len(changes)) if changes else None
                assert wanted is None and got is None or wanted is not None and abs(wanted-got)<1e-12
                if tid!='base-00000':
                    interactions=[(metrics[tid][control][i]['conflict']-metrics[tid]['prompt_on'][i]['conflict'])-(metrics['base-00000'][control][i]['conflict']-metrics['base-00000']['prompt_on'][i]['conflict']) for i in ids if metrics[tid][control][i]['conflict'] is not None]
                    wanted=r['contrasts'][cohort][tid][control+'_effect_change_from_base_conflicting_original_fact_accuracy']['point']
                    got=float(sum(interactions)/len(interactions)) if interactions else None
                    assert wanted is None and got is None or wanted is not None and abs(wanted-got)<1e-12
                integrity['paired_contrasts_checked']+=1
    for path,h in plan['input_sha256'].items():assert sha(path)==h,path
    for path,h in r['input_sha256'].items():assert sha(path)==h,path
    assert_main_paused()
    out={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_integrity_and_arithmetic_only',
         'checks':dict(integrity),'historical_original_answer_reproduction':repro,
         'all_measurement_sources_unchanged':True,'report_sha256':sha(SIDE/'results.json'),
         'scope':'Independent Fraction scoring and contrasts; same registered factual labels. Does not establish the research claims or validate bootstrap coverage.'}
    save_new(SIDE/'results-validation.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
