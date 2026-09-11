"""Check saved answers independently and verify the actual training budget."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
from shared import *
from selfsight.data.questions import normalize_answer
from selfsight.v4.observe import PROMPTED_PREAMBLE
from selfsight.training.paired import _seed_from_parts


def main():
    plan=read(SIDE/'plan.json');verify_inputs(plan)
    result=read(SIDE/'results.json');schedule=read(SIDE/'schedule.json');probe=read(SIDE/'probe.json')
    bank=read(OLD/'fixed-bank.json');checks=Counter()
    for task in plan['tasks']:
        folder=SIDE/'runs'/task['id'];done=read(folder/'complete.json');started=read(folder/'started.json')
        assert done['training_sha256']==sha(folder/'training.jsonl')
        assert started['plan_sha256']==sha(SIDE/'plan.json')
        training=lines(folder/'training.jsonl');assert [r['step'] for r in training]==list(range(1,33))
        for rd in range(4):
            selected=schedule[str(rd)]['generation'][task['arm']];gc=rc=0
            replay_key={'original':'original','zero_weight':'original','balanced_absence':'balanced_absence','alternate_ids':'alternate_'+task['arm']}[task['variant']]
            replay=schedule[str(rd)][replay_key]
            for row in training[rd*8:rd*8+8]:
                step=row['step'];assert len(row['micro'])==8
                assert abs(row['learning_rate_used']-1e-4*min(1,step/8))<1e-12
                for m,entry in enumerate(row['micro']):
                    kind='replay' if m%4==0 else 'generation'
                    assert entry['kind']==kind
                    assert entry['latent_seed']==_seed_from_parts(20260906,rd,(step-1)%8,m,'replay' if kind=='replay' else 't2i')
                    if kind=='replay':
                        expected=replay[rc];rc+=1
                        assert entry['sample_id']==expected['sample_id'] and entry['answer']==expected['answer']
                        assert entry['weight']==(0 if task['variant']=='zero_weight' else 1)
                    else:
                        expected=selected[gc%len(selected)];gc+=1
                        assert entry['sample_id']==expected['candidate_id'] and entry['weight']==1
                    checks['training_microbatches']+=1
            assert rc==16 and gc==48
        for step in range(33):
            row=read(folder/'probes'/f'step-{step:05d}.json');assert row['rng_restored']
            lookup={(r['condition'],r['id']):r for r in row['answers']};assert len(lookup)==len(row['answers'])==200
            for qrow in probe['rows']:
                for cond in ('prompt_on','prompt_off'):
                    a=lookup[cond,qrow['id']]['answer'];q=AtomicQuestion.from_dict(qrow['question'])
                    if cond=='prompt_on':q=replace(q,text=PROMPTED_PREAMBLE.format(prompt=qrow['prompt'],question=q.text))
                    assert normalize_answer(a['raw_answer'],q)==a['normalized_answer']
                    assert not a['abstain'] and not a.get('error')
                    assert lookup[cond,qrow['id']]['truth']==qrow['truth'];checks['probe_answers']+=1
        for step in (0,16,32):
            path=folder/'measurements'/f'step-{step:05d}.jsonl';complete=read(path.with_name(path.stem+'-complete.json'))
            assert sha(path)==complete['answers_sha256']
            assert complete['parameter_digest_before']==complete['parameter_digest_after']
            lookup={(r['condition'],r['prompt_id'],r['candidate_id']):r for r in lines(path)};assert len(lookup)==240
            for cond in ('prompt_on','prompt_blank','prompt_off'):
                for pool in bank['pools']:
                    scores={'original':{},'expanded':{}}
                    for c in pool['candidates']:
                        record=lookup[cond,pool['prompt_id'],c['candidate_id']]['observation']
                        assert record['rgb_sha256']==c['rgb_sha256']
                        assert record['observer_id']=='showlab/show-o2-1.5B' and record['observer_revision']=='07ec16589d4fc5422a74dddbbc4b2cd11e551039'
                        ans=record['answers'];assert len(ans)==len(pool['questions']);hits=[]
                        for a,qd in zip(ans,pool['questions']):
                            q=AtomicQuestion.from_dict(qd)
                            if cond!='prompt_off':q=replace(q,text=PROMPTED_PREAMBLE.format(prompt=pool['prompt'] if cond=='prompt_on' else '',question=q.text))
                            assert a['question_id']==q.question_id
                            parsed=normalize_answer(a['raw_answer'],q);assert parsed==a['normalized_answer']
                            assert not a.get('error') and not a['abstain']
                            hits.append(int(parsed==q.expected_answer));checks['full_answers']+=1
                        for version,n in (('original',pool['n_original_questions']),('expanded',len(hits))):scores[version][c['candidate_id']]=F(sum(hits[:n]),n)
                    for version,values in scores.items():
                        top=[c for c in pool['candidates'] if values[c['candidate_id']]==max(values.values())]
                        selected=max(pool['candidates'],key=lambda c:(values[c['candidate_id']],-c['sampling_seed'],c['candidate_id']))
                        reported=next(p for p in result['per_pool'][task['id']][str(step)][cond] if p['prompt_id']==pool['prompt_id'])[version]
                        assert reported['selected_candidate_id']==selected['candidate_id']
                        assert abs(reported['top_correct_rate']-float(F(sum(c['correct'] for c in top),len(top))))<1e-12
                        checks['pool_scores']+=1
    assert checks['training_microbatches']==6*32*8
    assert checks['probe_answers']==6*33*200
    assert checks['full_answers']==6*3*1818
    save_new(SIDE/'results-validation.json',{'created_utc':now(),'status':'PASS_integrity_and_arithmetic_only',
        'checks':dict(checks),'results_sha256':sha(SIDE/'results.json'),'limitations':'Does not establish representativeness or causal generalization beyond these interventions.'})
    print(json.dumps(dict(checks),indent=2))


if __name__=='__main__':main()
