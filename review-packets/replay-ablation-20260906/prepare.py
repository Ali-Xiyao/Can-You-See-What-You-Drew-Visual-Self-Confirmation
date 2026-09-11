"""Freeze actual training inputs and a held-out factual probe before new runs."""
from collections import Counter, defaultdict
from dataclasses import asdict
import random
from shared import *
from selfsight.v4.train import load_training_corpus, restrict_replay
from selfsight.v4.factual_truth import factual_answer, question_scope
from selfsight.v4.spec import canonical_noun


def zero_question(q):
    scope=question_scope(q); assert scope and scope.kind=='existence'
    return {'question_id':q['question_id']+':count-zero','atom_id':q['atom_id']+':count-zero',
        'family':'count','text':f'How many {scope.noun}s are in this picture? Answer with a single number.',
        'expected_answer':'0','question_format':'open','choices':[],'choice_order_seed':0}


def scene_round_robin(rows,n,salt):
    grouped=defaultdict(list)
    for r in stable_order(rows,salt,lambda r:r['id']): grouped[r['scene_sha256']].append(r)
    selected=[]
    while len(selected)<n:
        active=sorted(k for k,v in grouped.items() if v)
        if not active: raise ValueError(f'Insufficient probe facts: {len(selected)} / {n}')
        for key in active:
            selected.append(grouped[key].pop(0))
            if len(selected)==n: break
    return selected


def main():
    assert_main_paused(); assert not (SIDE/'plan.json').exists()
    inputs={}
    def track(path):
        path=Path(path).resolve(); inputs[str(path)]=sha(path); return path
    oldplan=read(track(OLD/'plan.json')); bank=read(track(OLD/'fixed-bank.json'))
    split=read(track(MAIN/'split.json'))
    corpus=restrict_replay(load_training_corpus(split['runs']),split['train'])
    assert len(corpus.replay)==706
    verifications={}
    for run in split['runs']:
        track(ROOT/run/'manifest.jsonl')
        for row in lines(track(ROOT/run/'verified.jsonl')):
            verifications[str(Path(row['image_path']).resolve())]=row
    schedule={}; replay_original=[]
    for round_index in range(4):
        folder=MAIN/'rounds'/f'round-{round_index:03d}'
        selection=read(track(folder/'selection.json')); done=read(track(folder/'DONE.json'))
        cursors={arm['replay_cursor_start'] for arm in done['arms']}; assert len(cursors)==1
        cursor=cursors.pop(); rep=[asdict(corpus.replay[(cursor+i)%len(corpus.replay)]) for i in range(16)]
        replay_original+=rep
        naive={r['candidate_id']:r for r in lines(track(folder/'observations/naive.jsonl'))}
        gold={Path(r['image_path']).stem:r for r in lines(track(folder/'ladder/rfo_gold/manifest.jsonl'))}
        generation={}
        for arm in ('naive','rfo_gold'):
            decisions={d['prompt_id']:d for d in selection['decisions'][arm]}
            generation[arm]=[]
            for pid in selection['paired_prompt_ids']:
                cid=decisions[pid]['selected_candidate_id']; item=(naive if arm=='naive' else gold)[cid]
                p=str(Path(item['image_path']).resolve());track(p)
                generation[arm].append({'prompt_id':pid,'candidate_id':cid,'image_path':p,'prompt':corpus.specs[pid].prompt})
            assert len(generation[arm])==done['paired']
        schedule[str(round_index)]={'generation':generation,'original':rep}
    assert len(replay_original)==64
    original_ids={r['sample_id'] for r in replay_original}; original_images={r['image_path'] for r in replay_original}
    for arm in ('naive','rfo_gold'):
        eligible=defaultdict(list)
        for x in corpus.replay:
            if x.sample_id not in original_ids and x.image_path not in original_images: eligible[x.answer].append(asdict(x))
        for answer, rows in eligible.items():
            eligible[answer]=stable_order(rows,'alternate-replay-v1:'+arm,lambda r:r['sample_id'])
        cursors=Counter()
        for rd in schedule.values():
            other=[]
            for orig in rd['original']:
                options=eligible[orig['answer']]; i=cursors[orig['answer']]; assert i<len(options)
                item=options[i];cursors[orig['answer']]+=1;other.append(item)
                assert item['sample_id'] not in original_ids and item['image_path'] not in original_images
            rd['alternate_'+arm]=other
    from common import neg_questions
    family_seen=Counter(); transformed=Counter()
    for rd in schedule.values():
        balanced=[]
        for orig in rd['original']:
            is_exist=orig['answer']=='yes'; fam='existence' if is_exist else 'count'
            family_seen[fam]+=1
            item=dict(orig)
            if family_seen[fam]%2==0:
                spec=corpus.specs[orig['prompt_id']].to_dict()
                choices=neg_questions(spec,bank['vocabulary'],len(bank['vocabulary'])-len({canonical_noun(o['object']) for o in spec['objects']}))
                verdict=verifications[str(Path(orig['image_path']).resolve())]
                assert verdict['image_correct'] is True
                for q in choices:
                    qd=as_serializable(q)
                    if not is_exist: qd=zero_question(qd)
                    truth=factual_answer(qd,verdict)
                    if truth.known and truth.answer==('no' if is_exist else '0'):
                        item.update(question=qd['text'],answer=truth.answer,sample_id=orig['sample_id']+':balanced-absence-v1')
                        item['fact_provenance']={'question':qd,'reason':truth.reason,'verification_sha256':sha(ROOT/next(r for r in split['runs'] if r.endswith(orig['sample_id'].split(':')[0]))/'verified.jsonl')}
                        transformed[item['answer']]+=1;break
                else: raise ValueError('No verified absent target for training image')
            balanced.append(item)
        rd['balanced_absence']=balanced
    for rd in schedule.values():
        for key, values in rd.items():
            if key!='generation':
                for x in values: track(x['image_path'])
    groups=defaultdict(list)
    for pool in bank['pools']:
        if not pool['primary_scene_disjoint']: continue
        for c in pool['candidates']:
            for q,fact in zip(pool['questions'],c['facts']):
                if not fact['known']: continue
                variants=[(q,fact['answer'],'original' if ':negative-exists-v1:' not in q['question_id'] else 'negative_existence')]
                if ':negative-exists-v1:' in q['question_id'] and fact['answer']=='no':
                    variants.append((zero_question(q),'0','negative_count'))
                for question,truth,origin in variants:
                    group=answer_group(question,truth)
                    groups[group].append({'id':sha256_json([pool['prompt_id'],c['candidate_id'],question['question_id']]),
                        'prompt_id':pool['prompt_id'],'prompt':pool['prompt'],'scene_sha256':pool['scene_sha256'],
                        'image_path':c['image_path'],'image_file_sha256':c['image_file_sha256'],'question':question,
                        'truth':truth,'group':group,'origin':origin})
    probe=[]
    for group,rows in sorted(groups.items()):
        # Preserve all original-scope zero/no items when scarce; fill with known absent nouns.
        if group in ('count_zero','existence_no'):
            original=[r for r in rows if r['origin']=='original']
            chosen=scene_round_robin(original,min(25,len(original)),group+'-original') if original else []
            chosen+=scene_round_robin([r for r in rows if r['origin']!='original'],25-len(chosen),group+'-extra') if len(chosen)<25 else []
        else: chosen=scene_round_robin(rows,25,group)
        probe+=chosen
    assert len(probe)==len({r['id'] for r in probe})==100
    for r in probe: track(r['image_path'])
    probe_counts={'groups':dict(Counter(r['group'] for r in probe)),'origins':dict(Counter(r['origin'] for r in probe)),
                  'n_scenes':len({r['scene_sha256'] for r in probe}),'n_images':len({r['image_path'] for r in probe})}
    tasks=[{'id':f'{arm}-{variant}','arm':arm,'variant':variant,'device':device} for device,arm,variant in [
        ('cuda:0','naive','original'),('cuda:0','naive','balanced_absence'),('cuda:0','naive','zero_weight'),
        ('cuda:1','rfo_gold','original'),('cuda:1','rfo_gold','alternate_ids'),('cuda:1','naive','alternate_ids')]]
    save_new(SIDE/'schedule.json',schedule);save_new(SIDE/'probe.json',{'rows':probe,'counts':probe_counts})
    for p in (SIDE/'schedule.json',SIDE/'probe.json',SIDE/'PROTOCOL.md',ROOT/oldplan['config_path'],ROOT/oldplan['targets_path']):track(p)
    for p in list(SIDE.glob('*.py')):track(p)
    # Training helpers are inputs too; source changes after registration abort the controller.
    for p in [ROOT/'src/selfsight/v4/train.py',ROOT/'src/selfsight/backbones/showo2.py',ROOT/'src/selfsight/training/checkpoint.py']:track(p)
    base=next(t for t in oldplan['tasks'] if t['step']==0)
    plan={'created_utc':now(),'tasks':tasks,'steps':32,'full_measurement_steps':[0,16,32],
          'deadline':oldplan['deadline'],'max_controller_seconds':6*3600,'config_path':oldplan['config_path'],
          'targets_path':oldplan['targets_path'],'base':base,'input_sha256':inputs,'probe_counts':probe_counts,
          'balanced_replay_replaced_targets':dict(transformed),'original_replay_targets':dict(Counter(r['answer'] for r in replay_original)),
          'research_goal_complete':False}
    save_new(SIDE/'plan.json',plan)
    print(json.dumps({k:v for k,v in plan.items() if k not in ('input_sha256','base')},indent=2))


if __name__=='__main__': main()
