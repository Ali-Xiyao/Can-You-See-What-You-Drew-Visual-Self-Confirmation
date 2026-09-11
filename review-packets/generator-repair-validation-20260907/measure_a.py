from common import *
import argparse,contextlib
from dataclasses import replace

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    p=verify();corpus=read(SIDE/'corpus.json');scenes={r['spec']['spec_id']:r for r in corpus['rows'] if r['split']=='A'}
    from selfsight.backbones.showo2 import Showo2GenerationBatch
    from selfsight.training.checkpoint import _rng_state,_restore_rng_state
    from selfsight.v4.train import parameter_digest,trainable_snapshot,seed_training
    from selfsight.v4.observe import PROMPTED_PREAMBLE
    from selfsight.schemas import AtomicQuestion,as_serializable
    import torch
    backbone,config=make_backbone(p,'cuda:0')
    wait_for(SIDE/'generation/base/complete.json',args.deadline)
    targets=[{**r,'cohort':'training','scene_key':r['prompt_id']} for r in read(SIDE/'loss-training.json')]
    targets += [{**r,'cohort':'new_base','scene_key':r['spec_id']} for r in lines(SIDE/'generation/base/images.jsonl') if r['candidate_index']==0]
    assert len(targets)==32
    for model in p['models']:
        before=load_model(backbone,model,config);folder=SIDE/'measurement'/model['id'];folder.mkdir(parents=True,exist_ok=False)
        rows=[]
        with (folder/'loss.jsonl').open('x',encoding='utf-8') as log:
            for r in targets:
                for noise_index in range(2):
                    main_paused();assert time.time()<args.deadline
                    latent_seed=seed('loss',r['cohort'],r['scene_key'],noise_index)
                    batch=Showo2GenerationBatch(prompts=(r['prompt'],),images=(r['image_path'],),sample_ids=(r['scene_key'],),latent_seed=latent_seed)
                    rng=_rng_state()
                    with torch.no_grad():
                        value=float(backbone.generation_loss(batch).detach().cpu())
                        repeat=float(backbone.generation_loss(batch).detach().cpu()) if model['id']=='base' else None
                    _restore_rng_state(rng)
                    assert torch.isfinite(torch.tensor(value))
                    row={'model':model['id'],'cohort':r['cohort'],'scene_key':r['scene_key'],'image_path':r['image_path'],
                         'image_sha256':sha(r['image_path']),'latent_seed':latent_seed,'noise_index':noise_index,'loss':value,'base_repeat_loss':repeat}
                    rows.append(row);append(log,row)
        assert parameter_digest(trainable_snapshot(backbone.model))==before
        save_new(folder/'loss-complete.json',{'at_utc':now(),'n_instances':64,'n_base_repeats':64 if model['id']=='base' else 0,
            'data_sha256':sha(folder/'loss.jsonl'),'parameter_digest':before,'model_eval':True})
        print(f"{model['id']} fixed-noise losses complete",flush=True)
    wait_for(SIDE/'generation/complete.json',args.deadline)
    images=[r for model in p['models'] for r in lines(SIDE/'generation'/model['id']/'images.jsonl')]
    assert len(images)==128
    for model in p['models']:
        before=load_model(backbone,model,config);folder=SIDE/'measurement'/model['id'];n_answers=0;n_images=0
        with (folder/'observations.jsonl').open('x',encoding='utf-8') as log,(folder/'cycle.jsonl').open('x',encoding='utf-8') as cyclelog:
            for row in images:
                main_paused();assert time.time()<args.deadline
                scene=scenes[row['spec_id']];raw=tuple(AtomicQuestion.from_dict(q) for q in scene['questions'])
                for cond in p['conditions']:
                    rng=_rng_state();seed_training(seed('observe',row['spec_id'],row['candidate_index'],row['model']))
                    qs=raw if cond=='prompt_off' else tuple(replace(q,text=PROMPTED_PREAMBLE.format(prompt=row['prompt'] if cond=='prompt_on' else '',question=q.text)) for q in raw)
                    observation=backbone.observe_atoms(row['image_path'],qs);_restore_rng_state(rng)
                    assert observation.rgb_sha256==row['rgb_sha256']
                    assert [a.question_id for a in observation.answers]==[q.question_id for q in raw]
                    append(log,{'observer':model['id'],'generator':row['model'],'spec_id':row['spec_id'],'scene_sha256':row['scene_sha256'],
                        'candidate_index':row['candidate_index'],'seed':row['seed'],'condition':cond,'n_original':scene['n_original'],
                        'image_path':row['image_path'],'observation':as_serializable(observation)})
                    n_answers+=len(observation.answers)
                rng=_rng_state();value=backbone.cycle_consistency_score(row['image_path'],row['prompt']);_restore_rng_state(rng)
                append(cyclelog,{'observer':model['id'],'generator':row['model'],'spec_id':row['spec_id'],'candidate_index':row['candidate_index'],
                                 'image_path':row['image_path'],'cycle':value})
                n_images+=1
                if n_images%8==0:print(f"{model['id']} cross-observation {n_images}/128",flush=True)
        after=parameter_digest(trainable_snapshot(backbone.model));assert after==before
        save_new(folder/'complete.json',{'at_utc':now(),'n_answers':n_answers,'n_images':n_images,'n_conditions':3,
            'parameter_digest_before':before,'parameter_digest_after':after,'observations_sha256':sha(folder/'observations.jsonl'),'cycle_sha256':sha(folder/'cycle.jsonl')})
    save_new(SIDE/'measurement/complete.json',{'at_utc':now(),'models':4,'new_optimizer_steps':0})
    print('Phase A paired losses and crossed observations complete',flush=True)

if __name__=='__main__':main()
