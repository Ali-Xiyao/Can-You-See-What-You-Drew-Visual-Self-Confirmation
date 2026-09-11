from bcommon import *
import argparse, gc
from dataclasses import replace

def generate(p, deadline):
    wait_room('cuda:0',deadline)
    from selfsight.v4.train import parameter_digest, trainable_snapshot
    backbone,config=make_backbone(p,'cuda:0')
    model=p['models'][0]; before=load_model(backbone,model,config)
    scenes={r['spec']['spec_id']:r for r in scene_rows()}; total=0
    for batch in p['batches']:
        folder=B/'generation'/batch['id']; folder.mkdir(parents=True,exist_ok=False)
        with (folder/'manifest.jsonl').open('x',encoding='utf-8') as log:
            for sid in batch['scene_ids']:
                scene=scenes[sid]
                for index,s in enumerate(scene['sampling_seeds']):
                    check_time(deadline); started=time.time()
                    c=backbone.generate_images([scene['spec']['prompt']],[s],folder,'base-B')[0]
                    append(log,{'model':'base','spec_id':sid,'scene_sha256':scene['scene_sha256'],
                        'candidate_index':index,'seed':s,'prompt':scene['spec']['prompt'],
                        'image_path':c.image_path,'rgb_sha256':c.rgb_sha256,'file_sha256':sha(c.image_path),
                        'spec':scene['spec'],'seconds':time.time()-started,'device':'cuda:0'})
                    total+=1
                    print(f'generated {total}/900 ({batch["id"]})',flush=True)
        assert parameter_digest(trainable_snapshot(backbone.model))==before
        save_new(folder/'complete.json',{'at_utc':now(),'n_images':len(batch['scene_ids'])*6,
            'manifest_sha256':sha(folder/'manifest.jsonl'),'parameter_digest':before})
    save_new(B/'generation/complete.json',{'at_utc':now(),'n_images':total,'parameter_digest':before,'new_optimizer_steps':0})

def observe(p, deadline, shard):
    device=f'cuda:{shard}'; wait_room(device,deadline)
    from selfsight.schemas import AtomicQuestion,as_serializable
    from selfsight.v4.observe import PROMPTED_PREAMBLE
    from selfsight.training.checkpoint import _rng_state,_restore_rng_state
    from selfsight.v4.train import parameter_digest,trainable_snapshot,seed_training
    from selfsight.utils.hashing import rgb_sha256
    scenes={r['spec']['spec_id']:r for r in scene_rows()}
    assigned=set(p['observation_shards'][str(shard)])
    images=[r for r in all_images(p) if r['spec_id'] in assigned]
    assert len(images)==6*len(assigned)
    for r in images:
        assert sha(r['image_path'])==r['file_sha256'] and rgb_sha256(r['image_path'])==r['rgb_sha256']
    backbone,config=make_backbone(p,device)
    for model in p['models']:
        before=load_model(backbone,model,config)
        folder=B/'observations'/f'gpu{shard}'/model['id'];folder.mkdir(parents=True,exist_ok=False)
        n=0
        with (folder/'answers.jsonl').open('x',encoding='utf-8') as log:
            for i,row in enumerate(images):
                check_time(deadline);scene=scenes[row['spec_id']]
                raw=tuple(AtomicQuestion.from_dict(q) for q in scene['questions'])
                for cond in p['conditions']:
                    rng=_rng_state();seed_training(seed('B-observe',row['spec_id'],row['candidate_index']))
                    qs=raw if cond=='prompt_off' else tuple(replace(q,text=PROMPTED_PREAMBLE.format(
                        prompt=row['prompt'] if cond=='prompt_on' else '',question=q.text)) for q in raw)
                    observation=backbone.observe_atoms(row['image_path'],qs);_restore_rng_state(rng)
                    assert observation.rgb_sha256==row['rgb_sha256']
                    assert [a.question_id for a in observation.answers]==[q.question_id for q in raw]
                    append(log,{'observer':model['id'],'generator':'base','spec_id':row['spec_id'],
                        'scene_sha256':row['scene_sha256'],'candidate_index':row['candidate_index'],
                        'seed':row['seed'],'condition':cond,'n_original':scene['n_original'],
                        'image_path':row['image_path'],'device':device,'observation':as_serializable(observation)})
                    n+=len(observation.answers)
                    assert not any(a.error for a in observation.answers),'Observer error saved; stop without retry'
                if (i+1)%12==0:print(f'{device} {model["id"]} observed {i+1}/{len(images)}, {n} answers',flush=True)
        after=parameter_digest(trainable_snapshot(backbone.model));assert after==before
        save_new(folder/'complete.json',{'at_utc':now(),'n_images':len(images),'n_answers':n,
            'parameter_digest_before':before,'parameter_digest_after':after,'answers_sha256':sha(folder/'answers.jsonl')})
    save_new(B/'observations'/f'gpu{shard}'/'complete.json',{'at_utc':now(),'n_scenes':len(assigned),'models':4})

def detect_batch(p, deadline, batch_id):
    import torch
    from types import SimpleNamespace
    from selfsight.v4.train import seed_training
    batch=next(b for b in p['batches'] if b['id']==batch_id)
    folder=B/'generation'/batch_id
    assert not (folder/'verified.jsonl').exists()
    data=lines(folder/'manifest.jsonl');assert len(data)==len(batch['scene_ids'])*6
    assert sha(folder/'manifest.jsonl')==read(folder/'complete.json')['manifest_sha256']
    for r in data:assert sha(r['image_path'])==r['file_sha256']
    pipeline=pipeline_module()
    pipeline.await_room=lambda device,*args,**kwargs:wait_room(device,deadline)
    hashes={}
    for name in ('qwen3vl','internvl'):
        check_time(deadline);seed_training(seed('B-detector',batch_id,name))
        out=folder/f'detections.{name}.jsonl';assert not out.exists()
        pipeline.stage_detect(SimpleNamespace(manifest=folder/'manifest.jsonl',detector=name,
            device='cuda:1',output=out,overwrite=False))
        rs=lines(out)
        assert len(rs)==len(data) and {r['image_path'] for r in rs}=={r['image_path'] for r in data}
        assert all('detections' in r and 'error' not in r for r in rs),'Detector error saved; stop without single-model fallback'
        hashes[name]=sha(out);gc.collect();torch.cuda.empty_cache()
    check_time(deadline)
    pipeline.stage_crop(SimpleNamespace(run=folder,primary='qwen3vl',secondary='internvl',device='cuda:1',overwrite=False))
    crops=folder/'detections.crops.jsonl'
    if crops.exists():
        assert all('error' not in r for r in lines(crops)),'Crop computation error saved; stop without retry'
        hashes['crops']=sha(crops)
    pipeline.stage_verify(SimpleNamespace(run=str(folder),primary='qwen3vl',secondary='internvl'))
    rs=lines(folder/'verified.jsonl');assert len(rs)==len(data)
    hashes['verified']=sha(folder/'verified.jsonl')
    save_new(folder/'verification-complete.json',{'at_utc':now(),'n_images':len(rs),'sha256':hashes,
        'detector_snapshot_digest':p['detector_snapshot_digest'],'device':'cuda:1'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['generate','observe','detect-batch'])
    ap.add_argument('--deadline',type=float,required=True);ap.add_argument('--shard',type=int,choices=[0,1]);ap.add_argument('--batch')
    a=ap.parse_args();p=verify_plan()
    if a.mode=='generate':generate(p,a.deadline)
    elif a.mode=='observe':observe(p,a.deadline,a.shard)
    else:detect_batch(p,a.deadline,a.batch)
    print(f'{a.mode} complete',flush=True)

if __name__=='__main__':main()
