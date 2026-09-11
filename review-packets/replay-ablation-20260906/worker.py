"""Bounded offline replay interventions; never write into the original run."""
import argparse, os, time
from dataclasses import replace
from contextlib import contextmanager
from collections import defaultdict
import yaml
from shared import *


@contextmanager
def observing(backbone):
    import torch
    from selfsight.training.checkpoint import _rng_state, _restore_rng_state
    state=_rng_state(); training=backbone.model.training
    try:
        backbone.model.eval()
        with torch.no_grad(): yield
    finally:
        backbone.model.train(training);_restore_rng_state(state)
        after=_rng_state()
        assert state['python']==after['python']
        assert (state['numpy'][1]==after['numpy'][1]).all()
        assert torch.equal(state['torch_cpu'],after['torch_cpu'])
        assert all(torch.equal(a,b) for a,b in zip(state.get('torch_cuda',[]),after.get('torch_cuda',[])))


def checked_observation(backbone,image,questions):
    observation=backbone.observe_atoms(image,questions)
    assert [a.question_id for a in observation.answers]==[q.question_id for q in questions]
    assert all(a.error is None and not a.abstain for a in observation.answers)
    return observation


def probe_step(backbone,probe,folder,step):
    from selfsight.v4.observe import PROMPTED_PREAMBLE
    groups=defaultdict(list)
    for r in probe['rows']: groups[r['image_path']].append(r)
    records=[]
    with observing(backbone):
        for image,rows in sorted(groups.items()):
            for condition in ('prompt_on','prompt_off'):
                qs=tuple(AtomicQuestion.from_dict(r['question']) for r in rows)
                if condition=='prompt_on': qs=tuple(replace(q,text=PROMPTED_PREAMBLE.format(prompt=r['prompt'],question=q.text)) for q,r in zip(qs,rows))
                result=checked_observation(backbone,image,qs)
                for r,a in zip(rows,result.answers):
                    records.append({'id':r['id'],'condition':condition,'group':r['group'],'origin':r['origin'],
                                    'truth':r['truth'],'scene_sha256':r['scene_sha256'],'answer':as_serializable(a)})
    assert len(records)==200
    result={'step':step,'at_utc':now(),'rng_restored':True,'n_answers':200,'summary':summarize_probe(records),'answers':records}
    save_new(folder/'probes'/f'step-{step:05d}.json',result)
    print(json.dumps({'event':'probe','step':step,'summary':result['summary']}),flush=True)


def full_measurement(backbone,bank,folder,step,deadline):
    from selfsight.v4.observe import PROMPTED_PREAMBLE
    from selfsight.v4.train import seed_training, parameter_digest, trainable_snapshot
    before=parameter_digest(trainable_snapshot(backbone.model)); count=0
    path=folder/'measurements'/f'step-{step:05d}.jsonl'
    with observing(backbone), path.open('x',encoding='utf-8') as out:
        for pool in bank['pools']:
            raw=tuple(AtomicQuestion.from_dict(q) for q in pool['questions'])
            for c in pool['candidates']:
                assert time.time()<deadline
                for condition in ('prompt_on','prompt_blank','prompt_off'):
                    seed_training(int(sha256_json([bank['original_bank_fingerprint'],pool['prompt_id'],c['candidate_id'],'observe'])[:8],16))
                    qs=raw if condition=='prompt_off' else tuple(replace(q,text=PROMPTED_PREAMBLE.format(prompt=pool['prompt'] if condition=='prompt_on' else '',question=q.text)) for q in raw)
                    result=checked_observation(backbone,c['image_path'],qs)
                    assert result.rgb_sha256==c['rgb_sha256']
                    out.write(json.dumps({'step':step,'condition':condition,'prompt_id':pool['prompt_id'],
                               'candidate_id':c['candidate_id'],'observation':as_serializable(result)},ensure_ascii=False)+'\n');out.flush()
                    count+=len(result.answers)
    after=parameter_digest(trainable_snapshot(backbone.model));assert before==after
    assert count==1818
    save_new(folder/'measurements'/f'step-{step:05d}-complete.json',{'step':step,'at_utc':now(),'n_answers':count,
        'rng_restored':True,'parameter_digest_before':before,'parameter_digest_after':after,'answers_sha256':sha(path)})
    print(f'Full measurement completed step {step}, {count} answers',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();plan=read(SIDE/'plan.json');verify_inputs(plan)
    config=yaml.safe_load(Path(plan['config_path']).read_text(encoding='utf-8'))
    targets=read(plan['targets_path']); assert not targets.get('forbidden_modules_selected')
    training=config['training'];seed=int(config['seed']);deadline=min(plan['deadline'],args.deadline)
    schedule=read(SIDE/'schedule.json'); probe=read(SIDE/'probe.json');bank=read(OLD/'fixed-bank.json')
    import torch
    from selfsight.backbones.showo2 import Showo2Adapter,Showo2GenerationBatch,Showo2ReplayBatch
    from selfsight.training.checkpoint import load_checkpoint,save_checkpoint
    from selfsight.v4.train import seed_training,parameter_digest,trainable_snapshot
    from selfsight.training.paired import _seed_from_parts
    seed_training(seed)
    backbone=Showo2Adapter(backbone_config=config['model']['backbone_config'],device=args.device,dtype=config['hardware']['precision'],lazy=False)
    lora=training['lora'];seed_training(seed)
    backbone.attach_lora(target_modules=tuple(targets['target_modules']),rank=int(lora['rank']),alpha=int(lora['alpha']),
                         dropout=float(lora['dropout']),gradient_checkpointing=bool(training['gradient_checkpointing']))
    parameters=[p for p in backbone.model.parameters() if p.requires_grad]
    for task in [t for t in plan['tasks'] if t['device']==args.device]:
        assert_main_paused(); assert time.time()<deadline
        folder=SIDE/'runs'/task['id'];folder.mkdir(parents=True,exist_ok=False)
        for sub in ('probes','measurements','checkpoints'): (folder/sub).mkdir()
        optimizer=torch.optim.AdamW(parameters,lr=float(training['learning_rate']),weight_decay=float(training['weight_decay']))
        warmup=max(1,round(int(training['rounds'])*int(training['optimizer_steps_per_round'])*float(training['warmup_ratio'])))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda s:min(1.,float(s+1)/warmup))
        state=load_checkpoint(Path(plan['base']['checkpoint']['path']),model=backbone.model,optimizer=optimizer,scheduler=scheduler,expected_config_digest=sha256_json(config))
        assert state['step']==0
        base_digest=parameter_digest(trainable_snapshot(backbone.model));assert base_digest==plan['base']['adapter_parameter_digest']
        backbone.model.train()
        save_new(folder/'started.json',{'at_utc':now(),'pid':os.getpid(),'task':task,'plan_sha256':sha(SIDE/'plan.json'),
                                     'base_parameter_digest':base_digest,'warmup_steps':warmup,'deadline':deadline})
        probe_step(backbone,probe,folder,0);full_measurement(backbone,bank,folder,0,deadline)
        with (folder/'training.jsonl').open('x',encoding='utf-8') as log:
            for rd in range(4):
                data=schedule[str(rd)]; generation=data['generation'][task['arm']]
                key={'original':'original','zero_weight':'original','balanced_absence':'balanced_absence','alternate_ids':'alternate_'+task['arm']}[task['variant']]
                replay=data[key];gc=rc=0
                for local_step in range(8):
                    assert_main_paused();assert time.time()<deadline
                    optimizer.zero_grad(set_to_none=True);micro=[];start=time.time()
                    for m in range(8):
                        use_replay=m%4==0
                        batch_seed=_seed_from_parts(seed,rd,local_step,m,'replay' if use_replay else 't2i')
                        if use_replay:
                            item=replay[rc];rc+=1
                            loss=backbone.understanding_replay_loss(Showo2ReplayBatch(images=(item['image_path'],),questions=(item['question'],),answers=(item['answer'],),sample_ids=(item['sample_id'],),latent_seed=batch_seed))
                            weight=0. if task['variant']=='zero_weight' else 1.
                            meta={'kind':'replay','sample_id':item['sample_id'],'answer':item['answer'],'weight':weight}
                        else:
                            item=generation[gc%len(generation)];gc+=1
                            loss=backbone.generation_loss(Showo2GenerationBatch(prompts=(item['prompt'],),images=(item['image_path'],),sample_ids=(item['candidate_id'],),latent_seed=batch_seed))
                            weight=1.;meta={'kind':'generation','sample_id':item['candidate_id'],'weight':weight}
                        assert torch.isfinite(loss),'Nonfinite loss'
                        value=float(loss.detach().cpu());(loss*weight/8).backward()
                        micro.append({**meta,'micro':m,'latent_seed':batch_seed,'loss':value})
                        del loss
                    norm=torch.nn.utils.clip_grad_norm_(parameters,float(training['max_grad_norm']),error_if_nonfinite=True)
                    lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                    step=rd*8+local_step+1
                    log.write(json.dumps({'step':step,'round':rd,'at_utc':now(),'seconds':time.time()-start,
                                         'learning_rate_used':lr,'gradient_norm':float(norm.detach().cpu()),'micro':micro})+'\n');log.flush()
                    print(f"{task['id']} optimizer step {step}/32",flush=True)
                    probe_step(backbone,probe,folder,step)
                    if step%8==0:
                        digest=parameter_digest(trainable_snapshot(backbone.model))
                        save_checkpoint(folder/'checkpoints'/f'step-{step:05d}',model=backbone.model,optimizer=optimizer,scheduler=scheduler,
                            config_digest=sha256_json(config),config_values=config,step=step,round_index=rd,
                            metadata={'task':task,'plan_sha256':sha(SIDE/'plan.json'),'parameter_digest':digest,'offline_frozen_generation':True})
                    if step in plan['full_measurement_steps']: full_measurement(backbone,bank,folder,step,deadline)
                assert gc==48 and rc==16
        save_new(folder/'complete.json',{'at_utc':now(),'task':task,'steps':32,'n_probe_answers':6600,
            'n_full_answers':5454,'parameter_digest':parameter_digest(trainable_snapshot(backbone.model)),
            'training_sha256':sha(folder/'training.jsonl'),'research_goal_complete':False})
        del optimizer,scheduler;torch.cuda.empty_cache()
        print(f"Completed {task['id']}",flush=True)


if __name__=='__main__':main()
