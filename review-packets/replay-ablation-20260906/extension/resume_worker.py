"""Continue the last fixed condition from its full saved state, on the same GPU."""
import os,time,json,importlib.util
import yaml
from extension_common import *


def copy_prefix(source,folder,step):
    copies={}
    for sub in ('probes','measurements','checkpoints'):(folder/sub).mkdir()
    paths=[source/'probes'/f'step-{s:05d}.json' for s in range(step+1)]
    for s in (0,16,32):
        complete=source/'measurements'/f'step-{s:05d}-complete.json'
        data=source/'measurements'/f'step-{s:05d}.jsonl'
        if s<=step and complete.exists():
            assert read(complete)['answers_sha256']==sha(data)
            paths.extend([complete,data])
    for src in paths:
        relative=src.relative_to(source);dst=folder/relative
        assert dst.resolve().is_relative_to(EXT.resolve()) and not dst.exists()
        payload=src.read_bytes();dst.write_bytes(payload);assert sha(src)==sha(dst)
        copies[str(relative)]={'source':str(src),'sha256':sha(src)}
    return copies


def main():
    extplan,plan=verify_extension();resume=read(EXT/'resume-source.json')
    task=next(t for t in plan['tasks'] if t['id']==extplan['target_task'])
    assert task['arm']=='naive' and task['variant']=='alternate_ids' and task['device']=='cuda:1'
    source=Path(resume['source_directory']);checkpoint=Path(resume['checkpoint_path'])
    assert sha(checkpoint/'manifest.json')==resume['checkpoint_manifest_sha256']
    manifest=read(checkpoint/'manifest.json');start=resume['resume_step'];assert start in (24,32)
    config=yaml.safe_load(Path(plan['config_path']).read_text(encoding='utf-8'));training=config['training'];seed=int(config['seed'])
    targets=read(plan['targets_path']);assert not targets.get('forbidden_modules_selected')
    schedule=read(PARENT/'schedule.json');probe=read(PARENT/'probe.json');bank=read(OLD/'fixed-bank.json')
    deadline=extplan['new_deadline'];assert time.time()<deadline
    prefix=retained_training(source,start)
    raw_prefix=b''.join((source/'training.jsonl').read_bytes().splitlines(keepends=True)[:start])
    spec=importlib.util.spec_from_file_location('frozen_replay_worker',PARENT/'worker.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    import torch
    from selfsight.backbones.showo2 import Showo2Adapter,Showo2GenerationBatch,Showo2ReplayBatch
    from selfsight.training.checkpoint import load_checkpoint,save_checkpoint
    from selfsight.v4.train import seed_training,parameter_digest,trainable_snapshot
    from selfsight.training.paired import _seed_from_parts
    seed_training(seed)
    backbone=Showo2Adapter(backbone_config=config['model']['backbone_config'],device=task['device'],dtype=config['hardware']['precision'],lazy=False)
    lora=training['lora'];seed_training(seed)
    backbone.attach_lora(target_modules=tuple(targets['target_modules']),rank=int(lora['rank']),alpha=int(lora['alpha']),dropout=float(lora['dropout']),gradient_checkpointing=bool(training['gradient_checkpointing']))
    parameters=[p for p in backbone.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=float(training['learning_rate']),weight_decay=float(training['weight_decay']))
    warmup=max(1,round(int(training['rounds'])*int(training['optimizer_steps_per_round'])*float(training['warmup_ratio'])))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda s:min(1.,float(s+1)/warmup))
    state=load_checkpoint(checkpoint,model=backbone.model,optimizer=optimizer,scheduler=scheduler,expected_config_digest=sha256_json(config))
    assert state['step']==start and state['round_index']==start//8-1
    digest=parameter_digest(trainable_snapshot(backbone.model));assert digest==manifest['metadata']['parameter_digest']
    assert {int(v['step'].item()) for v in optimizer.state.values() if 'step' in v}=={start}
    assert scheduler.last_epoch==start and abs(optimizer.param_groups[0]['lr']-1e-4)<1e-12
    backbone.model.train()
    folder=EXT/'logical-run';folder.mkdir(exist_ok=False)
    copies=copy_prefix(source,folder,start)
    save_new(folder/'started.json',{'at_utc':now(),'pid':os.getpid(),'task':task,'plan_sha256':sha(PARENT/'plan.json'),
        'extension_plan_sha256':sha(EXT/'plan.json'),'base_parameter_digest':read(source/'started.json')['base_parameter_digest'],
        'resume_parameter_digest':digest,'resume_step':start,'resume_checkpoint':str(checkpoint),'warmup_steps':warmup,'deadline':deadline})
    save_new(EXT/'prefix-provenance.json',{'at_utc':now(),'resume_step':start,'copied_files':copies,
        'training_prefix_sha256':__import__('hashlib').sha256(raw_prefix).hexdigest(),
        'source_training_sha256':sha(source/'training.jsonl'),'superseded_tail_rule':'Only records above the persisted resume step are replaced by the resumed trajectory, never counted twice.'})
    checkdir=EXT/'resume-verification';(checkdir/'probes').mkdir(parents=True)
    frozen.probe_step(backbone,probe,checkdir,start)
    old=read(source/'probes'/f'step-{start:05d}.json')['answers'];new=read(checkdir/'probes'/f'step-{start:05d}.json')['answers']
    a={(r['condition'],r['id']):r['answer']['normalized_answer'] for r in old};b={(r['condition'],r['id']):r['answer']['normalized_answer'] for r in new}
    assert a==b and len(a)==200,'Restored checkpoint probe does not reproduce exactly'
    save_new(checkdir/'complete.json',{'at_utc':now(),'matches':200,'total':200,'parameter_digest':digest,'optimizer_step':start,'scheduler_epoch':scheduler.last_epoch,'same_device':task['device']})
    print(f'Restored full state at step{start}; 200/200 checkpoint probe answers reproduced',flush=True)
    with (folder/'training.jsonl').open('x',encoding='utf-8') as log:
        log.writelines(prefix);log.flush()
        for rd in range(start//8,4):
            data=schedule[str(rd)];generation=data['generation'][task['arm']];replay=data['alternate_'+task['arm']];gc=rc=0
            for local_step in range(8):
                assert_main_paused();assert time.time()<deadline
                optimizer.zero_grad(set_to_none=True);micro=[];begin=time.time()
                for m in range(8):
                    use_replay=m%4==0;batch_seed=_seed_from_parts(seed,rd,local_step,m,'replay' if use_replay else 't2i')
                    if use_replay:
                        item=replay[rc];rc+=1
                        loss=backbone.understanding_replay_loss(Showo2ReplayBatch(images=(item['image_path'],),questions=(item['question'],),answers=(item['answer'],),sample_ids=(item['sample_id'],),latent_seed=batch_seed))
                        meta={'kind':'replay','sample_id':item['sample_id'],'answer':item['answer'],'weight':1.}
                    else:
                        item=generation[gc%len(generation)];gc+=1
                        loss=backbone.generation_loss(Showo2GenerationBatch(prompts=(item['prompt'],),images=(item['image_path'],),sample_ids=(item['candidate_id'],),latent_seed=batch_seed))
                        meta={'kind':'generation','sample_id':item['candidate_id'],'weight':1.}
                    assert torch.isfinite(loss);value=float(loss.detach().cpu());(loss/8).backward()
                    micro.append({**meta,'micro':m,'latent_seed':batch_seed,'loss':value});del loss
                norm=torch.nn.utils.clip_grad_norm_(parameters,float(training['max_grad_norm']),error_if_nonfinite=True)
                lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();step=rd*8+local_step+1
                log.write(json.dumps({'step':step,'round':rd,'at_utc':now(),'seconds':time.time()-begin,'learning_rate_used':lr,'gradient_norm':float(norm.detach().cpu()),'micro':micro})+'\n');log.flush()
                print(f'Continuation optimizer step {step}/32',flush=True)
                frozen.probe_step(backbone,probe,folder,step)
                if step%8==0:
                    digest=parameter_digest(trainable_snapshot(backbone.model))
                    save_checkpoint(folder/'checkpoints'/f'step-{step:05d}',model=backbone.model,optimizer=optimizer,scheduler=scheduler,
                        config_digest=sha256_json(config),config_values=config,step=step,round_index=rd,
                        metadata={'task':task,'plan_sha256':sha(PARENT/'plan.json'),'extension_plan_sha256':sha(EXT/'plan.json'),'parameter_digest':digest,'offline_frozen_generation':True})
            assert gc==48 and rc==16
    if not (folder/'measurements/step-00032-complete.json').exists():frozen.full_measurement(backbone,bank,folder,32,deadline)
    assert len(lines(folder/'training.jsonl'))==32
    assert (folder/'training.jsonl').read_bytes().startswith(raw_prefix)
    overlap=[]
    for step in range(start+1,33):
        p=source/'probes'/f'step-{step:05d}.json'
        if p.exists():
            try:previous=read(p)
            except json.JSONDecodeError:
                overlap.append({'step':step,'source_partial_not_compared':True});continue
            a={(r['condition'],r['id']):r['answer']['normalized_answer'] for r in previous['answers']}
            b={(r['condition'],r['id']):r['answer']['normalized_answer'] for r in read(folder/'probes'/p.name)['answers']}
            overlap.append({'step':step,'n':len(a),'same_normalized':sum(a[k]==b[k] for k in a)})
    save_new(EXT/'overlap-reproduction.json',{'at_utc':now(),'rows':overlap,'rule':'Both histories retained; no result-based selection of a trajectory'})
    save_new(folder/'complete.json',{'at_utc':now(),'task':task,'steps':32,'n_probe_answers':6600,'n_full_answers':5454,
        'parameter_digest':parameter_digest(trainable_snapshot(backbone.model)),'training_sha256':sha(folder/'training.jsonl'),
        'extension_plan_sha256':sha(EXT/'plan.json'),'resumed_from_step':start,'new_optimizer_steps':32-start,'n_resume_verification_answers':200,'research_goal_complete':False})
    print('Extended final condition complete at the original planned step32',flush=True)


if __name__=='__main__':main()
