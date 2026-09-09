"""Read saved adapters and run observations only, in isolated output files."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import os
import time
import yaml

from common import *
from selfsight.v4.observe import PROMPTED_PREAMBLE
from selfsight.v4.checkpoint_probe import validate_observation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    args=parser.parse_args()
    plan=read(SIDE/'plan.json')
    assert_main_paused()
    for path,digest in plan['input_sha256'].items():
        assert sha(path)==digest,path
    bank=read(SIDE/'fixed-bank.json')
    config=yaml.safe_load(Path(plan['config_path']).read_text(encoding='utf-8'))
    targets=read(plan['targets_path'])
    assert not targets.get('forbidden_modules_selected')
    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.training.checkpoint import load_checkpoint
    from selfsight.v4.train import seed_training, trainable_snapshot, parameter_digest
    seed_training(int(config['seed']))
    backbone=Showo2Adapter(backbone_config=config['model']['backbone_config'], device=args.device,
                          dtype=config['hardware']['precision'], lazy=False)
    assert backbone.model_id==config['model']['trainable_id']
    lora=config['training']['lora']
    seed_training(int(config['seed']))
    backbone.attach_lora(target_modules=tuple(targets['target_modules']), rank=int(lora['rank']),
                         alpha=int(lora['alpha']),dropout=float(lora['dropout']),
                         gradient_checkpointing=bool(config['training']['gradient_checkpointing']))
    for task in [t for t in plan['tasks'] if t['device']==args.device]:
        assert_main_paused()
        assert time.time()<plan['deadline']
        folder=SIDE/'observations'/task['id']
        folder.mkdir(parents=True, exist_ok=False)
        checkpoint=task['checkpoint']
        path=Path(checkpoint['path'])
        assert sha(path/'manifest.json')==checkpoint['manifest_sha256']
        assert sha(path/'adapter.pt')==checkpoint['adapter_sha256']
        state=load_checkpoint(path,model=backbone.model,optimizer=None,scheduler=None,
                              expected_config_digest=sha256_json(config))
        assert state['step']==task['step']
        before=parameter_digest(trainable_snapshot(backbone.model))
        assert before==task['adapter_parameter_digest']
        backbone.model.eval()
        save_new(folder/'started.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'task':task,
                    'worker_pid':os.getpid(),'plan_sha256':sha(SIDE/'plan.json'), 'adapter_parameter_digest':before})
        completed=errors=abstained=0
        original_lookup={(r['prompt_id'],r['candidate_id']):r['observation'] for r in lines(task['old_observations'])}
        reproduction_hits=reproduction_n=0
        repro_mismatches=[]
        with (folder/'answers.jsonl').open('x',encoding='utf-8') as out:
            for pool in bank['pools']:
                raw_questions=tuple(AtomicQuestion.from_dict(q) for q in pool['questions'])
                for c in pool['candidates']:
                    assert sha(c['image_path'])==c['image_file_sha256']
                    order=sorted(plan['conditions'],key=lambda cond:sha256_json([pool['prompt_id'],c['candidate_id'],cond,'order-v1']))
                    for condition in order:
                        assert time.time()<plan['deadline']
                        seed=int(sha256_json([bank['original_bank_fingerprint'],pool['prompt_id'],c['candidate_id'],'observe'])[:8],16)
                        seed_training(seed)
                        questions=tuple(replace(q,text=PROMPTED_PREAMBLE.format(prompt=pool['prompt'],question=q.text)) for q in raw_questions) if condition=='prompt_on' else raw_questions
                        result=backbone.observe_atoms(c['image_path'],questions)
                        validate_observation(result,pool,c,identity=(backbone.model_id,backbone.revision))
                        actual=as_serializable(result)
                        errors+=sum(a.error is not None for a in result.answers)
                        abstained+=sum(a.abstain for a in result.answers)
                        if condition=='prompt_on':
                            old=original_lookup[pool['prompt_id'],c['candidate_id']]['answers']
                            assert [x['question_id'] for x in old]==[q.question_id for q in questions[:pool['n_original_questions']]]
                            for previous,current in zip(old,result.answers):
                                match=(previous['normalized_answer'],previous['abstain'],previous.get('error'))==(current.normalized_answer,current.abstain,current.error)
                                reproduction_hits+=match
                                reproduction_n+=1
                                if not match: repro_mismatches.append({'prompt_id':pool['prompt_id'],'candidate_id':c['candidate_id'],'question_id':current.question_id,'old':previous,'new':as_serializable(current)})
                        request={'task_id':task['id'],'condition':condition,'prompt_id':pool['prompt_id'],
                                 'candidate_id':c['candidate_id'],'seed':seed,'question_sha256':sha256_json([as_serializable(q) for q in questions]),
                                 'checkpoint_adapter_sha256':checkpoint['adapter_sha256'],'image_rgb_sha256':c['rgb_sha256']}
                        record={**request,'measurement_id':sha256_json(request),'observation':actual}
                        out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
                        completed+=1
                    print(f"{task['id']} images {completed//2}/{plan['n_images']}",flush=True)
        assert completed==plan['n_images']*2
        after=parameter_digest(trainable_snapshot(backbone.model))
        assert before==after
        save_new(folder/'complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'task':task,
                 'n_observations':completed,'n_errors':errors,'n_abstained':abstained,
                 'old_prompted_answer_reproduction':{'matches':reproduction_hits,'total':reproduction_n,'mismatches':repro_mismatches},
                 'adapter_parameter_digest_before':before,'adapter_parameter_digest_after':after,
                 'answers_sha256':sha(folder/'answers.jsonl'),'plan_sha256':sha(SIDE/'plan.json')})
        print(f"completed {task['id']}; old-answer reproduction {reproduction_hits}/{reproduction_n}",flush=True)
    assert_main_paused()


if __name__=='__main__':
    main()
