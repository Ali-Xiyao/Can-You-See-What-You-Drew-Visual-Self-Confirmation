"""Helpers for the explicitly authorized budget extension, without changing old inputs."""
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parent
PARENT=EXT.parent
sys.path.insert(0,str(PARENT))
from shared import *


def verify_extension():
    plan=read(EXT/'plan.json');parent=read(PARENT/'plan.json')
    assert sha(PARENT/'plan.json')==plan['parent_plan_sha256']
    verify_inputs(parent)
    for path,digest in plan['input_sha256'].items():assert sha(path)==digest,path
    assert plan['target_task']=='naive-alternate_ids'
    assert plan['new_deadline']==min(plan['old_deadline']+7200,parent['deadline'])
    return plan,parent


def newest_checkpoint(source):
    for step in (32,24):
        path=source/'checkpoints'/f'step-{step:05d}'
        if not path.exists():continue
        manifest=read(path/'manifest.json')
        assert manifest['step']==step and manifest['round_index']==step//8-1
        for name,digest in manifest['files'].items():assert sha(path/name)==digest
        return path,manifest
    raise RuntimeError('No intact checkpoint at step24 or32; no base restart permitted')


def retained_training(source,step):
    raw=(source/'training.jsonl').read_text(encoding='utf-8').splitlines(keepends=True)
    assert len(raw)>=step
    prefix=raw[:step]
    assert [__import__('json').loads(r)['step'] for r in prefix]==list(range(1,step+1))
    return prefix


def task_folder(task_id):
    replacement=EXT/'logical-run'
    if task_id=='naive-alternate_ids' and (replacement/'complete.json').exists():return replacement
    return PARENT/'runs'/task_id
