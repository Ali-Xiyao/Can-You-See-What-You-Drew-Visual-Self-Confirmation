"""Bounded controller for two independent GPU observation workers."""
import argparse
from datetime import datetime, timezone
import os
import subprocess
import time
import yaml

from common import *


def preflight():
    assert_main_paused()
    plan=read(SIDE/'plan.json')
    for path,h in plan['input_sha256'].items():
        assert sha(path)==h,path
    config=yaml.safe_load(Path(plan['config_path']).read_text(encoding='utf-8'))
    for task in plan['tasks']:
        c=task['checkpoint']; p=Path(c['path'])
        assert sha(p/'manifest.json')==c['manifest_sha256']
        manifest=read(p/'manifest.json')
        assert manifest['step']==task['step'] and manifest['config_digest']==sha256_json(config)
        assert sha(p/'adapter.pt')==manifest['files']['adapter.pt']==c['adapter_sha256']
        assert sha(p/'training_state.pt')==manifest['files']['training_state.pt']
    assert time.time()<plan['deadline']
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    plan=preflight()
    if args.preflight:
        print('Read-only preflight passed: five saved checkpoints, sources, images, questions and paused main run.',flush=True)
        return
    import msvcrt
    assert not (SIDE/'controller-started.json').exists()
    with (MAIN/'supervisor.lock').open('r+b') as ownership:
        ownership.seek(0);msvcrt.locking(ownership.fileno(),msvcrt.LK_NBLCK,1)
        recovery=read(ROOT/'review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json')
        env=os.environ.copy();env.update(recovery['environment_allowlist'])
        for key in recovery['environment_remove']: env.pop(key,None)
        env['PYTHONDONTWRITEBYTECODE']='1'
        env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
        save_new(SIDE/'controller-started.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'plan_sha256':sha(SIDE/'plan.json')})
        children=[];logs=[]
        try:
            for device in ('cuda:0','cuda:1'):
                log=(SIDE/f"worker-{device[-1]}.log").open('x',encoding='utf-8');logs.append(log)
                command=[str(ROOT/'envs/showo2/python.exe'),'-B','-u',str(SIDE/'worker.py'),'--device',device]
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                children.append((device,child,command))
            save_new(SIDE/'workers.json',[{'device':d,'pid':p.pid,'command':c} for d,p,c in children])
            while any(p.poll() is None for _,p,_ in children):
                assert_main_paused()
                assert time.time()<plan['deadline'],'Original budget exhausted'
                failures=[(d,p.returncode) for d,p,_ in children if p.poll() not in (None,0)]
                if failures: raise RuntimeError(f'Worker failed: {failures}; no automatic retry')
                time.sleep(5)
            assert all(p.returncode==0 for _,p,_ in children)
            for task in plan['tasks']:
                complete=read(SIDE/'observations'/task['id']/'complete.json')
                assert complete['n_observations']==160
                assert complete['answers_sha256']==sha(SIDE/'observations'/task['id']/'answers.jsonl')
            save_new(SIDE/'controller-complete.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'exit_code':0,'n_checkpoints':5,'research_goal_complete':False})
        except BaseException as exc:
            # Only processes created and retained by this controller are stopped.
            for _,p,_ in children:
                if p.poll() is None: p.terminate()
            for _,p,_ in children:
                if p.poll() is None: p.wait(timeout=30)
            save_new(SIDE/'controller-failed.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'error':str(exc),'automatic_retry':False})
            raise
        finally:
            for log in logs: log.close()


if __name__=='__main__':
    main()
