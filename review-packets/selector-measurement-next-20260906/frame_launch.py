"""Wait for the original observations, then append the registered blank-prompt condition."""
from datetime import datetime, timezone
import os
import subprocess
import time
from common import *
from launch import preflight


def main():
    plan=read(SIDE/'plan.json')
    frozen=read(SIDE/'frame-source-freeze.json')
    for path,h in frozen.items(): assert sha(path)==h,path
    assert not (SIDE/'frame-controller-started.json').exists()
    save_new(SIDE/'frame-controller-started.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'source_freeze_sha256':sha(SIDE/'frame-source-freeze.json')})
    while not (SIDE/'controller-complete.json').exists():
        assert not (SIDE/'controller-failed.json').exists(),'Original controller failed'
        assert time.time()<plan['deadline']
        time.sleep(5)
    assert read(SIDE/'controller-complete.json')['exit_code']==0
    preflight()
    import msvcrt
    with (MAIN/'supervisor.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        recovery=read(ROOT/'review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json')
        env=os.environ.copy();env.update(recovery['environment_allowlist'])
        for key in recovery['environment_remove']:env.pop(key,None)
        env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        children=[];logs=[]
        try:
            for device in ('cuda:0','cuda:1'):
                log=(SIDE/f'frame-worker-{device[-1]}.log').open('x',encoding='utf-8');logs.append(log)
                command=[str(ROOT/'envs/showo2/python.exe'),'-B','-u',str(SIDE/'frame_worker.py'),'--device',device]
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                children.append((device,child,command))
            save_new(SIDE/'frame-workers.json',[{'device':d,'pid':p.pid,'command':c} for d,p,c in children])
            while any(p.poll() is None for _,p,_ in children):
                assert_main_paused();assert time.time()<plan['deadline']
                assert not any(p.poll() not in (None,0) for _,p,_ in children),'Frame worker failed'
                time.sleep(5)
            assert all(p.returncode==0 for _,p,_ in children)
            for task in plan['tasks']:
                complete=read(SIDE/'matched-frame'/task['id']/'complete.json')
                assert complete['n_observations']==80
                assert complete['answers_sha256']==sha(SIDE/'matched-frame'/task['id']/'answers.jsonl')
            save_new(SIDE/'frame-controller-complete.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'exit_code':0,'n_checkpoints':5})
        except BaseException as exc:
            for _,p,_ in children:
                if p.poll() is None:p.terminate()
            for _,p,_ in children:
                if p.poll() is None:p.wait(timeout=30)
            save_new(SIDE/'frame-controller-failed.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'error':str(exc),'automatic_retry':False})
            raise
        finally:
            for log in logs:log.close()


if __name__=='__main__':main()
