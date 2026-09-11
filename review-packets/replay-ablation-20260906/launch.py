"""Own two hidden workers, bounded runtime, original-run lock, no automatic retry."""
import argparse,os,subprocess,time,shutil
from shared import *


def preflight():
    plan=read(SIDE/'plan.json');verify_inputs(plan)
    cp=plan['base']['checkpoint'];p=Path(cp['path'])
    assert sha(p/'manifest.json')==cp['manifest_sha256']
    manifest=read(p/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(p/name)==digest
    assert sha(p/'adapter.pt')==cp['adapter_sha256']
    assert shutil.disk_usage(ROOT).free>12*1024**3
    assert time.time()<plan['deadline']
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    plan=preflight()
    if args.preflight:
        print('Read-only preflight passed: frozen schedule, verified probe facts, six tasks, intact base, paused original run.');return
    import msvcrt
    assert not (SIDE/'controller-started.json').exists()
    with (MAIN/'supervisor.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        recovery=read(ROOT/'review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json')
        env=os.environ.copy();env.update(recovery['environment_allowlist'])
        for key in recovery['environment_remove']:env.pop(key,None)
        env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        deadline=min(plan['deadline'],time.time()+plan['max_controller_seconds'])
        save_new(SIDE/'controller-started.json',{'at_utc':now(),'pid':os.getpid(),'deadline':deadline,'plan_sha256':sha(SIDE/'plan.json')})
        workers=[];logs=[]
        try:
            for device in ('cuda:0','cuda:1'):
                log=(SIDE/f'worker-{device[-1]}.log').open('x',encoding='utf-8');logs.append(log)
                cmd=[str(ROOT/'envs/showo2/python.exe'),'-B','-u',str(SIDE/'worker.py'),'--device',device,'--deadline',str(deadline)]
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                workers.append((device,p,cmd))
            save_new(SIDE/'workers.json',[{'device':d,'pid':p.pid,'command':cmd} for d,p,cmd in workers])
            while any(p.poll() is None for _,p,_ in workers):
                assert_main_paused();assert time.time()<deadline,'Registered runtime exhausted'
                failed=[(d,p.returncode) for d,p,_ in workers if p.poll() not in (None,0)]
                assert not failed,failed
                time.sleep(5)
            assert all(p.returncode==0 for _,p,_ in workers)
            for task in plan['tasks']:assert read(SIDE/'runs'/task['id']/'complete.json')['steps']==32
            save_new(SIDE/'controller-complete.json',{'at_utc':now(),'exit_code':0,'n_tasks':6,'research_goal_complete':False})
        except BaseException as exc:
            for _,p,_ in workers:
                if p.poll() is None:p.terminate()
            for _,p,_ in workers:
                if p.poll() is None:p.wait(timeout=30)
            save_new(SIDE/'controller-failed.json',{'at_utc':now(),'error':str(exc),'automatic_retry':False})
            raise
        finally:
            for log in logs:log.close()


if __name__=='__main__':main()
