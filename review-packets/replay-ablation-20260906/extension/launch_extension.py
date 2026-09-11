"""Wait for the exact old processes, then resume only the unfinished final condition."""
import argparse,os,subprocess,time,shutil
import psutil
from extension_common import *


def capture_process(pid,script,exe):
    try:p=psutil.Process(pid)
    except psutil.NoSuchProcess:return None
    assert Path(p.exe()).resolve()==Path(exe).resolve()
    assert any(a.endswith('.py') and Path(a).resolve()==Path(script).resolve() for a in p.cmdline())
    return {'pid':pid,'create_time':p.create_time(),'script':str(script),'exe':str(exe)}


def still_running(identity):
    if identity is None:return False
    try:p=psutil.Process(identity['pid']);return p.is_running() and p.create_time()==identity['create_time']
    except psutil.NoSuchProcess:return False


def preflight():
    plan,parent=verify_extension();source=PARENT/'runs'/plan['target_task']
    cp,manifest=newest_checkpoint(source)
    assert manifest['config_digest']==sha256_json(__import__('yaml').safe_load(Path(parent['config_path']).read_text(encoding='utf-8')))
    assert shutil.disk_usage(ROOT).free>4*1024**3
    assert time.time()<plan['new_deadline']
    old=read(PARENT/'controller-started.json')
    worker=next(w for w in read(PARENT/'workers.json') if w['device']=='cuda:1')
    identities=[capture_process(old['pid'],PARENT/'launch.py',ROOT/'envs/core/python.exe'),
                capture_process(worker['pid'],PARENT/'worker.py',ROOT/'envs/showo2/python.exe')]
    return plan,parent,identities


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    plan,parent,identities=preflight()
    if args.preflight:print('Read-only extension preflight passed: exact processes, intact saved state, frozen schedule and bounded additional time.');return
    assert not (EXT/'controller-started.json').exists()
    save_new(EXT/'controller-started.json',{'at_utc':now(),'pid':os.getpid(),'new_deadline':plan['new_deadline'],
                'phase':'waiting_for_original_processes','original_processes':identities,'extension_plan_sha256':sha(EXT/'plan.json')})
    child=None
    try:
        while any(still_running(p) for p in identities):
            assert_main_paused();assert time.time()<plan['new_deadline'];time.sleep(5)
        source=PARENT/'runs'/plan['target_task']
        if (source/'complete.json').exists():
            save_new(EXT/'controller-complete.json',{'at_utc':now(),'exit_code':0,'extension_gpu_used':False,'reason':'Original final task completed before continuation was needed'})
            return
        assert time.time()>=plan['old_deadline']-5,'Original process stopped before the old budget; do not treat unrelated errors as a budget extension'
        failed=read(PARENT/'controller-failed.json')
        assert 'Registered runtime exhausted' in failed['error'] or 'cuda:1' in failed['error'],failed
        for task in parent['tasks']:
            if task['id']!=plan['target_task']:assert read(PARENT/'runs'/task['id']/'complete.json')['steps']==32
        cp,manifest=newest_checkpoint(source);step=manifest['step'];retained_training(source,step)
        selection={'selected_utc':now(),'checkpoint_path':str(cp),'checkpoint_manifest_sha256':sha(cp/'manifest.json'),
            'resume_step':step,'source_directory':str(source),'original_failure':failed,'original_failure_sha256':sha(PARENT/'controller-failed.json'),
            'extension_plan_sha256':sha(EXT/'plan.json')}
        save_new(EXT/'resume-source.json',selection)
        import msvcrt
        with (MAIN/'supervisor.lock').open('r+b') as lock:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            recovery=read(ROOT/'review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json')
            env=os.environ.copy();env.update(recovery['environment_allowlist'])
            for key in recovery['environment_remove']:env.pop(key,None)
            env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            cmd=[str(ROOT/'envs/showo2/python.exe'),'-B','-u',str(EXT/'resume_worker.py')]
            with (EXT/'worker.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                save_new(EXT/'worker-started.json',{'at_utc':now(),'pid':child.pid,'command':cmd,'resume_step':step})
                while child.poll() is None:
                    assert_main_paused();assert time.time()<plan['new_deadline'],'Extended runtime exhausted';time.sleep(5)
                assert child.returncode==0,f'Continuation worker failed: {child.returncode}'
            assert read(EXT/'logical-run/complete.json')['steps']==32
            save_new(EXT/'controller-complete.json',{'at_utc':now(),'exit_code':0,'extension_gpu_used':True,'resume_step':step,
                'logical_final_steps':32,'research_goal_complete':False,'original_failure_preserved':True})
    except BaseException as exc:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        save_new(EXT/'controller-failed.json',{'at_utc':now(),'error':str(exc),'automatic_retry':False});raise


if __name__=='__main__':main()
