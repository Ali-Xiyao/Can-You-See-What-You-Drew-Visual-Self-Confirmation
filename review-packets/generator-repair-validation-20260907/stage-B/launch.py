"""Two sequential device lanes; bounded, no retry, owned-process cleanup."""
from bcommon import *
import shutil, msvcrt, psutil

def main():
    p=verify_plan();assert not (B/'controller-started.json').exists()
    assert shutil.disk_usage(ROOT).free > 8*1024**3
    recovery=read(ROOT/'review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json')
    env=os.environ.copy();env.update(recovery['environment_allowlist'])
    for key in recovery['environment_remove']:env.pop(key,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    children={};logs=[];deadline=time.time()+p['max_seconds'];next_batch=0
    started_observation=set();detector_complete=False
    def start(name, environment, args):
        log=(B/(name+'.log')).open('x',encoding='utf-8');logs.append(log)
        cmd=[str(ROOT/f'envs/{environment}/python.exe'),'-B','-u',str(B/'worker.py'),*args,'--deadline',str(deadline)]
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        children[name]=proc
        save_new(B/(name+'-started.json'),{'at_utc':now(),'pid':proc.pid,'command':cmd})
        print(f'started {name} PID {proc.pid}',flush=True)
    def done(name):return name in children and children[name].poll()==0
    try:
        with (MAIN/'supervisor.lock').open('r+b') as lock:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            save_new(B/'controller-started.json',{'at_utc':now(),'pid':os.getpid(),'deadline':deadline,
                'plan_sha256':sha(B/'plan.json'),'stage':'B','new_optimizer_steps':0})
            start('generate','showo2',['generate'])
            while True:
                check_time(deadline)
                for name,proc in children.items():assert proc.poll() in (None,0),f'{name}: exit {proc.returncode}'
                active_detector=any(name.startswith('detect-') and proc.poll() is None for name,proc in children.items())
                if next_batch<len(p['batches']) and not active_detector:
                    batch=p['batches'][next_batch]
                    if (B/'generation'/batch['id']/'complete.json').exists():
                        start('detect-'+batch['id'],'observer',['detect-batch','--batch',batch['id']]);next_batch+=1
                detector_complete=next_batch==len(p['batches']) and all(done('detect-'+b['id']) for b in p['batches'])
                if done('generate'):
                    for shard in (0,1):
                        if shard not in started_observation and (shard==0 or detector_complete):
                            start(f'observe-gpu{shard}','showo2',['observe','--shard',str(shard)]);started_observation.add(shard)
                if detector_complete and done('observe-gpu0') and done('observe-gpu1'):break
                time.sleep(5)
            save_new(B/'gpu-complete.json',{'at_utc':now(),'n_images':900,'expected_answers':108000})
            log=(B/'analysis.log').open('x',encoding='utf-8');logs.append(log)
            cmd=[str(ROOT/'envs/core/python.exe'),'-B',str(B/'analyze.py')]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);children['analysis']=proc
            while proc.poll() is None:check_time(deadline);time.sleep(5)
            assert proc.returncode==0,'Stage B analysis failed'
            result=read(B/'analysis-auto/results.json')
            save_new(B/'controller-complete.json',{'at_utc':now(),'exit_code':0,'results_sha256':sha(B/'analysis-auto/results.json'),
                'pending_review_images':result['gold']['review_images'],'research_goal_complete':False,'new_optimizer_steps':0})
    except BaseException as exc:
        # Only terminate children launched by this controller; never GPU PIDs in general.
        for proc in children.values():
            if proc.poll() is None:
                try:
                    parent=psutil.Process(proc.pid)
                    owned=parent.children(recursive=True)
                    for child in owned:child.terminate()
                    proc.terminate()
                    _,alive=psutil.wait_procs(owned,timeout=5)
                    for child in alive:child.kill()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
                except psutil.NoSuchProcess:pass
        if not (B/'controller-failed.json').exists():save_new(B/'controller-failed.json',{'at_utc':now(),'error':str(exc),'automatic_retry':False})
        raise
    finally:
        for log in logs:log.close()

if __name__=='__main__':main()
