from common import *
import subprocess,shutil

def main():
    p=verify();assert not (SIDE/'controller-started.json').exists()
    assert shutil.disk_usage(ROOT).free>8*1024**3
    recovery=read(ROOT/'review-packets/factual-diagnostics-20260906/continuation/recovery-expected.json')
    env=os.environ.copy();env.update(recovery['environment_allowlist'])
    for name in recovery['environment_remove']:env.pop(name,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    import msvcrt
    children=[];logs=[];started=time.time();deadline=started+p['phase_A_max_seconds']
    try:
        with (MAIN/'supervisor.lock').open('r+b') as lock:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            save_new(SIDE/'controller-started.json',{'at_utc':now(),'pid':os.getpid(),'deadline':deadline,'phase':'A','plan_sha256':sha(SIDE/'plan.json')})
            for script in ('generate_a.py','measure_a.py'):
                log=(SIDE/(script+'.log')).open('x',encoding='utf-8');logs.append(log)
                cmd=[str(ROOT/'envs/showo2/python.exe'),'-B','-u',str(SIDE/script),'--deadline',str(deadline)]
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                children.append((script,proc));save_new(SIDE/(script+'-started.json'),{'at_utc':now(),'pid':proc.pid,'command':cmd})
            while any(proc.poll() is None for _,proc in children):
                main_paused();assert time.time()<deadline,'Phase A budget exhausted'
                for script,proc in children:assert proc.poll() in (None,0),f'{script} failed with exit {proc.returncode}'
                time.sleep(5)
            for script,proc in children:assert proc.returncode==0,script
            log=(SIDE/'analysis.log').open('x',encoding='utf-8');logs.append(log)
            cmd=[str(ROOT/'envs/core/python.exe'),'-B',str(SIDE/'analyze_a.py')]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            children.append(('analyze_a.py',proc))
            while proc.poll() is None:
                assert time.time()<deadline;time.sleep(2)
            assert proc.returncode==0,'CPU analysis/validation failed'
            save_new(SIDE/'controller-complete.json',{'at_utc':now(),'phase':'A','exit_code':0,'stage_B_started':False,'results_sha256':sha(SIDE/'stage-A-results.json'),'research_goal_complete':False})
    except BaseException as exc:
        for _,proc in children:
            if proc.poll() is None:proc.terminate()
        for _,proc in children:
            if proc.poll() is None:proc.wait(timeout=30)
        if not (SIDE/'controller-failed.json').exists():save_new(SIDE/'controller-failed.json',{'at_utc':now(),'error':str(exc),'automatic_retry':False})
        raise
    finally:
        for log in logs:log.close()

if __name__=='__main__':main()
