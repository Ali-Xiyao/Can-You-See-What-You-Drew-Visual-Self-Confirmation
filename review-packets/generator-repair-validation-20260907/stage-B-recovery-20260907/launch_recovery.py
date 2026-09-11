"""Explicit one-shot continuation with the original remaining time budget."""
from runtime_common import *
import os, shutil, msvcrt, psutil


def terminate_owned(proc):
    if proc.poll() is not None:
        return
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
        for child in children:
            try: child.terminate()
            except psutil.NoSuchProcess: pass
        proc.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try: child.kill()
            except psutil.NoSuchProcess: pass
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
    except psutil.NoSuchProcess:
        pass


def main():
    recovery = verify_recovery()
    p = original_plan()
    assert read(R / 'diagnostic-resolution.json')['action'] == 'quarantine_json_parse_failure_for_human'
    assert not (R / 'controller-started.json').exists(), 'No automatic duplicate launches'
    assert shutil.disk_usage(ROOT).free > 8 * 1024**3
    deadline = recovery['deadline']
    children = {}; logs = []; next_batch = 0; gpu1_observer = False

    def start(name, executable, script, args):
        log = (R / (name + '.log')).open('x', encoding='utf-8'); logs.append(log)
        cmd = [str(ROOT / f'envs/{executable}/python.exe'), '-B', '-u', str(R / script), *args]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        children[name] = proc
        save_new(R / (name + '-started.json'), {'at_utc': now(), 'pid': proc.pid, 'command': cmd})
        print(f'started {name} PID {proc.pid}', flush=True)

    def done(name):
        return name in children and children[name].poll() == 0

    try:
        with (MAIN / 'supervisor.lock').open('r+b') as lock:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            check_time(deadline)
            save_new(R / 'controller-started.json', {'at_utc': now(), 'pid': os.getpid(),
                'deadline': deadline, 'recovery_plan_sha256': sha(R / 'recovery-plan.json'),
                'original_failed_attempt': str(OLD), 'new_optimizer_steps': 0})
            start('observe-gpu0', 'showo2', 'observation_resume.py', ['--shard', '0', '--deadline', str(deadline)])
            while True:
                check_time(deadline)
                for name, proc in children.items():
                    assert proc.poll() in (None, 0), f'{name}: exit {proc.returncode}'
                active_detector = any(name.startswith('detect-') and proc.poll() is None for name, proc in children.items())
                if next_batch < len(p['batches']) and not active_detector:
                    batch = p['batches'][next_batch]
                    start('detect-' + batch['id'], 'observer', 'detector_recovery.py',
                          ['detect-batch', '--batch', batch['id'], '--deadline', str(deadline), '--device', 'cuda:1'])
                    next_batch += 1
                detectors_done = next_batch == len(p['batches']) and all(done('detect-' + b['id']) for b in p['batches'])
                if detectors_done and not gpu1_observer:
                    start('observe-gpu1', 'showo2', 'observation_resume.py', ['--shard', '1', '--deadline', str(deadline)])
                    gpu1_observer = True
                if detectors_done and done('observe-gpu0') and done('observe-gpu1'):
                    break
                time.sleep(5)
            save_new(R / 'gpu-complete.json', {'at_utc': now(), 'n_images': 900, 'expected_answers': 108000,
                'original_failed_attempt_preserved': True})
            start('analysis', 'core', 'analyze_recovery.py', [])
            while children['analysis'].poll() is None:
                check_time(deadline); time.sleep(5)
            assert done('analysis'), 'Recovery analysis failed'
            result = read(R / 'analysis-recovery/results.json')
            save_new(R / 'controller-complete.json', {'at_utc': now(), 'exit_code': 0,
                'results_sha256': sha(R / 'analysis-recovery/results.json'),
                'pending_review_images': result['gold']['review_images'],
                'research_goal_complete': False, 'new_optimizer_steps': 0})
    except BaseException as exc:
        for proc in children.values():
            terminate_owned(proc)
        if not (R / 'controller-failed.json').exists():
            save_new(R / 'controller-failed.json', {'at_utc': now(), 'error': str(exc), 'automatic_retry': False})
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
