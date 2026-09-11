"""Bounded diagnostic supervisor; no retry and no budget extension."""
from rcommon import *
import os


def main():
    inventory = read(R / 'reuse-inventory.json')
    frozen = {str(R / name): sha(R / name) for name in
              ('rcommon.py', 'diagnose_one.py', 'run_diagnostic.py', 'RECOVERY.md', 'reuse-inventory.json')}
    save_new(R / 'diagnostic-plan.json', {'at_utc': now(), 'deadline': inventory['deadline'],
        'original_plan_sha256': sha(OLD / 'plan.json'), 'frozen_files': frozen,
        'original_artifacts': inventory['original_artifacts'], 'one_attempt_only': True})
    log = (R / 'diagnostic.log').open('x', encoding='utf-8')
    cmd = [str(ROOT / 'envs/observer/python.exe'), '-B', '-u', str(R / 'diagnose_one.py')]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    save_new(R / 'diagnostic-process.json', {'at_utc': now(), 'pid': proc.pid, 'command': cmd})
    stop_at = min(inventory['deadline'], time.time() + 1200)
    try:
        while proc.poll() is None:
            if time.time() >= stop_at:
                proc.terminate(); proc.wait(timeout=20)
                raise TimeoutError('Bounded diagnostic timeout')
            time.sleep(2)
        assert proc.returncode == 0, f'diagnostic exited {proc.returncode}'
    finally:
        if proc.poll() is None:
            proc.terminate()
        log.close()
    print('Diagnostic complete; inspect diagnostic-result.json.', flush=True)


if __name__ == '__main__':
    main()
