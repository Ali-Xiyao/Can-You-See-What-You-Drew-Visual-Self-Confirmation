"""Launch the reviewed cold-start dispatcher exactly once and without a window."""
from runtime_common import *
import shutil
import psutil


def main():
    assert not (R / 'controller-process.json').exists()
    assert not (R / 'controller-started.json').exists()
    verify_recovery()
    check_time(approved_deadline())
    assert shutil.disk_usage(ROOT).free > 8 * 1024**3
    previous = R.parent / 'stage-B-recovery-20260907'
    for proc in psutil.process_iter(['name']):
        if 'python' not in (proc.info['name'] or '').lower():
            continue
        try:
            cmd = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        assert not any(str(previous).lower() in arg.lower() and arg.endswith('.py')
                       for arg in cmd), 'Prior recovery still has a Python process'
    command = [str(ROOT / 'envs/core/python.exe'), '-B', '-u', str(R / 'scheduler.py')]
    with (R / 'controller.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    save_new(R / 'controller-process.json', {'at_utc': now(), 'pid': proc.pid,
             'create_time': psutil.Process(proc.pid).create_time(), 'command': command})
    print(json.dumps({'pid': proc.pid, 'deadline': approved_deadline()}), flush=True)


if __name__ == '__main__':
    main()
