"""Start exactly one hidden recovery controller after the reviewed freeze."""
from runtime_common import *
import freeze_recovery


def main():
    assert not (R / 'controller-process.json').exists()
    freeze_recovery.main()
    with (R / 'controller.log').open('x', encoding='utf-8') as log:
        cmd = [str(ROOT / 'envs/core/python.exe'), '-B', '-u', str(R / 'launch_recovery.py')]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    save_new(R / 'controller-process.json', {'at_utc': now(), 'pid': proc.pid, 'command': cmd})
    print(json.dumps({'pid': proc.pid, 'deadline': read(R / 'recovery-plan.json')['deadline']}), flush=True)


if __name__ == '__main__':
    main()
