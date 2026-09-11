"""Route unchanged registered secondary analyses to the reviewed R2 run."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys
import time

R = Path(__file__).resolve().parent
S = R.parent / 'selection-opportunity-secondary-20260908'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    path = Path(path).resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        assert Path(existing.__file__).resolve() == path, 'Refusing a different module with the same name'
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def bind_route(secondary, recovery_dir=R):
    """Override the bound inventory default explicitly; statistics stay original."""
    recovery_dir = Path(recovery_dir).resolve()
    original_inventory = getattr(secondary, '_r2_original_inventory', secondary.inventory)
    secondary._r2_original_inventory = original_inventory
    secondary.R = recovery_dir

    def inventory(plan, recovery_dir=None):
        if recovery_dir is not None:
            assert Path(recovery_dir).resolve() == secondary.R, 'Inventory route must remain R2'
        return original_inventory(plan, recovery_dir=secondary.R)

    secondary.inventory = inventory
    return secondary


def prepare():
    secondary = load_module('analyze_secondary', S / 'analyze_secondary.py')
    registration = secondary.registration(require_source_freeze=True)
    sys.path.insert(0, str(R))
    if 'rcommon' in sys.modules:
        assert Path(sys.modules['rcommon'].__file__).resolve() == R / 'rcommon.py'
    runtime = load_module('r2_secondary_runtime', R / 'runtime_common.py')
    recovery_plan = runtime.verify_recovery()
    assert runtime.R.resolve() == R, 'Recovery validation must target R2'
    bind_route(secondary)
    return secondary, runtime, registration, recovery_plan


def completed_results(secondary, plan, results_path=None):
    expected = secondary.R / 'analysis-recovery/results.json'
    path = expected if results_path is None else Path(results_path)
    assert path.resolve() == expected.resolve(), 'Only R2 completed primary results may be read'
    # Do not open results/per_pool, even for inspection, before all three markers.
    for marker in (secondary.R / 'controller-complete.json',
                   secondary.R / 'analysis-recovery/recovery-integrity.json',
                   secondary.R / 'analysis-recovery/integrity.json'):
        assert marker.is_file(), 'Secondary results unavailable before completion: ' + str(marker)
    # Original function verifies the three results hashes and original plan hash,
    # then calls its frozen statistical functions without any metric changes.
    return secondary.completed_results(path, plan)


def workload_report(secondary):
    workload = load_module('r2_secondary_workload', S / 'audit_review_workload.py')
    assert workload.secondary is secondary, 'Workload must share the routed secondary module'
    return workload.audit()


def produce(mode, results_path=None):
    secondary, runtime, registration, recovery_plan = prepare()
    plan = secondary.read(secondary.B / 'plan.json')
    if mode == 'inventory':
        report = secondary.inventory(plan, recovery_dir=R)
    elif mode == 'workload':
        report = workload_report(secondary)
    elif mode == 'results':
        report = completed_results(secondary, plan, results_path)
    else:
        raise ValueError(mode)
    source_paths = (S / 'analyze_secondary.py', S / 'audit_review_workload.py',
                    secondary.B / 'analyze.py', secondary.B / 'stats.py',
                    secondary.B / 'bcommon.py')
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),
        registration=registration,
        routing={'recovery_dir': str(R), 'mode': mode,
            'route_source_sha256': {str(Path(__file__).resolve()): sha(__file__)},
            'original_sources_sha256': {str(p): sha(p) for p in source_paths},
            'registration_sha256': sha(S / 'REGISTRATION.json'),
            'code_freeze_sha256': sha(S / 'CODE_FREEZE.json'),
            'recovery_plan': {'path': str(R / 'recovery-plan.json'),
                              'sha256': sha(R / 'recovery-plan.json'),
                              'original_plan_sha256': recovery_plan['original_plan_sha256']},
            'scientific_functions_unchanged': True})
    folder = R / ('secondary-' + mode + '-' +
                  datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(time.time_ns()))
    assert folder.resolve().parent == R
    folder.mkdir(exist_ok=False)
    output = folder / 'results.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return output


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inventory', action='store_true')
    group.add_argument('--workload', action='store_true')
    group.add_argument('--results', nargs='?', type=Path,
                       const=R / 'analysis-recovery/results.json')
    args = parser.parse_args()
    mode = 'inventory' if args.inventory else 'workload' if args.workload else 'results'
    output = produce(mode, args.results)
    print(json.dumps({'output': str(output), 'sha256': sha(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
