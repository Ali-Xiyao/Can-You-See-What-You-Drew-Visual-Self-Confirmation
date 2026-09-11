"""Freeze this one continuation after code, prefix and controller validation."""
from runtime_common import *


def main():
    assert not (R / 'recovery-plan.json').exists()
    validation = read(R / 'cpu-validation-final.json')
    assert validation['passed']
    assert read(R / 'observer-preflight.json')['passed']
    for source, expected in validation['tested_source_sha256'].items():
        assert sha(source) == expected, source
    inventory = read(R / 'reuse-inventory.json')
    assert inventory['reused_observation_answers'] == 42156
    assert inventory['completed_batches'] == ['batch-00', 'batch-01']
    assert sum(row['answers'] for row in inventory['observations']) == 42156
    for row in inventory['observations']:
        assert sha(row['path']) == sha(row['source_path']) == row['sha256']
        assert Path(row['path']).stat().st_size == row['original_prefix_bytes']
    for source, expected in inventory['original_artifacts'].items():
        assert stream_sha(source) == expected, source
    plan = original_plan()
    assert sha(R / 'plan.json') == sha(OLD / 'plan.json')
    frozen = {str(path): sha(path) for path in R.glob('*.py')}
    for name in ('CONTINUATION.md', 'plan.json', 'reuse-inventory.json',
                 'budget-approval.json', 'PARSE_FAILURE_ADDENDUM.md',
                 'diagnostic-plan.json', 'diagnostic-result.json',
                 'diagnostic-resolution.json', 'observer-preflight.json',
                 'cpu-validation-final.json'):
        frozen[str(R / name)] = sha(R / name)
    save_new(R / 'recovery-plan.json', {
        'at_utc': now(), 'original_plan_sha256': sha(OLD / 'plan.json'),
        'deadline': approved_deadline(),
        'original_deadline': read(OLD / 'controller-started.json')['deadline'],
        'frozen_files': frozen, 'original_artifacts': inventory['original_artifacts'],
        'prior_recovery_plan_sha256': inventory['prior_recovery_plan_sha256'],
        'prior_failure': inventory['original_failure'],
        'reused_observation_answers': 42156, 'n_images': 900,
        'expected_answers': plan['expected_answers'],
        'completed_batches_at_resume': inventory['completed_batches'],
        'new_optimizer_steps': 0, 'thresholds_unchanged': True,
        'unresolved_facts_remain_unknown': True,
        'analysis_source': str(OLD / 'analyze.py')})
    verify_recovery()
    print(json.dumps({'recovery_plan_sha256': sha(R / 'recovery-plan.json'),
                      'deadline': approved_deadline(), 'reused_answers': 42156}), flush=True)


if __name__ == '__main__':
    main()
