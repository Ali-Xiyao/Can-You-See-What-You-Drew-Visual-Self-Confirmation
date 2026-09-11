"""Freeze tested recovery code and exact legacy provenance before continuation."""
from runtime_common import *
import shutil


def main():
    assert not (R / 'recovery-plan.json').exists()
    diagnostic = read(R / 'diagnostic-resolution.json')
    assert diagnostic['action'] == 'quarantine_json_parse_failure_for_human'
    inventory = read(R / 'reuse-inventory.json')
    assert (R / 'cpu-validation-final.json').exists()
    assert read(R / 'cpu-validation-final.json')['passed']
    p = original_plan()
    for item in inventory['observations']:
        assert sha(item['path']) == item['sha256'], 'Prefix changed before launch'
    shutil.copyfile(OLD / 'plan.json', R / 'plan.json')
    frozen = {str(f): sha(f) for f in sorted(R.glob('*.py'))}
    for f in ('RECOVERY.md', 'plan.json', 'reuse-inventory.json', 'diagnostic-plan.json',
              'diagnostic-result.json', 'diagnostic-resolution.json', 'cpu-validation-final.json',
              'budget-approval.json', 'PARSE_FAILURE_ADDENDUM.md'):
        frozen[str(R / f)] = sha(R / f)
    save_new(R / 'recovery-plan.json', {'at_utc': now(), 'original_plan_sha256': sha(OLD / 'plan.json'),
        'deadline': approved_deadline(), 'original_deadline': inventory['deadline'], 'frozen_files': frozen,
        'original_artifacts': inventory['original_artifacts'],
        'reused_observation_answers': 23640, 'n_images': 900, 'expected_answers': p['expected_answers'],
        'diagnostic': diagnostic, 'new_optimizer_steps': 0,
        'analysis_source': str(OLD / 'analyze.py'), 'thresholds_unchanged': True,
        'unresolved_facts_remain_unknown': True})
    print('Recovery plan frozen; user-approved deadline extension recorded.', flush=True)


if __name__ == '__main__':
    main()
