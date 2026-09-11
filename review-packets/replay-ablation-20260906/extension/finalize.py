"""Run frozen statistics and validation with one explicitly documented input-path override."""
import sys
from extension_common import *


def run_frozen(filename):
    expected=read(PARENT/'analysis-test-freeze.json')['files'] if filename=='analyze_ablation.py' else read(EXT/'plan.json')['input_sha256']
    path=PARENT/filename;assert sha(path)==expected[str(path)]
    source=path.read_text(encoding='utf-8');needle="folder=SIDE/'runs'/task['id']"
    assert source.count(needle)==1
    amended=source.replace(needle,"folder=task_folder(task['id'])")
    namespace={'__name__':'__main__','__file__':str(path),'task_folder':task_folder}
    oldargv=sys.argv;sys.argv=[str(path)]
    try:exec(compile(amended,str(path), 'exec'),namespace)
    finally:sys.argv=oldargv


def main():
    verify_extension();assert read(EXT/'controller-complete.json')['exit_code']==0
    for task in read(PARENT/'plan.json')['tasks']:assert read(task_folder(task['id'])/'complete.json')['steps']==32
    if not (EXT/'analysis-input-resolution.json').exists():
        save_new(EXT/'analysis-input-resolution.json',{'at_utc':now(),'folders':{t['id']:str(task_folder(t['id'])) for t in read(PARENT/'plan.json')['tasks']},
            'only_code_substitution':"folder=SIDE/'runs'/task['id'] -> folder=task_folder(task['id'])",
            'statistics_and_criteria_unchanged':True,'source_sha256':{n:sha(PARENT/n) for n in ('analyze_ablation.py','validate_completed.py')}})
    if not (PARENT/'results.json').exists():run_frozen('analyze_ablation.py')
    if not (PARENT/'results-validation.json').exists():run_frozen('validate_completed.py')
    assert read(PARENT/'results-validation.json')['results_sha256']==sha(PARENT/'results.json')
    print('Frozen analysis and independent validation completed with disclosed continuation provenance')


if __name__=='__main__':main()
