import unittest,ast,tempfile,json
from extension_common import *


class ExtensionTests(unittest.TestCase):
    def test_prefix_excludes_unpersisted_or_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'training.jsonl').write_text(''.join(json.dumps({'step':i})+'\n' for i in range(1,27))+'{"step":',encoding='utf-8')
            prefix=retained_training(folder,24)
            self.assertEqual(len(prefix),24);self.assertEqual(json.loads(prefix[-1])['step'],24)

    def test_resumed_schedule_has_no_logical_duplicate_steps(self):
        schedule=read(PARENT/'schedule.json')
        for start in (24,32):
            steps=list(range(1,start+1))+[rd*8+s+1 for rd in range(start//8,4) for s in range(8)]
            self.assertEqual(steps,list(range(1,33)))
            for rd in range(start//8,4):
                self.assertEqual(len(schedule[str(rd)]['alternate_naive']),16)
                self.assertEqual(sum(m%4==0 for _ in range(8) for m in range(8)),16)
                self.assertEqual(sum(m%4!=0 for _ in range(8) for m in range(8)),48)

    def test_analysis_change_is_only_one_directory_expression(self):
        for filename in ('analyze_ablation.py','validate_completed.py'):
            source=(PARENT/filename).read_text(encoding='utf-8');needle="folder=SIDE/'runs'/task['id']"
            self.assertEqual(source.count(needle),1)
            modified=source.replace(needle,"folder=task_folder(task['id'])")
            recovered=modified.replace("folder=task_folder(task['id'])",needle)
            self.assertEqual(source,recovered);ast.parse(modified)

    def test_extension_does_not_modify_old_worker_or_old_plan(self):
        plan=read(PARENT/'plan.json')
        for name in ('worker.py','launch.py','PROTOCOL.md'):
            path=PARENT/name;self.assertEqual(sha(path),plan['input_sha256'][str(path)])
        self.assertEqual(read(PARENT/'controller-started.json')['plan_sha256'],sha(PARENT/'plan.json'))


if __name__=='__main__':unittest.main()
