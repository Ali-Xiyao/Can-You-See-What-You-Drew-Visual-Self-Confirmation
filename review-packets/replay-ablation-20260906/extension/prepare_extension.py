from extension_common import *


def main():
    parent=read(PARENT/'plan.json');old=read(PARENT/'controller-started.json')
    assert not (EXT/'plan.json').exists();verify_inputs(parent)
    paths=list(EXT.glob('*.py'))+[EXT/'BUDGET_EXTENSION.md',PARENT/'analyze_ablation.py',PARENT/'validate_completed.py',PARENT/'analysis-test-freeze.json']
    save_new(EXT/'plan.json',{'created_utc':now(),'authorization':'User explicitly asked to extend the hard deadline after the partial results were known.',
        'target_task':'naive-alternate_ids','old_deadline':old['deadline'],'new_deadline':min(old['deadline']+7200,parent['deadline']),
        'extension_seconds':7200,'parent_plan_sha256':sha(PARENT/'plan.json'),'input_sha256':{str(p):sha(p) for p in paths},
        'allowed_resume_steps':[24,32],'final_optimizer_step':32,'same_device':'cuda:1','required_restored_probe_reproduction':{'matches':200,'total':200},
        'criteria_changed':False,'sample_schedule_changed':False,'research_goal_complete':False})
    print('Budget extension and immutable source hashes recorded; no old records modified.')


if __name__=='__main__':main()
