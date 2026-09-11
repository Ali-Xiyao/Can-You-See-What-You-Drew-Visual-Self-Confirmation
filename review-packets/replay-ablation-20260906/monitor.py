"""Read-only progress and the registered two-consecutive-step absence alarms."""
from shared import *


def main():
    plan=read(SIDE/'plan.json');out={'at_utc':now(),'original_main_state':read(MAIN/'state.json')['status'],
        'controller_complete':(SIDE/'controller-complete.json').exists(),
        'controller_failed':read(SIDE/'controller-failed.json') if (SIDE/'controller-failed.json').exists() else None,'tasks':{}}
    for task in plan['tasks']:
        folder=SIDE/'runs'/task['id']
        if not folder.exists():out['tasks'][task['id']]={'status':'queued'};continue
        rows={r['step']:r for p in sorted((folder/'probes').glob('step-*.json')) for r in [read(p)]}
        alarms={}
        if 0 in rows:
            for cond in ('prompt_on','prompt_off'):
                for group in ('existence_no','count_zero'):
                    base=rows[0]['summary'][cond][group]['accuracy']
                    for step in sorted(rows):
                        if step<2 or step-1 not in rows:continue
                        if all(rows[s]['summary'][cond][group]['accuracy']<=base-.2+1e-12 for s in (step-1,step)):
                            alarms[cond+':'+group]={'first_persistent_step':step,'start_step':step-1,'base_accuracy':base};break
        latest=max(rows,default=-1);training=folder/'training.jsonl'
        trained=lines(training) if training.exists() else []
        full={p.stem:sum(1 for _ in p.open(encoding='utf-8')) for p in (folder/'measurements').glob('step-*.jsonl')}
        out['tasks'][task['id']]={'status':'complete' if (folder/'complete.json').exists() else 'running',
            'optimizer_steps':len(trained),'latest_probe_step':latest,'full_observation_rows':full,
            'last_probe_accuracy':{c:{g:x['accuracy'] for g,x in d.items()} for c,d in rows[latest]['summary'].items()} if latest>=0 else None,
            'absence_alarms':alarms}
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
