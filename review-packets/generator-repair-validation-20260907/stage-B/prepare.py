"""Freeze implementation, schedules, inputs, and detector snapshots before B."""
from bcommon import *
import statistics,shutil

def main():
    from common import verify as verify_A
    from selfsight.v4.detectors import QWEN3VL_8B, INTERNVL35_8B
    from selfsight.v4.factual_truth import question_scope
    ap=verify_A();assert not (B/'plan.json').exists() and not (B/'generation').exists()
    a=read(SIDE/'stage-A-results.json');validation=read(SIDE/'stage-A-validation.json')
    assert a['B_recommended_for_fixed_image_validation'] and validation['status']=='PASS_integrity_and_endpoint_arithmetic'
    assert validation['results_sha256']==sha(SIDE/'stage-A-results.json')
    scenes=scene_rows();assert len(scenes)==150 and sum(len(s['spec']['objects'])==2 for s in scenes)==75
    for scene in scenes:
        for q in scene['questions']:
            scope=question_scope(parser_question(q));assert scope is not None
            if ':diagnostic-absence:' in q['atom_id']:
                assert scope.noun==q['atom_id'].split(':diagnostic-absence:')[1] and scope.colour is None
    # Deterministic, family-balanced physical observation assignment before images.
    shards={'0':[],'1':[]}
    for size in (2,3):
        ids=sorted([s['spec']['spec_id'] for s in scenes if len(s['spec']['objects'])==size],key=lambda x:digest(['B-device',x]))
        shards['0']+=ids[:50];shards['1']+=ids[50:]
    batches=[{'id':f'batch-{i//25:02d}','scene_ids':[s['spec']['spec_id'] for s in scenes[i:i+25]]} for i in range(0,150,25)]
    inputs=dict(ap['input_sha256'])
    for path in [SIDE/'plan.json',SIDE/'stage-A-results.json',SIDE/'stage-A-validation.json',SIDE/'verify_a.py',*B.glob('*.py'),B/'IMPLEMENTATION.md',
        ROOT/'scripts/v4_run_pipeline.py',ROOT/'src/selfsight/v4/factual_truth.py',ROOT/'src/selfsight/v4/verifier.py',
        ROOT/'src/selfsight/v4/detectors.py',ROOT/'src/selfsight/utils/hashing.py',ROOT/'src/selfsight/schemas.py',ROOT/'src/selfsight/v4/train.py']:
        inputs[str(path.resolve())]=stream_sha(path)
    detector_hashes={};detector_stat={}
    for folder in (Path(QWEN3VL_8B),Path(INTERNVL35_8B)):
        files=sorted(f for f in folder.iterdir() if f.is_file() and (f.suffix in {'.safetensors','.json','.py','.txt','.jinja'}))
        assert len([f for f in files if f.suffix=='.safetensors'])==4
        for f in files:
            print('hashing frozen detector '+str(f),flush=True)
            before=f.stat();h=stream_sha(f);after=f.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            detector_hashes[str(f)]=h;detector_stat[str(f)]=[after.st_size,after.st_mtime_ns]
    gen=statistics.mean(r['seconds'] for r in lines(SIDE/'generation/base/images.jsonl'))
    latency=[x['latency_ms']/1000 for m in ap['models'] for r in lines(SIDE/'measurement'/m['id']/'observations.jsonl') for x in r['observation']['answers']]
    mean=statistics.mean(latency)
    estimate={'A_generation_seconds_per_image':gen,'A_observation_seconds_per_answer':mean,
        'generation_hours_at_A_rate':900*gen/3600,'observation_gpu0_hours_at_A_rate':72000*mean/3600,
        'observation_gpu1_hours_at_A_rate':36000*mean/3600,
        'detector_seconds_per_image_sensitivity':[5,10,20],
        'two_detector_hours_sensitivity':[900*2*s/3600 for s in (5,10,20)],
        'crops_loading_queue_are_additional':True,'disk_free_bytes':shutil.disk_usage(ROOT).free,
        'deadline_is_budget_not_completion_guarantee':True}
    p={'created_utc':now(),'input_sha256':inputs,'models':ap['models'],'config_path':ap['config_path'],
        'targets_path':ap['targets_path'],'conditions':ap['conditions'],'observer_revision':'07ec16589d4fc5422a74dddbbc4b2cd11e551039',
        'batches':batches,'observation_shards':shards,'detector_sha256':detector_hashes,'detector_stat':detector_stat,
        'detector_snapshot_digest':digest(detector_hashes),'max_seconds':86400,'expected_answers':108000,
        'estimate':estimate,'new_optimizer_steps':0,'D_star_test':False,'D_g_test':False}
    save_new(B/'plan.json',p)
    print(json.dumps({'plan_sha256':sha(B/'plan.json'),'estimate':estimate,'scene_counts':{k:len(v) for k,v in shards.items()}},indent=2))

if __name__=='__main__':main()
