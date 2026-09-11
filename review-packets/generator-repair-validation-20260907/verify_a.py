"""Post-completion independent integrity and endpoint arithmetic checks."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
from PIL import Image
from datetime import datetime,timezone

SIDE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=read(SIDE/'plan.json');r=read(SIDE/'stage-A-results.json');done=read(SIDE/'controller-complete.json')
    assert done['exit_code']==0 and done['results_sha256']==sha(SIDE/'stage-A-results.json')
    assert r['plan_sha256']==sha(SIDE/'plan.json')
    frozen={s['spec']['spec_id']:s for s in read(SIDE/'corpus.json')['rows'] if s['split']=='A'}
    images={};losses={};image_count=loss_count=answer_count=cycle_count=0
    for model in plan['models']:
        name=model['id'];folder=SIDE/'generation'/name;g=read(folder/'complete.json')
        assert g['parameter_digest_before']==g['parameter_digest_after']==model['parameter_digest']
        assert g['manifest_sha256']==sha(folder/'images.jsonl')
        images[name]={}
        for row in rows(folder/'images.jsonl'):
            sid=row['spec_id'];idx=row['candidate_index'];scene=frozen[sid]
            assert row['prompt']==scene['spec']['prompt'] and row['spec']==scene['spec']
            assert row['seed']==scene['sampling_seeds'][idx] and row['file_sha256']==sha(Path(row['image_path']))
            images[name][sid,idx]=row['image_path'];image_count+=1
        assert len(images[name])==32
        folder=SIDE/'measurement'/name;v=read(folder/'complete.json');lc=read(folder/'loss-complete.json')
        assert v['parameter_digest_before']==v['parameter_digest_after']==lc['parameter_digest']==model['parameter_digest']
        assert v['observations_sha256']==sha(folder/'observations.jsonl') and v['cycle_sha256']==sha(folder/'cycle.jsonl')
        assert lc['data_sha256']==sha(folder/'loss.jsonl')
        losses[name]={(s['cohort'],s['scene_key'],s['noise_index']):s for s in rows(folder/'loss.jsonl')};assert len(losses[name])==64;loss_count+=64
        obs=rows(folder/'observations.jsonl');assert len(obs)==384
        for row in obs:
            scene=frozen[row['spec_id']]
            assert row['image_path']==images.get(row['generator'],{}).get((row['spec_id'],row['candidate_index']),row['image_path'])
            assert row['n_original']==scene['n_original']
            assert len(row['observation']['answers'])==len(scene['questions'])
            assert all(not a.get('error') and not a['abstain'] for a in row['observation']['answers'])
            answer_count+=len(row['observation']['answers'])
        cycle=rows(folder/'cycle.jsonl');assert len(cycle)==128;cycle_count+=128
    for model in plan['models'][1:]:
        name=model['id'];values=[]
        for sid in sorted(frozen):
            rms=[]
            for idx in (0,1):
                with Image.open(images['base'][sid,idx]) as f:a=np.asarray(f.convert('RGB'),dtype=np.float64)
                with Image.open(images[name][sid,idx]) as f:b=np.asarray(f.convert('RGB'),dtype=np.float64)
                rms.append(float(np.sqrt(np.mean((a-b)**2))/255))
            values.append(statistics.mean(rms))
        assert abs(statistics.mean(values)-r['output_movement'][name]['rmse_by_scene']['mean'])<1e-7
        for cohort in ('training','new_base'):
            ds=[]
            for key,a in losses['base'].items():
                b=losses[name][key]
                assert a['image_sha256']==b['image_sha256'] and a['latent_seed']==b['latent_seed']
                if key[0]==cohort:ds.append(b['loss']-a['loss'])
            assert len(ds)==32
            assert abs(statistics.mean(ds)-r['loss_response'][name][cohort]['signed_difference']['mean'])<1e-12
    for name in images:
        for row in rows(SIDE/'measurement'/name/'observations.jsonl'):
            assert row['image_path']==images[row['generator']][row['spec_id'],row['candidate_index']]
    repeats=read(SIDE/'generation/base/repeats.json');assert len(repeats)==8
    for row in repeats:
        with Image.open(row['source_image']) as f:a=np.asarray(f.convert('RGB'))
        with Image.open(row['image_path']) as f:b=np.asarray(f.convert('RGB'))
        assert np.array_equal(a,b)
    assert all(s['base_repeat_loss']==s['loss'] for s in losses['base'].values())
    assert (image_count,loss_count,answer_count,cycle_count)==(128,256,15360,512)
    output={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_integrity_and_endpoint_arithmetic',
        'images':128,'identical_base_repeats':8,'paired_loss_instances':256,'identical_base_loss_repeats':64,
        'observation_answers':15360,'cycle_scores':512,'results_sha256':sha(SIDE/'stage-A-results.json'),
        'limitations':'Not an independent population or image-quality validation; no D* or D_g finding.'}
    with (SIDE/'stage-A-validation.json').open('x',encoding='utf-8') as f:json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
