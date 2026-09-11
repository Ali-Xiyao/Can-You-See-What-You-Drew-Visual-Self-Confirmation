from common import *
import numpy as np
from PIL import Image
from collections import defaultdict

def interval(values):
    x=np.asarray(values,dtype=float);assert len(x)>0 and np.isfinite(x).all()
    draws=x[np.random.default_rng(20260907).integers(len(x),size=(5000,len(x)))].mean(axis=1)
    return {'n_scenes':len(x),'mean':float(x.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'ci90':np.quantile(draws,[.05,.95]).tolist(),'lower_one_sided95':float(np.quantile(draws,.05))}

def image_difference(a,b):
    with Image.open(a) as f:x=np.asarray(f.convert('RGB'),dtype=np.float32)/255
    with Image.open(b) as f:y=np.asarray(f.convert('RGB'),dtype=np.float32)/255
    assert x.shape==y.shape
    return {'rmse':float(np.sqrt(np.mean((x-y)**2))),'mae':float(np.abs(x-y).mean()),
            'fraction_pixels_changed':float(np.any(x!=y,axis=-1).mean())}

def keyed_loss(rows):
    out={(r['cohort'],r['scene_key'],r['noise_index']):r for r in rows};assert len(out)==len(rows)==64
    return out

def main():
    p=verify();assert (SIDE/'generation/complete.json').exists() and (SIDE/'measurement/complete.json').exists()
    corpus=read(SIDE/'corpus.json');scene_data={r['spec']['spec_id']:r for r in corpus['rows'] if r['split']=='A'}
    images={m['id']:{(r['spec_id'],r['candidate_index']):r for r in lines(SIDE/'generation'/m['id']/'images.jsonl')} for m in p['models']}
    base=images['base'];assert len(base)==32
    repeat=read(SIDE/'generation/base/repeats.json')
    repeat_metrics=[image_difference(r['source_image'],r['image_path']) for r in repeat];assert len(repeat_metrics)==8
    rmax=max(r['rmse'] for r in repeat_metrics);threshold=rmax+p['image_response_margin']
    out={'created_utc':now(),'plan_sha256':sha(SIDE/'plan.json'),'stage':'A','output_movement':{},'loss_response':{},
         'crossed_scores':{},'crossed_cycle':{},'fixed_base_image_answer_drift':{},'base_repeat':{'n':8,'max_rmse':rmax,'image_threshold':threshold,'rows':repeat_metrics},
         'limitations':['16 new compositional scenes, one trained seed, descriptive intervals.','No image quality verdict or D*/D_g conclusion.','No model weights updated in this experiment.']}
    for m in p['models']:
        name=m['id'];assert images[name].keys()==base.keys()
        complete=read(SIDE/'generation'/name/'complete.json');assert complete['manifest_sha256']==sha(SIDE/'generation'/name/'images.jsonl')
        assert complete['parameter_digest_before']==complete['parameter_digest_after']==m['parameter_digest']
        for row in images[name].values():assert sha(row['image_path'])==row['file_sha256']
        if name=='base':continue
        values=[];details=[]
        for sid in sorted(scene_data):
            pair=[image_difference(base[sid,index]['image_path'],images[name][sid,index]['image_path']) for index in (0,1)]
            values.append(float(np.mean([r['rmse'] for r in pair])));details.append({'spec_id':sid,'seeds':pair})
        metric=interval(values);changed=sum(x>threshold for x in values)
        out['output_movement'][name]={'rmse_by_scene':metric,'scenes_above_threshold':changed,'threshold':threshold,
            'measurable_output_change':bool(metric['lower_one_sided95']>threshold and changed>=p['minimum_changed_scenes']),'details':details}
    baseloss=keyed_loss(lines(SIDE/'measurement/base/loss.jsonl'))
    noise=max(abs(r['loss']-r['base_repeat_loss']) for r in baseloss.values());out['base_repeat']['max_absolute_loss_difference']=noise
    for m in p['models']:
        name=m['id'];current=keyed_loss(lines(SIDE/'measurement'/name/'loss.jsonl'));assert current.keys()==baseloss.keys()
        cohorts=defaultdict(lambda:defaultdict(list))
        for k,b in baseloss.items():
            a=current[k];assert a['latent_seed']==b['latent_seed'] and a['image_sha256']==b['image_sha256']
            cohorts[k[0]][k[1]].append(a['loss']-b['loss'])
        report={}
        all_abs=[]
        for cohort,scene_values in cohorts.items():
            signed=[float(np.mean(v)) for v in scene_values.values()];absolute=[float(np.mean(np.abs(v))) for v in scene_values.values()]
            all_abs+=absolute;report[cohort]={'signed_difference':interval(signed),'absolute_difference':interval(absolute)}
        report['all_scene_absolute_difference']=interval(all_abs)
        report['measurable_target_response']=bool(report['all_scene_absolute_difference']['lower_one_sided95']>noise+p['loss_response_margin'])
        out['loss_response'][name]=report
    answer_maps={};validated_answers=0
    from selfsight.data.questions import normalize_answer
    from selfsight.schemas import AtomicQuestion
    from selfsight.v4.observe import PROMPTED_PREAMBLE
    from dataclasses import replace
    for m in p['models']:
        name=m['id'];folder=SIDE/'measurement'/name;complete=read(folder/'complete.json')
        assert complete['observations_sha256']==sha(folder/'observations.jsonl') and complete['cycle_sha256']==sha(folder/'cycle.jsonl')
        assert complete['parameter_digest_before']==complete['parameter_digest_after']==m['parameter_digest']
        rows=lines(folder/'observations.jsonl');assert len(rows)==384
        unique={(r['generator'],r['spec_id'],r['candidate_index'],r['condition']) for r in rows};assert len(unique)==384
        score=defaultdict(lambda:defaultdict(list));answers={};bad=0
        for r in rows:
            scene=scene_data[r['spec_id']];obs=r['observation'];assert len(obs['answers'])==len(scene['questions'])
            assert obs['rgb_sha256']==images[r['generator']][r['spec_id'],r['candidate_index']]['rgb_sha256']
            hits=[]
            for a,qd in zip(obs['answers'],scene['questions']):
                q=AtomicQuestion.from_dict(qd)
                if r['condition']!='prompt_off':q=replace(q,text=PROMPTED_PREAMBLE.format(prompt=scene['spec']['prompt'] if r['condition']=='prompt_on' else '',question=q.text))
                assert a['question_id']==q.question_id and normalize_answer(a['raw_answer'],q)==a['normalized_answer']
                hits.append(int(not a.get('error') and not a['abstain'] and a['normalized_answer']==q.expected_answer));bad+=bool(a.get('error') or a['abstain']);validated_answers+=1
                answers[r['generator'],r['spec_id'],r['candidate_index'],r['condition'],q.question_id]=a['normalized_answer']
            score[r['generator'],r['condition']][r['spec_id']].append(sum(hits[:scene['n_original']])/scene['n_original'])
        answer_maps[name]=answers
        out['crossed_scores'][name]={'error_or_abstain_answers':bad,'by_generator_and_context':{
            f'{g}:{c}':interval([float(np.mean(v)) for v in scenes.values()]) for (g,c),scenes in score.items()}}
        cyc=defaultdict(lambda:defaultdict(list))
        for r in lines(folder/'cycle.jsonl'):cyc[r['generator']][r['spec_id']].append(r['cycle'])
        out['crossed_cycle'][name]={g:interval([float(np.mean(v)) for v in scenes.values()]) for g,scenes in cyc.items()}
    for name,a in answer_maps.items():
        out['fixed_base_image_answer_drift'][name]={}
        for cond in p['conditions']:
            keys=[k for k in a if k[0]=='base' and k[3]==cond]
            out['fixed_base_image_answer_drift'][name][cond]={'total':len(keys),'changed_normalized':sum(a[k]!=answer_maps['base'][k] for k in keys)}
    out['validated_observation_answers']=validated_answers;assert validated_answers==15360
    out['B_recommended_for_fixed_image_validation']=any(v['measurable_output_change'] for v in out['output_movement'].values())
    out['long_training_authorized_by_result']=False
    save_new(SIDE/'stage-A-results.json',out)
    print(json.dumps({'output_movement':{k:{x:v[x] for x in ('measurable_output_change','scenes_above_threshold','threshold','rmse_by_scene')} for k,v in out['output_movement'].items()},
        'loss_response':{k:v['measurable_target_response'] for k,v in out['loss_response'].items()},'B_recommended':out['B_recommended_for_fixed_image_validation'],
        'validated_answers':validated_answers},indent=2))

if __name__=='__main__':main()
