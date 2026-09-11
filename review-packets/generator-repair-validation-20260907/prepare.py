from common import *
import random,itertools,math,statistics
import numpy as np
from selfsight.v4.spec import canonical_noun
from selfsight.v4.tasks import _check,PLAUSIBLE_COLORS,COLOR_WORDS


def fresh_specs(old):
    existing={tuple(sorted({canonical_noun(o['object']) for o in r['objects']})) for r in old}
    colors={}
    for row in old:
        for o in row['objects']:
            noun=canonical_noun(o['object']);color=o['color']
            if color in COLOR_WORDS and color in PLAUSIBLE_COLORS.get(noun,{color}):colors.setdefault(noun,set()).add(color)
    made=[];used=set(existing)
    for size in (2,3):
        pairs=sorted(itertools.combinations(sorted(colors),size),key=lambda ns:digest(['new-noun-set',size,ns]))
        chosen=[ns for ns in pairs if ns not in used][:83]
        assert len(chosen)==83
        for index,nouns in enumerate(chosen):
            rng=random.Random(seed('scene',nouns));ordered=list(nouns);rng.shuffle(ordered)
            if size==2 and ordered[0]=='leaf':ordered.reverse()
            objects=[{'object':noun,'color':rng.choice(sorted(colors[noun])),'count':2 if size==2 and j==0 else 1} for j,noun in enumerate(ordered)]
            def phrase(o):
                noun=o['object'];plural=noun+('es' if noun.endswith(('s','x','ch','sh')) else 's')
                if noun=='tomato':plural='tomatoes'
                return f"{'two' if o['count']==2 else 'one'} {o['color']} {plural if o['count']==2 else noun}"
            spec={'spec_id':f'grv{size}-{index:04d}','prompt':', '.join(map(phrase,objects)),'objects':objects,'relations':[],
                  'surface':'','metadata':{'author':'deterministic_recombination','source':'old_accepted_object_color_vocabulary','seed':20260907}}
            _check(spec,size);used.add(nouns)
            made.append({'split':'A' if index<8 else 'B','spec':spec,'noun_set':list(nouns),'scene_sha256':digest(sorted((o['object'],o['color'],o['count']) for o in objects))})
    return sorted(made,key=lambda r:(r['split'],digest(r['spec']['spec_id']))),sorted(colors),existing


def main():
    main_paused();assert not (SIDE/'plan.json').exists()
    inputs={}
    def track(path):
        path=Path(path).resolve();inputs[str(path)]=sha(path);return path
    old=[r for name in ('2plus1','1plus1plus1') for r in lines(track(ROOT/f'data/v4/corpus_{name}.jsonl'))]
    prior=read(track(PRIOR/'plan.json'));result=read(track(PRIOR/'results.json'))
    specs,vocab,existing=fresh_specs(old)
    for row in specs:
        row.update(questions(row['spec'],vocab))
        row['sampling_seeds']=[seed('generation',row['spec']['spec_id'],i) for i in range(2 if row['split']=='A' else 6)]
    corpus={'created_utc':now(),'source_old_specs':len(old),'old_noun_sets':len(existing),'vocabulary':vocab,'rows':specs,
            'counts':{'A_scenes':16,'A_per_model_images':32,'base_repeats':8,'B_scenes':150,'B_base_images':900},
            'novelty':'No shared unordered canonical noun set with any old formal spec or the other new split; new compositional sampling distribution.'}
    save_new(SIDE/'corpus.json',corpus);track(SIDE/'corpus.json')
    scheduled=read(track(PRIOR/'schedule.json'));training={}
    for rd in scheduled.values():
        for r in rd['generation']['naive']:training.setdefault(r['prompt_id'],r)
    training=sorted(training.values(),key=lambda r:digest(['loss-training',r['prompt_id']]))[:16]
    assert len(training)==16
    for r in training:track(r['image_path'])
    save_new(SIDE/'loss-training.json',training);track(SIDE/'loss-training.json')
    models=[]
    for name in ('base','naive-original','naive-balanced_absence','naive-zero_weight'):
        cp=Path(prior['base']['checkpoint']['path']) if name=='base' else PRIOR/'runs'/name/'checkpoints/step-00032'
        manifest=read(track(cp/'manifest.json'))
        assert manifest['step']==(0 if name=='base' else 32)
        for filename,value in manifest['files'].items():assert sha(cp/filename)==value
        track(cp/'adapter.pt')
        pd=prior['base']['adapter_parameter_digest'] if name=='base' else read(track(PRIOR/'runs'/name/'complete.json'))['parameter_digest']
        models.append({'id':name,'step':manifest['step'],'checkpoint':str(cp.resolve()),'parameter_digest':pd,'adapter_sha256':sha(cp/'adapter.pt')})
    source_changes=[r['original']['top_correct_rate']-b['original']['top_correct_rate'] for r,b in zip(result['per_pool']['naive-original']['32']['prompt_blank'],result['per_pool']['naive-original']['0']['prompt_blank']) if r['primary_scene_disjoint']]
    x=np.asarray(source_changes);d=x[np.random.default_rng(20260907).integers(len(x),size=(5000,len(x)))].mean(1);lo,hi=np.quantile(d,[.025,.975])
    precision={'created_utc':now(),'n_frozen':150,'candidate_images_per_scene':6,'old_half_width':float((hi-lo)/2),
        'rough_scaled_n':math.ceil(14*((hi-lo)/2/.05)**2),'scenario_sd':{str(s):{'half_width_at150':1.96*s/math.sqrt(150),'n_for_half_width5pp':math.ceil((1.96*s/.05)**2)} for s in (.25,.35,.5)},
        'human_workload_scenarios':{'5_percent':45,'10_percent':90,'20_percent':180},'precision_guaranteed':False,'automatic_sample_size_increase':False}
    save_new(SIDE/'precision-plan.json',precision);track(SIDE/'precision-plan.json')
    clips={}
    for name in ('naive-original','naive-balanced_absence','naive-zero_weight'):
        rs=lines(track(PRIOR/'runs'/name/'training.jsonl'));norms=[r['gradient_norm'] for r in rs]
        clips[name]={'mean_norm':statistics.mean(norms),'median_norm':statistics.median(norms),'first8_mean':statistics.mean(norms[:8]),
            'clipped_steps':sum(n>1 for n in norms),'mean_gradient_scale':statistics.mean(min(1,1/(n+1e-6)) for n in norms),
            'per_step':[{'step':r['step'],'total_norm_before_clip':r['gradient_norm'],'clip_coefficient':min(1,1/(r['gradient_norm']+1e-6)),'generation_component_norm':None} for r in rs]}
    save_new(SIDE/'historical-clipping.json',{'rows':clips,'interpretation':'Gradient coefficient only; cannot identify parameter update fraction or separate gradient components from the logged total.'})
    for f in SIDE.glob('*.py'):track(f)
    track(SIDE/'PROTOCOL.md')
    for name in ('src/selfsight/backbones/showo2.py','src/selfsight/training/checkpoint.py','src/selfsight/v4/probe.py','src/selfsight/v4/observe.py','src/selfsight/v4/spec.py','src/selfsight/v4/tasks.py','src/selfsight/data/questions.py',
                 'configs/v4_decoupling_pilot.yaml','configs/backbones/showo2_1p5b.yaml', 'configs/models.lock.yaml',
                 'runs/readiness/showo2-1p5b/a4-lora-targets-r1.json'):
        track(ROOT/name)
    plan={'created_utc':now(),'models':models,'config_path':prior['config_path'],'targets_path':prior['targets_path'],
        'input_sha256':inputs,'phase_A_max_seconds':21600,'stage_B_scenes':150,'stage_B_not_automatically_started':True,
        'conditions':['prompt_on','prompt_blank','prompt_off'],'generation_device':'cuda:1','measurement_device':'cuda:0',
        'image_response_margin':1/255,'loss_response_margin':1e-6,'minimum_changed_scenes':8,'bootstrap_resamples':5000,
        'seed':20260907,'research_goal_complete':False}
    save_new(SIDE/'plan.json',plan)
    print(json.dumps({'counts':corpus['counts'],'precision':precision,'models':[m['id'] for m in models]},indent=2))

if __name__=='__main__':main()
