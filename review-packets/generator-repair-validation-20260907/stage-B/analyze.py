"""Frozen Stage B analysis. No filtering on observed selection performance."""
from bcommon import *
from stats import *
from collections import Counter,defaultdict
from dataclasses import replace
import argparse, base64, html, re

def verifications(p, human=None):
    pipeline=pipeline_module();out={}
    from selfsight.v4.spec import SceneSpec
    from selfsight.v4.verifier import verify as verify_image
    for batch in p['batches']:
        folder=B/'generation'/batch['id'];complete=read(folder/'verification-complete.json')
        for key,h in complete['sha256'].items():
            path=folder/('verified.jsonl' if key=='verified' else f'detections.{key}.jsonl')
            assert sha(path)==h,path
        first={r['image_path']:r['detections'] for r in lines(folder/'detections.qwen3vl.jsonl')}
        second={r['image_path']:r['detections'] for r in lines(folder/'detections.internvl.jsonl')}
        original={r['image_path']:r for r in lines(folder/'verified.jsonl')}
        crops=pipeline.cached_crops(folder/'detections.crops.jsonl')
        for r in lines(folder/'manifest.jsonl'):
            path=r['image_path'];assert path in first and path in second and path in original
            assert path not in out
            if human is not None and path in human:
                v=verify_image(path,SceneSpec.from_dict(r['spec']),pipeline._Replay('qwen3vl',first,crops),
                    pipeline._Replay('internvl',second,crops),human_labels=human).to_dict()
                out[path]=v
            else:out[path]=original[path]
    assert len(out)==900
    return out

def load_answers(p, scenes, images):
    from selfsight.schemas import AtomicQuestion
    from selfsight.data.questions import normalize_answer
    from selfsight.v4.observe import PROMPTED_PREAMBLE
    from selfsight.v4.factual_truth import canonical_answer
    scores={};answers={};n_answers=0;abstentions=Counter();response_counts=Counter()
    for shard in (0,1):
        assigned=set(p['observation_shards'][str(shard)])
        for m in p['models']:
            folder=B/'observations'/f'gpu{shard}'/m['id'];done=read(folder/'complete.json')
            assert done['answers_sha256']==sha(folder/'answers.jsonl')
            assert done['parameter_digest_before']==done['parameter_digest_after']==m['parameter_digest']
            rows=lines(folder/'answers.jsonl');assert len(rows)==len(assigned)*6*3
            count=0
            for r in rows:
                sid=r['spec_id'];idx=r['candidate_index'];cond=r['condition'];scene=scenes[sid]
                assert sid in assigned and cond in p['conditions'] and r['observer']==m['id'] and r['generator']=='base'
                assert r['device']==f'cuda:{shard}' and r['scene_sha256']==scene['scene_sha256']
                image=images[sid,idx]
                assert r['image_path']==image['image_path'] and r['seed']==image['seed']
                obs=r['observation'];assert obs['rgb_sha256']==image['rgb_sha256']
                assert obs['observer_id']=='showlab/show-o2-1.5B' and obs['observer_revision']==p['observer_revision']
                assert len(obs['answers'])==len(scene['questions']) and r['n_original']==scene['n_original']
                hits=[]
                for qd,a in zip(scene['questions'],obs['answers']):
                    q=AtomicQuestion.from_dict(qd)
                    if cond!='prompt_off':q=replace(q,text=PROMPTED_PREAMBLE.format(prompt=scene['spec']['prompt'] if cond=='prompt_on' else '',question=q.text))
                    assert a['question_id']==q.question_id and not a.get('error')
                    assert normalize_answer(a['raw_answer'],q)==a['normalized_answer']
                    key=(m['id'],cond,sid,idx,q.question_id);assert key not in answers
                    value=None if a['abstain'] else canonical_answer(parser_question(qd),a['normalized_answer'])
                    answers[key]=value
                    hits.append(int(value==q.expected_answer))
                    abstentions[m['id'],cond]+=int(a['abstain'])
                    response_counts[m['id'],cond,qd['family'],str(value)]+=1
                    n_answers+=1;count+=1
                key=(m['id'],cond,sid,idx);assert key not in scores
                scores[key]={'original':sum(hits[:scene['n_original']])/scene['n_original'],'extended':sum(hits)/len(hits)}
            assert count==done['n_answers']
    assert n_answers==108000 and len(scores)==10800
    return scores,answers,{'validated_answers':n_answers,'abstentions':{':'.join(k):v for k,v in abstentions.items()},
        'response_counts':{':'.join(k):v for k,v in response_counts.items()}}

def build_facts(scenes,images,verified):
    facts={};unknown=Counter();review=[]
    for (sid,idx),im in images.items():
        v=verified[im['image_path']];reasons=[]
        if image_gold(v) is None:reasons.append('whole_image_unknown')
        for q in scenes[sid]['questions']:
            f=fact(q,v);facts[sid,idx,q['question_id']]=f
            if not f.known:unknown[f.reason]+=1;reasons.append(f.reason)
        if reasons:review.append({**im,'verification':v,'reasons':sorted(set(reasons))})
    return facts,unknown,review

def selection(p,scenes,images,verified,scores):
    ids=sorted(scenes);labels={sid:[image_gold(verified[images[sid,i]['image_path']]) for i in range(6)] for sid in ids}
    weights={};per_pool=[];summaries={};comparisons={}
    random=np.full(6,1/6)
    for m in p['models']:
        for cond in p['conditions']:
            bounds=[];randdiff=[];fixed=[];ties=[];matches=[];extbounds=[]
            for sid in ids:
                ss=[scores[m['id'],cond,sid,i]['original'] for i in range(6)]
                es=[scores[m['id'],cond,sid,i]['extended'] for i in range(6)]
                w=top_weights(ss);weights[m['id'],cond,sid]=w
                ew=top_weights(es);y=labels[sid]
                top=[i for i in range(6) if w[i]>0]
                chosen=min(top,key=lambda i:(images[sid,i]['seed'],images[sid,i]['candidate_index']))
                fw=np.zeros(6);fw[chosen]=1
                b=weighted_bounds(w,y);bounds.append(b)
                rb=paired_selection(w,random,y);randdiff.append(rb)
                fixed.append(weighted_bounds(fw,y));extbounds.append(weighted_bounds(ew,y))
                ties.append(len(top));matches.append(float(np.mean(ss)))
                per_pool.append({'model':m['id'],'condition':cond,'spec_id':sid,'original_scores':ss,'extended_scores':es,
                    'labels':y,'top_indices':top,'top_size':len(top),'all_tied':len(top)==6,'fixed_seed_choice':chosen,
                    'top_correct_rate_bounds':b,'selection_gain_bounds':rb,'base_rate_bounds':weighted_bounds(random,y),
                    'extended_top_indices':[i for i in range(6) if ew[i]>0]})
            summaries[m['id']+':'+cond]={'top_correct_rate':bounded_summary(bounds),'selection_gain':bounded_summary(randdiff),
                'fixed_seed_correct_rate':bounded_summary(fixed),'extended_top_correct_rate_descriptive':bounded_summary(extbounds),
                'mean_top_size':float(np.mean(ties)),'all_tied_pools':sum(t==6 for t in ties),
                'request_match':interval(matches)}
    for m in p['models'][1:]:
        for cond in p['conditions']:
            bs=[paired_selection(weights[m['id'],cond,sid],weights['base',cond,sid],labels[sid]) for sid in ids]
            s=bounded_summary(bs);lo,hi=s['ci95_worst_completion'];s['ci95_half_width']=(hi-lo)/2
            s['precision_target_met']=(hi-lo)/2<=.05
            comparisons[m['id']+':'+cond]=s
    contexts={}
    for a,b in [('prompt_blank','prompt_on'),('prompt_off','prompt_on'),('prompt_off','prompt_blank')]:
        contexts[a+'-'+b]=bounded_summary([paired_selection(weights['base',a,s],weights['base',b,s],labels[s]) for s in ids])
    decision=context_decision(summaries['base:prompt_on']['selection_gain'],summaries['base:prompt_blank']['selection_gain'],contexts['prompt_blank-prompt_on'])
    return {'summaries':summaries,'paired_changes':comparisons,'base_context_changes':contexts,
        'recommend_future_prompt_blank':decision,'random_baseline':bounded_summary([weighted_bounds(random,labels[s]) for s in ids])},per_pool

def ability(p,scenes,answers,facts):
    out={}
    groups={'existence_yes':('existence','yes'),'existence_no':('existence','no'),
            'count_positive':('count','positive'),'count_zero':('count','0')}
    for m in p['models'][1:]:
        model={}
        for cond in ('prompt_on','prompt_off'):
            for group,(family,target) in groups.items():
                rs=[];base_acc=[];model_acc=[]
                for sid,scene in sorted(scenes.items()):
                    ds=[];unknown=0;ba=[];ma=[]
                    for idx in range(6):
                        for q in scene['questions']:
                            if q['family']!=family:continue
                            f=facts[sid,idx,q['question_id']]
                            if not f.known:unknown+=1;continue
                            if target=='positive':member=int(f.answer)>0
                            else:member=f.answer==target
                            if not member:continue
                            b=int(answers['base',cond,sid,idx,q['question_id']]==f.answer)
                            a=int(answers[m['id'],cond,sid,idx,q['question_id']]==f.answer)
                            ds.append(a-b);ba.append(b);ma.append(a)
                    rs.append({'spec_id':sid,'differences':ds,'unknown_family':unknown})
                    if ba:base_acc.append(float(np.mean(ba)));model_acc.append(float(np.mean(ma)))
                result=ability_summary(rs)
                result['base_accuracy_known_only']=interval(base_acc) if base_acc else None
                result['model_accuracy_known_only']=interval(model_acc) if model_acc else None
                model[cond+':'+group]=result
        out[m['id']]={'groups':model,'all_groups_retained':all(x['retained'] for x in model.values())}
    return out

def make_review(folder, rows):
    folder.mkdir(exist_ok=False)
    save_new(folder/'index.json',{'created_utc':now(),'images':rows,'selection':'All image-level or predeclared-question factual unknowns; no observer answer filtering.'})
    pages=[]
    for start in range(0,len(rows),10):
        cards=[]
        for number,row in enumerate(rows[start:start+10],start+1):
            src='data:image/png;base64,'+base64.b64encode(Path(row['image_path']).read_bytes()).decode()
            cards.append('<article><h2>'+str(number)+' · '+html.escape(row['spec_id'])+' / '+str(row['candidate_index'])+
                '</h2><p><b>原请求：</b>'+html.escape(row['prompt'])+'</p><img width="432" src="'+src+'">'+
                '<p>请按图片实际内容列出物体、主色和数量；每个物体一条。无法确定则留空，继续记为未知。</p>'+
                '<textarea rows="4" data-path="'+html.escape(row['image_path'],quote=True)+'" placeholder=\'[{"object":"apple","color":"red"}]\'></textarea>'+
                '<details><summary>自动裁定记录（可选查看）</summary><pre>'+html.escape(json.dumps(row['verification'],ensure_ascii=False,indent=2))+'</pre></details></article>')
        script='''<script>function download(){const rows=[];for(const e of document.querySelectorAll('textarea')){if(!e.value.trim())continue;let d;try{d=JSON.parse(e.value);if(!Array.isArray(d)||d.some(x=>typeof x.object!=='string'||typeof x.color!=='string'))throw Error();}catch{alert('请使用物体列表 JSON；无法确定请留空。');e.focus();return;}rows.push(JSON.stringify({image_path:e.dataset.path,detections:d}));}if(!rows.length){alert('尚无裁定；空白仍为未知。');return;}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{type:'application/x-ndjson'}));a.download='human-labels.jsonl';a.click();}</script>'''
        name=f'review-{start//10+1:03d}.html';pages.append(name)
        content='<!doctype html><html lang="zh"><meta charset="utf-8"><title>B事实复核</title><style>body{font:16px sans-serif;max-width:960px;margin:32px auto}article{border-bottom:2px solid #ddd;padding:24px}textarea{width:95%}pre{white-space:pre-wrap}</style><h1>B阶段独立事实复核</h1><p>原请求仅用于辨认任务；事实记录必须以图中实际对象为准。请勿把期望对象补进图像。模糊、融合或无法数清的请保留未知。</p><button onclick="download()">导出已填写裁定</button>'+''.join(cards)+script+'</html>'
        with (folder/name).open('x',encoding='utf-8') as f:f.write(content)
    save_new(folder/'pages.json',{'pages':pages,'n_images':len(rows),'automatic_labels_added':0})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--human',type=Path);ap.add_argument('--tag',default='auto');a=ap.parse_args()
    assert re.fullmatch('[a-zA-Z0-9_-]+',a.tag)
    p=verify_plan(full=True);assert (B/'gpu-complete.json').exists()
    folder=B/('analysis-'+a.tag);assert not folder.exists()
    scenes={r['spec']['spec_id']:r for r in scene_rows()};assert len(scenes)==150
    rows=all_images(p);images={(r['spec_id'],r['candidate_index']):r for r in rows}
    from selfsight.utils.hashing import rgb_sha256
    for (sid,idx),r in images.items():
        scene=scenes[sid]
        assert r['seed']==scene['sampling_seeds'][idx] and r['prompt']==scene['spec']['prompt'] and r['spec']==scene['spec']
        assert r['scene_sha256']==scene['scene_sha256'] and r['model']=='base' and r['device']=='cuda:0'
        assert sha(r['image_path'])==r['file_sha256'] and rgb_sha256(r['image_path'])==r['rgb_sha256']
    human=None
    if a.human:
        hr=lines(a.human);human={r['image_path']:r['detections'] for r in hr}
        assert len(human)==len(hr) and set(human)<={r['image_path'] for r in rows}
        for ds in human.values():
            assert isinstance(ds,list) and all(isinstance(d.get('object'),str) and isinstance(d.get('color'),str) for d in ds)
    v=verifications(p,human);scores,answers,integrity=load_answers(p,scenes,images)
    facts,unknown,review=build_facts(scenes,images,v)
    selected,per_pool=selection(p,scenes,images,v,scores);retained=ability(p,scenes,answers,facts)
    for m in p['models'][1:]:
        for c in p['conditions']:
            s=selected['paired_changes'][m['id']+':'+c]
            s['decision']=retention_decision(retained[m['id']]['all_groups_retained'],s)
    out={'created_utc':now(),'plan_sha256':sha(B/'plan.json'),'stage':'B','n_scenes':150,'n_images':900,
        'integrity':integrity,'gold':{'known_image_labels':sum(image_gold(x) is not None for x in v.values()),
            'unknown_image_labels':sum(image_gold(x) is None for x in v.values()),'resolution_counts':dict(Counter(x['resolution'] for x in v.values())),
            'known_question_facts':sum(x.known for x in facts.values()),'unknown_question_facts':sum(not x.known for x in facts.values()),
            'unknown_reasons':dict(unknown),'review_images':len(review)},
        'selection':selected,'ability':retained,'per_pool':per_pool,
        'human_input_sha256':sha(a.human) if a.human else None,'D_star_supported':False,'D_g_supported':False,
        'limitations':['New compositional validation distribution; one training seed.','Used for configuration selection, not independent D* confirmation.',
            'Unresolved facts retain conservative bounds; no assumption of detector perfection.','No new training, natural outcome trajectory, BoN or gradient measurement.']}
    folder.mkdir();save_new(folder/'results.json',out)
    save_new(folder/'integrity.json',{'at_utc':now(),'status':'PASS_integrity','images':900,'answers':integrity['validated_answers'],
        'pool_scores':len(per_pool),'results_sha256':sha(folder/'results.json'),'detector_sha256_rechecked':True})
    make_review(folder/'human-review',review)
    report=['# B阶段固定图验证','',f'完整性核验：900张图片、{integrity["validated_answers"]}条回答、{len(per_pool)}个池级评分。',
        f'整图未知{out["gold"]["unknown_image_labels"]}/900；题级事实未知{out["gold"]["unknown_question_facts"]}/{len(facts)}；需要复核{len(review)}张。','',
        '| 模型 | 能力保留全部通过 | prompt_blank选择判定 | 配对变化最坏补全90%区间 |','|---|---|---|---|']
    for m in p['models'][1:]:
        s=selected['paired_changes'][m['id']+':prompt_blank'];ci=s['ci90_worst_completion']
        report.append(f'| {m["id"]} | {retained[m["id"]]["all_groups_retained"]} | {s["decision"]} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |')
    report+=['',f'下一版本prompt_blank条件推荐规则触发：{selected["recommend_future_prompt_blank"]}。',
        '完整条件、未知界、分题族分母及精度见results.json。D*与D_g均未在本轮测量或确认。']
    with (folder/'RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    print(json.dumps({'gold':out['gold'],'selection_decisions':{k:v['decision'] for k,v in selected['paired_changes'].items()},
        'recommend_future_prompt_blank':selected['recommend_future_prompt_blank']},indent=2))

if __name__=='__main__':main()
