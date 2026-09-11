import itertools,copy
import numpy as np
import pytest
from bcommon import *
from stats import *

def negative(noun='apple'):
    return {'atom_id':'s:diagnostic-absence:'+noun,'question_id':'s:diagnostic-absence:'+noun+':count','family':'count',
        'text':f'How many {noun} objects are in this picture? Answer with a single whole number only.'}

def test_negative_count_actual_objects_not_zero():
    q=negative();old=copy.deepcopy(q)
    v={'resolution':'agreed','detections':[{'object':'apple','color':'red'},{'object':'apple','color':'green'}]}
    assert fact(q,v).answer=='2' and q==old
    # The legacy parser really would count the wrong category without this adapter.
    from selfsight.v4.factual_truth import factual_answer
    assert factual_answer(q,v).answer=='0'

def test_all_frozen_negative_scopes_and_truth():
    from selfsight.v4.factual_truth import question_scope
    for scene in scene_rows():
        for q in scene['questions']:
            s=question_scope(parser_question(q));assert s is not None
            if ':diagnostic-absence:' in q['atom_id']:
                noun=q['atom_id'].split(':diagnostic-absence:')[1]
                assert s.noun==noun and s.colour is None
                v={'resolution':'agreed','detections':[{'object':noun,'color':'red'}]*2}
                assert fact(q,v).answer==('2' if q['family']=='count' else 'yes')

def test_disputed_and_pending_facts_stay_unknown():
    q=negative();v={'resolution':'agreed_verdict','image_correct':False,'detections':[],
        'disputed':[{'object':'apple','color':'red'}]}
    assert not fact(q,v).known and image_gold(v)==0
    v['resolution']='pending_human';assert image_gold(v) is None and not fact(q,v).known
    assert not fact(q,None).known
    v={'resolution':'agreed','detections':[{'object':'unnameable','color':'red'}]}
    assert not fact(q,v).known

def test_scope_colour_and_template_guard():
    q={'family':'count','atom_id':'s:count:0','text':'How many red books are in this picture? Answer with a single whole number only.'}
    v={'resolution':'agreed','detections':[{'object':'notebook','color':'red'},{'object':'book','color':'blue'}]}
    assert fact(q,v).answer=='1'
    q=negative();q['text']=q['text'].replace('apple','pear')
    with pytest.raises(AssertionError):parser_question(q)

def test_linear_unknown_bounds_exhaustive():
    rng=np.random.default_rng(14)
    for _ in range(40):
        a=rng.integers(0,4,6);b=rng.integers(0,4,6)
        w=top_weights(a)-top_weights(b);ys=[1,None,0,None,None,1]
        all_values=[]
        for fill in itertools.product((0,1),repeat=3):
            y=[1,fill[0],0,fill[1],fill[2],1];all_values.append(float(np.dot(w,y)))
        assert weighted_bounds(w,ys)==pytest.approx([min(all_values),max(all_values)])

def test_same_top_unknown_cancels_and_random_ties():
    w=top_weights([1]*6)
    assert list(w)==pytest.approx([1/6]*6)
    assert paired_selection(w,w,[None]*6)==[0,0]
    assert weighted_bounds(w,[1,1,0,0,0,0])==pytest.approx([1/3,1/3])
    s=bounded_summary([[0,0]]*150)
    assert s['point_identified'] and retention_decision(True,s)=='retained_equivalent'
    assert retention_decision(False,s)=='inconclusive'

def test_ability_exact_margin_counts_and_unknown_sensitivity():
    rs=[{'differences':[0,0,0],'unknown_family':0} for _ in range(50)]
    a=ability_summary(rs);assert a['retained'] and a['known_instances']==150
    assert not ability_summary(rs[:49])['retained']
    bad=[{'differences':[-1,-1,-1],'unknown_family':0} for _ in range(50)]
    assert not ability_summary(bad)['retained']
    unknown=[{'differences':[0,0,0],'unknown_family':10} for _ in range(50)]
    u=ability_summary(unknown);assert not u['retained'] and u['mean_bounds'][0] < -.7

def test_ability_membership_bounds_cover_completions():
    rows=[{'differences':[0,1],'unknown_family':1},{'differences':[],'unknown_family':1},
          {'differences':[-1],'unknown_family':1}]
    bound=ability_summary(rows)['mean_bounds']
    # Each unknown can be outside this truth group, or enter with any difference.
    for assignment in itertools.product((None,-1,0,1),repeat=3):
        values=[]
        for r,a in zip(rows,assignment):
            ds=r['differences']+([] if a is None else [a])
            if ds:values.append(np.mean(ds))
        mean=np.mean(values)
        assert bound[0]-1e-12 <= mean <= bound[1]+1e-12

def test_context_rule_requires_equivalence_and_both_gains():
    on=bounded_summary([[0,0]]*150);br=bounded_summary([[.2,.2]]*150);bo=bounded_summary([[.2,.2]]*150)
    assert context_decision(on,br,bo)
    on=bounded_summary([[-.2,.2]]*150)
    assert not context_decision(on,br,bo)

def test_fixed_corpus_sizes_and_generation_seed_uniqueness():
    scenes=scene_rows();assert len(scenes)==150
    assert sum(len(s['spec']['objects'])==2 for s in scenes)==75
    assert len({s['scene_sha256'] for s in scenes})==150
    assert sum(len(s['questions'])*6*4*3 for s in scenes)==108000
    assert all(len(s['sampling_seeds'])==6 and len(set(s['sampling_seeds']))==6 for s in scenes)

def test_full_pool_aggregation_with_known_selector_changes():
    from analyze import selection
    scenes={r['spec']['spec_id']:r for r in scene_rows()}
    models=['base','naive-original','naive-balanced_absence','naive-zero_weight']
    conds=['prompt_on','prompt_blank','prompt_off']
    p={'models':[{'id':m} for m in models],'conditions':conds}
    images={};verified={};scores={}
    patterns={'base':[1,1,0,0,0,0],'naive-original':[0,1,0,0,0,0],
        'naive-balanced_absence':[1,0,0,0,0,0],'naive-zero_weight':[1,1,0,0,0,0]}
    for sid in scenes:
        for i in range(6):
            path=f'{sid}:{i}';images[sid,i]={'image_path':path,'seed':5-i,'candidate_index':i}
            verified[path]={'resolution':'agreed','image_correct':i==0}
            for m in models:
                for c in conds:scores[m,c,sid,i]={'original':patterns[m][i],'extended':patterns[m][i]}
    s,rows=selection(p,scenes,images,verified,scores)
    assert len(rows)==1800
    assert s['random_baseline']['point']['mean']==pytest.approx(1/6)
    assert s['summaries']['base:prompt_blank']['top_correct_rate']['point']['mean']==pytest.approx(.5)
    # Minimum seed selects the wrong tied candidate; uniform ties retain .5.
    assert s['summaries']['base:prompt_blank']['fixed_seed_correct_rate']['point']['mean']==0
    assert s['paired_changes']['naive-original:prompt_blank']['point']['mean']==-.5
    assert s['paired_changes']['naive-balanced_absence:prompt_blank']['point']['mean']==.5
    assert s['paired_changes']['naive-zero_weight:prompt_blank']['point']['mean']==0

def test_source_compile_without_pyc():
    for path in B.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
