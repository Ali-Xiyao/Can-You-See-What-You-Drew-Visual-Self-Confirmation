from pathlib import Path
import json,hashlib,sys,time,os
from datetime import datetime,timezone
SIDE=Path(__file__).resolve().parent
ROOT=SIDE.parents[1]
PRIOR=ROOT/'review-packets/replay-ablation-20260906'
MAIN=ROOT/'runs/v4/decoupling-pilot-20260906'
sys.path.insert(0,str(ROOT/'src'))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def seed(*parts):return int(digest([20260907,*parts])[:8],16)%2147483647
def save_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def append(f,row):f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def main_paused():
    s=read(MAIN/'state.json')
    assert s['status']=='canary_complete' and s['completed_rounds']==s['through_round']==4 and not s['research_goal_complete']
    assert not (MAIN/'rounds/round-004').exists()
def verify():
    main_paused();p=read(SIDE/'plan.json')
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    return p
def wait_for(path,deadline):
    while not Path(path).exists():
        main_paused();assert time.time()<deadline,'Stage A runtime exhausted'
        if (SIDE/'controller-failed.json').exists():raise RuntimeError('Controller failed')
        time.sleep(3)

def questions(spec,vocabulary):
    from selfsight.v4.spec import SceneSpec,canonical_noun
    from selfsight.v4.probe import spec_questions
    from selfsight.schemas import AtomicQuestion,QuestionFamily,QuestionFormat,as_serializable
    original=spec_questions(SceneSpec.from_dict(spec))
    present={canonical_noun(o['object']) for o in spec['objects']}
    absent=sorted(set(vocabulary)-present,key=lambda n:digest(['absence',spec['spec_id'],n]))[:len(spec['objects'])]
    extra=[]
    for noun in absent:
        yes_first=seed('placement',spec['spec_id'],noun)%2==0
        a,b=('yes','no') if yes_first else ('no','yes')
        article='an' if noun[0] in 'aeiou' else 'a'
        atom=spec['spec_id']+':diagnostic-absence:'+noun
        extra.append(AtomicQuestion(question_id=atom+':exists',atom_id=atom,family=QuestionFamily.EXISTENCE,
            text=f'Is there {article} {noun} in this picture? Answer A or B only.\nA. {a}\nB. {b}',
            expected_answer='no',question_format=QuestionFormat.FORCED_CHOICE,choices=(a,b),choice_order_seed=seed(atom)))
        extra.append(AtomicQuestion(question_id=atom+':count',atom_id=atom,family=QuestionFamily.COUNT,
            text=f'How many {noun} objects are in this picture? Answer with a single whole number only.',
            expected_answer='0',question_format=QuestionFormat.OPEN))
    return {'n_original':len(original),'questions':[as_serializable(q) for q in (*original,*extra)]}

def make_backbone(plan,device):
    import yaml
    from selfsight.backbones.showo2 import Showo2Adapter
    from selfsight.v4.train import seed_training
    config=yaml.safe_load(Path(plan['config_path']).read_text())
    targets=read(plan['targets_path']);seed_training(20260906)
    backbone=Showo2Adapter(backbone_config=config['model']['backbone_config'],device=device,dtype=config['hardware']['precision'],lazy=False)
    l=config['training']['lora'];seed_training(20260906)
    backbone.attach_lora(target_modules=tuple(targets['target_modules']),rank=l['rank'],alpha=l['alpha'],dropout=l['dropout'],gradient_checkpointing=True)
    return backbone,config

def load_model(backbone,model,config):
    from selfsight.training.checkpoint import load_checkpoint
    from selfsight.utils.hashing import sha256_json
    from selfsight.v4.train import parameter_digest,trainable_snapshot
    state=load_checkpoint(model['checkpoint'],model=backbone.model,optimizer=None,scheduler=None,expected_config_digest=sha256_json(config))
    assert state['step']==model['step']
    actual=parameter_digest(trainable_snapshot(backbone.model));assert actual==model['parameter_digest']
    backbone.model.eval();return actual
