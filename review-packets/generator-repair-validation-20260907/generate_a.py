from common import *
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    p=verify();corpus=read(SIDE/'corpus.json');scenes=[r for r in corpus['rows'] if r['split']=='A']
    from selfsight.v4.train import parameter_digest,trainable_snapshot
    backbone,config=make_backbone(p,'cuda:1')
    save_new(SIDE/'generation/started.json',{'at_utc':now(),'pid':os.getpid(),'device':'cuda:1'})
    for model in p['models']:
        before=load_model(backbone,model,config);folder=SIDE/'generation'/model['id'];folder.mkdir(parents=True,exist_ok=False)
        manifest=[]
        with (folder/'images.jsonl').open('x',encoding='utf-8') as log:
            for scene in scenes:
                for index,s in enumerate(scene['sampling_seeds']):
                    main_paused();assert time.time()<args.deadline
                    start=time.time();c=backbone.generate_images([scene['spec']['prompt']],[s],folder,model['id'])[0]
                    r={'model':model['id'],'spec_id':scene['spec']['spec_id'],'scene_sha256':scene['scene_sha256'],
                       'candidate_index':index,'seed':s,'prompt':scene['spec']['prompt'],'image_path':c.image_path,
                       'rgb_sha256':c.rgb_sha256,'file_sha256':sha(c.image_path),'spec':scene['spec'],'seconds':time.time()-start}
                    manifest.append(r);append(log,r)
                    print(f"{model['id']} generated {len(manifest)}/32",flush=True)
        if model['id']=='base':
            repeats=[];repeatdir=folder/'repeats';repeatdir.mkdir()
            for row in manifest[::4]:
                assert time.time()<args.deadline
                c=backbone.generate_images([row['prompt']],[row['seed']],repeatdir,'base-repeat')[0]
                repeats.append({'spec_id':row['spec_id'],'candidate_index':row['candidate_index'],'seed':row['seed'],
                                'source_image':row['image_path'],'image_path':c.image_path,'rgb_sha256':c.rgb_sha256,'file_sha256':sha(c.image_path)})
            save_new(folder/'repeats.json',repeats)
        after=parameter_digest(trainable_snapshot(backbone.model));assert before==after
        save_new(folder/'complete.json',{'at_utc':now(),'n_images':len(manifest),'parameter_digest_before':before,'parameter_digest_after':after,'manifest_sha256':sha(folder/'images.jsonl')})
    save_new(SIDE/'generation/complete.json',{'at_utc':now(),'n_images':128,'base_repeat_images':8,'new_optimizer_steps':0})
    print('Phase A paired generation complete',flush=True)

if __name__=='__main__':main()
