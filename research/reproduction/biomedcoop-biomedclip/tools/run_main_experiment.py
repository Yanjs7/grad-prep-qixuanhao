#!/usr/bin/env python3
"""Instrumented main experiments around the unmodified author implementations.

Only prompt methods use the author's forward_backward. LP follows its published
validation-only regularization search on the same support/validation identities.
"""
import argparse
from collections import Counter
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path('/root/BiomedCoOp')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--method',choices=['BiomedCoOp','CoOp','LinearProbe','ZeroShot'],required=True)
    parser.add_argument('--shots',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    assert args.dataset in plan['datasets'] and args.shots in plan['shots'] and args.seed in plan['seeds']
    assert file_sha(__file__)==plan['runner_sha256'], 'Runner differs from frozen source'
    lock=(ROOT/'logs/gpu-study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(HF_HOME=str(ROOT/'hf-cache'),HF_HUB_CACHE=str(ROOT/'hf-cache/hub'),HUGGINGFACE_HUB_CACHE=str(ROOT/'hf-cache/hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
    os.chdir(ROOT/'code');sys.path.insert(0,str(ROOT/'code/Dassl.pytorch'));sys.path.insert(0,str(ROOT/'code'))
    import numpy as np
    import torch
    import train as author
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    from dassl.data import DataManager
    from dassl.data.data_manager import DatasetWrapper
    from open_clip.src.open_clip import create_model_from_pretrained,get_tokenizer
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=True
    torch.cuda.reset_peak_memory_stats()
    output=ROOT/'runs'/plan['run_id']/args.dataset/(args.method if args.method=='ZeroShot' else f'{args.method}/shots_{args.shots}/seed_{args.seed}')
    output.mkdir(parents=True,exist_ok=True)
    completed=output/'completed.json'
    if completed.exists():
        old=json.loads(completed.read_text());assert old['plan_sha256']==file_sha(args.plan)
        assert old['prediction_sha256']==file_sha(output/'predictions.csv')
        print('Verified completed run; skipped',str(output));return
    attempt=len(list(output.glob('attempt-*.jsonl')))+1
    history=output/f'attempt-{attempt}.jsonl'
    config_method='CoOp' if args.method=='CoOp' else 'BiomedCoOp'
    config_path='configs/trainers/CoOp/vit_b16.yaml' if args.method=='CoOp' else 'configs/trainers/BiomedCoOp/few_shot/'+args.dataset+'.yaml'
    ns=SimpleNamespace(root=str(ROOT/'data'),output_dir=str(output),resume='',seed=args.seed,source_domains=None,target_domains=None,transforms=None,trainer=('CoOp_BiomedCLIP' if args.method=='CoOp' else 'BiomedCoOp_BiomedCLIP'),backbone='',head='',dataset_config_file='configs/datasets/'+args.dataset+'.yaml',config_file=config_path,opts=['DATASET.NUM_SHOTS',str(args.shots),'DATALOADER.TRAIN_X.BATCH_SIZE',str(plan['train_batch']),'DATALOADER.TEST.BATCH_SIZE',str(plan['test_batch']),'DATALOADER.NUM_WORKERS',str(plan['workers']),'TEST.FINAL_MODEL','last_step','TRAINER.'+config_method.upper()+'.PREC','fp32'])
    cfg=author.setup_cfg(ns)
    assert cfg.OPTIM.MAX_EPOCH==plan['prompt_epochs']
    set_random_seed(args.seed)
    invocation={'started_utc':now(),'method':args.method,'dataset':args.dataset,'shots':args.shots,'seed':args.seed,'plan_sha256':file_sha(args.plan),'attempt':attempt,'status':'initializing','configuration':cfg.dump()}
    started=time.monotonic()
    def event(**record):
        record.update(utc=now(),attempt=attempt)
        with history.open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        atomic_json(output/'progress.json',dict(method=args.method,dataset=args.dataset,shots=args.shots,seed=args.seed,**record))
    try:
        if args.method in ['BiomedCoOp','CoOp']:
            trainer=build_trainer(cfg);dm=trainer.dm;model=trainer.model
            encoder=model.image_encoder
        else:
            dm=DataManager(cfg)
            model,_=create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            model=model.float().eval().cuda();encoder=model.visual
        def datum(d):return {'image':str(Path(d.impath).relative_to(ROOT/'data')),'label':int(d.label),'classname':d.classname}
        support={'train':[datum(d) for d in dm.dataset.train_x],'val':[datum(d) for d in dm.dataset.val],'classnames':dm.dataset.classnames,'test_count':len(dm.dataset.test)}
        support_sha=digest(support)
        if (output/'support.json').exists():assert json.loads((output/'support.json').read_text())==support
        else:atomic_json(output/'support.json',support)
        common=ROOT/'reports'/plan['run_id']/'supports'/args.dataset/f'shots_{args.shots}-seed_{args.seed}.json'
        if args.method!='ZeroShot':
            if common.exists():assert json.loads(common.read_text())==support, 'Methods have different supports'
            else:atomic_json(common,support)
            assert all(n==args.shots for n in Counter(d['label'] for d in support['train']).values())
            assert set(d['image'] for d in support['train']).isdisjoint(d['image'] for d in support['val'])
        invocation.update(support_sha256=support_sha,train_labels=len(support['train']),validation_labels=len(support['val']),classes=len(dm.dataset.classnames))
        # Bind evaluation cache to official weights, exact ordered paths, sizes and mtimes,
        # and deterministic evaluation transform. Labels are saved for output checking.
        samples=[datum(d) for d in dm.dataset.test]
        cache_identity={'weight_sha256':plan['model_sha256'],'source_commit':plan['code_commit'],'transform':str(dm.test_loader.dataset.transform),'samples':[(d.impath,Path(d.impath).stat().st_size,Path(d.impath).stat().st_mtime_ns,int(d.label)) for d in dm.dataset.test],'precision':'fp32','batch':plan['test_batch']}
        cache_id=digest(cache_identity)
        cache=ROOT/'feature-cache'/args.dataset/(cache_id+'.pt');cache.parent.mkdir(parents=True,exist_ok=True)
        encoder.eval()
        assert not any(isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for module in encoder.modules()), 'Mutable vision normalization invalidates fixed feature cache'
        assert all(not p.requires_grad for p in encoder.parameters()) or args.method in ['ZeroShot','LinearProbe']
        @torch.no_grad()
        def features(loader):
            tensors=[];labels=[];paths=[]
            for batch in loader:
                x=encoder(batch['img'].cuda().float());x=x/x.norm(dim=-1,keepdim=True)
                assert torch.isfinite(x).all()
                tensors.append(x.cpu());labels.append(batch['label'].cpu());paths.extend(batch['impath'])
            return torch.cat(tensors),torch.cat(labels),paths
        if cache.exists():
            saved=torch.load(cache,map_location='cpu');assert saved['identity_sha256']==cache_id
            test_features,test_labels=saved['features'],saved['labels']
            assert saved['paths']==[d.impath for d in dm.dataset.test]
        else:
            event(status='extracting deterministic test features',total_images=len(samples))
            test_features,test_labels,paths=features(dm.test_loader)
            assert paths==[d.impath for d in dm.dataset.test]
            temp=cache.with_suffix('.tmp');torch.save({'identity_sha256':cache_id,'features':test_features,'labels':test_labels,'paths':paths},temp);temp.replace(cache)
            atomic_json(cache.with_suffix('.json'),{'identity_sha256':cache_id,'identity':cache_identity,'file_sha256':file_sha(cache),'images':len(paths)})
        assert test_labels.tolist()==[d['label'] for d in samples]
        invocation['test_cache_identity']=cache_id
        if args.method in ['BiomedCoOp','CoOp']:
            set_random_seed(args.seed) # Stable training RNG independent of cached support creation.
            enabled={name:p for name,p in model.named_parameters() if p.requires_grad}
            assert enabled and all(name=='prompt_learner.ctx' for name in enabled),list(enabled)
            resume=output/'latest-resume.pt';start_epoch=0
            if resume.exists():
                state=torch.load(resume,map_location='cpu')
                assert state['plan_sha256']==invocation['plan_sha256'] and state['support_sha256']==support_sha
                for name,p in enabled.items():p.data.copy_(state['trainable'][name].to(p.device))
                trainer.optim.load_state_dict(state['optimizer']);trainer.sched.load_state_dict(state['scheduler'])
                random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
                start_epoch=state['completed_epochs'];invocation['resumed_from_epoch']=start_epoch
            epochs=cfg.OPTIM.MAX_EPOCH
            for epoch in range(start_epoch,epochs):
                trainer.epoch=epoch;trainer.num_batches=len(trainer.train_loader_x);trainer.set_model_mode('train')
                for index,batch in enumerate(trainer.train_loader_x):
                    trainer.batch_idx=index
                    losses=trainer.forward_backward(batch)
                    assert all(math.isfinite(float(v)) for v in losses.values()), losses
                    event(status='training',epoch=epoch+1,total_epochs=epochs,batch=index+1,batches=trainer.num_batches,losses=losses,images=[str(Path(p).relative_to(ROOT/'data')) for p in batch['impath']],lr=trainer.get_current_lr())
                state={'plan_sha256':invocation['plan_sha256'],'support_sha256':support_sha,'completed_epochs':epoch+1,'trainable':{n:p.detach().cpu().clone() for n,p in enabled.items()},'optimizer':trainer.optim.state_dict(),'scheduler':trainer.sched.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
                temp=resume.with_suffix('.tmp');torch.save(state,temp);temp.replace(resume)
            trainer.save_model(epochs-1,str(output)) # Preserve the author's full final checkpoint.
            trainer.set_model_mode('eval')
            with torch.no_grad():
                text=model.text_encoder(model.prompt_learner(),model.tokenized_prompts)
                text=text/text.norm(dim=-1,keepdim=True)
                scale=model.logit_scale.exp()
                # Algebra check on support images; no test-based tuning.
                batch=next(iter(dm.val_loader));images=batch['img'].cuda().float()
                original_logits=trainer.model_inference(images)
                v=encoder(images);v=v/v.norm(dim=-1,keepdim=True)
                alternative=scale*v@text.t()
                assert torch.allclose(original_logits,alternative,atol=1e-5,rtol=1e-5)
            logits=torch.cat([(scale*x.cuda()@text.t()).detach().cpu() for x in test_features.split(plan['test_batch'])])
            probabilities=torch.softmax(logits,dim=1).numpy()
            invocation['completed_epochs']=epochs
        elif args.method=='ZeroShot':
            from trainers.prompt_templates import CUSTOM_TEMPLATES
            tokenizer=get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            prompts=[CUSTOM_TEMPLATES[cfg.DATASET.NAME].format(c.replace('_',' ')) for c in dm.dataset.classnames]
            with torch.no_grad():
                text=model.encode_text(torch.cat([tokenizer(p) for p in prompts]).cuda(),False);text=text/text.norm(dim=-1,keepdim=True)
                logits=torch.cat([(model.logit_scale.exp()*x.cuda()@text.t()).cpu() for x in test_features.split(plan['test_batch'])])
            probabilities=torch.softmax(logits,dim=1).numpy();invocation['prompts']=prompts
            invocation['train_labels']=0;invocation['validation_labels']=0
        else:
            from sklearn.linear_model import LogisticRegression
            from torchvision import transforms as T
            set_random_seed(args.seed)
            # Author LP uses one augmented feature pass and deterministic validation.
            transform=T.Compose([T.RandomResizedCrop(224,scale=(0.5,1),interpolation=T.InterpolationMode.BICUBIC),T.RandomHorizontalFlip(0.5),T.ToTensor(),T.Normalize(cfg.INPUT.PIXEL_MEAN,cfg.INPUT.PIXEL_STD)])
            loader=torch.utils.data.DataLoader(DatasetWrapper(cfg,dm.dataset.train_x,transform=transform,is_train=True),batch_size=plan['test_batch'],num_workers=plan['workers'],shuffle=False,drop_last=False)
            train_x,train_y,paths=features(loader);val_x,val_y,_=features(dm.val_loader)
            assert len(train_y)==len(support['train'])
            search=[]
            def fit(c):
                classifier=LogisticRegression(solver='lbfgs',max_iter=1000,penalty='l2',C=float(c),random_state=args.seed).fit(train_x.numpy(),train_y.numpy())
                accuracy=float((classifier.predict(val_x.numpy())==val_y.numpy()).mean())
                search.append({'C':float(c),'validation_accuracy':accuracy})
                return classifier,accuracy
            coarse=[1e6,1e4,1e2,1,1e-2,1e-4,1e-6]
            trials=[fit(c) for c in coarse];best=coarse[int(np.argmax([v for _,v in trials]))];left,right=best*0.1,best*10
            for step in range(8):
                clf_left,a_left=fit(left);clf_right,a_right=fit(right)
                if a_left<a_right:classifier=clf_right;left,right=10**((math.log10(right)+math.log10(left))/2),right
                else:classifier=clf_left;left,right=left,10**((math.log10(right)+math.log10(left))/2)
                event(status='linear probe validation search',step=step+1,total_steps=8)
            probabilities=classifier.predict_proba(test_features.numpy())
            assert classifier.classes_.tolist()==list(range(len(dm.dataset.classnames)))
            import joblib
            joblib.dump(classifier,output/'classifier.joblib');atomic_json(output/'validation-search.json',search)
            invocation['selected_C']=float(classifier.C)
        assert probabilities.shape==(len(samples),len(dm.dataset.classnames)) and np.isfinite(probabilities).all()
        prediction=probabilities.argmax(axis=1)
        with (output/'predictions.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['image','label','prediction',*['probability_'+str(k) for k in range(probabilities.shape[1])]])
            for record,pred,p in zip(samples,prediction,probabilities):writer.writerow([record['image'],record['label'],int(pred),*map(float,p)])
        invocation.update(status='complete',finished_utc=now(),seconds=time.monotonic()-started,own_peak_allocated_bytes=torch.cuda.max_memory_allocated(),own_peak_reserved_bytes=torch.cuda.max_memory_reserved(),test_count=len(samples),accuracy=float((prediction==test_labels.numpy()).mean()*100),prediction_sha256=file_sha(output/'predictions.csv'))
        atomic_json(output/'invocation.json',invocation);atomic_json(completed,invocation);event(status='complete',accuracy=invocation['accuracy'])
        print(json.dumps(invocation,ensure_ascii=False),flush=True)
    except BaseException as error:
        invocation.update(status='failed',finished_utc=now(),seconds=time.monotonic()-started,error_type=type(error).__name__,error=str(error))
        atomic_json(output/f'failed-attempt-{attempt}.json',invocation);event(status='failed',error_type=type(error).__name__,error=str(error));raise

if __name__=='__main__':main()
