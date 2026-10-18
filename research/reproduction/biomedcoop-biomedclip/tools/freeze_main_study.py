#!/usr/bin/env python3
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path('/root/BiomedCoOp')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def bound(path):
    s=path.stat();return {'path':str(path),'sha256':sha(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
target=ROOT/'reports/main-study-freeze-v1-20261014'
if target.exists():raise RuntimeError('Existing freeze retained')
env=json.loads((ROOT/'logs/environment-state.json').read_text());assert env['status']=='complete'
for dataset in ['btmri','chmnist']:
    probe=json.loads((ROOT/f'reports/cuda-probe-{dataset}16-20261014/probe-proof.json').read_text())
    assert probe['status']=='passed' and probe['actual_cuda_backward'] and probe['finite_nonzero_prompt_gradient']
manifest=json.loads((ROOT/'manifests/download-manifest.json').read_text())
prep=json.loads((ROOT/'logs/preparation-state.json').read_text());assert prep['models_cached'] and 'source' in prep['prepared']
model=ROOT/'models/biomedclip/open_clip_pytorch_model.bin'
assert sha(model)==next(x['checksum'].removeprefix('sha256:') for x in manifest['files'] if x['id']=='biomedclip-weights')
datasets=['btmri','busi','chmnist','covid','ctkidney','dermamnist','kneexray','kvasir','lungcolon','octmnist','retina']
names=['BTMRI','BUSI','CHMNIST','COVID_19','CTKidney','DermaMNIST','KneeXray','Kvasir','LungColon','OCTMNIST','RETINA']
source=[]
for path in sorted((ROOT/'code').rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc','.pkl']:source.append(bound(path))
runner=ROOT/'tools/run_main_experiment.py'
plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':'main-v1-20261014','scope':'complete main experiments and the three agreed comparators; no extensions or ablations','datasets':datasets,'dataset_directories':dict(zip(datasets,names)),'shots':[1,2,4,8,16],'seeds':[1,2,3],'methods':['BiomedCoOp','CoOp','LinearProbe','ZeroShot'],'prompt_runs':330,'linear_probe_runs':165,'zero_shot_evaluations':11,'total_jobs':506,'precision':'fp32','train_batch':4,'test_batch':16,'workers':2,'prompt_epochs':100,'model_selection':'last_step, not test-selected','code_commit':manifest['revisions']['code'],'model_sha256':sha(model),'model_binding':bound(model),'runner_sha256':sha(runner),'source_bindings':source,'runtime_environment':env['validation'],'deviations':['PyTorch2.1.0+cu121/torchvision0.16.0 satisfy published torch>=2.0.1, instead of example torch2.0.1/cu118','Test batch100 to16; workers8 to2','CoOp batch32 from published file overridden to4 to match paper protocol and device; published CoOp LR0.002 retained','Prompt training RNG reset after model construction to avoid support-cache creation altering training RNG','LinearProbe uses exact shared train and min(K,4)-per-class validation supports; published entry point has a missing return and different validation resampling','Deterministic frozen visual test features cached with weight/transform/path/stat binding; final prompted logits algebra checked on validation images','Tensorboard2.15.2 replaces unpinned tb-nightly; unused lint/wilds not imported or installed','Per-epoch compact resume includes trainable weights/optimizer/scheduler/random states; unmodified author full final checkpoint retained'], 'constraints':['All public anonymous resources','No test parameter/model selection','Images are not confirmed independent patients','Patient and pretraining overlap unknown','Class descriptions are not patient reports','No GPU/VRAM occupancy keepalive or interference with other processes']}
target.mkdir()
(target/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
(target/'freeze-proof.json').write_text(json.dumps({'plan_sha256':sha(target/'plan.json'),'plan':str(target/'plan.json'),'probe_files':[bound(ROOT/f'reports/cuda-probe-{d}16-20261014/probe-proof.json') for d in ['btmri','chmnist']],'manifest':bound(ROOT/'manifests/download-manifest.json'),'runner':bound(runner),'runtime_freeze':bound(ROOT/'reports/runtime-freeze.txt')},indent=2)+'\n')
print(json.dumps({'plan':str(target/'plan.json'),'sha256':sha(target/'plan.json'),'jobs':506}))
