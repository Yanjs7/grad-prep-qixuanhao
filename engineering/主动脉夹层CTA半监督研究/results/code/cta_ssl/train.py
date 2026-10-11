import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from torch.nn import functional as F

from .common import config, digest, save_json, seed_all
from .data import read_manifest, load_patient, augment, labeled_schedule
from .identity import check_identity_partitions
from .model import PatientMIL, update_ema
from .precision import autocast_context, check_checkpoint_precision, precision_policy


def metrics(labels, scores, threshold=0.5):
    labels=np.asarray(labels,dtype=int)
    scores=np.asarray(scores,dtype=float)
    tn,fp,fn,tp=confusion_matrix(labels,scores>=threshold,labels=[0,1]).ravel()
    return dict(n=int(len(labels)),positive=int((labels==1).sum()),negative=int((labels==0).sum()),
                auroc=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
                sensitivity=float(tp/(tp+fn)) if tp+fn else None,
                specificity=float(tn/(tn+fp)) if tn+fp else None,
                tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),threshold=float(threshold))


def threshold_on_validation(labels,scores):
    if len(set(labels))!=2:
        raise ValueError('Validation requires both classes for threshold selection')
    fpr,tpr,thresholds=roc_curve(labels,scores)
    valid=np.isfinite(thresholds)&(thresholds>=0)&(thresholds<=1)
    if not valid.any():
        return 0.5
    candidates=np.where(valid)[0]
    return float(thresholds[candidates[np.argmax((tpr-fpr)[valid])]])


@torch.no_grad()
def predict(model, rows, device, settings=None):
    model.eval()
    result=[]
    for _,row in rows.iterrows():
        x,pos=load_patient(row,device)
        with autocast_context(settings or {},device,prediction=True):
            logit,attention,_=model(x,pos)
        result.append(dict(patient_id=row.patient_id,source=row.source,label=int(row.label),
                           score=float(torch.sigmoid(logit.float()).item()),
                           influence_weights=json.dumps(attention.float().cpu().tolist())))
    return pd.DataFrame(result)


def select_rows(df,ids):
    selected=df[df.patient_id.isin(ids)]
    if set(selected.patient_id)!=set(ids):
        raise ValueError('Fold file references absent, rejected or duplicate patients')
    return selected


def checked_split(df, splitfile, fold):
    allfolds=json.loads(Path(splitfile).read_text())
    split=next(x for x in allfolds['folds'] if int(x['fold'])==fold)
    sets=[set(split[x]) for x in ['train','val','test']]
    unknown=set(allfolds.get('unlabeled',[]))
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)) or any(x&unknown for x in sets):
        raise ValueError('Patient leakage between training, validation, test or unlabeled pool')
    groups={key:select_rows(df,split[key]) for key in ['train','val','test']}
    groups['unlabeled']=select_rows(df,unknown)
    if any((rows.label==-1).any() for key,rows in groups.items() if key!='unlabeled'):
        raise ValueError('Unknown labels in a labeled split')
    if not groups['unlabeled'].empty and set(groups['unlabeled'].label)!={-1}:
        raise ValueError('Unlabeled pool must have diagnostic labels hidden as -1')
    check_identity_partitions(groups, allfolds.get('leakage_groups'))
    return groups


def validate_preprocessing(df, cfg):
    for _,row in df.iterrows():
        if row.source=='synthetic':
            if not cfg['training'].get('allow_synthetic',False):
                raise ValueError('Synthetic data are restricted to technical tests')
            continue
        sidecar=Path(row.prepared_path).with_suffix('.json')
        if not sidecar.is_file():
            raise ValueError(f'Prepared input provenance missing: {row.patient_id}')
        record=json.loads(sidecar.read_text())
        from .preprocess import PREPROCESSING_VERSION, file_identity
        from .patches import cache_key
        if record.get('preprocessing_version')!=PREPROCESSING_VERSION:
            raise ValueError(f'Prepared input uses an obsolete preprocessing version: {row.patient_id}')
        inputs=record.get('inputs',{})
        if inputs.get('image')!=file_identity(row.image_path) or inputs.get('predicted_roi')!=file_identity(row.roi_path):
            raise ValueError(f'Prepared inputs refer to changed source files: {row.patient_id}')
        if record.get('cache_key')!=cache_key(inputs) or record.get('prepared_sha256')!=digest(row.prepared_path):
            raise ValueError(f'Prepared input integrity check failed: {row.patient_id}')
        if record['config']!=cfg['preprocess'] or record['patient_id']!=row.patient_id:
            raise ValueError(f'Prepared inputs differ from config or patient ID: {row.patient_id}')
        from .intensity import calibration, geometry
        if record.get('intensity_calibration')!=calibration(row):
            raise ValueError(f'Prepared inputs have a different or unrecorded HU calibration: {row.patient_id}')
        if record.get('geometry_transform')!=geometry(row):
            raise ValueError(f'Prepared inputs have a different or unrecorded geometry: {row.patient_id}')


def initialize_teacher(model, parent=None, preserve_ema=False):
    teacher=copy.deepcopy(model).eval()
    if parent is not None and preserve_ema:
        if 'teacher' not in parent:
            raise ValueError('Preserving EMA requires the parent teacher checkpoint')
        teacher.load_state_dict(parent['teacher'],strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def run(cfgpath, phase, fold, init=None, resume=None):
    cfg=config(cfgpath)
    settings=cfg['training']
    precision_policy(settings)
    seed=int(cfg.get('seed',42))+fold
    seed_all(seed)
    df=read_manifest(cfg['manifest'])
    if (df.source=='synthetic').any() and not settings.get('allow_synthetic',False):
        raise ValueError('Synthetic cases are permitted only in explicitly marked technical tests')
    validate_preprocessing(df,cfg)
    groups=checked_split(df,cfg['folds'],fold)
    selection_source=settings.get('selection_source','AortaSeg')
    val=groups['val'][groups['val'].source==selection_source]
    if set(val.label)!={0,1}:
        raise ValueError(f'Validation selection source {selection_source} must contain both classes')
    if phase=='s1' and groups['unlabeled'].empty:
        raise ValueError('S1 requires a nonempty reviewed unlabeled pool')
    if phase!='s0' and not init and not resume:
        raise ValueError('S0+ and S1 require a shared S0 checkpoint')
    requested_device=settings.get('device','auto')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested_device=='auto' else requested_device)
    with autocast_context(settings,device):
        pass
    model=PatientMIL(cfg['model'],load_pretrained=not bool(init or resume)).to(device)
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=settings['lr'],weight_decay=settings.get('weight_decay',1e-4))
    total=int(settings['initial_steps'] if phase=='s0' else settings['continuation_steps'])
    output=Path(cfg['output'])/cfg['model']['encoder']/f'seed{seed}'/f'fold{fold}'/phase
    output.mkdir(parents=True,exist_ok=True)
    existing=output/'latest.pt'
    if existing.exists() and not resume:
        raise FileExistsError(f'Run already exists; use --resume or a new output root: {output}')
    meta=dict(fold=fold,seed=seed,phase=phase,model_config=cfg['model'],preprocess=cfg['preprocess'],
              manifest_sha256=digest(cfg['manifest']),folds_sha256=digest(cfg['folds']),
              train_ids=groups['train'].patient_id.tolist(),val_ids=groups['val'].patient_id.tolist(),
              test_ids=groups['test'].patient_id.tolist())
    # Keep historical pilot behavior explicit and require a separate new run
    # to change continuation semantics. Old checkpoints default to copy_student.
    meta['continuation_teacher_policy']='preserve_parent_ema' if settings.get('preserve_ema_on_init',False) else 'copy_student'
    meta['compute_precision_policy']=precision_policy(settings)
    parent=None
    if init:
        parent=torch.load(init,map_location=device,weights_only=True)
        for key in ['fold','seed','model_config','preprocess','manifest_sha256','folds_sha256']:
            if parent['meta'][key]!=meta[key]:
                raise ValueError(f'S0 checkpoint and current run differ in {key}')
        if parent['meta']['phase']!='s0':
            raise ValueError('Continuation initializer must be an S0 checkpoint')
        check_checkpoint_precision(settings,parent['meta'])
        if parent['meta'].get('continuation_teacher_policy','copy_student')!=meta['continuation_teacher_policy']:
            raise ValueError('S0 and continuation teacher policies differ')
        model.load_state_dict(parent['student'])
        optim.load_state_dict(parent['optimizer'])
        meta['parent_sha256']=digest(init)
    teacher=initialize_teacher(model,parent,settings.get('preserve_ema_on_init',False))
    start,best=0,-1.0
    if resume:
        checkpoint=torch.load(resume,map_location=device,weights_only=True)
        for key in ['fold','seed','phase','model_config','preprocess','manifest_sha256','folds_sha256']:
            if checkpoint['meta'][key]!=meta[key]:
                raise ValueError(f'Resume configuration differs in {key}')
        if checkpoint['total_steps']!=total:
            raise ValueError('Resume must preserve the original step budget')
        check_checkpoint_precision(settings,checkpoint['meta'])
        if checkpoint['meta'].get('continuation_teacher_policy','copy_student')!=meta['continuation_teacher_policy']:
            raise ValueError('Resume must preserve the continuation teacher policy')
        model.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        optim.load_state_dict(checkpoint['optimizer'])
        start,best=int(checkpoint['step']),float(checkpoint['best'])
        meta=checkpoint['meta']
    schedule=labeled_schedule(groups['train'],total,seed+10000)
    u_rng=np.random.default_rng(seed+20000)
    u_indices=groups['unlabeled'].index.to_numpy()
    u_schedule=u_rng.choice(u_indices,size=total,replace=True) if len(u_indices) else []
    logpath=output/'train.jsonl'
    if resume and logpath.exists():
        old=logpath.read_text().splitlines()
        logpath.rename(logpath.with_name(f'train.before-resume-{time.time_ns()}.jsonl'))
        kept=[line for line in old if int(json.loads(line)['step'])<=start]
        logpath.write_text('\n'.join(kept)+'\n' if kept else '')
    save_json(output/'run.json',dict(config=cfg,meta=meta,device=str(device),
                                   note='Fixed continuation steps and labeled visits; S1 uses additional compute.'))
    started=time.monotonic()
    for step in range(start,total):
        model.train()
        # Deterministic per-step augmentation, unaffected by unlabeled RNG consumption.
        row=df.loc[schedule[step]]
        x,pos=load_patient(row,device)
        x=augment(x,seed*100000+step,strong=False)
        lr=float(settings['lr'])*(.1+.9*.5*(1+math.cos(math.pi*step/max(total,1))))
        for group in optim.param_groups:
            group['lr']=lr
        optim.zero_grad(set_to_none=True)
        with autocast_context(settings,device):
            logit,_,_=model(x,pos)
            label=torch.tensor(float(row.label),device=device)
            supervised=F.binary_cross_entropy_with_logits(logit.float(),label)
            consistency=torch.zeros((),device=device)
            weight=0.0
            # A zero-weight consistency ablation must follow the identical
            # supervised graph, rather than building needless zero-grad branches.
            if phase=='s1' and float(settings['consistency_max'])>0:
                urow=df.loc[int(u_schedule[step])]
                ux,upos=load_patient(urow,device)
                weak=augment(ux,seed*100000+step+3000000,strong=False)
                strong=augment(ux,seed*100000+step+6000000,strong=True)
                with torch.no_grad():
                    target=torch.sigmoid(teacher(weak,upos)[0].float())
                prediction=torch.sigmoid(model(strong,upos)[0].float())
                consistency=F.mse_loss(prediction,target)
                weight=float(settings['consistency_max'])*min(1.0,(step+1)/max(1,settings['consistency_ramp_steps']))
            loss=supervised+weight*consistency
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss; no successful completion claimed')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optim.step()
        update_ema(teacher,model,float(settings['ema']))
        record=dict(step=step+1,labeled_patient=row.patient_id,supervised=float(supervised.detach()),
                    consistency=float(consistency.detach()),unlabeled_weight=weight,lr=lr)
        evaluate=(step+1)%int(settings['evaluate_every'])==0 or step+1==total
        if evaluate:
            predictions=predict(teacher,val,device,settings=settings)
            auc=float(roc_auc_score(predictions.label,predictions.score))
            record.update(validation_source=selection_source,validation_auroc=auc)
            improved=auc>best
            best=max(best,auc)
            state=dict(student=model.state_dict(),teacher=teacher.state_dict(),optimizer=optim.state_dict(),
                       step=step+1,total_steps=total,best=best,meta=meta)
            torch.save(state,output/'latest.tmp')
            (output/'latest.tmp').replace(output/'latest.pt')
            if improved:
                torch.save(state,output/'best.tmp')
                (output/'best.tmp').replace(output/'best.pt')
        with logpath.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if evaluate:
            print(json.dumps(record),flush=True)
    save_json(output/'completed.json',dict(steps=total,seconds_this_invocation=time.monotonic()-started,
               peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None))
    print(f'Completed {phase}: {output}',flush=True)
    return output


def evaluate(cfgpath, checkpoint, fold, output):
    cfg=config(cfgpath)
    df=read_manifest(cfg['manifest'])
    validate_preprocessing(df,cfg)
    groups=checked_split(df,cfg['folds'],fold)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    check_checkpoint_precision(cfg['training'],state['meta'])
    if state['meta']['fold']!=fold or state['meta']['manifest_sha256']!=digest(cfg['manifest']) or state['meta']['folds_sha256']!=digest(cfg['folds']):
        raise ValueError('Checkpoint data provenance differs from evaluation files')
    if state['meta']['model_config']!=cfg['model'] or state['meta']['preprocess']!=cfg['preprocess']:
        raise ValueError('Checkpoint architecture/preprocessing differs from evaluation config')
    requested_device=cfg['training'].get('device','auto')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested_device=='auto' else requested_device)
    model=PatientMIL(cfg['model'],load_pretrained=False).to(device)
    model.load_state_dict(state['teacher'])
    val=predict(model,groups['val'],device,settings=cfg['training'])
    selection_source=cfg['training'].get('selection_source','AortaSeg')
    selected=val[val.source==selection_source]
    threshold=threshold_on_validation(selected.label,selected.score)
    result=predict(model,groups['test'],device,settings=cfg['training'])
    result['prediction']=(result.score>=threshold).astype(int)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'predictions.csv',index=False)
    val.to_csv(output/'validation_predictions.csv',index=False)
    summary=dict(threshold_selection_source=selection_source,fold=fold,
                 checkpoint_sha256=digest(checkpoint),
                 manifest_sha256=digest(cfg['manifest']),folds_sha256=digest(cfg['folds']),
                 mixed_source_secondary=metrics(result.label,result.score,threshold),
                 by_source={str(source):metrics(rows.label,rows.score,threshold)
                            for source,rows in result.groupby('source')},
                 note='Small-sample exploratory evaluation; scores are not calibrated clinical probabilities.')
    save_json(output/'metrics.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
