#!/usr/bin/env python3
"""Real, bounded author-model CUDA backward and one support-set epoch.

Uses only an official training support set. Does not evaluate test predictions,
modify the author source, or create formal study results.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path('/root/BiomedCoOp')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='btmri')
parser.add_argument('--output', required=True)
args = parser.parse_args()
lock = (ROOT/'logs/gpu-study.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out = Path(args.output)
if out.exists(): raise RuntimeError('Existing probe output retained')
out.mkdir(parents=True)
os.chdir(ROOT / 'code')
sys.path.insert(0, str(ROOT / 'code' / 'Dassl.pytorch'))
sys.path.insert(0, str(ROOT / 'code'))
os.environ.update(HF_HOME=str(ROOT / 'hf-cache'), HF_HUB_CACHE=str(ROOT / 'hf-cache/hub'), HUGGINGFACE_HUB_CACHE=str(ROOT / 'hf-cache/hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import torch
import numpy as np
import train as author
from dassl.engine import build_trainer
from dassl.utils import set_random_seed

report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'dataset': args.dataset, 'method': 'BiomedCoOp_BiomedCLIP', 'scope': 'one actual support-set epoch; training-only, not formal budget', 'precision': 'fp32'}
start = time.monotonic()
try:
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    namespace = SimpleNamespace(root=str(ROOT/'data'), output_dir=str(out/'trainer'), resume='', seed=1, source_domains=None, target_domains=None, transforms=None, trainer='BiomedCoOp_BiomedCLIP', backbone='', head='', dataset_config_file='configs/datasets/'+args.dataset+'.yaml', config_file='configs/trainers/BiomedCoOp/few_shot/'+args.dataset+'.yaml', opts=['DATASET.NUM_SHOTS','16','DATALOADER.TEST.BATCH_SIZE','16','DATALOADER.NUM_WORKERS','2','TRAINER.BIOMEDCOOP.PREC','fp32'])
    cfg = author.setup_cfg(namespace)
    set_random_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    enabled = {name: p for name, p in trainer.model.named_parameters() if p.requires_grad}
    assert enabled and all(name.endswith('prompt_learner.ctx') or name == 'prompt_learner.ctx' for name in enabled), list(enabled)
    original = {name: p.detach().clone() for name, p in enabled.items()}
    support = [{'image': str(Path(d.impath).relative_to(ROOT/'data')), 'label': int(d.label), 'classname': d.classname} for d in trainer.dm.dataset.train_x]
    serialized = json.dumps(support, sort_keys=True).encode()
    report['support_sha256'] = hashlib.sha256(serialized).hexdigest()
    report['support'] = support
    report['n_classes'] = trainer.num_classes
    report['support_count'] = len(support)
    report['batches'] = len(trainer.train_loader_x)
    report['configured_train_batch'] = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
    report['config'] = cfg.dump()
    losses = []
    original_fb = trainer.forward_backward
    def checked_forward_backward(batch):
        result = original_fb(batch)
        assert all(math.isfinite(float(v)) for v in result.values()), result
        gradients = {name: {'finite': bool(torch.isfinite(p.grad).all()), 'l1': float(p.grad.abs().sum())} for name,p in enabled.items() if p.grad is not None}
        assert gradients and all(g['finite'] and g['l1'] > 0 for g in gradients.values()), gradients
        assert all(p.grad is None for name,p in trainer.model.named_parameters() if not p.requires_grad)
        losses.append({'batch': len(losses)+1, **result, 'gradients': gradients})
        return result
    trainer.forward_backward = checked_forward_backward
    torch.cuda.synchronize()
    epoch_start = time.monotonic()
    trainer.epoch = 0
    trainer.run_epoch()
    torch.cuda.synchronize()
    report['training_epoch_seconds'] = time.monotonic()-epoch_start
    report['losses'] = losses
    report['updated_trainable_parameters'] = {name: float((p.detach()-original[name]).abs().sum()) for name,p in enabled.items()}
    assert all(v>0 for v in report['updated_trainable_parameters'].values())
    report.update(status='passed', actual_cuda_backward=True, finite_nonzero_prompt_gradient=True, frozen_backbone_no_grad=True, own_peak_allocated_bytes=torch.cuda.max_memory_allocated(), own_peak_reserved_bytes=torch.cuda.max_memory_reserved())
except BaseException as error:
    report.update(status='failed', error_type=type(error).__name__, error=str(error))
    raise
finally:
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['invocation_seconds'] = time.monotonic()-start
    (out/'probe-proof.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['support','losses','config']},ensure_ascii=False),flush=True)
