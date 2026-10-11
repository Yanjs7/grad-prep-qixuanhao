#!/usr/bin/env python3
"""Isolated formal train/evaluate phase; report this process's CUDA usage."""
import argparse, datetime, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path('/root/CTA')
sys.path.insert(0, str(ROOT / 'code/cta-ssl/src'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def main():
    import torch, yaml
    from cta_ssl.train import run, evaluate
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--phase', choices=['s0', 's0plus', 's1'], required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--action', choices=['train', 'evaluate'], default='train')
    p.add_argument('--init'); p.add_argument('--resume'); p.add_argument('--checkpoint')
    args = p.parse_args()
    proof = json.loads((ROOT / 'reports/full-study-freeze-20261010/config-freeze-proof.json').read_text())
    assert sha(args.config) == proof['config_sha256']
    for filename, expected in proof['runtime_code_sha256'].items():
        assert sha(filename) == expected, filename
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert sha(cfg['manifest']) == proof['manifest_sha256']
    assert sha(cfg['folds']) == proof['folds_sha256']
    assert sha(proof['original_mass_checkpoint']) == proof['original_mass_checkpoint_sha256']
    assert torch.cuda.is_available() and cfg['training']['device'] == 'cuda'
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    out = Path(cfg['output']) / 'mass' / f'seed{42 + args.fold}' / f'fold{args.fold}' / args.phase
    out.mkdir(parents=True, exist_ok=True)
    started = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(),
                   action=args.action, fold=args.fold, phase=args.phase, config_sha256=sha(args.config),
                   initial_memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),
                   initial_cuda_free_mib=torch.cuda.mem_get_info()[0] / 2**20,
                   initial_parent_sha256=sha(args.init) if args.init else None)
    invocation = out / f'{args.action}-invocation-{time.time_ns()}.json'
    save(invocation, started)
    try:
        if args.action == 'train':
            run(args.config, args.phase, args.fold, args.init, args.resume)
        else:
            assert args.checkpoint
            evaluate(args.config, args.checkpoint, args.fold, out / 'evaluation')
        torch.cuda.synchronize()
        started.update(status='completed', finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       seconds=time.monotonic() - begin, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                       final_memory_events=Path('/sys/fs/cgroup/memory.events').read_text())
    except Exception as e:
        started.update(status='failed', error=repr(e), seconds=time.monotonic() - begin,
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                       final_memory_events=Path('/sys/fs/cgroup/memory.events').read_text())
        raise
    finally:
        save(invocation, started)

if __name__ == '__main__':
    main()
