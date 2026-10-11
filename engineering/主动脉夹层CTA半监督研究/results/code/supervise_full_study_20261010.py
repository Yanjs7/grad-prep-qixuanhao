#!/usr/bin/env python3
"""Freeze a verified protocol, then run fixed budgets with recoverable phase checkpoints."""
import argparse, datetime, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

R = Path('/root/CTA')
P = R / 'reports/full-study-freeze-20261010'
C = R / 'code/cta-ssl/configs/full-study-titan24-v4-frozen-20261010.yaml'
OUT = R / 'runs/full-titan24-v4-20261010'
PY = R / '.venv/bin/python'
WRAPPER = R / 'tools/run_full_study_phase_20261010.py'
JOB = P / 'full-study-job.json'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)

def freeze():
    import yaml
    assert not C.exists() and not (P / 'config-freeze-proof.json').exists()
    data = json.loads((P / 'manifest-folds-freeze-proof.json').read_text())
    backward = json.loads((P / 'real-cuda-fold0-control/result.json').read_text())
    profile = json.loads((P / 'real-cuda-fold0-profile/result.json').read_text())
    for proof in (backward, profile):
        assert proof['status'] == 'passed' and not proof['test_cases_used']
        assert proof['manifest_sha256'] == data['manifest_sha256']
        assert proof['folds_sha256'] == data['folds_sha256']
    assert profile['supervised_pass_cases'] == 156 and profile['s1_profile_steps'] == 32
    assert data['all_current_preprocessing_controls_passed'] and data['all_five_partition_controls_passed']
    draft = R / 'code/cta-ssl/configs/full-study-titan24-draft-20261008.yaml'
    cfg = yaml.safe_load(draft.read_text())
    assert sha(draft) == backward['config_sha256'] == profile['config_sha256']
    cfg.update(manifest=data['manifest'], folds=data['folds'], output=str(OUT))
    cfg['training']['device'] = 'cuda'
    assert cfg['seed'] == 42 and cfg['training']['initial_steps'] == 600
    assert cfg['training']['continuation_steps'] == 400 and cfg['training']['preserve_ema_on_init']
    assert cfg['training']['compute_precision'] == 'float32' and not cfg['training']['amp']
    assert cfg['model']['encoder_chunk_size'] == 32 and cfg['model']['head_precision'] == 'float32'
    C.write_text('# Frozen after exact QC, fixed grouped folds and genuine CUDA profiling.\n' + yaml.safe_dump(cfg, sort_keys=False))
    files = [R / f'code/cta-ssl/src/cta_ssl/{n}.py' for n in
             ['train', 'model', 'data', 'identity', 'precision', 'common']]
    proof = dict(at_utc=now(), config=str(C), config_sha256=sha(C), original_draft_sha256=sha(draft),
                 manifest_sha256=data['manifest_sha256'], folds_sha256=data['folds_sha256'],
                 groups_sha256=data['groups_sha256'], output=str(OUT),
                 backward_proof_sha256=sha(P / 'real-cuda-fold0-control/result.json'),
                 fold0_profile_sha256=sha(P / 'real-cuda-fold0-profile/result.json'),
                 runtime_code_sha256={str(f): sha(f) for f in files},
                 wrapper_sha256=sha(WRAPPER), supervisor_sha256=sha(__file__),
                 original_mass_checkpoint=cfg['model']['checkpoint'], original_mass_checkpoint_sha256=sha(cfg['model']['checkpoint']),
                 budgets=dict(s0=600, s0plus=400, s1=400, folds=5),
                 shared_parent_policy='Same fold S0 best validation checkpoint: matching student, optimizer and preserved EMA',
                 selection_source='AortaSeg', no_test_based_selection=True,
                 profile_scope=profile['scope'], config_frozen=True)
    save(P / 'config-freeze-proof.json', proof)
    print(json.dumps(proof), flush=True)

def phase_dir(fold, phase):
    return OUT / 'mass' / f'seed{42 + fold}' / f'fold{fold}' / phase

def latest_step(directory):
    p = directory / 'train.jsonl'
    if not p.exists(): return 0
    lines = p.read_text().splitlines()
    return int(json.loads(lines[-1])['step']) if lines else 0

def supervise(resume):
    proof = json.loads((P / 'config-freeze-proof.json').read_text())
    assert sha(C) == proof['config_sha256'] and sha(WRAPPER) == proof['wrapper_sha256']
    assert sha(__file__) == proof['supervisor_sha256']
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid(): continue
        try:
            argv = (item / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if str(WRAPPER).encode() in argv:
            raise RuntimeError('An actual study child is still running; do not start a duplicate')
    locks = []
    for name in ['taad-localizer-batch.lock', 'full-study-training.lock']:
        f = (R / 'logs' / name).open('a+'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(f)
    if JOB.exists() and not resume: raise FileExistsError('Use explicit --resume for an existing study job')
    previous = json.loads(JOB.read_text()) if JOB.exists() else {}
    history = previous.get('history', [])
    status = dict(stage='starting', supervisor_pid=os.getpid(), config=str(C), config_sha256=sha(C),
                  at_utc=now(), formal_training_started=previous.get('formal_training_started', False), total_steps=7000, history=history,
                  output=str(OUT), training_complete=False, evaluation_complete=False)
    save(JOB, status)
    def launch(fold, phase, action, parent=None):
        directory = phase_dir(fold, phase); total = 600 if phase == 's0' else 400
        marker = directory / ('completed.json' if action == 'train' else 'evaluation/metrics.json')
        if marker.exists():
            if action == 'train': assert json.loads(marker.read_text())['steps'] == total
            return
        command = [str(PY), '-u', str(WRAPPER), '--config', str(C), '--fold', str(fold), '--phase', phase, '--action', action]
        if action == 'train':
            latest = directory / 'latest.pt'
            if parent is not None: command += ['--init', str(parent)]
            if latest.exists():
                assert resume; command += ['--resume', str(latest)]
            elif (directory / 'train.jsonl').exists():
                assert resume
                (directory / 'train.jsonl').rename(directory / f'train.before-retry-{time.time_ns()}.jsonl')
        else: command += ['--checkpoint', str(directory / 'best.pt')]
        log = R / 'logs' / f'full-titan24-v4-fold{fold}-{phase}-{action}-{time.time_ns()}.log'
        start = time.monotonic(); stamp = now()
        with log.open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=R / 'code/cta-ssl',
                                     env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'))
            while child.poll() is None:
                step = latest_step(directory)
                status.update(stage=action + '_running', fold=fold, phase=phase, action=action, child_pid=child.pid,
                              command=command, log=str(log), current_step=step, phase_total_steps=total,
                              at_utc=now(), phase_started_at_utc=stamp,
                              formal_training_started=status['formal_training_started'] or step > 0)
                save(JOB, status)
                time.sleep(10)
            exit_code = child.returncode
        event = dict(fold=fold, phase=phase, action=action, at_utc=now(), started_at_utc=stamp,
                     log=str(log), command=command, exit_code=exit_code, seconds=time.monotonic() - start)
        history.append(event); status['history'] = history
        status.update(at_utc=now(), child_pid=None, current_step=latest_step(directory))
        if exit_code != 0:
            status.update(stage='failed_requires_inspection', error=f'{action} fold {fold} {phase} exit {exit_code}')
            save(JOB, status); raise RuntimeError(status['error'])
        assert marker.exists(), str(marker)
        status['completed_training_steps'] = sum((600 if p == 's0' else 400) for f in range(5)
                                                for p in ['s0', 's0plus', 's1'] if (phase_dir(f, p) / 'completed.json').exists())
        save(JOB, status); print(json.dumps(event), flush=True)
    for fold in range(5):
        launch(fold, 's0', 'train')
        parent = phase_dir(fold, 's0') / 'best.pt'
        parent_record = P / f'fold{fold}-shared-parent.json'
        record = dict(fold=fold, parent=str(parent), sha256=sha(parent), selection='S0 AortaSeg validation AUROC best; full 600-step S0 budget completed')
        if parent_record.exists(): assert json.loads(parent_record.read_text()) == record
        else: save(parent_record, record)
        for phase in ['s0plus', 's1']:
            assert sha(parent) == record['sha256']
            launch(fold, phase, 'train', parent)
        # Check shared S0 parent and identical supervised access without test scores.
        logs = [phase_dir(fold, p) / 'train.jsonl' for p in ['s0plus', 's1']]
        visits = [[json.loads(line)['labeled_patient'] for line in log.read_text().splitlines()] for log in logs]
        assert len(visits[0]) == len(visits[1]) == 400 and visits[0] == visits[1]
        runs = [json.loads((phase_dir(fold, p) / 'run.json').read_text()) for p in ['s0plus', 's1']]
        assert all(r['meta']['parent_sha256'] == record['sha256'] for r in runs)
        save(P / f'fold{fold}-continuation-comparison-controls.json', dict(fold=fold, parent_sha256=record['sha256'],
             supervised_visits_equal=True, steps=400, supervised_visits_sha256=hashlib.sha256(json.dumps(visits[0]).encode()).hexdigest(),
             preserved_parent_ema=True, seeds_equal=runs[0]['meta']['seed'] == runs[1]['meta']['seed']))
    status.update(training_complete=True, stage='all_training_complete', at_utc=now()); save(JOB, status)
    for fold in range(5):
        for phase in ['s0', 's0plus', 's1']: launch(fold, phase, 'evaluate')
    status.update(stage='training_and_evaluation_complete_pending_final_report', evaluation_complete=True, at_utc=now(), child_pid=None)
    save(JOB, status)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'run'], required=True)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    else: supervise(args.resume)
