#!/usr/bin/env python3
"""Package a verified completed report and small protocol evidence, without volumes or weights."""
import argparse
import datetime
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
import yaml

ROOT = Path('/root/CTA')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = Path(args.report).resolve()
    assert report.is_relative_to(ROOT / 'reports')
    proof = json.loads((report / 'report_proof.json').read_text())
    assert proof['all_planned_training_complete'] and proof['all_planned_evaluation_complete']
    assert (proof['labeled_scans'], proof['phase_prediction_rows'], proof['training_step_rows']) == (243, 729, 7000)
    for name, identity in proof['files'].items():
        source = (report / name).resolve()
        assert source.is_relative_to(report) and source.stat().st_size == identity['bytes']
        assert sha(source) == identity['sha256'], name
    out = Path(args.output).resolve()
    assert out.is_relative_to(ROOT / 'reports') and not out.exists()
    archive = out.with_suffix('.tar.gz')
    assert not archive.exists()
    out.mkdir(parents=True)

    def copy(source, destination):
        source = Path(source)
        assert source.is_file() and not source.is_symlink()
        assert source.suffix in {'.json', '.jsonl', '.csv', '.yaml', '.md', '.py', '.png', '.pdf'}, source
        target = out / destination
        assert not target.exists(), destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)

    for source in sorted(report.iterdir()):
        assert source.is_file(), source
        copy(source, source.name)

    freeze_dir = ROOT / 'reports/full-study-freeze-20261010'
    frozen = json.loads((freeze_dir / 'config-freeze-proof.json').read_text())
    cfg = yaml.safe_load(Path(frozen['config']).read_text())
    job = json.loads((freeze_dir / 'full-study-job.json').read_text())
    assert job['training_complete'] and job['evaluation_complete']
    for field in ['config', 'manifest', 'folds']:
        source = Path(frozen['config'] if field == 'config' else cfg[field])
        assert sha(source) == frozen[field + '_sha256']
        copy(source, Path('protocol') / source.name)
    groups = ROOT / 'data/manifests/near-duplicate-identity-groups-v4-20261009.json'
    assert sha(groups) == frozen['groups_sha256']
    copy(groups,
         'protocol/near-duplicate-identity-groups-v4-20261009.json')
    copy(freeze_dir / 'config-freeze-proof.json', 'evidence/config-freeze-proof.json')
    copy(freeze_dir / 'full-study-job.json', 'evidence/full-study-job.json')
    copy(freeze_dir / 'final-report-statistical-unit-controls-20261010.json',
         'evidence/final-report-statistical-unit-controls-20261010.json')
    for fold in range(5):
        for suffix in ['shared-parent', 'continuation-comparison-controls']:
            source = freeze_dir / f'fold{fold}-{suffix}.json'
            copy(source, Path('evidence') / source.name)
        for phase in ['s0', 's0plus', 's1']:
            folder = Path(job['output']) / 'mass' / f'seed{42+fold}' / f'fold{fold}' / phase
            dest = Path('evidence') / f'fold{fold}' / phase
            for name in ['run.json', 'completed.json']:
                copy(folder / name, dest / name)
            for source in sorted(folder.glob('*-invocation-*.json')):
                copy(source, dest / source.name)
            copy(folder / 'evaluation/metrics.json', dest / 'evaluation-metrics.json')
    for name in ['generate_full_study_final_v2_20261010.py', 'generate_experiment_report.py',
                 'run_full_study_phase_20261010.py', 'supervise_full_study_20261010.py']:
        source = ROOT / 'tools' / name
        copy(source, Path('code') / name)
    for name, expected in frozen['runtime_code_sha256'].items():
        source = Path(name)
        assert sha(source) == expected
        copy(source, Path('code/cta_ssl') / source.name)
    assert sha(ROOT / 'tools/generate_full_study_final_v2_20261010.py') == proof['generator_sha256']
    copy(freeze_dir / 'final-report-statistical-unit-controls-v2-20261010.json',
         'evidence/final-report-statistical-unit-controls-v2-20261010.json')
    copy(Path(__file__).resolve(), Path('code') / Path(__file__).name)
    files = {str(f.relative_to(out)): {'bytes': f.stat().st_size, 'sha256': sha(f)}
             for f in sorted(out.rglob('*')) if f.is_file()}
    index = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'purpose': 'Completed Chinese report, curves, case CSV and fixed protocol evidence',
             'contains_ct_npz_or_model_weights': False, 'is_full_model_backup': False,
             'report_generator_sha256': proof['generator_sha256'], 'files': files}
    (out / 'delivery_file_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    with tarfile.open(archive, 'w:gz') as tar:
        for source in sorted(out.rglob('*')):
            if source.is_file():
                tar.add(source, arcname=str(source.relative_to(out)), recursive=False)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': sha(archive), 'files': len(files) + 1}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
