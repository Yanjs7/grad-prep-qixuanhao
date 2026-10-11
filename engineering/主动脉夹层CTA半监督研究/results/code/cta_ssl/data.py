from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import approved


def read_manifest(path, require_prepared=True):
    df = pd.read_csv(path, keep_default_na=False, dtype={'exact_voxel_sha256': str, 'leakage_group_id': str})
    for column in ['patient_id', 'source', 'image_path', 'label', 'qc', 'label_basis']:
        if column not in df:
            raise ValueError(f'Missing manifest column {column}')
    if df.patient_id.duplicated().any():
        raise ValueError('One scan per patient required; duplicate patient IDs found')
    if df.image_path.duplicated().any():
        raise ValueError('Duplicate image paths found')
    numeric = pd.to_numeric(df.label, errors='raise')
    if not set(numeric).issubset({-1,0,1}):
        raise ValueError('Labels must be -1 unknown, 0 non-AD, 1 AD')
    df['label'] = numeric.astype(int)
    df = df[df.qc.map(approved)].copy()
    if df.empty:
        raise ValueError('No QC-approved patients; inventory is not a reviewed manifest')
    real = df[df.source != 'synthetic']
    # Scan names and paths alone missed actual TAAD and DB duplicate inputs.
    # Require the completed voxel audit for formal training, including its
    # retained representative, so excluded copies cannot enter the SSL pool.
    if require_prepared and not real.empty:
        for column in ['exact_voxel_sha256', 'duplicate_representative']:
            if column not in real:
                raise ValueError(f'Formal inputs require completed duplicate audit: {column}')
        if not real.exact_voxel_sha256.map(lambda value: bool(re.fullmatch(r'[0-9a-f]{64}', str(value)))).all():
            raise ValueError('Formal inputs require valid exact voxel audit hashes')
    if 'exact_voxel_sha256' in real:
        hashes = real.exact_voxel_sha256[real.exact_voxel_sha256 != '']
        if hashes.duplicated().any():
            raise ValueError('Identical voxel inputs cannot be repeated across labeled or unlabeled pools')
    if 'duplicate_representative' in real:
        aliases = real.duplicate_representative.ne('') & real.duplicate_representative.ne(real.patient_id)
        if aliases.any():
            raise ValueError('Excluded duplicate aliases cannot enter preprocessing or training')
    for _, row in real[real.label != -1].iterrows():
        if str(row.get('label_review_flag', '')).strip():
            if (row.get('label_review_resolution', '') != 'reviewed'
                    or not str(row.get('label_review_evidence', '')).strip()):
                raise ValueError(f'{row.patient_id}: supervised label review remains unresolved')
    for _, row in df.iterrows():
        if not Path(row.image_path).is_file():
            raise FileNotFoundError(row.image_path)
        if row.label != -1 and not row.label_basis.strip():
            raise ValueError('Labeled cases require recorded diagnostic evidence')
        if row.source!='synthetic':
            from .intensity import calibration, geometry
            calibration(row)
            geometry(row)
        if require_prepared:
            if row.source!='synthetic' and row.get('roi_basis','')!='external_pretrained_input_only':
                raise ValueError(f'Input-only ROI provenance missing for {row.patient_id}')
            if 'prepared_path' not in df or not Path(row.prepared_path).is_file():
                raise ValueError(f'Prepared patches missing for {row.patient_id}')
            if 'prepared_qc' not in df or not approved(row.prepared_qc):
                raise ValueError(f'Patch coverage QC not approved for {row.patient_id}')
    return df


def load_patient(row, device='cpu'):
    import torch
    with np.load(row.prepared_path, allow_pickle=False) as data:
        patches = torch.from_numpy(data['patches'].astype(np.float32)).to(device)
        positions = torch.from_numpy(data['positions'].astype(np.float32)).to(device)
    if patches.ndim != 5 or patches.shape[1] != 1 or positions.shape != (len(patches),3):
        raise ValueError('Expected K,1,D,H,W patches and K,3 positions')
    if not torch.isfinite(patches).all() or not torch.isfinite(positions).all():
        raise ValueError('Non-finite prepared inputs')
    return patches, positions


def augment(patches, seed, strong=False):
    import torch
    # Intensity-only views: anatomical patch correspondence remains exact.
    gen = torch.Generator(device=patches.device).manual_seed(seed)
    gain = 1 + (torch.rand((), generator=gen, device=patches.device)-0.5) * (0.16 if strong else 0.06)
    noise = torch.randn(patches.shape, generator=gen, device=patches.device) * (0.025 if strong else 0.008)
    return patches * gain + noise


def labeled_schedule(rows, steps, seed):
    indices = {label: list(rows.index[rows.label == label]) for label in [0,1]}
    if not all(indices.values()):
        raise ValueError('Both diagnostic classes are required in the training set')
    rng = np.random.default_rng(seed)
    return [int(rng.choice(indices[int(rng.integers(2))])) for _ in range(steps)]
