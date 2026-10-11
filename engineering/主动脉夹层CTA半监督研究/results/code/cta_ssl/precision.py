"""Explicit compute precision, with the historical pilot policy preserved."""
import torch


def precision_policy(settings):
    value = settings.get('compute_precision', 'legacy')
    if value not in {'legacy', 'float32', 'bfloat16_native'}:
        raise ValueError('compute_precision must be legacy, float32 or bfloat16_native')
    return value


def autocast_context(settings, device, prediction=False):
    policy = precision_policy(settings)
    if policy == 'float32':
        # Explicitly disable even an enclosing autocast context, in both phases.
        return torch.autocast(device_type=device.type, enabled=False)
    if policy == 'bfloat16_native':
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError('Native CUDA BF16 is required by compute_precision')
        enabled = True
    else:
        # Historical prediction ignored training.amp. Preserve that behavior only
        # for old configurations/checkpoints, including Torch's emulation gate.
        enabled = (device.type == 'cuda' and (prediction or settings.get('amp', True))
                   and torch.cuda.is_bf16_supported())
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def check_checkpoint_precision(settings, meta):
    recorded = meta.get('compute_precision_policy', 'legacy')
    if precision_policy(settings) != recorded:
        raise ValueError('Checkpoint and current compute precision policies differ')
