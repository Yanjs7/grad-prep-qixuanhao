import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(inc, outc, 3, stride, 1, bias=False),
            nn.GroupNorm(8, outc), nn.ReLU(),
            nn.Conv3d(outc, outc, 3, 1, 1, bias=False), nn.GroupNorm(8, outc))
        self.skip = nn.Identity() if inc == outc and stride == 1 else nn.Sequential(
            nn.Conv3d(inc, outc, 1, stride, bias=False), nn.GroupNorm(8, outc))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv3d(1, 32, 7, 2, 3, bias=False), nn.GroupNorm(8, 32), nn.ReLU()]
        previous = 32
        for stage, channels in enumerate([32, 64, 128, 256]):
            layers.extend([ResidualBlock(previous, channels, 1 if stage == 0 else 2),
                           ResidualBlock(channels, channels)])
            previous = channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return F.adaptive_avg_pool3d(self.layers(x), 1).flatten(1)


class MassEncoder(nn.Module):
    def __init__(self, cfg, load_pretrained=True):
        super().__init__()
        vendor = Path(cfg['vendor'])
        if not (vendor / 'models/modules/encoder.py').exists():
            raise FileNotFoundError(f'MASS source missing: {vendor}')
        sys.path.insert(0, str(vendor))
        from models.modules.encoder import UNet_Encoder
        self.core = UNet_Encoder(in_ch=1, channels=[32,64,128,256,512],
                                 scale=[2]*4, num_block=[2]*4, pool=True,
                                 norm='in', kernel_size=[3]*5)
        if load_pretrained:
            raw = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
            if 'ema_model' in raw:
                state = raw['ema_model']
                self.checkpoint_section = 'ema_model'
            elif 'ema_state_dict' in raw:
                state = raw['ema_state_dict']
                self.checkpoint_section = 'ema_state_dict'
            elif 'model' in raw:
                state = raw['model']
                self.checkpoint_section = 'model'
            elif 'model_state_dict' in raw:
                state = raw['model_state_dict']
                self.checkpoint_section = 'model_state_dict'
            else:
                state = raw
                self.checkpoint_section = 'root'
            state = {k.removeprefix('module.'): v for k,v in state.items()}
            encoder_state = {k.removeprefix('encoder.'):v for k,v in state.items()
                             if k.startswith('encoder.')}
            if not encoder_state:
                raise ValueError('No encoder weights found; refusing random fallback')
            self.core.load_state_dict(encoder_state, strict=True)
            del raw, state, encoder_state
        self.trainable = cfg.get('trainable', 'last_stage')
        if self.trainable not in {'frozen', 'last_stage', 'all'}:
            raise ValueError('Unknown MASS trainable setting')
        if self.trainable != 'all':
            for p in self.core.parameters():
                p.requires_grad_(False)
        if self.trainable == 'last_stage':
            for p in self.core.down4.parameters():
                p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.trainable != 'all':
            self.core.eval()
            if self.trainable == 'last_stage':
                self.core.down4.train(mode)
        return self

    def forward(self, x):
        return F.adaptive_avg_pool3d(self.core(x)[0], 1).flatten(1)


class PatientMIL(nn.Module):
    def __init__(self, cfg, load_pretrained=True):
        super().__init__()
        self.cfg = cfg
        if cfg['encoder'] not in {'mass','resnet18'} or cfg.get('pooling','attention') not in {'attention','max'}:
            raise ValueError('Unsupported encoder or pooling configuration')
        if cfg.get('head_precision', 'autocast') not in {'autocast', 'float32'}:
            raise ValueError('head_precision must be autocast or float32')
        self.encoder = MassEncoder(cfg, load_pretrained) if cfg['encoder'] == 'mass' else ResNet18()
        dim = 512 if cfg['encoder'] == 'mass' else 256
        self.position = nn.Sequential(nn.Linear(dim + 3, dim), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Linear(dim, 1)

    def forward(self, patches, positions):
        # One patient's [K,1,D,H,W], preserving a single patient-level loss.
        features = []
        trainable = any(p.requires_grad for p in self.encoder.parameters())
        chunk_size = int(self.cfg.get('encoder_chunk_size', 1))
        if chunk_size < 1:
            raise ValueError('encoder_chunk_size must be positive')
        # Encode genuine patches together; attention still pools the whole case.
        for chunk in patches.split(chunk_size):
            if not trainable:
                with torch.no_grad():
                    encoded = self.encoder(chunk)
            elif self.training and self.cfg.get('activation_checkpointing', True):
                encoded = checkpoint(self.encoder, chunk, use_reentrant=False)
            else:
                encoded = self.encoder(chunk)
            features.append(encoded)
        local = torch.cat(features)
        # Casting a rounded BF16 logit afterwards does not recover its precision.
        # An explicit FP32 head keeps attention reduction and patient logits in
        # FP32 while the expensive encoder can still use mixed precision.
        fp32_head = self.cfg.get('head_precision', 'autocast') == 'float32'
        context = torch.autocast(device_type=local.device.type, enabled=False) if fp32_head else nullcontext()
        with context:
            head_features = local.float() if fp32_head else local
            joint = self.position(torch.cat([head_features, positions.to(head_features.dtype)], dim=1))
            if self.cfg.get('pooling', 'attention') == 'max':
                pooled = joint.max(dim=0).values
                weights = torch.ones(len(joint), device=joint.device, dtype=joint.dtype) / len(joint)
            else:
                weights = torch.softmax(self.attention(joint).flatten(), dim=0)
                pooled = (joint * weights[:, None]).sum(dim=0)
            logit = self.classifier(pooled).squeeze(-1)
        return logit, weights, local


@torch.no_grad()
def update_ema(teacher, student, decay):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(decay).add_(sp.detach(), alpha=1-decay)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)
