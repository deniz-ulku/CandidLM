"""LoRA adapters for CandidLM (used on the RTX 3070: RL, persona, style).

The 1.08B base fits in 8 GB in BF16 (about 2.2 GB), so no 4-bit tricks are needed for training
adapters. Default targets: attention, DeltaNet projections and the always-on shared expert.
"""

from __future__ import annotations

import fnmatch
import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = [
    "layers.*.self_attn.q_proj", "layers.*.self_attn.k_proj", "layers.*.self_attn.v_proj",
    "layers.*.self_attn.o_proj",
    "layers.*.linear_attn.in_proj_qkv", "layers.*.linear_attn.in_proj_z", "layers.*.linear_attn.out_proj",
    "layers.*.mlp.shared.gate_proj", "layers.*.mlp.shared.up_proj", "layers.*.mlp.shared.down_proj",
    "layers.*.mlp.gate_proj", "layers.*.mlp.up_proj", "layers.*.mlp.down_proj",
]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.merged = False

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        return y + (self.drop(x) @ self.lora_A.t().to(x.dtype) @ self.lora_B.t().to(x.dtype)) * self.scale

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale
            self.merged = False


def add_lora(model: nn.Module, targets=None, r: int = 16, alpha: float = 32.0, dropout: float = 0.0) -> list[str]:
    """Freeze the model and wrap matching nn.Linear layers with LoRA. Returns wrapped names."""
    targets = targets or DEFAULT_TARGETS
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(fnmatch.fnmatch(full, t) for t in targets):
                setattr(mod, cname, LoRALinear(child, r, alpha, dropout))
                wrapped.append(full)
    return wrapped


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora(model: nn.Module, sd: dict) -> None:
    missing = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing.unexpected_keys]
    if bad:
        raise ValueError(f"adapter keys not in model: {bad[:5]}")


def merge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()
