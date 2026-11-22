"""Muon for hidden 2D (and stacked-expert 3D) matrices, AdamW for everything else."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise G (last two dims) with the quintic Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon (momentum + orthogonalised update). Update RMS is matched to AdamW's
    (0.2 * sqrt(max(rows, cols)), as in Moonlight) so both can share one LR schedule."""

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum" not in st:
                    st["momentum"] = torch.zeros_like(g)
                buf = st["momentum"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                u = newton_schulz(g, group["ns_steps"])
                scale = 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"] * scale)


def _is_muon_param(name: str, p: torch.Tensor) -> bool:
    if p.ndim < 2:
        return False
    if "embed_tokens" in name or "lm_head" in name or "router" in name or "conv1d" in name:
        return False
    return True


class CombinedOptimizer:
    """Muon + AdamW behind one step()/zero_grad()/state_dict() interface."""

    def __init__(self, model: torch.nn.Module, lr: float, weight_decay: float = 0.1,
                 betas=(0.9, 0.95), adam_eps: float = 1e-8):
        muon, adam_decay, adam_no_decay = [], [], []
        self.names = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_muon_param(n, p):
                muon.append(p)
            elif p.ndim >= 2:
                adam_decay.append(p)
            else:
                adam_no_decay.append(p)
        self.muon = Muon(muon, lr=lr, weight_decay=weight_decay) if muon else None
        groups = []
        if adam_decay:
            groups.append(dict(params=adam_decay, weight_decay=weight_decay))
        if adam_no_decay:
            groups.append(dict(params=adam_no_decay, weight_decay=0.0))
        fused = torch.cuda.is_available()
        self.adam = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=adam_eps, fused=fused) if groups else None
        self.opts = [o for o in (self.muon, self.adam) if o is not None]
        self.counts = dict(muon=sum(p.numel() for p in muon),
                           adam=sum(p.numel() for p in adam_decay + adam_no_decay))

    def set_lr(self, lr: float) -> None:
        for o in self.opts:
            for g in o.param_groups:
                g["lr"] = lr

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad(set_to_none=True)

    def state_dict(self):
        return {"muon": self.muon.state_dict() if self.muon else None,
                "adam": self.adam.state_dict() if self.adam else None}

    def load_state_dict(self, sd):
        if self.muon and sd.get("muon"):
            self.muon.load_state_dict(sd["muon"])
        if self.adam and sd.get("adam"):
            self.adam.load_state_dict(sd["adam"])


def wsd_lr(step: int, total: int, peak: float, warmup: int, decay_frac: float = 0.2, final_frac: float = 0.1):
    """Warmup-Stable-Decay schedule: linear warmup, flat, then linear decay over the last `decay_frac`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return peak
    t = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return peak * (1 - t * (1 - final_frac))
