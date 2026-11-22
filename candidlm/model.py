"""CandidLM model: hybrid Gated DeltaNet + gated GQA, fine-grained MoE, MTP head.

Written from scratch in plain PyTorch. The only optional dependency is the
`flash-linear-attention` (fla) package, whose Triton kernels are used for the
DeltaNet recurrence on GPU when available. A pure-PyTorch chunked version is
always available and is what the tests use.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import CandidConfig

try:  # optional fast kernels
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdr  # type: ignore
except Exception:  # pragma: no cover - fla is only installed on GPU boxes
    _fla_chunk_gdr = None


# --------------------------------------------------------------------------- norms


class RMSNorm(nn.Module):
    """Zero-centred RMSNorm: y = x / rms(x) * (1 + w). Matches Qwen3.5."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """RMSNorm followed by a SiLU output gate (used inside DeltaNet). Weight is one-centred."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        xf = self.weight * xf.to(dtype)
        return (xf * F.silu(gate.float())).to(dtype)


# --------------------------------------------------------------------------- rope


class Rotary(nn.Module):
    def __init__(self, rotary_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        freqs = positions[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; rope is applied on the first `cos.shape[-1]` dims only."""
    rd = cos.shape[-1]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


# --------------------------------------------------------------------------- full attention


class GatedAttention(nn.Module):
    """GQA with q/k RMSNorm, partial RoPE, and a sigmoid output gate (Qwen3.5 style)."""

    def __init__(self, cfg: CandidConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_kv_heads
        self.hd = cfg.head_dim
        h = cfg.hidden_size
        # q_proj produces [query | gate] per head, interleaved per head like Qwen3.5
        self.q_proj = nn.Linear(h, self.n_heads * self.hd * 2, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, h, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qg = self.q_proj(x).view(B, T, self.n_heads, self.hd * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, T, self.n_heads * self.hd)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, self.n_heads * self.hd)
        o = o * torch.sigmoid(gate)
        return self.o_proj(o)


# --------------------------------------------------------------------------- gated deltanet


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gated_delta_rule_recurrent(q, k, v, g, beta):
    """Reference token-by-token gated delta rule. q,k: [B,T,H,Dk]; v: [B,T,H,Dv]; g,beta: [B,T,H].

    S_t = exp(g_t) * S_{t-1} + beta_t * k_t (v_t - S_{t-1}^T k_t)^T ;  o_t = S_t^T q_t
    (q and k are expected already l2-normalised; q is scaled by 1/sqrt(Dk) here.)
    """
    dtype = q.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    q = q * Dk**-0.5
    S = q.new_zeros(B, H, Dk, Dv)
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt, qt = k[:, t], v[:, t], q[:, t]
        pred = torch.einsum("bhkv,bhk->bhv", S, kt)
        delta = (vt - pred) * beta[:, t, :, None]
        S = S + kt[..., :, None] * delta[..., None, :]
        out.append(torch.einsum("bhkv,bhk->bhv", S, qt))
    return torch.stack(out, dim=1).to(dtype)


def gated_delta_rule_chunked(q, k, v, g, beta, chunk: int = 64):
    """Chunked (WY / UT-transform) gated delta rule, same semantics as the recurrent version.

    Within a chunk the delta-rule updates are folded into a triangular solve, so the
    sequential loop runs over T/chunk steps instead of T.
    """
    dtype = q.dtype
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    q, k, v, g, beta = (t.float().transpose(1, 2) for t in (q, k, v, g, beta))  # [B,H,T,*]
    q = q * Dk**-0.5
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g, beta = (F.pad(t, (0, pad)) for t in (g, beta))
    n = (T + pad) // chunk
    q, k, v = (t.reshape(B, H, n, chunk, -1) for t in (q, k, v))
    g = g.reshape(B, H, n, chunk)
    beta = beta.reshape(B, H, n, chunk)

    gc = g.cumsum(-1)                                   # log decay from chunk start to t (inclusive)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    decay = (gc[..., :, None] - gc[..., None, :]).masked_fill(upper, float("-inf")).exp()  # [.., i, j] i>=j

    kb = k * beta[..., None]
    vb = v * beta[..., None]
    # A = I + strictly-lower(beta_i k_i.k_j decay_ij); solve A u = v*beta and A w = k*beta*exp(gc)
    A = (kb @ k.transpose(-1, -2)) * decay
    A = A.masked_fill(~torch.ones_like(upper).tril(-1), 0.0) + torch.eye(chunk, device=q.device)
    u = torch.linalg.solve_triangular(A, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(A, kb * gc.exp()[..., None], upper=False, unitriangular=True)

    attn = (q @ k.transpose(-1, -2)) * decay            # intra-chunk, causal incl. diagonal
    q_dec = q * gc.exp()[..., None]
    k_dec = k * (gc[..., -1:] - gc).exp()[..., None]
    chunk_decay = gc[..., -1].exp()[..., None, None]

    S = q.new_zeros(B, H, Dk, Dv)
    outs = []
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ S
        o = q_dec[:, :, i] @ S + attn[:, :, i] @ v_new
        outs.append(o)
        S = S * chunk_decay[:, :, i] + k_dec[:, :, i].transpose(-1, -2) @ v_new
    o = torch.stack(outs, dim=2).reshape(B, H, n * chunk, Dv)[:, :, :T]
    return o.transpose(1, 2).to(dtype)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (linear attention with delta-rule state updates)."""

    def __init__(self, cfg: CandidConfig):
        super().__init__()
        h = cfg.hidden_size
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        conv_dim = self.key_dim * 2 + self.value_dim
        self.kernel = cfg.linear_conv_kernel
        self.in_proj_qkv = nn.Linear(h, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(h, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(h, self.nv, bias=False)
        self.in_proj_a = nn.Linear(h, self.nv, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, self.kernel, groups=conv_dim, bias=False,
                                padding=self.kernel - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.nv))
        self.A_log = nn.Parameter(torch.empty(self.nv).uniform_(0.01, 16).log())
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, h, bias=False)
        self.use_kernel = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)
        qkv = F.silu(self.conv1d(qkv)[..., :T]).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.nk, self.dk)
        k = k.reshape(B, T, self.nk, self.dk)
        v = v.reshape(B, T, self.nv, self.dv)
        z = self.in_proj_z(x).reshape(B, T, self.nv, self.dv)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        if self.nv != self.nk:
            rep = self.nv // self.nk
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        if _fla_chunk_gdr is not None and x.is_cuda and self.use_kernel:
            o, _ = _fla_chunk_gdr(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        else:
            q, k = _l2norm(q.float()).to(q.dtype), _l2norm(k.float()).to(k.dtype)
            o = gated_delta_rule_chunked(q, k, v, g, beta)
        o = self.norm(o.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(B, T, self.value_dim)
        return self.out_proj(o)


# --------------------------------------------------------------------------- feed-forward


class MLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """Fine-grained MoE: 1 always-on shared expert + top-k of N routed experts.

    Routing uses sigmoid scores; selection adds a per-expert bias that is nudged
    after every step towards balanced load (aux-loss-free balancing, DeepSeek-V3
    style). The bias only changes *which* experts are picked, never the weights.
    """

    def __init__(self, cfg: CandidConfig):
        super().__init__()
        h, e, d = cfg.hidden_size, cfg.num_experts, cfg.expert_intermediate_size
        self.n_experts, self.top_k, self.scale = e, cfg.num_experts_per_tok, cfg.routed_scaling
        self.router = nn.Linear(h, e, bias=False)
        # stacked expert weights: [E, out, in]
        self.w_gate = nn.Parameter(torch.empty(e, d, h))
        self.w_up = nn.Parameter(torch.empty(e, d, h))
        self.w_down = nn.Parameter(torch.empty(e, h, d))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)
        self.shared = MLP(h, cfg.shared_expert_intermediate_size)
        self.register_buffer("route_bias", torch.zeros(e))
        self.register_buffer("load", torch.zeros(e), persistent=False)  # tokens per expert, this step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(self.router(x).float())                       # [N, E]
        _, idx = torch.topk(scores + self.route_bias, self.top_k, dim=-1)    # bias only for selection
        w = scores.gather(-1, idx)
        w = (w / w.sum(-1, keepdim=True)) * self.scale                       # [N, k]
        if self.training:
            self.load += torch.bincount(idx.flatten(), minlength=self.n_experts).to(self.load.dtype)

        out = torch.zeros_like(x)
        flat_idx = idx.flatten()
        order = flat_idx.argsort()
        counts = torch.bincount(flat_idx, minlength=self.n_experts).tolist()
        tok = (order // self.top_k)
        wts = w.flatten()[order].to(x.dtype)
        start = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            t = tok[start:start + c]
            xe = x[t]
            he = F.silu(xe @ self.w_gate[e].t()) * (xe @ self.w_up[e].t())
            ye = (he @ self.w_down[e].t()) * wts[start:start + c, None]
            out.index_add_(0, t, ye)
            start += c
        out = out + self.shared(x)
        return out.reshape(shape)

    @torch.no_grad()
    def update_bias(self, rate: float) -> None:
        """Push under-used experts up and over-used ones down, then reset the counter."""
        if self.load.sum() == 0:
            return
        err = self.load.mean() - self.load
        self.route_bias += rate * torch.sign(err)
        self.load.zero_()


# --------------------------------------------------------------------------- blocks / model


class Block(nn.Module):
    def __init__(self, cfg: CandidConfig, layer_idx: int, force_dense: bool = False, force_full: bool = False):
        super().__init__()
        self.kind = "full_attention" if force_full else cfg.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if self.kind == "full_attention":
            self.self_attn = GatedAttention(cfg)
        else:
            self.linear_attn = GatedDeltaNet(cfg)
        if cfg.is_moe_layer(layer_idx) and not force_dense:
            self.mlp = MoE(cfg)
        else:
            self.mlp = MLP(cfg.hidden_size, cfg.dense_intermediate_size)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin) if self.kind == "full_attention" else self.linear_attn(h)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class MTPHead(nn.Module):
    """One extra block that predicts token t+2 from (hidden_t, embedding of token t+1).

    Layout follows the Qwen3-Next / Qwen3.5 MTP module so the released MTP weights load:
    fc(cat[norm_e(emb), norm_h(hidden)]) -> full-attention block -> norm -> shared lm head.
    """

    def __init__(self, cfg: CandidConfig):
        super().__init__()
        h = cfg.hidden_size
        self.pre_fc_norm_embedding = RMSNorm(h, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(h, cfg.rms_norm_eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = nn.ModuleList([Block(cfg, 0, force_dense=True, force_full=True)])
        self.norm = RMSNorm(h, cfg.rms_norm_eps)

    def forward(self, hidden, next_emb, cos, sin):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(next_emb), self.pre_fc_norm_hidden(hidden)], -1))
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


def chunked_ce(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 1024,
               ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy over a huge vocab without materialising all logits at once.

    Each chunk's logits are recomputed in backward (checkpointing), so peak memory is
    chunk x vocab instead of tokens x vocab.
    """
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    n_valid = (targets != ignore_index).sum().clamp(min=1)

    def _piece(h, t):
        logits = (h @ weight.t()).float()
        return F.cross_entropy(logits, t, ignore_index=ignore_index, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, hidden.shape[0], chunk):
        h, t = hidden[s:s + chunk], targets[s:s + chunk]
        if torch.is_grad_enabled():
            total = total + checkpoint(_piece, h, t, use_reentrant=False)
        else:
            total = total + _piece(h, t)
    return total / n_valid


class CandidLM(nn.Module):
    def __init__(self, cfg: CandidConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = Rotary(cfg.rotary_dim, cfg.rope_theta)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.mtp = MTPHead(cfg) if cfg.mtp else None
        self.grad_checkpointing = False

    # -- helpers
    @property
    def head_weight(self) -> torch.Tensor:
        return self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    def moe_layers(self):
        return [l.mlp for l in self.layers if isinstance(l.mlp, MoE)]

    def update_router_biases(self) -> None:
        for m in self.moe_layers():
            m.update_bias(self.cfg.router_bias_update_rate)

    def num_params(self, active: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not active:
            return total
        inactive = 0
        for m in self.moe_layers():
            per_expert = m.w_gate[0].numel() + m.w_up[0].numel() + m.w_down[0].numel()
            inactive += per_expert * (m.n_experts - m.top_k)
        return total - inactive

    # -- forward
    def hidden_states(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor | None = None):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        T = x.shape[1]
        cos, sin = self.rotary(torch.arange(T, device=x.device), x.dtype)
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        return self.norm(x), cos, sin

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """Returns logits when targets is None, else a dict of losses.

        targets[t] is the token after input_ids[t]. The MTP head uses targets[t+1].
        """
        h, cos, sin = self.hidden_states(input_ids)
        if targets is None:
            return h @ self.head_weight.t()
        loss_main = chunked_ce(h, self.head_weight, targets)
        out = {"loss_main": loss_main, "loss": loss_main}
        if self.mtp is not None and h.shape[1] > 1:
            nxt = targets.clamp(min=0)  # embedding of token t+1
            m = self.mtp(h, self.embed_tokens(nxt), cos, sin)
            mtp_targets = torch.full_like(targets, -100)
            mtp_targets[:, :-1] = targets[:, 1:]
            loss_mtp = chunked_ce(m, self.head_weight, mtp_targets)
            out["loss_mtp"] = loss_mtp
            out["loss"] = loss_main + self.cfg.mtp_loss_weight * loss_mtp
        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 0.0):
        """Simple full-recompute sampler, good enough for evals and smoke tests."""
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -self.cfg.max_seq_len:])[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids
