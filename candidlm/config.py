"""Model configuration for CandidLM.

CandidLM is a hybrid Gated-DeltaNet / gated-GQA transformer with a fine-grained
Mixture-of-Experts feed-forward and a multi-token-prediction (MTP) head.

The default values match the layout of Qwen3.5-0.8B-Base (24 layers, hidden 1024,
18 DeltaNet + 6 full-attention layers) so its weights can be upcycled into it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class CandidConfig:
    vocab_size: int = 248320
    hidden_size: int = 1024
    num_layers: int = 24
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    # Layer pattern: every `full_attention_interval`-th layer is full attention,
    # the rest are Gated DeltaNet (3:1 for interval 4).
    full_attention_interval: int = 4

    # Full (gated GQA) attention
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25

    # Gated DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel: int = 4

    # Feed-forward
    dense_intermediate_size: int = 3584  # dense FFN width (first layers, MTP layer)
    num_dense_layers: int = 2            # first N layers stay dense
    moe: bool = True
    num_experts: int = 16                # routed experts
    num_experts_per_tok: int = 6         # top-k
    expert_intermediate_size: int = 448
    shared_expert_intermediate_size: int = 896
    routed_scaling: float = 6.0          # routed output is scaled by this (keeps scale of the dense FFN)
    router_bias_update_rate: float = 1e-3  # aux-loss-free balancing step size

    # Multi-token prediction
    mtp: bool = True
    mtp_loss_weight: float = 0.3

    max_seq_len: int = 8192

    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_layers)
            ]
        assert len(self.layer_types) == self.num_layers

    def is_moe_layer(self, i: int) -> bool:
        return self.moe and i >= self.num_dense_layers

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CandidConfig":
        with open(path) as f:
            return cls(**json.load(f))


def tiny_config(**overrides) -> CandidConfig:
    """A very small config for tests and laptop smoke runs."""
    base = dict(
        vocab_size=512,
        hidden_size=64,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        dense_intermediate_size=128,
        num_dense_layers=1,
        num_experts=8,
        num_experts_per_tok=6,
        expert_intermediate_size=16,
        shared_expert_intermediate_size=32,
        max_seq_len=256,
    )
    base.update(overrides)
    return CandidConfig(**base)
