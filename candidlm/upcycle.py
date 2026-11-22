"""Turn Qwen3.5 dense weights into a CandidLM MoE ("sparse upcycling").

Steps per MoE layer (dense FFN of width F = 3584 by default):
  1. Cut the FFN into F / expert_width slices (8 slices of 448).
  2. The first `shared/expert_width` slices (2 -> 896 wide) become the shared expert.
  3. The remaining slices (6) are assigned round-robin to the routed experts, so each
     slice has several copies. Copies after the first are partly re-randomised
     ("drop-upcycling") so the experts can specialise instead of staying clones.

Because the dense FFN output is exactly the sum of its slices, routing every token
to one copy of each slice with weight 1 reproduces the dense model. With top-6 over
6 slices and uniform weights that is exactly the init (see tests/test_upcycle.py).

Usage (on a machine that can reach Hugging Face):
  python -m candidlm.upcycle --src Qwen/Qwen3.5-0.8B-Base --out runs/upcycled
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch

from .config import CandidConfig
from .model import CandidLM, MoE

TEXT_PREFIXES = ("model.language_model.", "language_model.model.", "model.")


def config_from_hf(hf: dict, **overrides) -> CandidConfig:
    """Build a CandidConfig from a Qwen3.5 config.json (top-level or text_config)."""
    t = hf.get("text_config", hf)
    rope = t.get("rope_parameters") or t.get("rope_scaling") or {}
    cfg = dict(
        vocab_size=t["vocab_size"],
        hidden_size=t["hidden_size"],
        num_layers=t["num_hidden_layers"],
        rms_norm_eps=t.get("rms_norm_eps", 1e-6),
        tie_embeddings=hf.get("tie_word_embeddings", t.get("tie_word_embeddings", True)),
        num_attention_heads=t["num_attention_heads"],
        num_kv_heads=t["num_key_value_heads"],
        head_dim=t.get("head_dim", 256),
        rope_theta=rope.get("rope_theta", t.get("rope_theta", 10_000_000.0)),
        partial_rotary_factor=rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)),
        linear_num_key_heads=t["linear_num_key_heads"],
        linear_num_value_heads=t["linear_num_value_heads"],
        linear_key_head_dim=t["linear_key_head_dim"],
        linear_value_head_dim=t["linear_value_head_dim"],
        linear_conv_kernel=t.get("linear_conv_kernel_dim", 4),
        dense_intermediate_size=t["intermediate_size"],
        layer_types=t.get("layer_types") or [],
    )
    cfg.update(overrides)
    return CandidConfig(**cfg)


def split_text_state(sd: dict[str, torch.Tensor]):
    """Split a Qwen3.5 checkpoint into (text, mtp, vision, unknown) dicts with clean keys."""
    text, mtp, vision, other = {}, {}, {}, {}
    for k, v in sd.items():
        if k.startswith("mtp."):
            mtp[k[4:]] = v
        elif ".visual." in k or k.startswith("visual.") or k.startswith("model.visual."):
            vision[k] = v
        elif k == "lm_head.weight":
            text[k] = v
        else:
            for p in TEXT_PREFIXES:
                if k.startswith(p):
                    text[k[len(p):]] = v
                    break
            else:
                other[k] = v
    return text, mtp, vision, other


def _slices(w: torch.Tensor, width: int, dim: int):
    return list(torch.split(w, width, dim=dim))


@torch.no_grad()
def load_into(model: CandidLM, text_sd: dict, mtp_sd: dict | None = None, drop_ratio: float = 0.5,
              seed: int = 0, verbose: bool = True) -> dict:
    """Copy dense Qwen3.5 weights into `model`, upcycling FFNs where the model has MoE layers."""
    cfg = model.cfg
    gen = torch.Generator().manual_seed(seed)
    own = model.state_dict()
    loaded, report = set(), {"missing": [], "unused": []}

    def put(name: str, value: torch.Tensor):
        if own[name].shape != value.shape:
            raise ValueError(f"shape mismatch for {name}: ours {tuple(own[name].shape)} vs {tuple(value.shape)}")
        own[name].copy_(value.to(own[name].dtype))
        loaded.add(name)

    used = set()
    for k, v in text_sd.items():
        if k == "lm_head.weight":
            if model.lm_head is not None:
                put("lm_head.weight", v)
            used.add(k)
            continue
        m = re.match(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight", k)
        if m and isinstance(model.layers[int(m.group(1))].mlp, MoE):
            continue  # handled below
        if k in own:
            put(k, v)
            used.add(k)

    for i, layer in enumerate(model.layers):
        if not isinstance(layer.mlp, MoE):
            continue
        moe: MoE = layer.mlp
        g = text_sd[f"layers.{i}.mlp.gate_proj.weight"]   # [F, H]
        u = text_sd[f"layers.{i}.mlp.up_proj.weight"]     # [F, H]
        d = text_sd[f"layers.{i}.mlp.down_proj.weight"]   # [H, F]
        for n in ("gate_proj", "up_proj", "down_proj"):
            used.add(f"layers.{i}.mlp.{n}.weight")
        ew = cfg.expert_intermediate_size
        sw = cfg.shared_expert_intermediate_size
        assert sw % ew == 0 and g.shape[0] % ew == 0, "FFN width must split evenly into experts"
        n_shared = sw // ew
        gs, us, ds = _slices(g, ew, 0), _slices(u, ew, 0), _slices(d, ew, 1)
        # shared expert = first n_shared slices
        put(f"layers.{i}.mlp.shared.gate_proj.weight", torch.cat(gs[:n_shared], 0))
        put(f"layers.{i}.mlp.shared.up_proj.weight", torch.cat(us[:n_shared], 0))
        put(f"layers.{i}.mlp.shared.down_proj.weight", torch.cat(ds[:n_shared], 1))
        routed = list(range(n_shared, len(gs)))
        seen = set()
        for e in range(cfg.num_experts):
            s = routed[e % len(routed)]
            eg, eu, ed = gs[s].clone().float(), us[s].clone().float(), ds[s].clone().float()
            if s in seen and drop_ratio > 0:
                # drop-upcycling: re-init a random subset of intermediate units with matching stats
                mask = torch.rand(ew, generator=gen) < drop_ratio
                for w, axis in ((eg, 0), (eu, 0), (ed, 1)):
                    std = w.std().item()
                    noise = torch.randn(w.shape, generator=gen) * std
                    if axis == 0:
                        w[mask] = noise[mask]
                    else:
                        w[:, mask] = noise[:, mask]
            seen.add(s)
            moe.w_gate[e].copy_(eg)
            moe.w_up[e].copy_(eu)
            moe.w_down[e].copy_(ed)
        loaded.update({f"layers.{i}.mlp.w_gate", f"layers.{i}.mlp.w_up", f"layers.{i}.mlp.w_down"})
        # router: small random init, zero balancing bias
        torch.nn.init.normal_(moe.router.weight, std=0.006, generator=gen)
        loaded.add(f"layers.{i}.mlp.router.weight")

    if mtp_sd and model.mtp is not None:
        for k, v in mtp_sd.items():
            name = "mtp." + k
            if name in own and own[name].shape == v.shape:
                put(name, v)
                used.add("mtp:" + k)
            else:
                report["unused"].append("mtp." + k)

    report["unused"] += [k for k in text_sd if k not in used]
    report["missing"] = [k for k in own if k not in loaded and not k.endswith("route_bias")
                         and not k.startswith("rotary.")]
    if verbose:
        print(f"[upcycle] loaded {len(loaded)} tensors; missing {len(report['missing'])}; "
              f"unused {len(report['unused'])}")
        for k in report["missing"][:20]:
            print("  missing:", k)
        for k in report["unused"][:20]:
            print("  unused:", k)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-experts", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--dense", action="store_true", help="no MoE: plain dense copy (fallback path)")
    ap.add_argument("--drop-ratio", type=float, default=0.5)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = args.src if os.path.isdir(args.src) else snapshot_download(
        args.src, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
    with open(os.path.join(path, "config.json")) as f:
        hf_cfg = json.load(f)
    cfg = config_from_hf(hf_cfg, moe=not args.dense, num_experts=args.num_experts,
                         num_experts_per_tok=args.top_k)
    sd = {}
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(path, fn)))
    text, mtp, vision, other = split_text_state(sd)
    print(f"[upcycle] text {len(text)}, mtp {len(mtp)}, vision {len(vision)}, other {len(other)} tensors")

    model = CandidLM(cfg)
    report = load_into(model, text, mtp, drop_ratio=args.drop_ratio)
    os.makedirs(args.out, exist_ok=True)
    cfg.save(os.path.join(args.out, "config.json"))
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    if vision:
        torch.save(vision, os.path.join(args.out, "vision.pt"))  # kept for the vision phase
    with open(os.path.join(args.out, "upcycle_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    for fn in os.listdir(path):  # tokenizer files travel with the model
        if fn.startswith("tokenizer") or fn.endswith(".jinja") or fn in ("vocab.json", "merges.txt"):
            with open(os.path.join(path, fn), "rb") as a, open(os.path.join(args.out, fn), "wb") as b:
                b.write(a.read())
    print(f"[upcycle] total params {model.num_params()/1e9:.3f}B, "
          f"active {model.num_params(active=True)/1e9:.3f}B -> {args.out}")


if __name__ == "__main__":
    main()
