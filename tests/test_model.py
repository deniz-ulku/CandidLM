import torch
import pytest

from candidlm.config import tiny_config
from candidlm.model import CandidLM, gated_delta_rule_chunked, gated_delta_rule_recurrent, _l2norm
from candidlm.upcycle import load_into, split_text_state


def test_chunked_matches_recurrent():
    torch.manual_seed(0)
    B, T, H, Dk, Dv = 2, 77, 3, 8, 5
    q = _l2norm(torch.randn(B, T, H, Dk))
    k = _l2norm(torch.randn(B, T, H, Dk))
    v = torch.randn(B, T, H, Dv)
    g = -torch.rand(B, T, H)
    beta = torch.rand(B, T, H)
    a = gated_delta_rule_recurrent(q, k, v, g, beta)
    b = gated_delta_rule_chunked(q, k, v, g, beta, chunk=16)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def _hf_tiny():
    tr = pytest.importorskip("transformers")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    hc = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        tie_word_embeddings=True, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                   "partial_rotary_factor": 0.25},
    )
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(hc).eval()
    # make norms non-trivial so the test checks them
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
    return hc, m


def _ours_from_hf(hf_model, **over):
    cfg = tiny_config(rope_theta=10000.0, partial_rotary_factor=0.25, mtp=False, **over)
    ours = CandidLM(cfg).eval()
    sd = {"model." + k[len("model."):] if k.startswith("model.") else k: v
          for k, v in hf_model.state_dict().items()}
    text, mtp, vision, other = split_text_state(sd)
    rep = load_into(ours, text, mtp, drop_ratio=0.0, verbose=False)
    return ours, rep


def test_dense_matches_hf_qwen35():
    hc, hf = _hf_tiny()
    ours, rep = _ours_from_hf(hf, moe=False)
    assert rep["missing"] == [], rep
    ids = torch.randint(0, 512, (2, 40))
    with torch.no_grad():
        a = hf(ids).logits
        b = ours(ids)
    assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()


def test_upcycle_is_exact_with_one_copy_per_slice():
    hc, hf = _hf_tiny()
    # dense FFN 128 = 8 slices of 16; shared = 2 slices; 6 routed experts, top-6 -> exact
    ours, rep = _ours_from_hf(hf, moe=True, num_experts=6, num_experts_per_tok=6,
                              expert_intermediate_size=16, shared_expert_intermediate_size=32)
    with torch.no_grad():
        for m in ours.moe_layers():
            m.router.weight.zero_()  # uniform scores -> every weight is exactly 1
    ids = torch.randint(0, 512, (2, 40))
    with torch.no_grad():
        a = hf(ids).logits
        b = ours(ids)
    assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()


def test_upcycled_moe_is_close_and_trains():
    hc, hf = _hf_tiny()
    ours, _ = _ours_from_hf(hf, moe=True, num_experts=8, num_experts_per_tok=6,
                            expert_intermediate_size=16, shared_expert_intermediate_size=32)
    ids = torch.randint(0, 512, (2, 40))
    ours.train()
    out = ours(ids[:, :-1], ids[:, 1:])
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert ours.moe_layers()[0].w_gate.grad is not None
    ours.update_router_biases()
    assert ours.moe_layers()[0].route_bias.abs().sum() > 0


def test_mtp_and_params():
    cfg = tiny_config()
    m = CandidLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 33))
    out = m(ids[:, :-1], ids[:, 1:])
    assert set(out) >= {"loss", "loss_main", "loss_mtp"}
    assert m.num_params(active=True) < m.num_params()


def test_cached_decoding_matches_full_forward():
    torch.manual_seed(0)
    m = CandidLM(tiny_config()).eval()
    ids = torch.randint(0, 512, (2, 21))
    full = m(ids)
    cache = {}
    pre = m.step(ids[:, :13], cache)
    steps = [m.step(ids[:, t:t + 1], cache) for t in range(13, 21)]
    inc = torch.cat([pre] + steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()
    out, lp = m.generate_cached(ids[:, :5], max_new_tokens=6)
    ref = m.generate(ids[:, :5], max_new_tokens=6)
    assert torch.equal(out, ref)


def test_lora_wraps_and_merges():
    from candidlm.lora import add_lora, lora_state_dict, merge_all
    m = CandidLM(tiny_config()).eval()
    ids = torch.randint(0, 512, (1, 16))
    base = m(ids)
    names = add_lora(m, r=4)
    assert names and all(p.requires_grad == ("lora_" in n) for n, p in m.named_parameters())
    assert torch.allclose(m(ids), base, atol=1e-5)  # B starts at zero
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.02)
    adapted = m(ids)
    merge_all(m)
    assert torch.allclose(m(ids), adapted, atol=1e-4)
    assert len(lora_state_dict(m)) == 2 * len(names)
