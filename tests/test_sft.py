import numpy as np
import torch

from candidlm.config import tiny_config
from candidlm.model import CandidLM
from candidlm.sft import SFTLoader, pack, write_sft_shards
from candidlm.train import DEFAULTS, run


def _make_sft(d, with_teacher=True, k=4):
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(200):
        q = list(rng.integers(10, 500, rng.integers(3, 10)))
        a = [(t * 7) % 500 for t in q]  # learnable answer
        samples.append((q + a, [0] * len(q) + [1] * len(a)))
    write_sft_shards(pack(samples, 32, 0), d, per_shard=16)
    if with_teacher:
        import json, os
        man = json.load(open(os.path.join(d, "manifest.json")))
        for n in man["shards"][::2]:  # only half the shards get teacher logits
            p = os.path.join(d, n)
            sh = dict(np.load(p))
            N, W = sh["tokens"].shape
            sh["topk_idx"] = np.tile(sh["tokens"][:, 1:, None], (1, 1, k)).astype(np.int32)
            sh["topk_logp"] = np.log(np.full((N, W - 1, k), 1 / k)).astype(np.float16)
            np.savez(p, **sh)


def test_loader_masks_and_teacher(tmp_path):
    _make_sft(str(tmp_path))
    ld = SFTLoader(str(tmp_path), 8)
    b = ld.next_batch()
    assert (b["y"] == -100).any() and (b["y"] != -100).any()
    assert b["teacher"]["idx"].shape[:2] == b["x"].shape
    m = CandidLM(tiny_config())
    out = m(b["x"], b["y"], next_ids=b["next_ids"], teacher=b["teacher"], kd_weight=0.5)
    out["loss"].backward()
    assert torch.isfinite(out["loss_kd"])


def test_sft_training_resumes(tmp_path):
    _make_sft(str(tmp_path / "sft"))
    cfg = dict(DEFAULTS, tiny=True, data_type="sft", kd_weight=0.5, data_dir=str(tmp_path / "sft"),
               out_dir=str(tmp_path / "o"), seq_len=32, micro_batch=4, total_steps=12, warmup_steps=2,
               lr=3e-3, ckpt_minutes=1e9, log_every=1000, grad_checkpointing=False, freeze=[])
    a = run(dict(cfg, out_dir=str(tmp_path / "a")))
    b = run(dict(cfg, max_steps_this_run=5)) + run(cfg)
    assert np.allclose([r["loss"] for r in a], [r["loss"] for r in b], atol=1e-6)


class FakeQwenTok:
    """Mimics the parts of Qwen3.5's chat template that matter: refuses a prompt with no user turn,
    and drops <think> blocks from assistant turns that are not the last one."""

    def apply_chat_template(self, msgs, tokenize=True, return_dict=False, add_generation_prompt=False):
        if not any(m["role"] == "user" for m in msgs):
            raise ValueError("No user query found in messages.")
        text = ""
        for j, m in enumerate(msgs):
            c = m["content"]
            if m["role"] == "assistant" and j != len(msgs) - 1 and "</think>" in c:
                c = c.split("</think>")[-1].strip()
            text += f"<{m['role']}>{c}<end>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(ch) for ch in text]


def test_render_with_mask_real_template_rules():
    from candidlm.sft import render_with_mask
    tok = FakeQwenTok()
    msgs = [{"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>add</think>4"},
            {"role": "user", "content": "times 3?"},
            {"role": "assistant", "content": "<think>mul</think>12"}]
    ids, mask = render_with_mask(tok, msgs)
    text = "".join(chr(i) for i in ids)
    trained = "".join(chr(i) for i, m in zip(ids, mask) if m)
    assert text.startswith("<system>") and "12<end>" in text
    assert trained == "<think>mul</think>12<end>"  # last turn trained; rewritten earlier turn skipped safely
    plain = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."},
             {"role": "user", "content": "bye"}, {"role": "assistant", "content": "Goodbye."}]
    ids, mask = render_with_mask(tok, plain)
    assert "".join(chr(i) for i, m in zip(ids, mask) if m) == "Hello.<end>Goodbye.<end>"


def test_render_samples_recovers_rewritten_turns():
    from candidlm.sft import render_samples
    tok = FakeQwenTok()
    msgs = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "<think>add</think>4"},
            {"role": "user", "content": "times 3?"}, {"role": "assistant", "content": "<think>mul</think>12"}]
    trained = ["".join(chr(i) for i, m in zip(ids, mask) if m) for ids, mask in render_samples(tok, msgs)]
    assert trained == ["<think>mul</think>12<end>", "<think>add</think>4<end>"]
    plain = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."},
             {"role": "user", "content": "bye"}, {"role": "assistant", "content": "Goodbye."}]
    assert len(render_samples(tok, plain)) == 1  # nothing lost, no duplicates
