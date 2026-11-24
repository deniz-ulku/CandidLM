"""Phase 4 on the RTX 3070: GRPO reinforcement learning with a LoRA adapter, on checkable math.

For each prompt we sample G answers, score them (correct final answer = 1, plus small format
and length terms), and push up answers that beat their group's average. On-policy, one update
per batch, so no clipping or reference model is needed (the LoRA keeps changes small).

  python -m scripts.grpo --init runs/after_phase3 --out runs/grpo --steps 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import torch
import torch.nn.functional as F

from candidlm.config import CandidConfig
from candidlm.lora import add_lora, load_lora, lora_state_dict
from candidlm.model import CandidLM
from candidlm.sft import FORMAL_SYSTEM

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def final_number(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    cand = m[-1] if m else None
    if cand is None:
        nums = NUM.findall(text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").strip().rstrip(".")
    try:
        f = float(cand)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return cand


def reward(text: str, gold: str, max_len: int, n_tokens: int) -> float:
    r = 1.0 if final_number(text) == final_number(gold) else 0.0
    r += 0.1 if "\\boxed{" in text else 0.0
    if n_tokens >= max_len:  # ran out of room: likely a thinking loop
        r -= 0.2
    return r


def gsm8k(split="train"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [(r["question"], r["answer"].split("####")[-1].strip()) for r in ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=10)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CandidConfig.load(os.path.join(a.init, "config.json"))
    model = CandidLM(cfg)
    model.load_state_dict(torch.load(os.path.join(a.init, "model.pt"), map_location="cpu"))
    model.mtp = None  # not needed for RL; saves memory
    add_lora(model, r=a.rank, alpha=2 * a.rank)
    model.to(dev, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    os.makedirs(a.out, exist_ok=True)
    state_path = os.path.join(a.out, "grpo_state.pt")
    step = 0
    if os.path.exists(state_path):  # resume after a PC restart
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        load_lora(model, st["lora"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"[grpo] resumed at step {step}")
    data = gsm8k()
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    while step < a.steps:
        t0 = time.time()
        pick = random.Random(step)  # deterministic per step, so a resume continues the same curriculum
        batch = [data[i] for i in pick.sample(range(len(data)), a.prompts_per_step)]
        opt.zero_grad(set_to_none=True)
        stats = []
        for q, gold in batch:
            msgs = [{"role": "system", "content": FORMAL_SYSTEM},
                    {"role": "user", "content": q + "\nPut the final answer in \\boxed{}."}]
            p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
            p_ids = list(p_ids["input_ids"] if isinstance(p_ids, dict) else p_ids)
            prompt = torch.tensor([p_ids] * a.group, device=dev)
            model.eval()
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                seqs, _ = model.generate_cached(prompt, a.max_new, temperature=a.temperature,
                                                top_p=0.95, eos_id=tok.eos_token_id)
            comp = seqs[:, prompt.shape[1]:]
            lens = (comp != tok.eos_token_id).sum(1)
            texts = [tok.decode(c[:l]) for c, l in zip(comp, lens)]
            rs = torch.tensor([reward(t, gold, a.max_new, int(l)) for t, l in zip(texts, lens)], device=dev)
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            stats.append(rs.mean().item())
            if rs.std() < 1e-6:
                continue  # all answers equally good/bad: no signal
            model.train()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(seqs[:, :-1])
            logp = F.log_softmax(logits.float(), -1).gather(-1, seqs[:, 1:, None])[..., 0]
            mask = torch.zeros_like(logp)
            for i, l in enumerate(lens):
                mask[i, prompt.shape[1] - 1: prompt.shape[1] - 1 + int(l) + 1] = 1.0
            loss = -((logp * mask).sum(1) / mask.sum(1).clamp(min=1) * adv).mean() / len(batch)
            loss.backward()
            del logits, logp
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        rec = {"step": step, "reward": sum(stats) / len(stats), "sec": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[grpo] step {step} mean reward {rec['reward']:.3f} ({rec['sec']:.0f}s)", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"lora": lora_state_dict(model), "opt": opt.state_dict(), "step": step},
                       state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
    torch.save(lora_state_dict(model), os.path.join(a.out, "grpo_lora.pt"))


if __name__ == "__main__":
    main()
