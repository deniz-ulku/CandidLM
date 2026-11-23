"""Phase 2: add a teacher's top-K next-token distribution to SFT shards (for logit distillation).

The teacher must share CandidLM's tokenizer (any Qwen3.5 model does), so ids line up exactly.

  python -m scripts.teacher_logits --data data/sft --teacher Qwen/Qwen3.5-4B --k 32
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=None, help="stop after this many windows (budget)")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model.to(dev).eval()
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    done_rows = 0
    for name in man["shards"]:
        path = os.path.join(a.data, name)
        sh = dict(np.load(path))
        if "topk_idx" in sh:
            continue  # resumable: shard already has teacher logits
        tokens = sh["tokens"]
        N, W = tokens.shape
        idx = np.zeros((N, W - 1, a.k), dtype=np.int32)
        logp = np.zeros((N, W - 1, a.k), dtype=np.float16)
        for s in range(0, N, a.batch):
            x = torch.from_numpy(tokens[s:s + a.batch, :-1].astype(np.int64)).to(dev)
            lg = model(input_ids=x).logits.float()
            lp = torch.log_softmax(lg, -1)
            v, i = lp.topk(a.k, dim=-1)
            idx[s:s + a.batch] = i.cpu().numpy()
            logp[s:s + a.batch] = v.cpu().numpy().astype(np.float16)
        sh["topk_idx"], sh["topk_logp"] = idx, logp
        np.savez(path + ".tmp.npz", **sh)
        os.replace(path + ".tmp.npz", path)
        done_rows += N
        print(f"[teacher] {name}: {N} windows", flush=True)
        if a.max_rows and done_rows >= a.max_rows:
            break


if __name__ == "__main__":
    main()
