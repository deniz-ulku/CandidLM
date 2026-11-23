"""Build masked, packed SFT shards from open chat / reasoning datasets.

  python -m scripts.prepare_sft --mix configs/sft_phase3.json --out data/sft [--scale 0.01]

Each source: {"dataset", "config", "split", "messages": field, "windows": N, "max_tokens": L}.
Samples longer than the window are skipped (cut-off reasoning traces teach the wrong thing).
"""

import argparse
import json
import random

from candidlm.sft import pack, render_with_mask, write_sft_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    samples = []
    for name, src in json.load(open(a.mix)).items():
        want = int(src["samples"] * a.scale)
        ds = load_dataset(src["dataset"], src.get("config"), split=src.get("split", "train"), streaming=True)
        got = 0
        for row in ds:
            msgs = [{"role": m["role"], "content": m["content"]} for m in row[src.get("messages", "messages")]]
            ids, mask = render_with_mask(tok, msgs)
            if len(ids) > a.seq_len + 1 or sum(mask) == 0:
                continue
            samples.append((ids, mask))
            got += 1
            if got >= want:
                break
        print(f"[sft] {name}: {got} conversations", flush=True)
    random.Random(0).shuffle(samples)
    names = write_sft_shards(pack(samples, a.seq_len, pad), a.out)
    print(f"[sft] wrote {len(names)} shards to {a.out}", flush=True)


if __name__ == "__main__":
    main()
