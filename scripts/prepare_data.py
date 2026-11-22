"""Stream open datasets from Hugging Face, tokenise with the Qwen3.5 tokenizer, write uint32 shards.

  python -m scripts.prepare_data --mix configs/data_phase1.json --out data/phase1 [--upload denizulkuu/candidlm-data]

The mix file lists sources: {"name": {"dataset", "config", "split", "field" | "messages", "tokens", "weight"}}.
"""

from __future__ import annotations

import argparse
import json
import os

from candidlm.data import update_manifest, write_shards


def stream_texts(src: dict):
    from datasets import load_dataset
    ds = load_dataset(src["dataset"], src.get("config"), split=src.get("split", "train"), streaming=True)
    for row in ds:
        if "messages" in src:
            yield row[src["messages"]]
        else:
            t = row.get(src.get("field", "text"))
            if t:
                yield t


def token_stream(src: dict, tok, budget: int, batch: int = 256):
    eos = tok.eos_token_id
    produced, buf = 0, []

    def encode(items):
        if "messages" in src:
            return [tok.apply_chat_template(m, tokenize=True) for m in items]
        return tok(items, add_special_tokens=False)["input_ids"]

    for item in stream_texts(src):
        buf.append(item)
        if len(buf) == batch:
            for ids in encode(buf):
                ids = list(ids) + [eos]
                produced += len(ids)
                yield ids
                if produced >= budget:
                    return
            buf = []
    for ids in encode(buf) if buf else []:
        yield list(ids) + [eos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every token budget (e.g. 0.001 for a smoke test)")
    ap.add_argument("--upload", help="private HF dataset repo to upload the shards to")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    mix = json.load(open(a.mix))
    for name, src in mix.items():
        budget = int(src["tokens"] * a.scale)
        print(f"[data] {name}: {src['dataset']} -> {budget/1e6:.1f}M tokens", flush=True)
        entry = write_shards(token_stream(src, tok, budget), a.out, name)
        update_manifest(a.out, name, entry, src["weight"])
        print(f"[data] {name}: wrote {entry['tokens']/1e6:.1f}M tokens in {len(entry['shards'])} shards", flush=True)
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(a.upload, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=a.upload, repo_type="dataset", folder_path=a.out,
                          path_in_repo=os.path.basename(a.out.rstrip("/")))
        print(f"[data] uploaded to {a.upload}", flush=True)


if __name__ == "__main__":
    main()
