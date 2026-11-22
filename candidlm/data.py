"""Deterministic, resumable token loader over pre-tokenised uint32 shards.

Layout of a data dir:
  <dir>/<source>/shard_00000.bin   raw little-endian uint32 token ids, documents joined with EOS
  <dir>/manifest.json              {"sources": {"<source>": {"weight": 0.4, "shards": [...], "tokens": N}}}

Each source is cut into fixed windows of seq_len+1 tokens, visited in a seeded random
order. Which source feeds each sample is also seeded. The whole position is a few
integers (`state_dict()`), so a resumed run sees exactly the batches it would have seen.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch


class _Source:
    def __init__(self, root: str, name: str, info: dict, window: int, seed: int):
        self.name = name
        self.arrays = [np.memmap(os.path.join(root, name, s), dtype=np.uint32, mode="r") for s in info["shards"]]
        self.windows = []  # (shard, start)
        for si, a in enumerate(self.arrays):
            for st in range(0, len(a) - window + 1, window):
                self.windows.append((si, st))
        if not self.windows:
            raise ValueError(f"source {name} has fewer than {window} tokens")
        self.window = window
        self.seed = seed
        self.epoch = -1
        self.perm = None

    def get(self, cursor: int) -> np.ndarray:
        epoch, pos = divmod(cursor, len(self.windows))
        if epoch != self.epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self.perm = rng.permutation(len(self.windows))
            self.epoch = epoch
        si, st = self.windows[self.perm[pos]]
        return np.asarray(self.arrays[si][st:st + self.window], dtype=np.int64)


class TokenLoader:
    def __init__(self, root: str, seq_len: int, batch_size: int, seed: int = 1234,
                 weights: dict[str, float] | None = None):
        with open(os.path.join(root, "manifest.json")) as f:
            man = json.load(f)["sources"]
        if weights:
            man = {k: dict(v, weight=weights[k]) for k, v in man.items() if weights.get(k, 0) > 0}
        self.names = sorted(man)
        w = np.array([man[n].get("weight", 1.0) for n in self.names], dtype=np.float64)
        self.p = w / w.sum()
        self.sources = [_Source(root, n, man[n], seq_len + 1, seed + i) for i, n in enumerate(self.names)]
        self.seq_len, self.batch_size, self.seed = seq_len, batch_size, seed
        self.sample = 0                               # global sample counter
        self.cursors = [0] * len(self.sources)        # per-source window counter

    def next_batch(self):
        rows, srcs = [], []
        for _ in range(self.batch_size):
            r = np.random.default_rng([self.seed, 7, self.sample]).random()
            s = int(np.searchsorted(np.cumsum(self.p), r, side="right"))
            s = min(s, len(self.sources) - 1)
            rows.append(self.sources[s].get(self.cursors[s]))
            self.cursors[s] += 1
            self.sample += 1
            srcs.append(s)
        x = torch.from_numpy(np.stack(rows))
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    def state_dict(self) -> dict:
        return {"sample": self.sample, "cursors": list(self.cursors), "names": self.names, "seed": self.seed}

    def load_state_dict(self, sd: dict) -> None:
        if sd["names"] != self.names:
            raise ValueError(f"data sources changed: {sd['names']} vs {self.names}")
        self.sample = sd["sample"]
        self.cursors = list(sd["cursors"])


def write_shards(token_iter, out_dir: str, source: str, shard_tokens: int = 50_000_000) -> dict:
    """Write an iterator of token-id lists into uint32 shards. Returns the manifest entry."""
    d = os.path.join(out_dir, source)
    os.makedirs(d, exist_ok=True)
    shards, buf, total, idx = [], [], 0, 0

    def flush():
        nonlocal buf, idx
        if not buf:
            return
        name = f"shard_{idx:05d}.bin"
        np.concatenate(buf).astype(np.uint32).tofile(os.path.join(d, name))
        shards.append(name)
        buf, idx = [], idx + 1

    n_buf = 0
    for ids in token_iter:
        a = np.asarray(ids, dtype=np.uint32)
        buf.append(a)
        n_buf += len(a)
        total += len(a)
        if n_buf >= shard_tokens:
            flush()
            n_buf = 0
    flush()
    return {"shards": shards, "tokens": int(total)}


def update_manifest(out_dir: str, source: str, entry: dict, weight: float) -> None:
    path = os.path.join(out_dir, "manifest.json")
    man = {"sources": {}}
    if os.path.exists(path):
        with open(path) as f:
            man = json.load(f)
    man["sources"][source] = dict(entry, weight=weight)
    with open(path, "w") as f:
        json.dump(man, f, indent=2)
