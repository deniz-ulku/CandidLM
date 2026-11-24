"""Supervised fine-tuning data: chat rendering with loss masks, packing, shards, loader.

Shard format (`<dir>/sft_XXXXX.npz`):
  tokens    int32  [N, L+1]   packed conversations, padded with `pad_id`
  mask      uint8  [N, L+1]   1 where the token is trained (assistant turns), else 0
  topk_idx  int32  [N, L, K]  optional teacher top-K ids for predicting token t+1 from position t
  topk_logp f16    [N, L, K]  optional teacher log-probs for those ids
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

# CandidLM always answers formally; this goes into every training conversation.
FORMAL_SYSTEM = (
    "You are CandidLM. Always answer in a formal, polite and precise register, in the user's language "
    "(in Turkish, use the formal 'siz' form). Never switch to slang or a casual tone."
)


def _render(tok, msgs, **kw) -> list[int]:
    out = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, **kw)
    return list(out["input_ids"] if isinstance(out, dict) else out)


def render_with_mask(tok, messages: list[dict], system: str | None = FORMAL_SYSTEM, **tmpl_kwargs):
    """Tokenise a conversation with the model's chat template; mask = 1 on assistant tokens only.

    The whole conversation is rendered once. For each assistant turn i we render the prefix that
    ends right before it (with the generation prompt) and the prefix that ends with it; the tokens
    in between are that turn's answer. Prefixes are only ever cut at assistant turns, so they always
    contain a user message (Qwen3.5's template refuses a system-only prefix). If a template rewrites
    earlier turns (e.g. drops old <think> blocks), that turn's prefix won't line up and the turn is
    left untrained rather than mislabelled; the last turn always lines up.
    """
    msgs = list(messages)
    if system and (not msgs or msgs[0]["role"] != "system"):
        msgs = [{"role": "system", "content": system}] + msgs
    last = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=None)
    if last is None:
        return [], []
    msgs = msgs[: last + 1]  # nothing after the final answer is trained
    ids = _render(tok, msgs, **tmpl_kwargs)
    mask = [0] * len(ids)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not any(x["role"] == "user" for x in msgs[:i]):
            continue
        before = _render(tok, msgs[:i], add_generation_prompt=True, **tmpl_kwargs)
        upto = ids if i == last else _render(tok, msgs[: i + 1], **tmpl_kwargs)
        if upto[: len(before)] != before or ids[: len(upto)] != upto:
            continue
        for t in range(len(before), len(upto)):
            mask[t] = 1
    return ids, mask


def render_samples(tok, messages: list[dict], system: str | None = FORMAL_SYSTEM, max_turns: int = 3,
                   **tmpl_kwargs):
    """Like render_with_mask, but recovers turns the template rewrites: any assistant turn left
    untrained becomes its own sample (the conversation up to and including that turn).
    At most `max_turns` samples per conversation, latest turns first."""
    ids, mask = render_with_mask(tok, messages, system, **tmpl_kwargs)
    if not ids:
        return []
    out = [(ids, mask)]
    msgs = list(messages)
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][:-1]
    n_trained_turns = sum(1 for a, b in zip([0] + mask, mask) if b and not a)  # rising edges
    if n_trained_turns >= len(turns) + 1:
        return out  # every turn already trained in the full render
    for i in reversed(turns):
        if len(out) >= max_turns:
            break
        sub_ids, sub_mask = render_with_mask(tok, msgs[: i + 1], system, **tmpl_kwargs)
        if sum(sub_mask):
            out.append((sub_ids, sub_mask))
    return out


def pack(samples, seq_len: int, pad_id: int):
    """Greedy-pack (ids, mask) pairs into windows of seq_len+1 tokens. Over-long samples are cut."""
    W = seq_len + 1
    cur_ids, cur_mask = [], []
    for ids, mask in samples:
        ids, mask = ids[:W], mask[:W]
        if len(cur_ids) + len(ids) > W:
            pad = W - len(cur_ids)
            yield cur_ids + [pad_id] * pad, cur_mask + [0] * pad
            cur_ids, cur_mask = [], []
        cur_ids += ids
        cur_mask += mask
    if cur_ids:
        pad = W - len(cur_ids)
        yield cur_ids + [pad_id] * pad, cur_mask + [0] * pad


def write_sft_shards(windows, out_dir: str, per_shard: int = 4096) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    names, toks, masks = [], [], []

    def flush():
        if not toks:
            return
        n = f"sft_{len(names):05d}.npz"
        np.savez(os.path.join(out_dir, n), tokens=np.array(toks, dtype=np.int32), mask=np.array(masks, dtype=np.uint8))
        names.append(n)
        toks.clear()
        masks.clear()

    for ids, mask in windows:
        toks.append(ids)
        masks.append(mask)
        if len(toks) == per_shard:
            flush()
    flush()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"kind": "sft", "shards": names}, f)
    return names


class SFTLoader:
    """Deterministic, resumable loader over SFT shards (with optional teacher logits)."""

    def __init__(self, root: str, batch_size: int, seed: int = 1234):
        man = json.load(open(os.path.join(root, "manifest.json")))
        self.shards = [np.load(os.path.join(root, s)) for s in man["shards"]]
        self.index = [(si, r) for si, sh in enumerate(self.shards) for r in range(sh["tokens"].shape[0])]
        self.teacher_shards = {i for i, sh in enumerate(self.shards) if "topk_idx" in sh.files}
        self.k = next((self.shards[i]["topk_idx"].shape[-1] for i in self.teacher_shards), 0)
        self.batch_size, self.seed = batch_size, seed
        self.cursor = 0
        self._epoch, self._perm = -1, None

    def _row(self, c):
        epoch, pos = divmod(c, len(self.index))
        if epoch != self._epoch:
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(len(self.index))
            self._epoch = epoch
        return self.index[self._perm[pos]]

    def next_batch(self):
        rows = [self._row(self.cursor + i) for i in range(self.batch_size)]
        self.cursor += self.batch_size
        tok = torch.from_numpy(np.stack([self.shards[s]["tokens"][r] for s, r in rows]).astype(np.int64))
        msk = torch.from_numpy(np.stack([self.shards[s]["mask"][r] for s, r in rows]).astype(bool))
        x, nxt = tok[:, :-1].contiguous(), tok[:, 1:].contiguous()
        y = nxt.masked_fill(~msk[:, 1:], -100)
        batch = {"x": x, "y": y, "next_ids": nxt}
        if self.teacher_shards:  # rows from shards without teacher logits just skip the KD term
            T = x.shape[1]
            idx = np.zeros((len(rows), T, self.k), dtype=np.int64)
            logp = np.zeros((len(rows), T, self.k), dtype=np.float32)
            has = np.zeros(len(rows), dtype=bool)
            for j, (s, r) in enumerate(rows):
                if s in self.teacher_shards:
                    idx[j], logp[j], has[j] = self.shards[s]["topk_idx"][r], self.shards[s]["topk_logp"][r], True
            batch["teacher"] = {"idx": torch.from_numpy(idx), "logp": torch.from_numpy(logp),
                                "has": torch.from_numpy(has)}
        return batch

    def state_dict(self):
        return {"cursor": self.cursor, "n": len(self.index), "seed": self.seed}

    def load_state_dict(self, sd):
        if sd["n"] != len(self.index):
            raise ValueError("SFT data changed since the checkpoint")
        self.cursor = sd["cursor"]
