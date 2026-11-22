"""CandidLM training loop (our own; no HF Trainer).

  python -m candidlm.train --config configs/phase1_heal.json [--set key=value ...]

Resumes automatically: newest local checkpoint, else LATEST on the Hugging Face repo,
else starts from `init` (an upcycled model dir) or random weights.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import signal
import sys
import time

import torch

from . import checkpoint as ck
from .config import CandidConfig, tiny_config
from .data import TokenLoader
from .model import CandidLM
from .optim import CombinedOptimizer, wsd_lr

DEFAULTS = dict(
    phase="phase1_heal",
    init=None,                 # dir with config.json + model.pt (from candidlm.upcycle)
    tiny=False,                # use the tiny test config with random weights
    data_dir="data/phase1",
    data_weights=None,
    seq_len=4096,
    micro_batch=8,
    grad_accum=1,
    total_steps=1000,
    lr=3e-4,
    warmup_steps=50,
    decay_frac=0.2,
    weight_decay=0.1,
    grad_clip=1.0,
    freeze=["embed_tokens.*"],  # glob patterns of parameter names to freeze
    unfreeze_after_frac=None,   # e.g. 0.25 -> unfreeze everything after 25% of steps
    grad_checkpointing=True,
    out_dir="runs/phase1",
    ckpt_minutes=15.0,
    keep_local=2,
    hub_repo=None,             # e.g. "denizulkuu/candidlm-checkpoints"
    log_every=10,
    eval_every=200,
    eval_batches=8,
    seed=1234,
    max_steps_this_run=None,   # for tests / budget slicing
    compile=False,
)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (JSON values)")
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        with open(a.config) as f:
            cfg.update(json.load(f))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    return cfg


def apply_freeze(model, patterns):
    n_frozen = 0
    for n, p in model.named_parameters():
        frozen = any(fnmatch.fnmatch(n, pat) for pat in (patterns or []))
        p.requires_grad_(not frozen)
        n_frozen += p.numel() if frozen else 0
    return n_frozen


def build_model(tc: dict, device) -> CandidLM:
    if tc.get("tiny"):
        mcfg = tiny_config()
        model = CandidLM(mcfg)
    else:
        mcfg = CandidConfig.load(os.path.join(tc["init"], "config.json"))
        model = CandidLM(mcfg)
        sd = torch.load(os.path.join(tc["init"], "model.pt"), map_location="cpu")
        model.load_state_dict(sd, strict=True)
    return model.to(device)


def run(tc: dict) -> list[dict]:
    torch.manual_seed(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(tc["out_dir"], exist_ok=True)
    ckpt_root = os.path.join(tc["out_dir"], "checkpoints")
    hub = ck.HubSync(tc.get("hub_repo"), prefix=tc["phase"])

    model = build_model(tc, device)
    model.grad_checkpointing = tc["grad_checkpointing"]
    loader = TokenLoader(tc["data_dir"], tc["seq_len"], tc["micro_batch"], seed=tc["seed"],
                         weights=tc.get("data_weights"))

    step = 0
    state = None
    local = ck.latest_local(ckpt_root)
    if local is None and hub.enabled:
        local = hub.download_latest(os.path.join(tc["out_dir"], "hub_dl"))
    if local:
        print(f"[train] resuming from {local}", flush=True)
        state = ck.load_local(local)
        model.load_state_dict(state["model"])
        step = state["step"]

    unfrozen = state is not None and state.get("unfrozen", False)
    frozen = apply_freeze(model, [] if unfrozen else tc["freeze"])
    opt = CombinedOptimizer(model, lr=tc["lr"], weight_decay=tc["weight_decay"])
    if state is not None:
        opt.load_state_dict(state["optim"])
        loader.load_state_dict(state["data"])
        ck.set_rng_state(state["rng"])
    print(f"[train] params {model.num_params()/1e6:.1f}M (active {model.num_params(True)/1e6:.1f}M), "
          f"frozen {frozen/1e6:.1f}M, muon {opt.counts['muon']/1e6:.1f}M, adam {opt.counts['adam']/1e6:.1f}M, "
          f"device {device}", flush=True)

    stop = {"flag": False}

    def _sig(*_):
        print("[train] signal received: checkpointing and exiting", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    def save(final=False):
        payload = {"model": model.state_dict(), "optim": opt.state_dict(), "data": loader.state_dict(),
                   "rng": ck.rng_state(), "step": step, "phase": tc["phase"], "train_config": tc,
                   "model_config": model.cfg.to_dict(), "unfrozen": unfrozen}
        hub.wait()  # the previous upload must finish before we touch older local dirs
        path = ck.save_local(os.path.join(ckpt_root, f"step_{step:07d}"), payload)
        ck.prune_local(ckpt_root, tc["keep_local"])
        hub.upload_async(path, step, keep_forever=final)
        print(f"[ckpt] saved step {step} -> {path}", flush=True)

    history = []
    last_ckpt = time.time()
    t0, tokens_seen = time.time(), 0
    run_steps = 0
    log_path = os.path.join(tc["out_dir"], "log.jsonl")
    model.train()
    while step < tc["total_steps"]:
        # optional unfreeze point (e.g. attention unfrozen after 25% of healing)
        if (not unfrozen and tc.get("unfreeze_after_frac") is not None
                and step >= int(tc["total_steps"] * tc["unfreeze_after_frac"])):
            apply_freeze(model, [])
            unfrozen = True
            opt = CombinedOptimizer(model, lr=tc["lr"], weight_decay=tc["weight_decay"])
            print("[train] unfroze all parameters", flush=True)

        lr = wsd_lr(step, tc["total_steps"], tc["lr"], tc["warmup_steps"], tc["decay_frac"])
        opt.set_lr(lr)
        logs = {"loss": 0.0, "loss_main": 0.0, "loss_mtp": 0.0}
        for _ in range(tc["grad_accum"]):
            x, y = loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(x, y)
            (out["loss"] / tc["grad_accum"]).backward()
            for k in logs:
                if k in out:
                    logs[k] += out[k].item() / tc["grad_accum"]
            tokens_seen += x.numel()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc["grad_clip"])
        opt.step()
        opt.zero_grad()
        model.update_router_biases()
        step += 1
        run_steps += 1

        rec = dict(step=step, lr=lr, grad_norm=float(gn), **logs)
        history.append(rec)
        if step % tc["log_every"] == 0 or step == 1:
            dt = time.time() - t0
            rec["tok_per_s"] = tokens_seen / max(dt, 1e-9)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[train] step {step} loss {logs['loss']:.4f} main {logs['loss_main']:.4f} "
                  f"mtp {logs['loss_mtp']:.4f} lr {lr:.2e} gn {float(gn):.2f} "
                  f"{rec['tok_per_s']:.0f} tok/s", flush=True)
        if not math.isfinite(logs["loss"]):
            print("[train] non-finite loss, stopping without saving", flush=True)
            sys.exit(2)

        done = step >= tc["total_steps"]
        budget_hit = tc.get("max_steps_this_run") and run_steps >= tc["max_steps_this_run"]
        if done or stop["flag"] or budget_hit or (time.time() - last_ckpt) / 60 >= tc["ckpt_minutes"]:
            save(final=done)
            last_ckpt = time.time()
        if stop["flag"] or budget_hit:
            break

    hub.wait()
    if step >= tc["total_steps"]:
        with open(os.path.join(tc["out_dir"], "DONE"), "w") as f:
            f.write(str(step))
        hub.mark_done(step)
        print(f"[train] phase {tc['phase']} finished at step {step}", flush=True)
    return history


if __name__ == "__main__":
    run(parse())
