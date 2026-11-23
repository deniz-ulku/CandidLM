"""Turn a training checkpoint into a model dir (config.json + model.pt) that the next phase starts from.

  python -m scripts.export --ckpt runs/phase1/checkpoints/step_0007000 --out runs/after_phase1
"""

import argparse
import json
import os

import torch


def export(ckpt_dir: str, out: str) -> None:
    st = torch.load(os.path.join(ckpt_dir, "state.pt"), map_location="cpu", weights_only=False)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(st["model_config"], f, indent=2)
    torch.save(st["model"], os.path.join(out, "model.pt"))
    print(f"[export] step {st['step']} of {st['phase']} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a.ckpt, a.out)
