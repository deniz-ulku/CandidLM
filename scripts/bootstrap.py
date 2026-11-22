"""Get a fresh GPU box ready for a phase: data shards and the starting model.

Data: download from the private HF dataset repo; if it isn't there yet, build it once and upload it.
Model: phase 1 upcycles Qwen3.5-0.8B-Base (deterministic, about a minute); later phases pull the
previous phase's final checkpoint from the hub.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def sh(*cmd):
    print("[boot] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    tc = json.load(open(a.config))
    boot = tc.get("bootstrap", {})
    token = os.environ.get("HF_TOKEN")

    data_dir = tc["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "manifest.json")):
        from huggingface_hub import snapshot_download
        repo = boot.get("data_repo")
        name = os.path.basename(data_dir.rstrip("/"))
        got = False
        if repo:
            try:
                snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{name}/*"], token=token,
                                  local_dir=os.path.dirname(data_dir.rstrip("/")) or ".")
                got = os.path.exists(os.path.join(data_dir, "manifest.json"))
            except Exception as e:
                print(f"[boot] no data on hub yet ({e})", flush=True)
        if not got:
            cmd = [sys.executable, "-m", "scripts.prepare_data", "--mix", boot["data_mix"], "--out", data_dir]
            if repo:
                cmd += ["--upload", repo]
            sh(*cmd)

    init = tc.get("init")
    if init and not os.path.exists(os.path.join(init, "model.pt")):
        up = boot.get("upcycle")
        if up:
            cmd = [sys.executable, "-m", "candidlm.upcycle", "--src", up["src"], "--out", init,
                   "--num-experts", str(up.get("num_experts", 16)), "--top-k", str(up.get("top_k", 6))]
            if up.get("dense"):
                cmd.append("--dense")
            sh(*cmd)
        else:
            sys.exit(f"init dir {init} missing and no upcycle recipe in config")
    print("[boot] ready", flush=True)


if __name__ == "__main__":
    main()
