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
        if not got and boot.get("sft_mix"):
            sh(sys.executable, "-m", "scripts.prepare_sft", "--mix", boot["sft_mix"], "--out", data_dir,
               "--seq-len", str(tc["seq_len"]))
            if repo:  # only tokens + masks go to the hub (small); teacher logits stay on the box
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
                api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=data_dir, path_in_repo=name)
        elif not got:
            cmd = [sys.executable, "-m", "scripts.prepare_data", "--mix", boot["data_mix"], "--out", data_dir]
            if repo:
                cmd += ["--upload", repo]
            sh(*cmd)

    # Phase 2 (teacher logits) runs on the same box right before phase 3. Kept local only: ~3 GB would
    # blow the free Hugging Face quota. Resumable per shard, so a new box only redoes missing shards.
    t = boot.get("teacher")
    if t and not os.path.exists(os.path.join(tc["out_dir"], "DONE")):
        cmd = [sys.executable, "-m", "scripts.teacher_logits", "--data", data_dir,
               "--teacher", t["model"], "--k", str(t.get("k", 32))]
        if t.get("max_rows"):
            cmd += ["--max-rows", str(t["max_rows"])]
        sh(*cmd)

    init = tc.get("init")
    if init and not os.path.exists(os.path.join(init, "model.pt")):
        up = boot.get("upcycle")
        prev = boot.get("from_phase")
        if prev:
            from candidlm.checkpoint import HubSync
            from scripts.export import export
            ck = HubSync(tc["hub_repo"], prefix=prev).download_latest(os.path.join("runs", "hub_" + prev))
            if ck is None:
                sys.exit(f"no checkpoint of {prev} on the hub yet")
            export(ck, init)
        elif up:
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
