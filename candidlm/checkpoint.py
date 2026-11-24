"""Full-state checkpoints that survive losing the GPU.

A checkpoint holds model weights (incl. MoE balancing biases), optimizer states, step,
data position and RNG states. It is written locally (atomic rename), then uploaded to a
private Hugging Face repo in a background thread. `LATEST` on the hub is only updated
after an upload has fully finished, so a half-uploaded checkpoint is never resumed.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time

import numpy as np
import torch


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_local(path: str, payload: dict) -> str:
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "state.pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, os.path.join(path, "state.pt"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"step": payload["step"], "time": time.time(), "phase": payload.get("phase")}, f)
    return path


def load_local(path: str, map_location="cpu") -> dict:
    return torch.load(os.path.join(path, "state.pt"), map_location=map_location, weights_only=False)


class HubSync:
    """Uploads checkpoints to `repo_id` (private) under `<prefix>/step_XXXXXXX/`."""

    def __init__(self, repo_id: str | None, prefix: str, keep: int = 2, token: str | None = None):
        self.repo_id, self.prefix, self.keep = repo_id, prefix.strip("/"), keep
        self.token = token or os.environ.get("HF_TOKEN")
        self.thread: threading.Thread | None = None
        self.error: Exception | None = None
        self.api = None
        if repo_id:
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.token)
            self.api.create_repo(repo_id, private=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.api is not None

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def upload_async(self, local_dir: str, step: int, keep_forever: bool = False) -> None:
        if not self.enabled:
            return
        self.wait()  # never two uploads at once
        self.thread = threading.Thread(target=self._upload, args=(local_dir, step, keep_forever), daemon=True)
        self.thread.start()

    def _upload(self, local_dir: str, step: int, keep_forever: bool) -> None:
        name = f"step_{step:07d}"
        try:
            for attempt in range(5):
                try:
                    self.api.upload_folder(repo_id=self.repo_id, folder_path=local_dir,
                                           path_in_repo=f"{self.prefix}/{name}",
                                           commit_message=f"checkpoint {name}")
                    break
                except Exception as e:  # network blips: back off and retry
                    if attempt == 4:
                        raise
                    time.sleep(10 * 2 ** attempt)
            self.api.upload_file(path_or_fileobj=name.encode(), path_in_repo=f"{self.prefix}/LATEST",
                                 repo_id=self.repo_id, commit_message=f"LATEST -> {name}")
            if keep_forever:
                self.api.upload_file(path_or_fileobj=b"keep", path_in_repo=f"{self.prefix}/{name}/KEEP",
                                     repo_id=self.repo_id)
            self._prune()
            print(f"[ckpt] uploaded {name} to {self.repo_id}", flush=True)
        except Exception as e:
            self.error = e
            print(f"[ckpt] upload FAILED for {name}: {e}", flush=True)

    def _prune(self) -> None:
        files = self.api.list_repo_files(self.repo_id)
        steps = sorted({f.split("/")[1] for f in files if f.startswith(self.prefix + "/step_")})
        keep = {f.split("/")[1] for f in files if f.endswith("/KEEP") and f.startswith(self.prefix + "/")}
        pruned = False
        for s in [s for s in steps if s not in keep][:-self.keep]:
            self.api.delete_folder(path_in_repo=f"{self.prefix}/{s}", repo_id=self.repo_id,
                                   commit_message=f"prune {s}")
            pruned = True
        if pruned:
            # Deleted files still count toward the free Hugging Face storage quota while they sit in
            # git history; squashing drops them so the repo only ever holds the last `keep` checkpoints.
            try:
                self.api.super_squash_history(repo_id=self.repo_id, commit_message="squash old checkpoints")
            except Exception as e:
                print(f"[ckpt] history squash failed (storage may grow): {e}", flush=True)

    def mark_done(self, step: int) -> None:
        if self.enabled:
            self.wait()
            self.api.upload_file(path_or_fileobj=str(step).encode(), path_in_repo=f"{self.prefix}/DONE",
                                 repo_id=self.repo_id, commit_message=f"{self.prefix} done")

    def download_latest(self, dest: str) -> str | None:
        if not self.enabled:
            return None
        from huggingface_hub import hf_hub_download, snapshot_download
        try:
            p = hf_hub_download(self.repo_id, f"{self.prefix}/LATEST", token=self.token, force_download=True)
        except Exception:
            return None
        name = open(p).read().strip()
        snap = snapshot_download(self.repo_id, allow_patterns=[f"{self.prefix}/{name}/*"], token=self.token,
                                 local_dir=dest)
        return os.path.join(snap, self.prefix, name)


def latest_local(root: str) -> str | None:
    if not os.path.isdir(root):
        return None
    steps = sorted(d for d in os.listdir(root) if d.startswith("step_")
                   and os.path.exists(os.path.join(root, d, "state.pt")))
    return os.path.join(root, steps[-1]) if steps else None


def prune_local(root: str, keep: int = 2) -> None:
    steps = sorted(d for d in os.listdir(root) if d.startswith("step_"))
    for d in steps[:-keep]:
        shutil.rmtree(os.path.join(root, d), ignore_errors=True)
