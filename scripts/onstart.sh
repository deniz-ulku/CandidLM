#!/bin/bash
# Runs on every fresh Vast instance. Idempotent: safe to run again after a restart.
# Needs env: HF_TOKEN, CANDID_GIT, CANDID_BRANCH, CANDID_CONFIG, CANDID_PHASE, CANDID_HF_REPO
set -euo pipefail
# If anything below fails, tell the watcher through the hub (it then destroys this instance
# instead of paying for an idle GPU), with the end of the logs for debugging.
report_failure() {
  tail -n 80 /root/CandidLM/bootstrap.log /root/CandidLM/train.log /workspace/CandidLM/bootstrap.log \
    /workspace/CandidLM/train.log 2>/dev/null > /tmp/fail.txt || true
  python - <<'PY' || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="/tmp/fail.txt", repo_id=os.environ["CANDID_HF_REPO"],
    path_in_repo=f"{os.environ['CANDID_PHASE']}/FAILED", commit_message="run failed")
PY
}
trap report_failure ERR
cd /workspace 2>/dev/null || cd /root
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
if [ ! -d CandidLM ]; then git clone --depth 1 -b "$CANDID_BRANCH" "$CANDID_GIT" CandidLM; fi
cd CandidLM && git pull --ff-only || true
pip install -q -r requirements.txt
pip install -q flash-linear-attention || echo "fla not installed: falling back to the PyTorch DeltaNet"
export PYTHONUNBUFFERED=1
python -m scripts.bootstrap --config "$CANDID_CONFIG" 2>&1 | tee -a bootstrap.log
python -m candidlm.train --config "$CANDID_CONFIG" --set hub_repo="\"$CANDID_HF_REPO\"" 2>&1 | tee -a train.log
