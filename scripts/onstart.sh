#!/bin/bash
# Runs on every fresh Vast instance. Idempotent: safe to run again after a restart.
# Needs env: HF_TOKEN, CANDID_GIT, CANDID_BRANCH, CANDID_CONFIG, CANDID_PHASE, CANDID_HF_REPO
set -euo pipefail
cd /workspace 2>/dev/null || cd /root
if [ ! -d CandidLM ]; then git clone --depth 1 -b "$CANDID_BRANCH" "$CANDID_GIT" CandidLM; fi
cd CandidLM && git pull --ff-only || true
pip install -q -r requirements.txt
pip install -q flash-linear-attention || echo "fla not installed: falling back to the PyTorch DeltaNet"
export PYTHONUNBUFFERED=1
python -m scripts.bootstrap --config "$CANDID_CONFIG" 2>&1 | tee -a bootstrap.log
python -m candidlm.train --config "$CANDID_CONFIG" --set hub_repo="\"$CANDID_HF_REPO\"" 2>&1 | tee -a train.log
