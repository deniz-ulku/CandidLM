# CandidLM

A small hybrid MoE language model, built and trained by Deniz on a ~$5 GPU budget.

- **Architecture (ours):** 24 layers, hidden 1024, 18 Gated DeltaNet + 6 gated-GQA layers (3:1),
  first 2 layers dense, then fine-grained MoE: 16 routed experts (width 448) top-6 + 1 shared expert (896),
  aux-loss-free bias balancing, 1-token MTP head. ~1.08B total / ~0.77B active parameters.
- **Init:** sparse-upcycled from Qwen3.5-0.8B-Base (Apache 2.0). Our model code reproduces Qwen3.5
  exactly before upcycling, and the upcycled MoE reproduces it exactly at step 0 (see tests).
- **Training stack (ours, no HF Trainer):** Muon for hidden matrices + AdamW for the rest, BF16,
  chunked cross-entropy for the 248K vocab, full-state checkpoints to a private HF repo every 15 min,
  deterministic data order, auto-resume on any GPU.

## Layout

| Path | What |
| --- | --- |
| `candidlm/model.py` | Model: RMSNorm, gated GQA with partial RoPE, Gated DeltaNet (fla kernel or PyTorch), MoE, MTP |
| `candidlm/upcycle.py` | Qwen3.5 dense weights -> CandidLM MoE (drop-upcycling) |
| `candidlm/optim.py` | Muon + AdamW, WSD learning-rate schedule |
| `candidlm/data.py` | Resumable loader over uint32 token shards |
| `candidlm/checkpoint.py` | Local + Hugging Face checkpoints, `LATEST` pointer, pruning |
| `candidlm/train.py` | Training loop, freezing, SIGTERM-safe checkpointing |
| `scripts/prepare_data.py` | Stream + tokenise open datasets into shards |
| `scripts/bootstrap.py` | Prepare a fresh GPU box (data + starting model) |
| `scripts/onstart.sh` | What every Vast instance runs on boot |
| `scripts/vast_watcher.py` | Keeps one instance alive, re-rents after outbids, enforces the credit floor (dry run by default) |
| `configs/` | Phase configs and data mixes |

## Run

```bash
pip install -r requirements.txt
python -m pytest tests            # equivalence with Qwen3.5, exact upcycle, kill-and-resume
export VAST_API_KEY=... HF_TOKEN=...   # never commit these
python scripts/vast_watcher.py --phase phase1_heal            # dry run: shows the offer it would rent
python scripts/vast_watcher.py --phase phase1_heal --launch   # rents and keeps training alive
```

## Status

- [x] Phase 0: model, upcycle, optimizer, data, checkpoints, training loop, watcher; tests pass on CPU
- [ ] Phase 0b: first real upcycle of Qwen3.5-0.8B-Base and a 3070 smoke run (needs Hugging Face access)
- [ ] Phase 1: healing (~230M tokens, ~7.5 GPU-hours)
- [ ] Phase 2-3: teacher logits, distillation + reasoning SFT (code to come)
- [ ] Vision path: reuse Qwen3.5's vision encoder with our language model (code to come)
