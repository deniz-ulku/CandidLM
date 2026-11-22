"""Keep one CandidLM training instance alive on Vast.ai, cheaply, without overspending.

Every minute: is our instance running? If not (outbid, host died, never started), find the
cheapest offer that passes our filters, start it with our Docker image + onstart script,
and the training resumes from the latest Hugging Face checkpoint on its own.

It stops everything when the account credit drops below --min-credit, or when the phase
reports DONE on the hub.

DRY RUN BY DEFAULT: without --launch it only prints what it would rent. Nothing is spent.

  export VAST_API_KEY=...  HF_TOKEN=...
  python scripts/vast_watcher.py --phase phase1_heal                 # dry run, shows offers
  python scripts/vast_watcher.py --phase phase1_heal --launch        # really rents

Runs on your PC (any OS with Python 3.10+ and `requests`).
NOTE: the Vast REST endpoints below follow the public API docs as of late 2026; the first
dry run checks them before any money moves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

API = "https://console.vast.ai/api/v0"
LABEL = "candidlm"


def _h(key):
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def credit(key) -> float:
    r = requests.get(f"{API}/users/current/", headers=_h(key), timeout=30)
    r.raise_for_status()
    return float(r.json().get("credit", 0.0))


def my_instances(key) -> list[dict]:
    r = requests.get(f"{API}/instances/", headers=_h(key), params={"owner": "me"}, timeout=30)
    r.raise_for_status()
    return [i for i in r.json().get("instances", []) if i.get("label") == LABEL]


def search_offers(key, gpus: list[str], max_price: float, interruptible: bool) -> list[dict]:
    q = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_name": {"in": gpus},
        "gpu_ram": {"gte": 23000},
        "reliability2": {"gte": 0.98},
        "inet_down": {"gte": 300},
        "cuda_max_good": {"gte": 12.4},
        "disk_space": {"gte": 80},
        "type": "bid" if interruptible else "on-demand",
        "order": [["dph_total", "asc"]],
    }
    r = requests.post(f"{API}/bundles/", headers=_h(key), json=q, timeout=60)
    r.raise_for_status()
    offers = r.json().get("offers", [])
    price_key = "min_bid" if interruptible else "dph_total"
    offers = [o for o in offers if float(o.get(price_key) or 1e9) <= max_price]
    # cheapest per unit of speed: rank 4090 above 3090 at equal price
    speed = {"RTX 4090": 3.5, "RTX 3090": 1.0, "RTX 3090 Ti": 1.2, "RTX A5000": 0.9}
    offers.sort(key=lambda o: float(o.get(price_key) or 1e9) / speed.get(o.get("gpu_name"), 1.0))
    return offers


def launch(key, offer: dict, image: str, env: dict, onstart: str, interruptible: bool, bid: float):
    body = {
        "client_id": "me",
        "image": image,
        "disk": 80,
        "label": LABEL,
        "onstart": onstart,
        "env": env,
        "runtype": "ssh_direct",
    }
    if interruptible:
        body["price"] = bid
    r = requests.put(f"{API}/asks/{offer['id']}/", headers=_h(key), json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def destroy(key, inst_id):
    requests.delete(f"{API}/instances/{inst_id}/", headers=_h(key), timeout=30).raise_for_status()


def phase_done(hf_repo: str, phase: str, token: str) -> bool:
    try:
        from huggingface_hub import HfApi
        files = HfApi(token=token).list_repo_files(hf_repo)
        return f"{phase}/DONE" in files
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--config", help="training config path inside the repo, default configs/<phase>.json")
    ap.add_argument("--hf-repo", default="denizulkuu/candidlm-checkpoints")
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    ap.add_argument("--git", default="https://github.com/deniz-ulku/CandidLM.git")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--gpus", default="RTX 4090,RTX 3090")
    ap.add_argument("--max-price", type=float, default=0.40, help="$/h cap")
    ap.add_argument("--bid-margin", type=float, default=0.02, help="$/h over the min bid")
    ap.add_argument("--on-demand", action="store_true", help="no interruptible instances")
    ap.add_argument("--min-credit", type=float, default=0.50)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--launch", action="store_true", help="really rent GPUs (otherwise dry run)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("VAST_API_KEY")
    token = os.environ.get("HF_TOKEN")
    if not key or not token:
        sys.exit("set VAST_API_KEY and HF_TOKEN in your environment first")
    interruptible = not a.on_demand
    cfg = a.config or f"configs/{a.phase}.json"
    onstart = open(os.path.join(os.path.dirname(__file__), "onstart.sh")).read()
    env = {"HF_TOKEN": token, "CANDID_CONFIG": cfg, "CANDID_GIT": a.git, "CANDID_BRANCH": a.branch,
           "CANDID_PHASE": a.phase, "CANDID_HF_REPO": a.hf_repo}

    spent_log = os.path.expanduser("~/.candidlm_spend.jsonl")
    start_credit = None
    while True:
        try:
            c = credit(key)
            start_credit = c if start_credit is None else start_credit
            insts = my_instances(key)
            running = [i for i in insts if i.get("actual_status") == "running"]
            print(f"[watch] credit ${c:.2f} (spent ${start_credit - c:.2f} this session), "
                  f"instances: {[(i['id'], i.get('actual_status')) for i in insts]}", flush=True)
            with open(spent_log, "a") as f:
                f.write(json.dumps({"t": time.time(), "phase": a.phase, "credit": c}) + "\n")

            if c < a.min_credit or phase_done(a.hf_repo, a.phase, token):
                reason = "credit floor reached" if c < a.min_credit else "phase finished"
                print(f"[watch] {reason}: destroying instances and stopping", flush=True)
                if a.launch:
                    for i in insts:
                        destroy(key, i["id"])
                return

            # outbid / stopped / errored instances still bill storage: remove them
            for i in insts:
                if i.get("actual_status") in ("exited", "offline") or i.get("intended_status") == "stopped":
                    print(f"[watch] instance {i['id']} is {i.get('actual_status')}: destroying", flush=True)
                    if a.launch:
                        destroy(key, i["id"])

            starting = [i for i in insts if i.get("actual_status") in ("loading", "created", "scheduling")]
            if not running and not starting:
                offers = search_offers(key, a.gpus.split(","), a.max_price, interruptible)
                if not offers:
                    print("[watch] no offer under the price cap right now; waiting", flush=True)
                else:
                    o = offers[0]
                    price = float(o.get("min_bid") if interruptible else o.get("dph_total"))
                    bid = round(price + a.bid_margin, 4)
                    print(f"[watch] best offer {o['id']}: {o.get('gpu_name')} ${price:.3f}/h "
                          f"(bid ${bid:.3f}), reliability {o.get('reliability2')}", flush=True)
                    if a.launch:
                        print("[watch] launching:", launch(key, o, a.image, env, onstart, interruptible, bid),
                              flush=True)
                    else:
                        print("[watch] dry run: not launching (pass --launch to rent)", flush=True)
        except requests.RequestException as e:
            print(f"[watch] API error, will retry: {e}", flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
