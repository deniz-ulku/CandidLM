import numpy as np

from candidlm.data import update_manifest, write_shards
from candidlm.train import DEFAULTS, run


def _make_data(d):
    rng = np.random.default_rng(0)
    for name, w in (("web", 0.7), ("code", 0.3)):
        docs = (list((rng.integers(0, 512) + np.arange(rng.integers(20, 200)) * rng.integers(1, 4)) % 512)
                for _ in range(300))  # learnable: arithmetic runs
        update_manifest(d, name, write_shards(docs, d, name, shard_tokens=10_000), w)


def _cfg(tmp, out, **kw):
    c = dict(DEFAULTS, tiny=True, data_dir=str(tmp / "data"), out_dir=str(tmp / out), seq_len=32,
             micro_batch=4, total_steps=20, warmup_steps=3, lr=3e-3, ckpt_minutes=1e9, log_every=1000,
             grad_checkpointing=False, freeze=["embed_tokens.*"], unfreeze_after_frac=0.4)
    c.update(kw)
    return c


def test_kill_and_resume_is_identical(tmp_path):
    _make_data(str(tmp_path / "data"))
    straight = run(_cfg(tmp_path, "a"))
    part1 = run(_cfg(tmp_path, "b", max_steps_this_run=6))      # "outbid" after 6 steps
    part2 = run(_cfg(tmp_path, "b", max_steps_this_run=5))      # new GPU resumes, dies again
    part3 = run(_cfg(tmp_path, "b"))                            # finishes
    resumed = part1 + part2 + part3
    assert [r["step"] for r in resumed] == list(range(1, 21))
    a = [r["loss"] for r in straight]
    b = [r["loss"] for r in resumed]
    assert np.allclose(a, b, rtol=0, atol=1e-6), list(zip(a, b))
    assert a[-1] < a[0]
    assert (tmp_path / "b" / "DONE").exists()
