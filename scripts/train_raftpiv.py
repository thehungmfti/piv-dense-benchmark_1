#!/usr/bin/env python
"""Huấn luyện RAFT-PIV (tái hiện) theo file YAML.

    python scripts/train_raftpiv.py configs/raftpiv_wake.yaml [--steps N] [--device cuda]

Đường dẫn tương đối trong YAML tính từ thư mục gốc repo. Kho dữ liệu tổng hợp được sinh nếu chưa có.
Kết quả: <out_dir>/last.pt (tiếp tục được), best.pt (theo EPE validation), train_log.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pivbench.data.cai import list_cai_samples, load_cai_sample  # noqa: E402
from pivbench.methods.raft.data import MixedDataset, build_pool, make_sample  # noqa: E402
from pivbench.methods.raft.model import RAFTPIVConfig  # noqa: E402
from pivbench.methods.raft.train import train  # noqa: E402


def _p(x):
    p = Path(x)
    return p if p.is_absolute() else (ROOT / p).resolve()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device")
    ap.add_argument("--time-limit-h", type=float)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    m = dict(cfg["model"])
    m["enc_channels"] = tuple(m["enc_channels"])
    mcfg = RAFTPIVConfig(**m)

    d = cfg["data"]
    pool = d["pool"]
    pdir = _p(pool["dir"])
    if not (pdir / "meta.json").is_file():
        print(f"Sinh kho dữ liệu tổng hợp → {pdir}", flush=True)
        build_pool(pdir, pool["n"], pool["size"], pool.get("mix"), pool.get("seed", 0))
    cai_train, cai_test = [], []
    if d.get("cai") and d["cai"].get("root"):
        root = _p(d["cai"]["root"])
        if root.is_dir():
            cls = tuple(d["cai"].get("classes", ["cylinder", "backstep"]))
            cai_train = list_cai_samples(root, cls, "train")
            cai_test = list_cai_samples(root, cls, "test")
            print(f"Cai et al.: {len(cai_train)} mẫu train, {len(cai_test)} mẫu test", flush=True)
        else:
            print(f"Không thấy dữ liệu Cai tại {root} — chỉ dùng dữ liệu tổng hợp", flush=True)
    ds = MixedDataset([pdir], cai_train, d.get("cai", {}).get("prob", 0.3) if cai_train else 0.0,
                      crop=d.get("crop", 128), seed=1)

    v = cfg.get("validation", {})
    from pivbench.synth import SineField, generate_pair

    def _sine_val():  # dao động dưới pixel có nén (u(x)) và cắt (v(x)) — kiểm tra prior không nén được
        out = []
        for i, (lam, comp) in enumerate([(24, "u"), (48, "u"), (24, "v"), (48, "v")]):
            pr = generate_pair(SineField(0.5, lam, component=comp, axis="x", offset=3.0), (256, 256),
                               ppp=0.04, snr=20, seed=900 + i)
            out.append((pr["img1"], pr["img2"], np.stack([pr["u"], pr["v"]]), np.ones((256, 256), bool)))
        return out
    val = {"wake": [make_sample(v.get("wake_seed", 777) + i, 256, "wake") for i in range(v.get("wake_n", 24))]}
    if v.get("sine", True):
        val["sine"] = _sine_val()
    val["wake"] = [(x[0].astype(np.float32), x[1].astype(np.float32), x[2], x[3]) for x in val["wake"]]
    nc = v.get("cai_n_per_class", 24)
    for c in d.get("cai", {}).get("classes", []):
        items = [t for t in cai_test if t[2].name.startswith(c)][:nc]
        if items:
            val[f"cai_{c}"] = []
            for t in items:
                i1, i2, uu, vv = load_cai_sample(t)
                val[f"cai_{c}"].append((i1, i2, np.stack([uu, vv]), np.ones(i1.shape, bool)))

    t = cfg["train"]
    train(mcfg, ds, _p(cfg["out_dir"]), steps=a.steps or t["steps"], batch=t["batch"], lr=t["lr"],
          weight_decay=t["weight_decay"], gamma=t["gamma"], clip=t["clip"], val_sets=val,
          val_every=t["val_every"], ckpt_every=t["ckpt_every"], threads=t.get("threads"),
          device=a.device or t.get("device", "cpu"), time_limit_h=a.time_limit_h or t.get("time_limit_h"),
          init_from=_p(t["init_from"]) if t.get("init_from") else None)


if __name__ == "__main__":
    main()
