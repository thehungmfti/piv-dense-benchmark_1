#!/usr/bin/env python
"""Chạy RAFT-PIV (tái hiện) trên cặp ảnh của bạn.

    python scripts/infer_raftpiv.py img_a.tif img_b.tif -o out/          # một cặp
    python scripts/infer_raftpiv.py --dir data/my_run -o out/            # mọi cặp trong thư mục (a/b, 1/2)

Tuỳ chọn: --ckpt (mặc định checkpoints/raftpiv_wake/best.pt), --iters 16, --tile 512, --overlap 64,
--roi y0 y1 x0 x1, --mask mask.png (pixel ≠ 0 = vùng thân/không có hạt, ghi NaN), --device cuda,
--scale (px → đơn vị vật lý: nhân u, v), --compare (chạy thêm crosscorr & ofpiv để so sánh).

Đầu ra cho mỗi cặp: <tên>.npz (u, v [px/khoảng thời gian], x, y), <tên>.flo, <tên>.mat (nếu có scipy),
<tên>.png (|u| + vector + xoáy).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pivbench.data import find_pairs, load_pair, read_image  # noqa: E402
from pivbench.data.cai import write_flo  # noqa: E402
from pivbench.methods import get_method  # noqa: E402
from pivbench.metrics import vorticity  # noqa: E402


def _plot(u, v, img, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("ảnh 1", fontsize=9)
    sp = np.hypot(u, v)
    im = axs[1].imshow(sp, cmap="Blues")
    ny, nx = u.shape
    s = max(1, min(ny, nx) // 32)
    Y, X = np.mgrid[0:ny:s, 0:nx:s]
    axs[1].quiver(X, Y, u[::s, ::s], -v[::s, ::s], color="#eb6834", scale_units="xy", angles="xy",
                  scale=max(np.nanmax(sp), 1e-6) / s * 1.2, width=0.002)
    axs[1].set_title("|u| [px] + vector", fontsize=9)
    fig.colorbar(im, ax=axs[1], shrink=0.8)
    w = vorticity(np.nan_to_num(u), np.nan_to_num(v))
    lim = np.nanpercentile(np.abs(w), 99) or 1.0
    im2 = axs[2].imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].set_title("ω [1/khoảng thời gian]", fontsize=9)
    fig.colorbar(im2, ax=axs[2], shrink=0.8)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out_raftpiv"))
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--mask", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args(argv)
    if a.dir:
        pairs = find_pairs(a.dir)
    elif len(a.images) == 2:
        pairs = [tuple(a.images)]
    else:
        ap.error("cần 2 ảnh hoặc --dir")
    a.out.mkdir(parents=True, exist_ok=True)
    mask = None
    if a.mask:
        mask = read_image(a.mask) > 0
        if a.roi:
            y0, y1, x0, x1 = a.roi
            mask = mask[y0:y1, x0:x1]
    raft = get_method("raftpiv")
    params = dict(backend="local", iters=a.iters, tile=a.tile, overlap=a.overlap, device=a.device)
    if a.ckpt:
        params["checkpoint"] = a.ckpt
    for p1, p2 in pairs:
        i1, i2 = load_pair(p1, p2, roi=tuple(a.roi) if a.roi else None)
        t = time.time()
        u, v = raft(i1, i2, **params)
        dt = time.time() - t
        u, v = u * a.scale, v * a.scale
        if mask is not None:
            u = np.where(mask, np.nan, u)
            v = np.where(mask, np.nan, v)
        stem = p1.stem
        ny, nx = u.shape
        y, x = np.mgrid[0:ny, 0:nx]
        res = dict(u=u.astype(np.float32), v=v.astype(np.float32), x=x.astype(np.float32), y=y.astype(np.float32))
        if a.compare:
            for name, kw in [("crosscorr", dict(windows=(64, 32, 16), overlap=0.75)),
                             ("ofpiv", dict(init="crosscorr", shift_method="warp", n_iter=3))]:
                uu, vv = get_method(name)(i1, i2, **kw)
                res[f"u_{name}"], res[f"v_{name}"] = (uu * a.scale).astype(np.float32), (vv * a.scale).astype(np.float32)
        np.savez_compressed(a.out / f"{stem}.npz", **res)
        write_flo(a.out / f"{stem}.flo", np.nan_to_num(np.stack([u, v], -1)))
        try:
            from scipy.io import savemat

            savemat(a.out / f"{stem}.mat", res, do_compression=True)
        except Exception:  # noqa: BLE001
            pass
        _plot(u, v, i1, a.out / f"{stem}.png", f"RAFT-PIV — {p1.name} / {p2.name} ({ny}×{nx}, {dt:.1f} s)")
        print(f"{p1.name}: {ny}×{nx}, {dt:.1f} s, |u|max = {np.nanmax(np.hypot(u, v)):.3f} → {a.out / stem}.*")


if __name__ == "__main__":
    main()
