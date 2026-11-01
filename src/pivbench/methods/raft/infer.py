"""Suy luận RAFT-PIV: đệm ảnh, chạy theo ô (tile) chồng lấn cho ảnh lớn, ghép bằng trọng số cosin."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _multiple(model) -> int:
    c = model.cfg
    return c.down * 2 ** (c.corr_levels - 1)


@torch.no_grad()
def _run(model, a, b, iters, device):
    m = _multiple(model)
    h, w = a.shape
    ta = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None].to(device)
    tb = torch.from_numpy(np.ascontiguousarray(b, np.float32))[None, None].to(device)
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        ta = F.pad(ta, (0, pw, 0, ph), mode="reflect")
        tb = F.pad(tb, (0, pw, 0, ph), mode="reflect")
    flow = model(ta, tb, iters=iters, test_mode=True)[0, :, :h, :w].cpu().numpy()
    return flow


def _weights(n, overlap):
    w = np.ones(n)
    if overlap > 0:
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap + 2)[1:-1])
        w[:overlap] = r
        w[-overlap:] = r[::-1]
    return w


@torch.no_grad()
def predict(model, img1, img2, iters: int = 16, device: str = "cpu", tile: int | None = 512,
            overlap: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (u, v) cỡ ảnh. Ảnh lớn hơn ``tile`` được chia ô chồng lấn ``overlap`` px."""
    a = np.asarray(img1, np.float32)
    b = np.asarray(img2, np.float32)
    H, W = a.shape
    if tile is None or (H <= tile and W <= tile):
        f = _run(model, a, b, iters, device)
        return f[0].astype(np.float64), f[1].astype(np.float64)
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    acc = np.zeros((2, H, W))
    wsum = np.zeros((H, W))
    for y0 in ys:
        for x0 in xs:
            th, tw = min(tile, H), min(tile, W)
            f = _run(model, a[y0:y0 + th, x0:x0 + tw], b[y0:y0 + th, x0:x0 + tw], iters, device)
            wy = _weights(th, overlap if (y0 > 0 or y0 + th < H) else 0)
            wx = _weights(tw, overlap if (x0 > 0 or x0 + tw < W) else 0)
            # cạnh ảnh không giảm trọng số
            if y0 == 0:
                wy[:overlap] = 1
            if y0 + th == H:
                wy[-overlap:] = 1
            if x0 == 0:
                wx[:overlap] = 1
            if x0 + tw == W:
                wx[-overlap:] = 1
            wt = np.outer(wy, wx)
            acc[:, y0:y0 + th, x0:x0 + tw] += f * wt
            wsum[y0:y0 + th, x0:x0 + tw] += wt
    flow = acc / np.maximum(wsum, 1e-12)
    return flow[0], flow[1]
