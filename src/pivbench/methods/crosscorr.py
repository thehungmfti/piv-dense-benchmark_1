"""PIV tương quan chéo nhiều lượt với biến dạng cửa sổ (window deformation).

Baseline đơn giản, tự cài đặt (không phụ thuộc openpiv):
  * Tương quan chéo FFT (vòng, không đệm) trên cửa sổ đã trừ trung bình.
  * Nội suy dưới pixel 3 điểm Gauss.
  * Lượt k > 1: biến dạng đối xứng cả 2 ảnh theo trường dự báo dày đặc
    (I1(x − d/2), I2(x + d/2), nội suy bậc 3), tương quan để lấy hiệu chỉnh.
  * Kiểm định normalized median test (Westerweel & Scarano 2005), thay thế
    vector lỗi bằng trung vị lân cận; làm trơn nhẹ trường dự báo giữa các lượt.
  * Đầu ra dày đặc: nội suy bậc 3 lưới vector về từng pixel.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage


def _grid(n: int, win: int, step: int) -> np.ndarray:
    """Toạ độ góc trên-trái của các cửa sổ, căn giữa trong ảnh."""
    k = (n - win) // step + 1
    used = (k - 1) * step + win
    start = (n - used) // 2
    return start + step * np.arange(k)


def _subpixel_gauss(c: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nội suy đỉnh 3 điểm Gauss; c có shape (N, h, w)."""
    n, h, w = c.shape
    eps = 1e-10
    idx = np.arange(n)
    iy = np.clip(iy, 1, h - 2)
    ix = np.clip(ix, 1, w - 2)
    c0 = np.log(np.maximum(c[idx, iy, ix], eps))
    cym, cyp = np.log(np.maximum(c[idx, iy - 1, ix], eps)), np.log(np.maximum(c[idx, iy + 1, ix], eps))
    cxm, cxp = np.log(np.maximum(c[idx, iy, ix - 1], eps)), np.log(np.maximum(c[idx, iy, ix + 1], eps))
    with np.errstate(divide="ignore", invalid="ignore"):
        dy = (cym - cyp) / (2 * (cym - 2 * c0 + cyp))
        dx = (cxm - cxp) / (2 * (cxm - 2 * c0 + cxp))
    dy = np.where(np.isfinite(dy) & (np.abs(dy) < 1), dy, 0.0)
    dx = np.where(np.isfinite(dx) & (np.abs(dx) < 1), dx, 0.0)
    return iy + dy, ix + dx


def correlate_windows(img1: np.ndarray, img2: np.ndarray, win: int, step: int):
    """Tương quan chéo FFT trên lưới cửa sổ. Trả về (yc, xc, dy, dx, peak_ratio)."""
    ny, nx = img1.shape
    gy, gx = _grid(ny, win, step), _grid(nx, win, step)
    w1 = sliding_window_view(img1, (win, win))[gy][:, gx]
    w2 = sliding_window_view(img2, (win, win))[gy][:, gx]
    shp = w1.shape[:2]
    w1 = w1.reshape(-1, win, win).astype(np.float64)
    w2 = w2.reshape(-1, win, win).astype(np.float64)
    w1 = w1 - w1.mean(axis=(1, 2), keepdims=True)
    w2 = w2 - w2.mean(axis=(1, 2), keepdims=True)
    f1 = np.fft.rfft2(w1)
    f2 = np.fft.rfft2(w2)
    c = np.fft.fftshift(np.fft.irfft2(np.conj(f1) * f2, s=(win, win)), axes=(1, 2))
    norm = np.sqrt((w1**2).sum(axis=(1, 2)) * (w2**2).sum(axis=(1, 2))) + 1e-12
    c = c / norm[:, None, None]
    flat = c.reshape(c.shape[0], -1)
    imax = np.argmax(flat, axis=1)
    iy, ix = np.unravel_index(imax, (win, win))
    py, px = _subpixel_gauss(c - c.min(axis=(1, 2), keepdims=True) + 1e-6, iy, ix)
    # tỉ số đỉnh 1/đỉnh 2 (che vùng 3x3 quanh đỉnh 1)
    c2 = c.copy()
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            c2[np.arange(c.shape[0]), np.clip(iy + oy, 0, win - 1), np.clip(ix + ox, 0, win - 1)] = -np.inf
    pr = flat.max(axis=1) / np.maximum(c2.reshape(c.shape[0], -1).max(axis=1), 1e-12)
    dy = (py - win // 2).reshape(shp)
    dx = (px - win // 2).reshape(shp)
    yc = gy + (win - 1) / 2.0
    xc = gx + (win - 1) / 2.0
    return yc, xc, dy, dx, pr.reshape(shp)


def normalized_median_test(u: np.ndarray, v: np.ndarray, eps: float = 0.1, thr: float = 2.0) -> np.ndarray:
    """Trả về mặt nạ outlier (True = lỗi)."""
    bad = np.zeros(u.shape, bool)
    for comp in (u, v):
        p = np.pad(comp, 1, mode="edge")
        nb = sliding_window_view(p, (3, 3)).reshape(*comp.shape, 9)
        nb = np.delete(nb, 4, axis=-1)
        med = np.median(nb, axis=-1)
        res = np.median(np.abs(nb - med[..., None]), axis=-1)
        bad |= np.abs(comp - med) / (res + eps) > thr
    return bad


def _replace_outliers(u, v, bad):
    if not bad.any():
        return u, v
    u, v = u.copy(), v.copy()
    for comp in (u, v):
        c = np.where(bad, np.nan, comp)
        p = np.pad(c, 1, mode="edge")
        nb = sliding_window_view(p, (3, 3)).reshape(*c.shape, 9)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # lân cận toàn NaN → xử lý ngay dưới
            med = np.nanmedian(nb, axis=-1)
        med = np.where(np.isfinite(med), med, np.nanmedian(c))
        comp[bad] = med[bad]
    return u, v


def grid_to_dense(yc, xc, f, shape, order: int = 3) -> np.ndarray:
    """Nội suy trường trên lưới đều (yc, xc) về lưới pixel (spline bậc 3, ngoại suy 'nearest')."""
    ny, nx = shape
    sy = (yc[1] - yc[0]) if len(yc) > 1 else 1.0
    sx = (xc[1] - xc[0]) if len(xc) > 1 else 1.0
    gy = (np.arange(ny) - yc[0]) / sy
    gx = (np.arange(nx) - xc[0]) / sx
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    if f.shape[0] < 4 or f.shape[1] < 4:
        order = 1
    return ndimage.map_coordinates(f, [GY, GX], order=order, mode="nearest")


def _warp(img, dx, dy):
    ny, nx = img.shape
    Y, X = np.mgrid[0:ny, 0:nx].astype(np.float64)
    return ndimage.map_coordinates(img, [Y + dy, X + dx], order=3, mode="mirror")


def estimate(img1: np.ndarray, img2: np.ndarray, windows=(64, 32, 16), overlap: float = 0.5,
             iters_per_window=(1, 1, 2), median_eps: float = 0.1, median_thr: float = 2.0,
             smooth_sigma: float = 0.5, return_grid: bool = False, **_ignored):
    """Ước lượng dày đặc (u, v) cỡ ảnh bằng PIV nhiều lượt có biến dạng cửa sổ.

    ``windows``: cỡ cửa sổ từng lượt; ``overlap``: tỉ lệ chồng (0.5 → bước = win/2);
    ``iters_per_window``: số lượt lặp ở mỗi cỡ cửa sổ (số nguyên hoặc danh sách).
    ``return_grid=True`` trả thêm dict lưới vector cuối (xc, yc, u, v).
    """
    i1 = np.asarray(img1, dtype=np.float64)
    i2 = np.asarray(img2, dtype=np.float64)
    shape = i1.shape
    if np.isscalar(iters_per_window):
        iters_per_window = [int(iters_per_window)] * len(windows)
    U = np.zeros(shape)
    V = np.zeros(shape)
    first = True
    grid = None
    for win, nit in zip(windows, iters_per_window):
        win = int(win)
        step = max(1, int(round(win * (1 - overlap))))
        for _ in range(int(nit)):
            if first:
                a, b = i1, i2
            else:
                a, b = _warp(i1, -0.5 * U, -0.5 * V), _warp(i2, 0.5 * U, 0.5 * V)
            yc, xc, dy, dx, _pr = correlate_windows(a, b, win, step)
            # trường dự báo tại tâm cửa sổ
            if first:
                ug, vg = dx, dy
            else:
                ug = ndimage.map_coordinates(U, np.meshgrid(yc, xc, indexing="ij"), order=1) + dx
                vg = ndimage.map_coordinates(V, np.meshgrid(yc, xc, indexing="ij"), order=1) + dy
            bad = normalized_median_test(ug, vg, median_eps, median_thr)
            ug, vg = _replace_outliers(ug, vg, bad)
            if smooth_sigma and smooth_sigma > 0:
                ug_s = ndimage.gaussian_filter(ug, smooth_sigma, mode="nearest")
                vg_s = ndimage.gaussian_filter(vg, smooth_sigma, mode="nearest")
            else:
                ug_s, vg_s = ug, vg
            U = grid_to_dense(yc, xc, ug_s, shape)
            V = grid_to_dense(yc, xc, vg_s, shape)
            grid = dict(xc=xc, yc=yc, u=ug, v=vg, outliers=bad)
            first = False
    # Kết quả cuối: dùng lưới không làm trơn
    U = grid_to_dense(grid["yc"], grid["xc"], grid["u"], shape)
    V = grid_to_dense(grid["yc"], grid["xc"], grid["v"], shape)
    if return_grid:
        return U, V, grid
    return U, V
