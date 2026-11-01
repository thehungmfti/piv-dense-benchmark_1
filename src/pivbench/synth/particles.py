"""Bộ sinh cặp ảnh hạt PIV tổng hợp.

* Hạt Gauss, đường kính ảnh ``d_p`` (đường kính e⁻², mặc định 2.5 px); cường độ
  được tích phân giải tích trên diện tích pixel (hàm erf), không lấy mẫu điểm.
* Mật độ ``ppp`` (particles per pixel), cường độ đỉnh ngẫu nhiên.
* Tấm laser Gauss có độ dày ``sheet_thickness`` (đường kính e⁻²); dịch chuyển
  ngoài mặt phẳng ``w`` làm hạt sáng/tối đi hoặc biến mất (mất cặp).
* Nhiễu Gauss + Poisson theo SNR cho trước.
* Ảnh 2 được tạo bằng cách DỊCH VỊ TRÍ HẠT theo trường dịch chuyển (không warp
  ảnh): vị trí hạt được lấy ngẫu nhiên tại thời điểm giữa t = ½, rồi tích phân
  RK4 lùi nửa bước để được vị trí ở ảnh 1 và tiến nửa bước để được vị trí ở ảnh 2.
  Như vậy trường gốc trên lưới ứng với vị trí giữa — đúng với cách ước lượng đối
  xứng (central difference) của PIV. Trường được đánh giá tại vị trí hạt bằng
  nội suy bậc 3 (spline) với trường cho trên lưới, hoặc giải tích với trường
  sin / Lamb–Oseen.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.special import erf

from .fields import Field


@dataclass
class SynthParams:
    ppp: float = 0.05
    d_p: float = 2.5
    d_p_std: float = 0.0          # độ lệch chuẩn đường kính hạt (px)
    intensity_min: float = 0.6    # cường độ đỉnh ~ U(intensity_min, 1)
    snr: float | None = None      # I_peak / σ_noise (tuyến tính); None = không nhiễu
    poisson_fraction: float = 0.5  # tỉ lệ phương sai nhiễu tại đỉnh do Poisson
    background: float = 0.0
    sheet_thickness: float | None = None  # đường kính e⁻² của tấm laser (px)
    out_of_plane_w: float = 0.0   # dịch chuyển z trung bình (px)
    out_of_plane_std: float = 0.0  # độ lệch chuẩn dịch chuyển z (px, ngẫu nhiên từng hạt)
    rk4_substeps: int = 1         # số bước RK4 trong mỗi nửa bước
    bit_depth: int | None = None  # lượng tử hoá (ví dụ 8, 12); None = float
    seed: int | None = 0


def rk4_advect(field: Field, x: np.ndarray, y: np.ndarray, dt: float, nsub: int = 1):
    """Tích phân dx/dt = u(x) bằng RK4 trong thời gian ``dt`` (trường dừng)."""
    h = dt / nsub
    for _ in range(nsub):
        k1u, k1v = field(x, y)
        k2u, k2v = field(x + 0.5 * h * k1u, y + 0.5 * h * k1v)
        k3u, k3v = field(x + 0.5 * h * k2u, y + 0.5 * h * k2v)
        k4u, k4v = field(x + h * k3u, y + h * k3v)
        x = x + h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        y = y + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x, y


def render_particles(shape: tuple[int, int], x: np.ndarray, y: np.ndarray,
                     peak: np.ndarray, d_p: np.ndarray | float) -> np.ndarray:
    """Vẽ hạt Gauss tích phân trên pixel: I = I0·2πσ²·Px·Py, σ = d_p/4."""
    ny, nx = shape
    img = np.zeros(ny * nx, dtype=np.float64)
    d = np.broadcast_to(np.asarray(d_p, dtype=np.float64), x.shape)
    sigma = d / 4.0
    r = int(np.ceil(2.0 * float(np.max(d)))) + 1  # bán kính patch ≈ 4σ·2
    off = np.arange(-r, r + 1)
    ix0 = np.round(x).astype(np.int64)
    iy0 = np.round(y).astype(np.int64)
    s2 = (np.sqrt(2.0) * sigma)[:, None]
    # Tích phân 1D trên pixel [i-0.5, i+0.5]
    cx = ix0[:, None] + off[None, :]
    cy = iy0[:, None] + off[None, :]
    px = 0.5 * (erf((cx + 0.5 - x[:, None]) / s2) - erf((cx - 0.5 - x[:, None]) / s2))
    py = 0.5 * (erf((cy + 0.5 - y[:, None]) / s2) - erf((cy - 0.5 - y[:, None]) / s2))
    w = (peak * 2 * np.pi * sigma**2)[:, None, None] * py[:, :, None] * px[:, None, :]
    CY = np.broadcast_to(cy[:, :, None], w.shape)
    CX = np.broadcast_to(cx[:, None, :], w.shape)
    ok = (CX >= 0) & (CX < nx) & (CY >= 0) & (CY < ny)
    np.add.at(img, (CY[ok] * nx + CX[ok]), w[ok])
    return img.reshape(ny, nx)


def add_noise(img: np.ndarray, snr: float | None, poisson_fraction: float,
              rng: np.random.Generator, peak: float = 1.0) -> np.ndarray:
    """Nhiễu Poisson + Gauss sao cho σ_tổng tại đỉnh = peak / SNR.

    Với tỉ lệ ``f = poisson_fraction``: số photon tại đỉnh N = SNR²/f (để σ_Poisson
    tương đối = √f / SNR), còn σ_Gauss = √(1−f)·peak/SNR.
    """
    if snr is None or not np.isfinite(snr):
        return img
    f = float(np.clip(poisson_fraction, 0.0, 1.0))
    out = img
    if f > 0:
        n_peak = snr**2 / f
        out = rng.poisson(np.clip(img, 0, None) / peak * n_peak) * (peak / n_peak)
    if f < 1:
        out = out + rng.normal(0.0, np.sqrt(1 - f) * peak / snr, img.shape)
    return out


def generate_pair(field: Field, shape: tuple[int, int] = (256, 256),
                  params: SynthParams | None = None, **overrides):
    """Sinh cặp ảnh (img1, img2) và trường gốc (u, v) trên lưới pixel.

    Trả về dict: ``img1, img2, u, v, meta``.
    """
    p = params or SynthParams()
    if overrides:
        p = SynthParams(**{**asdict(p), **overrides})
    rng = np.random.default_rng(p.seed)
    ny, nx = shape
    u_true, v_true = field.on_grid(shape)
    dmax = float(np.max(np.hypot(u_true, v_true)))
    margin = 0.5 * dmax + 2 * p.d_p + 2
    # Gieo hạt tại t = ½ trên miền mở rộng để hạt đi vào ảnh không để lại vùng trống
    area = (nx + 2 * margin) * (ny + 2 * margin)
    n = rng.poisson(p.ppp * area)
    xm = rng.uniform(-margin, nx - 1 + margin, n)
    ym = rng.uniform(-margin, ny - 1 + margin, n)
    peak = rng.uniform(p.intensity_min, 1.0, n)
    dp = np.clip(rng.normal(p.d_p, p.d_p_std, n), 0.5, None) if p.d_p_std > 0 else np.full(n, p.d_p)

    x1, y1 = rk4_advect(field, xm, ym, -0.5, p.rk4_substeps)
    x2, y2 = rk4_advect(field, xm, ym, +0.5, p.rk4_substeps)

    peak1 = peak2 = peak
    n_lost = 0
    if p.sheet_thickness is not None:
        dz = p.sheet_thickness
        # z tại t=½ trong ±dz (bao cả hạt nằm rìa tấm laser)
        zm = rng.uniform(-dz, dz, n)
        w = p.out_of_plane_w + (rng.normal(0, p.out_of_plane_std, n) if p.out_of_plane_std > 0 else 0.0)
        z1, z2 = zm - 0.5 * w, zm + 0.5 * w
        prof = lambda z: np.exp(-8.0 * z**2 / dz**2)  # noqa: E731  — profile Gauss, e⁻² tại ±dz/2
        peak1, peak2 = peak * prof(z1), peak * prof(z2)
        # "Mất cặp": hạt sáng ở ảnh này nhưng < e⁻² ở ảnh kia
        thr = np.exp(-2.0)
        n_lost = int(np.sum((prof(z1) > thr) != (prof(z2) > thr)))

    img1 = render_particles(shape, x1, y1, peak1, dp) + p.background
    img2 = render_particles(shape, x2, y2, peak2, dp) + p.background
    img1 = add_noise(img1, p.snr, p.poisson_fraction, rng)
    img2 = add_noise(img2, p.snr, p.poisson_fraction, rng)
    img1 = np.clip(img1, 0, None)
    img2 = np.clip(img2, 0, None)
    if p.bit_depth:
        levels = 2**p.bit_depth - 1
        img1 = np.round(np.clip(img1, 0, 1) * levels) / levels
        img2 = np.round(np.clip(img2, 0, 1) * levels) / levels
    meta = dict(asdict(p), n_particles=int(n), n_lost_pairs=n_lost, field=field.name,
                max_displacement=dmax, shape=tuple(shape))
    return dict(img1=img1.astype(np.float32), img2=img2.astype(np.float32),
                u=u_true, v=v_true, meta=meta)
