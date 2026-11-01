"""Hàm truyền H(k) và hiệu chỉnh phổ năng lượng (P1, phần B).

Quy ước số sóng: k tính bằng chu kỳ/px (như ``pivbench.metrics.spectra``), λ = 1/k.
(Với số sóng góc κ = 2πk thì H(κ) = MTF(2π/κ) — cùng một hàm.)

Mô hình phổ đo
    E_meas(k) = |H(k)|² E_true(k) + N(k)

Hiệu chỉnh (``correct_spectrum``)
    inverse : E_corr = (E_meas − N) / |H|²
    wiener  : E_corr = (E_meas − N) · |H|² / (|H|⁴ + ε (1 − |H|²)²)
chỉ áp cho các k có |H(k)| > h_min (ngoài vùng đó trả NaN). Số hạng ổn định
ε(1 − |H|²)² triệt tiêu khi |H| = 1 nên H ≡ 1, N ≡ 0 trả về đúng phổ gốc, và khi |H| → 0 thì
E_corr → (E_meas − N)|H|²/ε → 0 (không khuếch đại nhiễu). (E_meas − N) được chặn dưới bởi 0.

Sàn nhiễu N(k)
    (a) ``noise_floor_tail``: hằng số = trung vị của E_meas trên dải k cao [k_lo, k_hi]
        (giả thiết sai số ngẫu nhiên trắng ở k cao và |H|²E_true ≈ 0 ở đó).
    (b) ``noise_floor_zero_shift``: phổ của trường ước lượng trên cặp ảnh dịch chuyển 0
        (cùng hạt, nhiễu ảnh độc lập) — N(k) phụ thuộc k.
    (c) ``noise_from_ensemble``: N = E_meas − |H|²E_true trên tập huấn luyện có trường gốc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ..metrics.errors import crop


# ---------------------------------------------------------------------------
# Phổ 2D trung bình vòng (tự phổ & phổ chéo), cùng chuẩn hoá với metrics.spectra
# ---------------------------------------------------------------------------
def _ring_bin(P: np.ndarray, nbins: int | None = None):
    ny, nx = P.shape
    K = np.hypot(*np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny)))
    dk = 1.0 / min(nx, ny)
    nbins = nbins or min(nx, ny) // 2
    edges = (np.arange(nbins + 1) + 0.5) * dk
    idx = np.digitize(K.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    s = np.bincount(idx[ok], P.ravel()[ok], nbins)
    k = 0.5 * (edges[:-1] + edges[1:])
    return k, s / (nx * ny) / dk


def _fft_windowed(c: np.ndarray, win: np.ndarray, wnorm: float, detrend: bool) -> np.ndarray:
    c = np.asarray(c, float)
    if detrend:
        c = c - c.mean()
    return np.fft.fft2(c * win) / np.sqrt(wnorm)


def radial_cross_spectrum(a_u, a_v, b_u, b_v, border: int = 0, detrend: bool = True):
    """Phổ chéo vòng S_ab(k) = ½ Re⟨â_u b̂_u* + â_v b̂_v*⟩ (Hann 2D).

    Với a = b cho đúng ``pivbench.metrics.spectra.radial_spectrum_2d``.
    Trả về (k, S_ab).
    """
    au, av, bu, bv = (crop(np.asarray(x, float), border) for x in (a_u, a_v, b_u, b_v))
    ny, nx = au.shape
    win = np.outer(signal.windows.hann(ny, sym=False), signal.windows.hann(nx, sym=False))
    wn = (win**2).mean()
    P = np.zeros((ny, nx))
    for a, b in ((au, bu), (av, bv)):
        P += np.real(_fft_windowed(a, win, wn, detrend) * np.conj(_fft_windowed(b, win, wn, detrend))) / (nx * ny)
    return _ring_bin(0.5 * P)


def radial_auto_spectrum(u, v, border: int = 0, detrend: bool = True):
    return radial_cross_spectrum(u, v, u, v, border, detrend)


# ---------------------------------------------------------------------------
# Hàm truyền
# ---------------------------------------------------------------------------
@dataclass
class TransferFunction:
    """H(k) cho trên các nút (k_nodes tăng dần), nội suy tuyến tính theo log k.

    Ngoài khoảng nút: k < k_nodes[0] → H = H[0] (thường ≈ 1, có thể đặt ``low_k_one=True`` để
    ép = 1); k > k_nodes[-1] → H = ``high_k_value`` (mặc định 0: không tin vùng chưa đo).
    """

    k_nodes: np.ndarray
    H_nodes: np.ndarray
    label: str = ""
    high_k_value: float = 0.0
    low_k_one: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        k = np.asarray(self.k_nodes, float)
        H = np.asarray(self.H_nodes, float)
        ok = np.isfinite(k) & np.isfinite(H) & (k > 0)
        o = np.argsort(k[ok])
        self.k_nodes, self.H_nodes = k[ok][o], H[ok][o]

    @classmethod
    def from_mtf(cls, wavelengths, mtf, label: str = "", **kw) -> "TransferFunction":
        """H(k) = MTF(λ = 1/k) đo trên trường sin."""
        lam = np.asarray(wavelengths, float)
        return cls(1.0 / lam, np.asarray(mtf, float), label=label, **kw)

    @classmethod
    def from_cross_spectrum(cls, k, S_et, S_tt, label: str = "", min_rel_power: float = 1e-6,
                            **kw) -> "TransferFunction":
        """H(k) = S_est,true(k) / S_true,true(k) (bỏ các k mà S_tt quá nhỏ)."""
        k = np.asarray(k, float)
        S_tt = np.asarray(S_tt, float)
        ok = S_tt > min_rel_power * np.nanmax(S_tt)
        H = np.full_like(k, np.nan)
        H[ok] = np.asarray(S_et, float)[ok] / S_tt[ok]
        return cls(k[ok], H[ok], label=label, **kw)

    @classmethod
    def identity(cls) -> "TransferFunction":
        return cls(np.array([1e-6, 1.0]), np.array([1.0, 1.0]), label="identity", high_k_value=1.0)

    def __call__(self, k) -> np.ndarray:
        k = np.asarray(k, float)
        lk = np.log(np.clip(k, 1e-12, None))
        H = np.interp(lk, np.log(self.k_nodes), self.H_nodes,
                      left=1.0 if self.low_k_one else self.H_nodes[0], right=self.high_k_value)
        return np.where(k > self.k_nodes[-1], self.high_k_value, H)


# ---------------------------------------------------------------------------
# Sàn nhiễu
# ---------------------------------------------------------------------------
def noise_floor_tail(k, E_meas, k_lo: float = 0.35, k_hi: float = 0.5) -> np.ndarray:
    """(a) Hằng số = trung vị E_meas trên [k_lo, k_hi] (chu kỳ/px)."""
    k = np.asarray(k, float)
    E = np.asarray(E_meas, float)
    sel = (k >= k_lo) & (k <= k_hi) & np.isfinite(E)
    n0 = float(np.median(E[sel])) if sel.any() else 0.0
    return np.full_like(E, n0)


def noise_floor_zero_shift(u0, v0, border: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(b) Phổ của trường ước lượng trên cặp ảnh dịch chuyển 0 → (k, N(k))."""
    return radial_auto_spectrum(u0, v0, border)


def noise_from_ensemble(E_meas, E_true, H) -> np.ndarray:
    """(c) N = E_meas − |H|² E_true (trung bình trên tập có trường gốc), chặn dưới 0."""
    return np.clip(np.asarray(E_meas) - np.asarray(H) ** 2 * np.asarray(E_true), 0.0, None)


# ---------------------------------------------------------------------------
# Hiệu chỉnh
# ---------------------------------------------------------------------------
def correct_spectrum(k, E_meas, H, N=0.0, mode: str = "wiener", eps: float = 1e-2,
                     h_min: float = 0.2, clip_negative: bool = True) -> np.ndarray:
    """E_corr(k) theo công thức ở docstring module.

    ``H``: TransferFunction, hàm gọi được hoặc mảng cùng cỡ k. ``N``: số hoặc mảng.
    Trả NaN tại các k có |H| ≤ h_min.
    """
    k = np.asarray(k, float)
    E = np.asarray(E_meas, float)
    Hk = np.abs(H(k) if callable(H) else np.broadcast_to(np.asarray(H, float), k.shape))
    S = E - np.broadcast_to(np.asarray(N, float), E.shape)
    if clip_negative:
        S = np.clip(S, 0.0, None)
    H2 = Hk**2
    if mode == "inverse":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = S / H2
    elif mode == "wiener":
        out = S * H2 / (H2**2 + eps * (1.0 - H2) ** 2)
    else:
        raise ValueError(f"mode không biết: {mode}")
    return np.where(Hk > h_min, out, np.nan)


def valid_wavenumber(k, E, E_true, tol: float = 0.10, k_max: float | None = None,
                     k_min: float | None = None) -> float:
    """Số sóng lớn nhất k* mà |E/E_true − 1| < tol với MỌI k ≤ k* (liên tục từ k nhỏ nhất).

    Bỏ các k có E_true ≤ 0 hoặc > k_max. NaN trong E (ngoài vùng hiệu chỉnh) coi như không đạt.
    Trả 0 nếu ngay bin đầu đã sai > tol.
    """
    k = np.asarray(k, float)
    E = np.asarray(E, float)
    Et = np.asarray(E_true, float)
    sel = np.isfinite(Et) & (Et > 0)
    if k_max is not None:
        sel &= k <= k_max
    if k_min is not None:
        sel &= k >= k_min
    kk, r = k[sel], E[sel] / Et[sel]
    good = np.isfinite(r) & (np.abs(r - 1.0) < tol)
    if good.size == 0 or not good[0]:
        return 0.0
    bad = np.flatnonzero(~good)
    return float(kk[-1] if bad.size == 0 else kk[bad[0] - 1])
