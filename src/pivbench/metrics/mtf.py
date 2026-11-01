"""Hàm truyền điều biến (MTF) từ trường sin.

Với trường gốc ``A sin(2π s/λ + φ)`` (s = x hoặc y), khớp bình phương tối thiểu
``a sin(2π s/λ + φ) + b cos(2π s/λ + φ) + c`` tại đúng bước sóng λ trên vùng đã bỏ biên
(trung bình theo trục vuông góc):

    MTF(λ) = A_est / A_true = √(a² + b²) / A_true,   Δφ = atan2(b, a).

Bước sóng cắt λ_c(t): bước sóng nhỏ nhất mà từ đó trở lên MTF ≥ t (nội suy tuyến tính
theo log λ tại điểm cắt đầu tiên khi đi từ λ lớn xuống λ nhỏ). Mặc định t ∈ {0.9, 0.5}.
"""

from __future__ import annotations

import numpy as np

from .errors import crop


def fit_sine(f: np.ndarray, wavelength: float, axis: str = "x", phase: float = 0.0,
             border: int = 0) -> dict:
    """Khớp biên độ & pha ở bước sóng cho trước. Trả về amplitude, phase_shift, offset, residual_rms."""
    f = np.asarray(f, float)
    ny, nx = f.shape
    Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    s = X if axis == "x" else Y
    th = 2 * np.pi * s / wavelength + phase
    fc, thc = crop(f, border).ravel(), crop(th, border).ravel()
    ok = np.isfinite(fc)
    A = np.column_stack([np.sin(thc[ok]), np.cos(thc[ok]), np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, fc[ok], rcond=None)
    a, b, c = coef
    res = fc[ok] - A @ coef
    return dict(amplitude=float(np.hypot(a, b)), phase_shift=float(np.arctan2(b, a)),
                offset=float(c), residual_rms=float(np.sqrt(np.mean(res**2))))


def mtf_from_sine(u_est, v_est, amplitude: float, wavelength: float, component: str = "u",
                  axis: str = "x", phase: float = 0.0, border: int = 0) -> dict:
    """MTF(λ) và độ lệch pha (rad) của thành phần mang sóng."""
    f = u_est if component == "u" else v_est
    r = fit_sine(f, wavelength, axis, phase, border)
    return dict(wavelength=float(wavelength), mtf=r["amplitude"] / amplitude,
                phase_error=r["phase_shift"], residual_rms=r["residual_rms"])


def cutoff_wavelength(wavelengths, mtf, threshold: float = 0.9) -> float:
    """λ_c tại MTF = threshold; NaN nếu MTF chưa bao giờ đạt ngưỡng; λ_min nếu luôn ≥ ngưỡng."""
    lam = np.asarray(wavelengths, float)
    m = np.asarray(mtf, float)
    o = np.argsort(lam)[::-1]  # λ giảm dần
    lam, m = lam[o], m[o]
    if not np.any(m >= threshold):
        return float("nan")
    i0 = int(np.argmax(m >= threshold))
    for i in range(i0, len(lam) - 1):
        if m[i] >= threshold > m[i + 1]:
            t = (m[i] - threshold) / (m[i] - m[i + 1])
            return float(np.exp(np.log(lam[i]) + t * (np.log(lam[i + 1]) - np.log(lam[i]))))
    return float(lam[-1])


def cutoffs(wavelengths, mtf, thresholds=(0.9, 0.5)) -> dict:
    return {f"lambda_c_{t:g}": cutoff_wavelength(wavelengths, mtf, t) for t in thresholds}
