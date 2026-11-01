"""Phổ năng lượng của trường vận tốc và của sai số.

* Phổ 1D theo x (dọc hàng) và theo y (dọc cột): Welch, cửa sổ Hann, trung bình
  trên tất cả hàng/cột. Đơn vị số sóng: chu kỳ/px (k), bước sóng λ = 1/k.
* Phổ 2D: FFT 2D với cửa sổ Hann 2D, trung bình theo vòng |k|.
* ``spectra_report``: luôn tính đồng thời phổ trường gốc, trường ước lượng và
  sai số (u_est − u_true).

Chuẩn hoá: phổ 1D là mật độ một phía (∫E dk ≈ phương sai). Phổ 2D chỉ lấy các vòng
|k| ≤ 0.5 chu kỳ/px nên ∫E dk ≈ π/4 phương sai với nhiễu trắng (bỏ các góc phổ).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .errors import crop


def spectrum_1d(f: np.ndarray, axis: int = 1, nperseg: int | None = None,
                detrend: str | bool = "constant") -> tuple[np.ndarray, np.ndarray]:
    """Phổ Welch (Hann) của mảng 2D dọc ``axis``, trung bình trên trục còn lại.

    ``axis=1`` → phổ theo x (k_x); ``axis=0`` → theo y.
    """
    f = np.asarray(f, float)
    n = f.shape[axis]
    nperseg = min(n, nperseg or n)
    k, P = signal.welch(f, fs=1.0, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                        detrend=detrend, axis=axis, scaling="density", return_onesided=True)
    other = 1 - axis
    return k, P.mean(axis=other)


def energy_spectra_1d(u, v, nperseg: int | None = None, border: int = 0) -> dict:
    """E_uu, E_vv và E = ½(E_uu + E_vv) theo k_x và k_y."""
    u = crop(np.asarray(u, float), border)
    v = crop(np.asarray(v, float), border)
    kx, Euu_x = spectrum_1d(u, 1, nperseg)
    _, Evv_x = spectrum_1d(v, 1, nperseg)
    ky, Euu_y = spectrum_1d(u, 0, nperseg)
    _, Evv_y = spectrum_1d(v, 0, nperseg)
    return dict(kx=kx, Euu_x=Euu_x, Evv_x=Evv_x, E_x=0.5 * (Euu_x + Evv_x),
                ky=ky, Euu_y=Euu_y, Evv_y=Evv_y, E_y=0.5 * (Euu_y + Evv_y))


def radial_spectrum_2d(u, v=None, border: int = 0, nbins: int | None = None,
                       detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Phổ 2D (Hann 2D) của ½(|û|²+|v̂|²), trung bình theo vòng |k| (chu kỳ/px).

    Trả về (k, E(k)) với E là mật độ theo |k| (đã nhân chu vi vòng 2π|k| để ∫E dk ≈ phương sai).
    """
    comps = [crop(np.asarray(u, float), border)]
    if v is not None:
        comps.append(crop(np.asarray(v, float), border))
    ny, nx = comps[0].shape
    win = np.outer(signal.windows.hann(ny, sym=False), signal.windows.hann(nx, sym=False))
    wnorm = (win**2).mean()
    P = np.zeros((ny, nx))
    for c in comps:
        if detrend:
            c = c - c.mean()
        F = np.fft.fft2(c * win)
        P += np.abs(F) ** 2 / (nx * ny * wnorm)  # Σ P/(nx ny) = phương sai
    P *= 0.5 if v is not None else 1.0
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    K = np.hypot(*np.meshgrid(kx, ky))
    dk = 1.0 / min(nx, ny)
    nbins = nbins or min(nx, ny) // 2
    edges = (np.arange(nbins + 1) + 0.5) * dk
    idx = np.digitize(K.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    s = np.bincount(idx[ok], P.ravel()[ok], nbins)
    # mật độ: tổng công suất trong vòng / (nx ny) / dk
    E = s / (nx * ny) / dk
    k = 0.5 * (edges[:-1] + edges[1:])
    return k, E


def spectra_report(u_est, v_est, u_true, v_true, border: int = 0, nperseg: int | None = None) -> dict:
    """Phổ 1D (x, y) và 2D của trường gốc, trường ước lượng và sai số."""
    eu = np.asarray(u_est, float) - u_true
    ev = np.asarray(v_est, float) - v_true
    out = {}
    for tag, (a, b) in dict(true=(u_true, v_true), est=(u_est, v_est), err=(eu, ev)).items():
        s1 = energy_spectra_1d(a, b, nperseg, border)
        k2, E2 = radial_spectrum_2d(a, b, border)
        for key, val in s1.items():
            if key in ("kx", "ky"):
                out[key] = val
            else:
                out[f"{tag}_{key}"] = val
        out["k2d"] = k2
        out[f"{tag}_E2d"] = E2
    return out
