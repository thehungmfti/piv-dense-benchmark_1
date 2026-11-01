"""Mô hình hoá độ phân giải hiệu dụng (P1, phần A).

* ``tophat_mtf``: hàm truyền lý thuyết của trung bình top-hat cỡ W (tương quan chéo một lượt,
  cửa sổ vuông, trường sin theo 1 trục): MTF(λ) = |sinc(W/λ)| = |sin(πW/λ)/(πW/λ)|.
* ``tophat_cutoff``: λ_c lý thuyết tại ngưỡng t (giải sinc(x) = t).
* ``particle_spacing``: khoảng cách trung bình giữa hạt ℓ_p = 1/√ppp (px).
* ``fit_power_law``: hồi quy log  log λ_c = log a + Σ b_i log X_i  (bình phương tối thiểu),
  trả về hệ số, R² trong không gian log và R² trong không gian tuyến tính.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def tophat_mtf(wavelength, window: float) -> np.ndarray:
    """|sinc(W/λ)| — np.sinc(x) = sin(πx)/(πx)."""
    lam = np.asarray(wavelength, float)
    return np.abs(np.sinc(window / lam))


def tophat_cutoff(window: float, threshold: float) -> float:
    """λ_c sao cho sinc(W/λ_c) = threshold (nhánh chính, 0 < W/λ < 1)."""
    x = brentq(lambda x: np.sinc(x) - threshold, 1e-9, 1.0 - 1e-12)
    return float(window / x)


def particle_spacing(ppp) -> np.ndarray:
    """ℓ_p = 1/√ppp (px)."""
    return 1.0 / np.sqrt(np.asarray(ppp, float))


def fit_power_law(y, X: dict) -> dict:
    """Khớp y = a · Π X_i^{b_i} bằng hồi quy tuyến tính trên log.

    ``X``: {tên: mảng}. Bỏ các hàng không hữu hạn / ≤ 0. Trả về
    dict(a, exponents={tên: b}, r2_log, r2_lin, n, rmse_log, pred=hàm dự đoán).
    """
    y = np.asarray(y, float)
    names = list(X)
    cols = [np.asarray(X[k], float) for k in names]
    ok = np.isfinite(y) & (y > 0)
    for c in cols:
        ok &= np.isfinite(c) & (c > 0)
    n = int(ok.sum())
    if n < len(names) + 2:
        return dict(a=np.nan, exponents={k: np.nan for k in names}, r2_log=np.nan, r2_lin=np.nan,
                    n=n, rmse_log=np.nan)
    A = np.column_stack([np.ones(n)] + [np.log(c[ok]) for c in cols])
    ly = np.log(y[ok])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pl = A @ coef
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2_log = 1 - np.sum((ly - pl) ** 2) / ss_tot if ss_tot > 0 else np.nan
    yl, pe = y[ok], np.exp(pl)
    ss_tot_l = np.sum((yl - yl.mean()) ** 2)
    r2_lin = 1 - np.sum((yl - pe) ** 2) / ss_tot_l if ss_tot_l > 0 else np.nan
    return dict(a=float(np.exp(coef[0])), exponents={k: float(b) for k, b in zip(names, coef[1:])},
                r2_log=float(r2_log), r2_lin=float(r2_lin), n=n,
                rmse_log=float(np.sqrt(np.mean((ly - pl) ** 2))))


def fit_linear(y, x) -> dict:
    """y = α·x + β (bình phương tối thiểu), R²; cùng với tỉ số trung bình y/x (đi qua gốc)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3:
        return dict(alpha=np.nan, beta=np.nan, r2=np.nan, ratio_mean=np.nan, ratio_std=np.nan,
                    ratio_cv=np.nan, r2_origin=np.nan, n=int(ok.sum()))
    y, x = y[ok], x[ok]
    alpha, beta = np.polyfit(x, y, 1)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - (alpha * x + beta)) ** 2) / ss_tot if ss_tot > 0 else np.nan
    # Qua gốc: y = c·x
    c = float(np.sum(x * y) / np.sum(x * x))
    r2o = 1 - np.sum((y - c * x) ** 2) / ss_tot if ss_tot > 0 else np.nan
    r = y / x
    return dict(alpha=float(alpha), beta=float(beta), r2=float(r2), c_origin=c, r2_origin=float(r2o),
                ratio_mean=float(r.mean()), ratio_std=float(r.std()),
                ratio_cv=float(r.std() / r.mean()), n=int(len(y)))
