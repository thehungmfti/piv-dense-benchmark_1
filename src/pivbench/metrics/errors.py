"""Sai số điểm: EPE, RMSE từng thành phần, bỏ biên."""

from __future__ import annotations

import numpy as np


def crop(a: np.ndarray, border: int | tuple[int, int] = 0) -> np.ndarray:
    """Bỏ ``border`` pixel ở mỗi cạnh (``(by, bx)`` để khác nhau theo trục)."""
    by, bx = (border, border) if np.isscalar(border) else border
    by, bx = int(by), int(bx)
    return a[by:a.shape[0] - by or None, bx:a.shape[1] - bx or None]


def endpoint_error(u, v, u_true, v_true) -> np.ndarray:
    return np.hypot(np.asarray(u) - u_true, np.asarray(v) - v_true)


def error_stats(u, v, u_true, v_true, border: int = 0, mask: np.ndarray | None = None) -> dict:
    """Thống kê sai số trên vùng đã bỏ biên (và trong ``mask`` nếu có).

    Trả về: epe_mean, epe_median, epe_p95, epe_max, rmse_u, rmse_v, rmse, bias_u, bias_v, n.
    Điểm NaN (ví dụ biên của phương pháp lưới) bị bỏ qua.
    """
    du = crop(np.asarray(u, float) - u_true, border)
    dv = crop(np.asarray(v, float) - v_true, border)
    ok = np.isfinite(du) & np.isfinite(dv)
    if mask is not None:
        ok &= crop(np.asarray(mask, bool), border)
    du, dv = du[ok], dv[ok]
    if du.size == 0:
        return {k: np.nan for k in ("epe_mean", "epe_median", "epe_p95", "epe_max", "rmse_u",
                                    "rmse_v", "rmse", "bias_u", "bias_v")} | {"n": 0}
    e = np.hypot(du, dv)
    return dict(
        epe_mean=float(e.mean()), epe_median=float(np.median(e)), epe_p95=float(np.percentile(e, 95)),
        epe_max=float(e.max()), rmse_u=float(np.sqrt(np.mean(du**2))), rmse_v=float(np.sqrt(np.mean(dv**2))),
        rmse=float(np.sqrt(np.mean(du**2 + dv**2))), bias_u=float(du.mean()), bias_v=float(dv.mean()),
        n=int(e.size),
    )


def mean_epe(u, v, u_true, v_true, border: int = 0) -> float:
    return error_stats(u, v, u_true, v_true, border)["epe_mean"]
