"""Sai số tại vùng gradient mạnh và sai số đỉnh xoáy."""

from __future__ import annotations

import numpy as np

from .errors import crop, error_stats


def velocity_gradients(u, v):
    """(∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y) bằng sai phân trung tâm bậc 2."""
    uy, ux = np.gradient(np.asarray(u, float))
    vy, vx = np.gradient(np.asarray(v, float))
    return ux, uy, vx, vy


def vorticity(u, v) -> np.ndarray:
    ux, uy, vx, vy = velocity_gradients(u, v)
    return vx - uy


def gradient_magnitude(u, v) -> np.ndarray:
    """‖∇u‖_F = √(u_x² + u_y² + v_x² + v_y²)."""
    ux, uy, vx, vy = velocity_gradients(u, v)
    return np.sqrt(ux**2 + uy**2 + vx**2 + vy**2)


def strong_gradient_mask(u_true, v_true, quantity: str = "vorticity", top_percent: float = 10.0,
                         border: int = 0) -> np.ndarray:
    """Mặt nạ top ``top_percent``% của |ω| hoặc ‖∇u‖ của trường GỐC (phân vị tính trong vùng bỏ biên)."""
    q = np.abs(vorticity(u_true, v_true)) if quantity in ("vorticity", "omega") else gradient_magnitude(u_true, v_true)
    thr = np.percentile(crop(q, border), 100 - top_percent)
    return q >= thr


def strong_gradient_errors(u, v, u_true, v_true, quantity: str = "vorticity", top_percent: float = 10.0,
                           border: int = 0, valid: np.ndarray | None = None) -> dict:
    """EPE trong/ngoài mặt nạ gradient mạnh (chỉ xét điểm ``valid`` nếu có)."""
    m = strong_gradient_mask(u_true, v_true, quantity, top_percent, border)
    ok = np.ones(m.shape, bool) if valid is None else np.asarray(valid, bool)
    inside = error_stats(u, v, u_true, v_true, border, m & ok)
    outside = error_stats(u, v, u_true, v_true, border, ~m & ok)
    return dict(mask_quantity=quantity, mask_top_percent=top_percent,
                epe_in=inside["epe_mean"], epe_out=outside["epe_mean"],
                rmse_in=inside["rmse"], rmse_out=outside["rmse"],
                ratio_in_out=inside["epe_mean"] / outside["epe_mean"] if outside["epe_mean"] else np.nan)


def vortex_peak_error(u, v, u_true, v_true, search_radius: int = 10, border: int = 0,
                      avg_radius: float = 2.0) -> dict:
    """So sánh đỉnh |ω| ước lượng với đỉnh gốc.

    Tìm đỉnh |ω_true| (vùng bỏ biên), rồi đỉnh |ω_est| trong bán kính ``search_radius`` quanh đó.
    Trả về omega_peak_true, omega_peak_est, omega_peak_rel_error (ω_est/ω_true − 1),
    peak_offset (px), và |u| cực đại gốc/ước lượng (đỉnh vận tốc tiếp tuyến).

    Vì max của trường có nhiễu bị thiên lệch dương, trả thêm ``omega_center_rel_error``:
    trung bình ω_est / ω_true trên đĩa bán kính ``avg_radius`` tại đúng vị trí đỉnh gốc.
    """
    wt = vorticity(u_true, v_true)
    we = vorticity(u, v)
    ny, nx = wt.shape
    b = int(border)
    sub = np.abs(wt[b:ny - b or None, b:nx - b or None])
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    iy, ix = iy + b, ix + b
    Y, X = np.mgrid[0:ny, 0:nx]
    near = (Y - iy) ** 2 + (X - ix) ** 2 <= search_radius**2
    we_n = np.where(near, np.abs(we), -np.inf)
    jy, jx = np.unravel_index(np.argmax(we_n), we.shape)
    pt, pe = float(wt[iy, ix]), float(we[jy, jx])
    disk = (Y - iy) ** 2 + (X - ix) ** 2 <= avg_radius**2
    wc_rel = float(we[disk].mean() / wt[disk].mean() - 1) if wt[disk].mean() else np.nan
    sp_t = np.hypot(u_true, v_true)
    sp_e = np.hypot(u, v)
    return dict(omega_peak_true=pt, omega_peak_est=pe, omega_peak_rel_error=pe / pt - 1 if pt else np.nan,
                omega_center_rel_error=wc_rel, peak_offset=float(np.hypot(jy - iy, jx - ix)),
                speed_peak_true=float(crop(sp_t, border).max()), speed_peak_est=float(crop(sp_e, border).max()))
