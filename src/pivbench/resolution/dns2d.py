"""DNS 2D (Navier–Stokes không nén, tuần hoàn) bằng phương pháp giả phổ — trường thay thế DNS.

Dùng khi không có dữ liệu PIV Challenge 2005 Case B / JHTDB trong ``data/``. Đây là rối 2D
suy giảm (decaying), KHÔNG phải rối 3D: phổ cuối ~k^-3 (dải enstrophy) với các xoáy kết hợp,
pha không ngẫu nhiên — khác về cấu trúc so với trường pha ngẫu nhiên k^-5/3.

Phương trình xoáy: ∂ω/∂t + u·∇ω = ν∇²ω − ν_h(−∇²)^p ω  (hyper-viscosity p = 4 tuỳ chọn),
khử răng cưa 2/3, tích phân RK4 bước thời gian cố định theo CFL.
"""

from __future__ import annotations

import numpy as np


def run_dns2d(n: int = 256, k0: float = 12.0, nu: float = 5e-4, t_end: float = 2.0,
              cfl: float = 0.5, hyper: float = 0.0, seed: int = 0, rms_out: float | None = 2.0):
    """Trả về (u, v) trên lưới n×n (đơn vị: chu kỳ miền = 2π; được chuẩn hoá rms nếu ``rms_out``).

    Khởi tạo: phổ năng lượng E(k) ∝ k⁴ exp(−2(k/k0)²) pha ngẫu nhiên, năng lượng ½⟨|u|²⟩ = ½.
    """
    rng = np.random.default_rng(seed)
    kk = np.fft.fftfreq(n, 1.0 / n)
    KX, KY = np.meshgrid(kk, kk)
    K2 = KX**2 + KY**2
    K = np.sqrt(K2)
    K2i = np.where(K2 > 0, 1.0 / np.where(K2 > 0, K2, 1.0), 0.0)
    dealias = (np.abs(KX) < n / 3) & (np.abs(KY) < n / 3)

    # ω̂ từ phổ E(k): |ω̂|² ∝ k² E(k)/(πk) trên vòng
    E = K**4 * np.exp(-2.0 * (K / k0) ** 2)
    amp = np.sqrt(np.where(K > 0, K * E / np.pi, 0.0))
    w_hat = amp * np.exp(2j * np.pi * rng.random(K.shape))
    w = np.real(np.fft.ifft2(w_hat))
    w_hat = np.fft.fft2(w)

    def vel(wh):
        psi = wh * K2i  # −∇²ψ = ω → ψ̂ = ω̂/k²
        return np.real(np.fft.ifft2(1j * KY * psi)), np.real(np.fft.ifft2(-1j * KX * psi))

    u, v = vel(w_hat)
    s = np.sqrt(np.mean(u**2 + v**2))
    w_hat /= s
    lin = -nu * K2 - hyper * K2**4

    def rhs(wh):
        u, v = vel(wh)
        wx = np.real(np.fft.ifft2(1j * KX * wh))
        wy = np.real(np.fft.ifft2(1j * KY * wh))
        nl = np.fft.fft2(u * wx + v * wy)
        return -nl * dealias + lin * wh

    dx = 2 * np.pi / n
    t = 0.0
    while t < t_end:
        u, v = vel(w_hat)
        dt = min(cfl * dx / max(np.abs(u).max(), np.abs(v).max(), 1e-6), t_end - t)
        k1 = rhs(w_hat)
        k2 = rhs(w_hat + 0.5 * dt * k1)
        k3 = rhs(w_hat + 0.5 * dt * k2)
        k4 = rhs(w_hat + dt * k3)
        w_hat = w_hat + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    u, v = vel(w_hat)
    if rms_out is not None:
        s = np.sqrt(np.mean(u**2 + v**2))
        u, v = u * rms_out / s, v * rms_out / s
    return u, v
