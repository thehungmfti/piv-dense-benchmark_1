"""Trường dịch chuyển tổng hợp cho dòng wake sau mô hình trục đối xứng đáy tù (blunt base).

Mặt phẳng đo là mặt kinh tuyến chứa trục: ``x`` (cột) theo dòng chảy, ``y`` (hàng) theo bán kính,
trục mô hình tại hàng ``y_axis``; đáy (base) tại cột ``x_base``, đường kính đáy ``D`` (px).
Mô hình bán thực nghiệm (không phải nghiệm Navier–Stokes), các thành phần:

1. Dòng trung bình: vận tốc tâm u_c(ξ), ξ = (x − x_base)/D, nội suy PCHIP qua các điểm điều khiển
   (dòng ngược cực đại −U_rev tại ≈ 0.5 L_r, điểm bám lại u_c = 0 tại ξ = L_r/D, hồi phục chậm
   về U∞ phía sau); profile thiếu hụt exp(−|η/δ|^p) với nửa bề rộng δ(ξ) (co lại ở vùng bám lại
   rồi nở ra) và số mũ p(ξ) từ ~8 (lớp cắt mỏng sát đáy, profile dạng "mũ") về 2 (Gauss, wake xa).
2. Xoáy cuộn của lớp cắt (Kelvin–Helmholtz) dọc hai lớp cắt, chiều quay theo dấu ∂u/∂y,
   khoảng cách tăng dần theo ξ.
3. Mode xoáy so le (dấu vết của mode xoắn ốc St ≈ 0.2 trên mặt kinh tuyến) phía sau L_r.
4. Rối k^-5/3 điều biến bởi bao lớp cắt + lõi wake.

Dòng trung bình (mặc định ``axisymmetric=True``) dựng từ HÀM DÒNG STOKES: u = (1/r)∂ψ/∂r,
v = −(1/r)∂ψ/∂x ⇒ ∂(r u)/∂x + ∂(r v)/∂r = 0, tức div₂D = −v/r ≠ 0 trên mặt kinh tuyến — đúng như
PIV 2D đo được trong dòng trục đối xứng (quan trọng: mạng huấn luyện chỉ với trường div₂D = 0 học prior
không nén được và triệt tiêu thành phần nén). Các thành phần xoáy/rối là phẳng, cộng qua hàm dòng
phẳng ψ (u = D_y ψ, v = −D_x ψ). ``axisymmetric=False``: toàn bộ qua ψ phẳng (div₂D = 0 rời rạc).
Vùng thân mô hình (x < x_base, |y − y_axis| < D/2) có u = v = 0 và được trả về qua ``body_mask``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.interpolate import PchipInterpolator

from .fields import GriddedField, synthetic_turbulence


@dataclass
class WakeParams:
    D: float = 120.0               # đường kính đáy [px]
    U_inf: float = 6.0             # dịch chuyển dòng tự do [px/khoảng thời gian]
    x_base: float = 20.0           # vị trí đáy (cột); âm = đáy ngoài khung ảnh bên trái
    y_axis: float | None = None    # hàng của trục; None = giữa ảnh
    L_r: float = 1.3               # chiều dài vùng tuần hoàn / D
    U_rev: float = 0.25            # dòng ngược cực đại / U∞
    delta0: float = 0.5            # nửa bề rộng wake tại đáy / D
    kh_strength: float = 0.5       # cường độ xoáy lớp cắt (0 = tắt)
    kh_spacing: float = 0.35       # khoảng cách xoáy lớp cắt tại đáy / D
    street_strength: float = 0.4   # cường độ mode xoáy so le (0 = tắt)
    street_wavelength: float = 3.0  # bước sóng mode so le / D
    street_phase: float = 0.0
    turb_rms: float = 0.06         # rms rối / U∞ (trước khi điều biến)
    turb_kmax: float | None = None  # số sóng cực đại (chu kỳ/miền); None = N/6
    angle_deg: float = 0.0         # góc tới nhỏ (xoay toàn trường quanh tâm đáy)
    with_body: bool = True
    axisymmetric: bool = True      # dòng trung bình theo liên tục trục đối xứng (hàm dòng Stokes)
    seed: int | None = 0


def _centerline(L):
    xi = np.array([0.0, 0.12 * L, 0.5 * L, L, 1.6 * L, 3.0 * L, 6.0 * L, 15.0 * L, 60.0 * L])
    return xi


def _ddx(f):
    return np.gradient(f, axis=1)


def _ddy(f):
    return np.gradient(f, axis=0)


class AxisymmetricWakeField(GriddedField):
    """Trường wake đáy tù (xem docstring module). ``body_mask`` có cùng kích thước ảnh."""

    name = "wake"

    def __init__(self, shape: tuple[int, int], params: WakeParams | None = None, margin: int = 24, **kw):
        p = params or WakeParams()
        if kw:
            p = WakeParams(**{**asdict(p), **kw})
        self.params = p
        self.shape = tuple(shape)
        rng = np.random.default_rng(p.seed)
        ny, nx = shape
        m = int(margin)
        NY, NX = ny + 2 * m, nx + 2 * m
        Y, X = np.mgrid[0:NY, 0:NX].astype(np.float64)
        X -= m
        Y -= m
        ya = (ny - 1) / 2.0 if p.y_axis is None else p.y_axis
        D, U = float(p.D), float(p.U_inf)
        # toạ độ quay quanh tâm đáy (góc tới)
        th = np.deg2rad(p.angle_deg)
        Xr = (X - p.x_base) * np.cos(th) + (Y - ya) * np.sin(th)
        Yr = -(X - p.x_base) * np.sin(th) + (Y - ya) * np.cos(th)
        xi = Xr / D
        eta = Yr / D
        L = p.L_r

        # 1) dòng trung bình (dạng u trong hệ quay), ψ = ∫ u dη
        uc_pts = np.array([-0.05, -0.6 * p.U_rev, -p.U_rev, 0.0, 0.3, 0.55, 0.75, 0.9, 0.98])
        uc = PchipInterpolator(_centerline(L), uc_pts, extrapolate=False)
        dl_pts = np.array([p.delta0, p.delta0 * 0.99, p.delta0 * 0.93, p.delta0 * 0.85,
                           p.delta0 * 0.9, p.delta0 * 1.05, p.delta0 * 1.3, p.delta0 * 1.8, p.delta0 * 3.0])
        dl = PchipInterpolator(_centerline(L), dl_pts, extrapolate=False)
        pe = PchipInterpolator(np.array([0, 0.5 * L, L, 3 * L, 60 * L]), np.array([8.0, 5.0, 3.0, 2.0, 2.0]),
                               extrapolate=False)
        xic = np.clip(xi, 0.0, 60 * L)
        ucx, dlx, px = uc(xic), dl(xic), pe(xic)
        # Phía trước đáy (dọc thân): lớp biên mỏng quanh thân, ngoài thân là U∞
        upstream = xi < 0
        deficit = (1.0 - ucx) * np.exp(-np.abs(eta / dlx) ** px)
        bl = 0.04 + 0.02 * np.clip(-xi, 0, 5)
        deficit_up = np.exp(-np.clip(np.abs(eta) - 0.5, 0, None) / bl)
        # chuyển tiếp trơn trong 0 < ξ < 0.15 (tránh bước nhảy ψ tại mặt phẳng đáy → v lớn giả)
        wgt = np.clip(xi / 0.15, 0.0, 1.0)
        wgt = wgt * wgt * (3 - 2 * wgt)
        deficit = np.where(upstream, deficit_up, wgt * deficit + (1 - wgt) * deficit_up)
        u_mean = U * (1.0 - deficit)
        # ψ theo hệ quay: tích phân u_mean theo η (pixel) dọc mỗi cột của lưới quay ≈ dùng lưới gốc
        # khi góc nhỏ: ψ(x, y) = Σ u dy (sai lệch do góc được hấp thụ vì u,v lấy lại từ ψ)
        um = vm = 0.0
        if p.axisymmetric:
            # Hàm dòng Stokes trong hệ quay (s = Yr có dấu, r = |s|):
            #   ψ_S = U[s|s|/2 − I],  I(x, s) = ∫_0^s deficit·|s'| ds'
            #   u = (1/r) ∂ψ_S/∂r = u_mean,   v_r = −(1/r) ∂ψ_S/∂x_r = U·(∂I/∂x_r)/r
            # ⇒ ∂(r u)/∂x + ∂(r v)/∂r = 0 (liên tục trục đối xứng); div₂D = −v/r ≠ 0 như PIV thật.
            G = deficit * np.abs(Yr)
            C = np.cumsum(G, axis=0)  # tích phân theo hàng (≈ theo r khi góc nhỏ)
            row_axis = ya + m + (X[0] - p.x_base) * np.tan(th)  # hàng của trục trên mỗi cột (chỉ số lưới)
            j0 = np.clip(np.floor(row_axis).astype(int), 0, NY - 2)
            fr = np.clip(row_axis - j0, 0, 1)
            cols = np.arange(NX)
            C_axis = C[j0, cols] * (1 - fr) + C[j0 + 1, cols] * fr
            Iint = (C - C_axis[None, :]) * np.cos(th)
            dIdx = np.cos(th) * _ddx(Iint) + np.sin(th) * _ddy(Iint)
            v_r = U * dIdx / np.maximum(np.abs(Yr), 0.5)
            um = u_mean * np.cos(th) - v_r * np.sin(th)
            vm = u_mean * np.sin(th) + v_r * np.cos(th)
            psi = np.zeros_like(u_mean)
        else:
            psi = np.cumsum(u_mean, axis=0) * np.cos(th)
            psi -= psi[int(round(ya)) + m, :][None, :] if 0 <= int(round(ya)) + m < NY else 0.0
            if th != 0:
                # thêm thành phần dòng tự do theo góc: ψ_∞ = U(y cosθ − x sinθ) đã gần đúng trong cumsum;
                # bổ sung −U x sinθ để dòng tự do có hướng θ
                psi -= U * np.sin(th) * X

        # 2) xoáy lớp cắt, 3) mode so le — ψ của xoáy Lamb–Oseen:
        # ψ = −Γ/(4π) [ln(r²) + E1(r²/rc²)]  (u = ∂ψ/∂y, v = −∂ψ/∂x)
        from scipy.special import exp1

        def add_vortex(xv, yv, gamma, rc):
            # (xv, yv) trong hệ quay → toạ độ ảnh
            xc = p.x_base + xv * np.cos(th) - yv * np.sin(th)
            yc = ya + xv * np.sin(th) + yv * np.cos(th)
            r2 = (X - xc) ** 2 + (Y - yc) ** 2 + 1e-9
            return -gamma / (4 * np.pi) * (np.log(r2) + exp1(r2 / rc**2))

        x_max = (nx + m - p.x_base) / D + 1.0
        if p.kh_strength > 0 and x_max > 0:
            for side in (-1.0, 1.0):
                s = max(0.0, -p.x_base / D - 1.0) if p.x_base < 0 else 0.0
                s = max(s, 0.15 * rng.random())
                while s < min(x_max, 2.5 * L):
                    spacing = p.kh_spacing * (1.0 + 0.8 * s / L) * rng.uniform(0.8, 1.2)
                    s += spacing
                    d = float(dl(np.clip(s, 0, 60 * L)))
                    dU = U * (1.0 - float(uc(np.clip(s, 0, 60 * L))))
                    # dấu: ω ≈ −∂u/∂y; phía η>0 (dưới ảnh) u tăng theo η ⇒ ω < 0
                    gamma = -side * 0.35 * p.kh_strength * dU * spacing * D * rng.uniform(0.5, 1.0)
                    rc = 0.25 * spacing * D * rng.uniform(0.7, 1.3)
                    yv = side * d * D * rng.uniform(0.85, 1.1)
                    psi += add_vortex(s * D, yv, gamma, rc)
        if p.street_strength > 0:
            lam = p.street_wavelength
            k = 0
            s0 = L + (p.street_phase % 1.0) * lam / 2
            while s0 + k * lam / 2 < x_max:
                s = s0 + k * lam / 2
                side = 1.0 if k % 2 == 0 else -1.0
                d = float(dl(np.clip(s, 0, 60 * L)))
                dU = U * (1.0 - float(uc(np.clip(s, 0, 60 * L))))
                ramp = min(1.0, (s - L) / L + 0.3)
                gamma = -side * p.street_strength * ramp * dU * lam * D * 0.08 * rng.uniform(0.7, 1.0)
                rc = 0.35 * d * D
                psi += add_vortex(s * D, side * 0.45 * d * D, gamma, rc)
                k += 1

        # 4) rối, điều biến bởi bao lớp cắt + lõi wake
        if p.turb_rms > 0:
            ut, vt = synthetic_turbulence((NY, NX), rms=1.0, k_min=2, k_max=p.turb_kmax or min(NY, NX) / 6,
                                          seed=int(rng.integers(2**31)))
            # hàm dòng của rối: giải ∇²ψ = −ω bằng FFT (miền tuần hoàn)
            w = _ddx(vt) - _ddy(ut)
            kx = 2 * np.pi * np.fft.fftfreq(NX)[None, :]
            ky = 2 * np.pi * np.fft.fftfreq(NY)[:, None]
            k2 = kx**2 + ky**2
            k2[0, 0] = 1.0
            psit = np.real(np.fft.ifft2(np.fft.fft2(w) / k2))
            dd = np.where(xi > 0, dlx, p.delta0)
            env = np.exp(-((np.abs(eta) - dd) / (0.12 + 0.1 * np.clip(xi, 0, 10))) ** 2)
            env = np.maximum(env, 0.6 * np.exp(-np.abs(eta / dd) ** 2) * np.clip(xi / L, 0, 1))
            env *= np.clip(xi * 4 + 1, 0, 1)  # tắt phía trước đáy
            psi += p.turb_rms * U * psit * env

        u = _ddy(psi) + um   # phần phẳng (xoáy, rối) từ ψ phẳng + dòng trung bình
        v = -_ddx(psi) + vm
        body = (Xr < 0) & (np.abs(Yr) < 0.5 * D) if p.with_body else np.zeros_like(X, bool)
        u[body] = 0.0
        v[body] = 0.0
        super().__init__(u, v, periodic=False, origin=(-m, -m), name="wake")
        self.body_mask = body[m:m + ny, m:m + nx]
        self.body_mask_grid = body
        self.psi = psi[m:m + ny, m:m + nx]


def random_wake_params(rng: np.random.Generator, shape: tuple[int, int], u_inf_range=(2.0, 10.0)) -> WakeParams:
    """Lấy mẫu ngẫu nhiên tham số wake (dùng sinh dữ liệu huấn luyện)."""
    ny, nx = shape
    D = float(np.exp(rng.uniform(np.log(0.3), np.log(2.0))) * min(ny, nx))
    L = float(rng.uniform(0.8, 1.6))
    # khung nhìn: sát đáy / vùng tuần hoàn / wake xa
    view = rng.choice(["base", "recirc", "far"], p=[0.4, 0.35, 0.25])
    if view == "base":
        x_base = float(rng.uniform(-0.1, 0.4) * nx)
    elif view == "recirc":
        x_base = float(-rng.uniform(0.2, 1.0) * L * D)
    else:
        x_base = float(-rng.uniform(1.0, 4.0) * L * D)
    y_axis = float((ny - 1) / 2 + rng.uniform(-0.6, 0.6) * D)
    return WakeParams(
        D=D, U_inf=float(rng.uniform(*u_inf_range)), x_base=x_base, y_axis=y_axis, L_r=L,
        U_rev=float(rng.uniform(0.1, 0.4)), delta0=float(rng.uniform(0.45, 0.55)),
        kh_strength=float(rng.uniform(0, 1.0)), kh_spacing=float(rng.uniform(0.2, 0.5)),
        street_strength=float(rng.uniform(0, 0.8)), street_wavelength=float(rng.uniform(2.0, 4.0)),
        street_phase=float(rng.random()), turb_rms=float(rng.uniform(0.0, 0.15)),
        angle_deg=float(rng.normal(0, 2.0)), with_body=True, seed=int(rng.integers(2**31)),
    )
