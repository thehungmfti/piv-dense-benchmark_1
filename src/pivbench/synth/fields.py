"""Trường dịch chuyển test (đơn vị: pixel / khoảng thời gian giữa 2 ảnh).

Quy ước toạ độ dùng trong toàn bộ pivbench:
  * ``x`` = chỉ số cột, ``y`` = chỉ số hàng, tâm pixel nằm tại toạ độ nguyên.
  * ``u`` là thành phần theo x (cột), ``v`` theo y (hàng).
  * Mảng trường có dạng ``(H, W)`` = ``(ny, nx)``.

Mỗi trường là một đối tượng ``Field`` có thể gọi tại toạ độ tuỳ ý
``field(x, y) -> (u, v)`` (dùng để tích phân quỹ đạo hạt) và lấy mẫu trên lưới
pixel bằng ``field.on_grid(shape)`` (dùng làm trường gốc khi đánh giá).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
from scipy import ndimage


class Field:
    """Trường dịch chuyển 2D dừng (không phụ thuộc thời gian)."""

    name = "field"

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def on_grid(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = shape
        y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
        return self(x, y)

    def max_displacement(self, shape: tuple[int, int]) -> float:
        u, v = self.on_grid(shape)
        return float(np.max(np.hypot(u, v)))


# ---------------------------------------------------------------------------
# (a) Sin 1D — dùng cho MTF
# ---------------------------------------------------------------------------
@dataclass
class SineField(Field):
    """``u = A sin(2π x / λ + φ)``, ``v = 0`` (mặc định, đúng như đặc tả).

    ``component`` chọn thành phần mang sóng ('u' hoặc 'v'), ``axis`` chọn hướng
    biến thiên ('x' hoặc 'y'). Ví dụ ``component='v', axis='x'`` cho trường cắt
    (không nén được), ``component='u', axis='x'`` (mặc định) là trường nén.
    """

    amplitude: float = 1.0
    wavelength: float = 32.0
    phase: float = 0.0
    component: str = "u"
    axis: str = "x"
    offset: float = 0.0
    name: str = "sine"

    def __call__(self, x, y):
        s = x if self.axis == "x" else y
        w = self.offset + self.amplitude * np.sin(2 * np.pi * np.asarray(s) / self.wavelength + self.phase)
        z = np.zeros_like(w)
        return (w, z) if self.component == "u" else (z, w)


# ---------------------------------------------------------------------------
# (b) Xoáy Lamb–Oseen
# ---------------------------------------------------------------------------
# Vận tốc tiếp tuyến u_θ(r) = Γ/(2πr) (1 − exp(−r²/r_c²)); cực đại tại
# r ≈ 1.12091 r_c với giá trị ≈ 0.638063 · Γ/(2π r_c).
_LO_RMAX = 1.1209083
_LO_UMAX_FACTOR = (1 - np.exp(-(_LO_RMAX**2))) / _LO_RMAX  # ≈ 0.638063


@dataclass
class LambOseenField(Field):
    """Xoáy Lamb–Oseen, tham số hoá theo dịch chuyển tiếp tuyến cực đại ``u_max`` (px)."""

    center: tuple[float, float] | None = None  # (x0, y0); None → tâm ảnh khi gọi on_grid
    core_radius: float = 20.0
    u_max: float = 4.0
    u_inf: tuple[float, float] = (0.0, 0.0)  # dịch chuyển nền đều
    name: str = "lamb_oseen"

    @property
    def circulation(self) -> float:
        return self.u_max * 2 * np.pi * self.core_radius / _LO_UMAX_FACTOR

    def __call__(self, x, y):
        x0, y0 = self.center if self.center is not None else (0.0, 0.0)
        dx = np.asarray(x, dtype=np.float64) - x0
        dy = np.asarray(y, dtype=np.float64) - y0
        r2 = dx * dx + dy * dy
        rc2 = self.core_radius**2
        # u_θ / r = Γ/(2π r²)(1 − e^{−r²/rc²}); giới hạn r→0 là Γ/(2π rc²)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(r2 > 1e-12 * rc2, -np.expm1(-r2 / rc2) / r2, 1.0 / rc2)
        f = f * self.circulation / (2 * np.pi)
        return self.u_inf[0] - f * dy, self.u_inf[1] + f * dx

    def on_grid(self, shape):
        if self.center is None:
            ny, nx = shape
            self.center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
        return super().on_grid(shape)

    def vorticity_peak(self) -> float:
        """ω(0) = Γ / (π r_c²) — đơn vị 1/khoảng thời gian (px/px)."""
        return self.circulation / (np.pi * self.core_radius**2)


# ---------------------------------------------------------------------------
# Trường cho trên lưới — nội suy bậc 3 (B-spline)
# ---------------------------------------------------------------------------
class GriddedField(Field):
    """Trường cho trên lưới pixel, đánh giá tại toạ độ bất kỳ bằng spline bậc 3.

    ``periodic=True`` dùng biên tuần hoàn (trường rối tổng hợp), ngược lại dùng
    phản xạ (mirror) ở biên.
    """

    name = "gridded"

    def __init__(self, u: np.ndarray, v: np.ndarray, periodic: bool = False,
                 origin: tuple[float, float] = (0.0, 0.0), order: int = 3, name: str | None = None):
        if u.shape != v.shape or u.ndim != 2:
            raise ValueError("u, v phải là mảng 2D cùng kích thước")
        self.u = np.asarray(u, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.periodic = periodic
        self.origin = origin  # toạ độ (x, y) của phần tử [0, 0] trong hệ toạ độ ảnh
        self.order = order
        self.mode = "grid-wrap" if periodic else "mirror"
        self._cu = ndimage.spline_filter(self.u, order=order, mode=self.mode)
        self._cv = ndimage.spline_filter(self.v, order=order, mode=self.mode)
        if name:
            self.name = name

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        coords = np.stack([(y - self.origin[1]).ravel(), (x - self.origin[0]).ravel()])
        kw = dict(order=self.order, mode=self.mode, prefilter=False)
        u = ndimage.map_coordinates(self._cu, coords, **kw).reshape(x.shape)
        v = ndimage.map_coordinates(self._cv, coords, **kw).reshape(x.shape)
        return u, v

    def on_grid(self, shape):
        if tuple(shape) == self.u.shape and self.origin == (0.0, 0.0):
            return self.u.copy(), self.v.copy()
        return super().on_grid(shape)


# ---------------------------------------------------------------------------
# (c) Trường rối 2D tổng hợp, phổ k^-5/3, pha ngẫu nhiên, div = 0
# ---------------------------------------------------------------------------
def synthetic_turbulence(shape: tuple[int, int], rms: float = 2.0, slope: float = -5.0 / 3.0,
                         k_min: float | None = None, k_max: float | None = None,
                         seed: int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sinh trường (u, v) tuần hoàn có phổ năng lượng E(k) ∝ k^slope.

    Phương pháp: dựng hàm dòng ψ trong không gian Fourier với biên độ
    |ψ̂(k)| ∝ sqrt(E(k) / (π k³)) và pha ngẫu nhiên đều, rồi lấy
    u = ∂ψ/∂y, v = −∂ψ/∂x bằng đạo hàm phổ ⇒ ∇·u = 0 chính xác (theo nghĩa phổ).

    ``k_min``/``k_max`` tính theo số sóng nguyên (chu kỳ trên toàn miền theo cạnh
    ngắn hơn); mặc định 2 … N/4 để tránh sóng ngắn hơn ~4 px vốn không biểu diễn
    được trên ảnh hạt. Trường được chuẩn hoá theo độ lệch chuẩn ``rms`` của |u|
    (mỗi thành phần có rms ≈ rms/√2).
    """
    ny, nx = shape
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(nx) * nx  # số sóng nguyên theo x (chu kỳ/miền)
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx * (min(nx, ny) / nx), ky * (min(nx, ny) / ny))
    K = np.hypot(KX, KY)
    n = min(nx, ny)
    k_min = 2.0 if k_min is None else k_min
    k_max = n / 4.0 if k_max is None else k_max
    band = (K >= k_min) & (K <= k_max)
    amp = np.zeros_like(K)
    amp[band] = np.sqrt(K[band] ** slope / (np.pi * K[band] ** 3))
    phase = np.exp(2j * np.pi * rng.random(K.shape))
    psi_hat = amp * phase
    # Đạo hàm phổ theo toạ độ pixel: ∂/∂x ↔ i·2π fx
    fx = 2j * np.pi * np.fft.fftfreq(nx)[None, :]
    fy = 2j * np.pi * np.fft.fftfreq(ny)[:, None]
    # Lấy phần thực ⇒ tương đương đối xứng Hermite hoá ψ̂, vẫn giữ div = 0
    u = np.real(np.fft.ifft2(fy * psi_hat))
    v = np.real(np.fft.ifft2(-fx * psi_hat))
    s = np.sqrt(np.mean(u**2 + v**2))
    if s > 0:
        u *= rms / s
        v *= rms / s
    return u, v


class TurbulenceField(GriddedField):
    name = "turbulence"

    def __init__(self, shape, rms=2.0, slope=-5.0 / 3.0, k_min=None, k_max=None, seed=0):
        u, v = synthetic_turbulence(shape, rms=rms, slope=slope, k_min=k_min, k_max=k_max, seed=seed)
        super().__init__(u, v, periodic=True, name="turbulence")
        self.params = dict(rms=rms, slope=slope, k_min=k_min, k_max=k_max, seed=seed)


# ---------------------------------------------------------------------------
# (d) Trường DNS từ file
# ---------------------------------------------------------------------------
@dataclass
class DNSFieldSpec:
    """Mô tả cách đọc một trường DNS lưu trong file HDF5/npz.

    ``scale`` đổi đơn vị vận tốc DNS sang px/khoảng thời gian (ví dụ
    ``scale = U_ref · Δt / Δx_px``). ``crop`` là (y0, y1, x0, x1).
    """

    path: str
    u_key: str = "u"
    v_key: str = "v"
    scale: float = 1.0
    crop: tuple[int, int, int, int] | None = None
    transpose: bool = False
    periodic: bool = False
    extra: dict = dc_field(default_factory=dict)


def load_dns_field(spec: DNSFieldSpec | str, **kwargs) -> GriddedField:
    """Nạp trường (u, v) từ file ``.npz`` hoặc ``.h5/.hdf5`` thành ``GriddedField``.

    Với HDF5, ``u_key``/``v_key`` có thể là đường dẫn dataset (``"/flow/u"``).
    Dataset 3D (ví dụ một lát z của JHTDB) cần chỉ số lát qua
    ``extra={"slice": k}`` (cắt theo trục đầu tiên).
    """
    if isinstance(spec, (str, Path)):
        spec = DNSFieldSpec(path=str(spec), **kwargs)
    path = Path(spec.path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as d:
            u, v = np.asarray(d[spec.u_key]), np.asarray(d[spec.v_key])
    elif suffix in (".h5", ".hdf5", ".he5"):
        import h5py

        with h5py.File(path, "r") as f:
            u, v = np.asarray(f[spec.u_key]), np.asarray(f[spec.v_key])
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {suffix} (cần .npz hoặc .h5)")
    if u.ndim == 3:
        k = spec.extra.get("slice", 0)
        u, v = u[k], v[k]
    if spec.transpose:
        u, v = u.T, v.T
    if spec.crop is not None:
        y0, y1, x0, x1 = spec.crop
        u, v = u[y0:y1, x0:x1], v[y0:y1, x0:x1]
    return GriddedField(u * spec.scale, v * spec.scale, periodic=spec.periodic, name=path.stem)


class DNSSource:
    """Giao diện (để trống) cho nguồn DNS ngoài: PIV Challenge, JHTDB...

    Lớp con cần cài đặt ``fetch(region) -> (u, v)`` trả về mảng px/khoảng thời gian.
    Ví dụ JHTDB: gọi ``givernylib``/``pyJHTDB`` với token cá nhân, lấy một lát 2D
    của ``isotropic1024coarse`` rồi nhân hệ số đổi đơn vị. Không cài sẵn vì cần
    đăng nhập/token — xem ``scripts/download_data.md``.
    """

    name = "dns_source"

    def fetch(self, region: dict) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError(
            "Chưa cài đặt truy cập trực tuyến (cần token JHTDB / tài khoản PIV Challenge). "
            "Hãy tải thủ công rồi dùng load_dns_field(path)."
        )

    def to_field(self, region: dict, scale: float = 1.0) -> GriddedField:
        u, v = self.fetch(region)
        return GriddedField(u * scale, v * scale, name=self.name)


class JHTDBSource(DNSSource):
    name = "jhtdb"


class PIVChallengeSource(DNSSource):
    name = "piv_challenge"


def make_field(kind: str, shape: tuple[int, int], **params) -> Field:
    """Factory dùng bởi run_benchmark: kind ∈ {sine, lamb_oseen, turbulence, dns, uniform, wake}."""
    kind = kind.lower()
    if kind == "sine":
        return SineField(**params)
    if kind in ("lamb_oseen", "vortex"):
        p = dict(params)
        if "center" in p and p["center"] is not None:
            p["center"] = tuple(p["center"])
        f = LambOseenField(**p)
        if f.center is None:
            f.center = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
        return f
    if kind == "turbulence":
        return TurbulenceField(shape, **params)
    if kind == "dns":
        return load_dns_field(DNSFieldSpec(**params))
    if kind == "uniform":
        return UniformField(**params)
    if kind == "wake":
        from .wake import AxisymmetricWakeField, WakeParams

        return AxisymmetricWakeField(shape, WakeParams(**params))
    raise ValueError(f"Loại trường không biết: {kind}")


@dataclass
class UniformField(Field):
    u0: float = 0.5
    v0: float = 0.25
    name: str = "uniform"

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        return np.full_like(x, self.u0), np.full_like(x, self.v0)
