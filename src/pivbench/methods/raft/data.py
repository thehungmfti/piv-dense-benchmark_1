"""Dữ liệu huấn luyện RAFT-PIV.

* ``build_pool``: sinh trước kho mẫu tổng hợp (wake đáy tù + trường tổng quát) ra file .npy
  (memmap) — ảnh hạt tạo bằng cách dịch vị trí hạt (RK4), trường gốc chính xác.
* ``MixedDataset``: trộn kho tổng hợp với bộ dữ liệu Cai et al. (cylinder/backstep), cắt ngẫu nhiên,
  lật/đảo thời gian, biến đổi cường độ.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ...synth import LambOseenField, SineField, TurbulenceField, UniformField, generate_pair
from ...synth.wake import AxisymmetricWakeField, random_wake_params


@dataclass
class ImagingRanges:
    """Khoảng tham số ảnh hạt lấy mẫu ngẫu nhiên (chỉnh cho thí nghiệm của bạn)."""

    ppp: tuple = (0.01, 0.12)          # log-uniform
    d_p: tuple = (1.5, 4.0)
    snr_none_prob: float = 0.3
    snr: tuple = (4.0, 40.0)           # log-uniform
    sheet_prob: float = 0.5
    sheet_thickness: tuple = (6.0, 16.0)
    oop_frac: tuple = (0.0, 0.3)       # σ_w / độ dày tấm
    background: tuple = (0.0, 0.08)
    quantize8_prob: float = 0.5
    u_inf: tuple = (1.0, 12.0)         # dịch chuyển dòng tự do [px]


def _loguni(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def _imaging(rng, r: ImagingRanges):
    kw = dict(ppp=_loguni(rng, *r.ppp), d_p=float(rng.uniform(*r.d_p)), d_p_std=float(rng.uniform(0, 0.4)),
              intensity_min=float(rng.uniform(0.3, 0.8)), background=float(rng.uniform(*r.background)),
              snr=None if rng.random() < r.snr_none_prob else _loguni(rng, *r.snr),
              poisson_fraction=float(rng.uniform(0.2, 0.8)), seed=int(rng.integers(2**31)))
    if rng.random() < r.sheet_prob:
        t = float(rng.uniform(*r.sheet_thickness))
        kw.update(sheet_thickness=t, out_of_plane_std=float(rng.uniform(*r.oop_frac)) * t)
    if rng.random() < r.quantize8_prob:
        kw["bit_depth"] = 8
    return kw


def make_sample(seed: int, size: int, kind: str, ranges: ImagingRanges | None = None):
    """Một mẫu (img1, img2, flow(2,H,W), valid(H,W)) kích thước size×size."""
    r = ranges or ImagingRanges()
    rng = np.random.default_rng(seed)
    shape = (size, size)
    body = None
    if kind == "wake":
        p = random_wake_params(rng, shape, r.u_inf)
        f = AxisymmetricWakeField(shape, p)
        body = f.body_mask
    elif kind == "vortex":
        f = LambOseenField(center=(rng.uniform(0, size), rng.uniform(0, size)),
                           core_radius=float(rng.uniform(4, size / 3)), u_max=float(rng.uniform(1, 8)),
                           u_inf=(float(rng.uniform(-6, 6)), float(rng.uniform(-3, 3))))
    elif kind == "turbulence":
        f = TurbulenceField(shape, rms=float(rng.uniform(0.5, 4)), k_max=float(rng.uniform(size / 16, size / 4)),
                            seed=int(rng.integers(2**31)))
        f.u += rng.uniform(-6, 6)
    elif kind == "compressible":
        # u, v là hai trường ngẫu nhiên trơn ĐỘC LẬP (div₂D ≠ 0, không có prior không nén được)
        from ...synth import GriddedField, synthetic_turbulence

        kmax = float(rng.uniform(size / 32, size / 6))
        ua, _ = synthetic_turbulence(shape, rms=1.0, k_max=kmax, slope=float(rng.uniform(-3, -1)),
                                     seed=int(rng.integers(2**31)))
        _, vb = synthetic_turbulence(shape, rms=1.0, k_max=kmax, slope=float(rng.uniform(-3, -1)),
                                     seed=int(rng.integers(2**31)))
        amp = float(rng.uniform(0.2, 3.0))
        f = GriddedField(ua * amp + rng.uniform(-8, 8), vb * amp + rng.uniform(-4, 4), periodic=True,
                         name="compressible")
    elif kind == "sine":
        f = SineField(amplitude=_loguni(rng, 0.1, 2.5), wavelength=_loguni(rng, 6, 4 * size),
                      phase=float(rng.uniform(0, 2 * np.pi)), component=str(rng.choice(["u", "v"])),
                      axis=str(rng.choice(["x", "y"])), offset=float(rng.uniform(-4, 4)))
    else:
        f = UniformField(float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)))
    pair = generate_pair(f, shape, **_imaging(rng, r))
    i1, i2 = pair["img1"], pair["img2"]
    valid = np.ones(shape, bool)
    if body is not None and body.any():
        level = float(rng.uniform(0.0, 0.3))
        noise = rng.normal(0, 0.02, shape)
        i1 = np.where(body, level + noise, i1)
        i2 = np.where(body, level + rng.normal(0, 0.02, shape), i2)
        valid = ~body
    # gradient chiếu sáng nhẹ (nhân)
    g = 1 + rng.uniform(-0.3, 0.3) * np.linspace(-1, 1, size)[None, :] + rng.uniform(-0.3, 0.3) * np.linspace(-1, 1, size)[:, None]
    i1 = np.clip(i1 * g, 0, None).astype(np.float16)
    i2 = np.clip(i2 * g * rng.uniform(0.9, 1.1), 0, None).astype(np.float16)
    flow = np.stack([pair["u"], pair["v"]]).astype(np.float32)
    return i1, i2, flow, valid


def _worker(args):
    return make_sample(*args)


def build_pool(out_dir, n: int, size: int = 128, mix: dict | None = None, seed: int = 0, workers: int = 4,
               ranges: ImagingRanges | None = None):
    """Sinh ``n`` mẫu vào ``out_dir`` (img1.npy, img2.npy, flow.npy, valid.npy, meta.json)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mix = mix or {"wake": 0.45, "compressible": 0.2, "sine": 0.12, "turbulence": 0.08, "vortex": 0.07,
                  "uniform": 0.08}
    kinds = list(mix)
    prob = np.array([mix[k] for k in kinds], float)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(kinds), size=n, p=prob / prob.sum())
    seeds = rng.integers(0, 2**31, size=n)
    I1 = np.lib.format.open_memmap(out / "img1.npy", "w+", np.float16, (n, size, size))
    I2 = np.lib.format.open_memmap(out / "img2.npy", "w+", np.float16, (n, size, size))
    FL = np.lib.format.open_memmap(out / "flow.npy", "w+", np.float32, (n, 2, size, size))
    VA = np.lib.format.open_memmap(out / "valid.npy", "w+", np.bool_, (n, size, size))
    args = [(int(seeds[i]), size, kinds[chosen[i]], ranges) for i in range(n)]
    with Pool(workers) as pool:
        for i, (a, b, f, v) in enumerate(pool.imap(_worker, args, chunksize=16)):
            I1[i], I2[i], FL[i], VA[i] = a, b, f, v
            if (i + 1) % 1000 == 0:
                print(f"  pool {i + 1}/{n}", flush=True)
    for a in (I1, I2, FL, VA):
        a.flush()
    meta = dict(n=n, size=size, mix=mix, seed=seed, kinds=[kinds[c] for c in chosen],
                ranges=asdict(ranges or ImagingRanges()))
    (out / "meta.json").write_text(json.dumps(meta))
    return out


class MixedDataset:
    """Trộn kho tổng hợp (memmap) với mẫu Cai et al.; trả về tensor numpy (1,H,W),(1,H,W),(2,H,W),(1,H,W)."""

    def __init__(self, pool_dirs: list, cai_samples: list | None = None, cai_prob: float = 0.3,
                 crop: int = 128, seed: int = 0):
        self.pools = []
        for d in pool_dirs:
            d = Path(d)
            self.pools.append(dict(i1=np.load(d / "img1.npy", mmap_mode="r"), i2=np.load(d / "img2.npy", mmap_mode="r"),
                                   fl=np.load(d / "flow.npy", mmap_mode="r"), va=np.load(d / "valid.npy", mmap_mode="r")))
        self.cai = cai_samples or []
        self.cai_prob = cai_prob if self.cai else 0.0
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self._cai_cache: dict = {}

    def _from_pool(self):
        p = self.pools[self.rng.integers(len(self.pools))]
        i = int(self.rng.integers(p["i1"].shape[0]))
        a, b = np.asarray(p["i1"][i], np.float32), np.asarray(p["i2"][i], np.float32)
        return a, b, np.asarray(p["fl"][i], np.float32), np.asarray(p["va"][i], np.float32), True

    def _from_cai(self):
        from ...data.cai import load_cai_sample

        t = self.cai[int(self.rng.integers(len(self.cai)))]
        a, b, u, v = load_cai_sample(t)
        return a, b, np.stack([u, v]).astype(np.float32), np.ones(a.shape, np.float32), False

    def sample(self):
        a, b, fl, va, midpoint = self._from_cai() if self.rng.random() < self.cai_prob else self._from_pool()
        H, W = a.shape
        c = self.crop
        if H > c or W > c:
            y0, x0 = int(self.rng.integers(0, H - c + 1)), int(self.rng.integers(0, W - c + 1))
            a, b = a[y0:y0 + c, x0:x0 + c], b[y0:y0 + c, x0:x0 + c]
            fl, va = fl[:, y0:y0 + c, x0:x0 + c], va[y0:y0 + c, x0:x0 + c]
        if self.rng.random() < 0.5:  # lật trái-phải: x → −x
            a, b, fl, va = a[:, ::-1], b[:, ::-1], fl[:, :, ::-1] * np.array([-1, 1], np.float32)[:, None, None], va[:, ::-1]
        if self.rng.random() < 0.5:  # lật trên-dưới: y → −y
            a, b, fl, va = a[::-1], b[::-1], fl[:, ::-1] * np.array([1, -1], np.float32)[:, None, None], va[::-1]
        if midpoint and self.rng.random() < 0.3:  # đảo thời gian (đúng chính xác vì trường gốc ở thời điểm giữa)
            a, b, fl = b, a, -fl
        # biến đổi cường độ: gamma + độ lợi
        g = float(self.rng.uniform(0.7, 1.4))
        a = np.clip(a, 0, None) ** g * float(self.rng.uniform(0.6, 1.5))
        b = np.clip(b, 0, None) ** g * float(self.rng.uniform(0.6, 1.5))
        return (np.ascontiguousarray(a)[None], np.ascontiguousarray(b)[None], np.ascontiguousarray(fl),
                np.ascontiguousarray(va)[None])

    def batch(self, n):
        items = [self.sample() for _ in range(n)]
        return [np.stack(x) for x in zip(*items)]
