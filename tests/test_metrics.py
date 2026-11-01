import numpy as np
import pytest

from pivbench.metrics import (cutoff_wavelength, error_stats, mtf_from_sine, radial_spectrum_2d,
                              spectrum_1d, strong_gradient_errors, vortex_peak_error)
from pivbench.synth import LambOseenField, SineField


@pytest.mark.parametrize("lam", [4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("component,axis", [("u", "x"), ("v", "x"), ("u", "y")])
def test_mtf_of_true_field_is_one(lam, component, axis):
    f = SineField(amplitude=0.7, wavelength=lam, component=component, axis=axis, phase=0.3)
    u, v = f.on_grid((256, 256))
    r = mtf_from_sine(u, v, 0.7, lam, component, axis, phase=0.3, border=16)
    assert r["mtf"] == pytest.approx(1.0, abs=1e-9)
    assert abs(r["phase_error"]) < 1e-9


def test_mtf_detects_attenuation_and_shift():
    f = SineField(amplitude=1.0, wavelength=32)
    u, v = f.on_grid((128, 128))
    u_est = 0.6 * np.sin(2 * np.pi * np.arange(128) / 32 + 0.2)[None, :].repeat(128, 0)
    r = mtf_from_sine(u_est, v, 1.0, 32)
    assert r["mtf"] == pytest.approx(0.6, abs=1e-9)
    assert r["phase_error"] == pytest.approx(0.2, abs=1e-9)


def test_cutoff_wavelength():
    lam = [4, 8, 16, 32, 64]
    m = [0.1, 0.4, 0.8, 0.95, 1.0]
    lc = cutoff_wavelength(lam, m, 0.9)
    assert 16 < lc < 32
    assert cutoff_wavelength(lam, [1] * 5, 0.9) == 4
    assert np.isnan(cutoff_wavelength(lam, [0.1] * 5, 0.9))


@pytest.mark.parametrize("axis", [0, 1])
def test_white_noise_1d_spectrum_is_flat(rng, axis):
    n = rng.normal(0.0, 1.0, (512, 512))
    k, P = spectrum_1d(n, axis=axis, nperseg=128)
    inner = P[1:-1]
    # mật độ một phía của nhiễu trắng phương sai 1, fs = 1: P = 2
    assert inner.mean() == pytest.approx(2.0, rel=0.03)
    assert inner.std() / inner.mean() < 0.05
    # không có xu hướng theo k: hệ số góc log-log ≈ 0
    slope = np.polyfit(np.log(k[1:-1]), np.log(inner), 1)[0]
    assert abs(slope) < 0.03


def test_white_noise_2d_spectrum_is_flat(rng):
    n = rng.normal(0.0, 1.0, (512, 512))
    k, E = radial_spectrum_2d(n)
    # mật độ 2D phẳng ⇔ E(|k|) / (2π|k|) không đổi
    d = (E / (2 * np.pi * k))[4:]
    assert d.std() / d.mean() < 0.1
    assert d.mean() == pytest.approx(1.0, rel=0.05)


def test_error_stats_and_border():
    u = np.zeros((20, 20))
    ut = np.zeros((20, 20))
    ut[0, :] = 10.0  # chỉ biên sai
    s = error_stats(u, u, ut, ut * 0, border=1)
    assert s["epe_mean"] == 0 and s["n"] == 18 * 18
    s = error_stats(u + 0.3, u - 0.4, ut * 0, ut * 0)
    assert s["epe_mean"] == pytest.approx(0.5)
    assert s["rmse_u"] == pytest.approx(0.3) and s["bias_v"] == pytest.approx(-0.4)


def test_strong_gradient_and_vortex_peak():
    f = LambOseenField(core_radius=12, u_max=3.0)
    ut, vt = f.on_grid((96, 96))
    r = vortex_peak_error(ut * 0.8, vt * 0.8, ut, vt, border=4)
    assert r["omega_peak_rel_error"] == pytest.approx(-0.2, abs=1e-9)
    assert r["peak_offset"] == 0
    assert r["omega_peak_true"] == pytest.approx(f.vorticity_peak(), rel=0.02)
    g = strong_gradient_errors(ut, vt, ut, vt)
    assert g["epe_in"] == 0 and g["epe_out"] == 0
