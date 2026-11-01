"""Tests P1: hàm MTF (đo & lý thuyết), mô hình λ_c, hàm truyền và hiệu chỉnh phổ."""

import numpy as np
import pytest

from pivbench.metrics.mtf import cutoff_wavelength, mtf_from_sine
from pivbench.resolution import (
    TransferFunction,
    correct_spectrum,
    fit_linear,
    fit_power_law,
    noise_floor_tail,
    particle_spacing,
    radial_auto_spectrum,
    radial_cross_spectrum,
    tophat_cutoff,
    tophat_mtf,
    valid_wavenumber,
)
from pivbench.metrics.spectra import radial_spectrum_2d
from pivbench.synth import TurbulenceField


# ---------------------------------------------------------------- MTF
def _sine(shape, A, lam, phase=0.0, comp="u"):
    ny, nx = shape
    X = np.mgrid[0:ny, 0:nx][1].astype(float)
    w = A * np.sin(2 * np.pi * X / lam + phase)
    return (w, np.zeros_like(w)) if comp == "u" else (np.zeros_like(w), w)


@pytest.mark.parametrize("lam", [6.0, 16.0, 64.0])
def test_mtf_identity_is_one(lam):
    u, v = _sine((64, 128), 0.7, lam)
    r = mtf_from_sine(u, v, 0.7, lam, border=8)
    assert r["mtf"] == pytest.approx(1.0, abs=1e-10)
    assert r["phase_error"] == pytest.approx(0.0, abs=1e-10)


def test_mtf_scaled_and_shifted():
    lam, A = 32.0, 2.0
    u, v = _sine((64, 256), 0.5 * A, lam, phase=0.3)  # biên độ giảm ½, lệch pha 0.3 rad
    r = mtf_from_sine(u, v, A, lam)
    assert r["mtf"] == pytest.approx(0.5, abs=1e-10)
    assert r["phase_error"] == pytest.approx(0.3, abs=1e-10)


def test_mtf_transverse_component():
    u, v = _sine((64, 128), 1.0, 16.0, comp="v")
    assert mtf_from_sine(u, v, 1.0, 16.0, component="v")["mtf"] == pytest.approx(1.0, abs=1e-10)
    assert mtf_from_sine(u, v, 1.0, 16.0, component="u")["mtf"] == pytest.approx(0.0, abs=1e-10)


def test_mtf_of_boxcar_filter_matches_sinc():
    """Trung bình trượt W px trên trường sin tuần hoàn ⇒ MTF = sinc(W/λ) (đúng tới sai số máy)."""
    from scipy.ndimage import uniform_filter1d

    W = 9
    for lam in (16.0, 32.0, 64.0):
        u, v = _sine((8, 256), 1.0, lam)
        uf = uniform_filter1d(u, W, axis=1, mode="wrap")
        m = mtf_from_sine(uf, v, 1.0, lam)["mtf"]
        assert m == pytest.approx(np.sin(np.pi * W / lam) / (W * np.sin(np.pi / lam)), abs=1e-6)
        # dạng liên tục sinc(W/λ) khớp tới ~1 % (rời rạc hoá)
        assert m == pytest.approx(float(tophat_mtf(lam, W)), rel=0.03)


def test_cutoff_wavelength_interpolation():
    lam = np.array([4, 8, 16, 32, 64])
    m = np.array([0.0, 0.2, 0.6, 0.95, 1.0])
    lc = cutoff_wavelength(lam, m, 0.5)
    assert 8 < lc < 16
    tt = (0.95 - 0.9) / (0.95 - 0.6)
    assert cutoff_wavelength(lam, m, 0.9) == pytest.approx(np.exp(np.log(32) + tt * (np.log(16) - np.log(32))))
    assert np.isnan(cutoff_wavelength(lam, 0.1 * m, 0.5))
    assert cutoff_wavelength(lam, np.ones(5), 0.9) == 4


def test_tophat_theory():
    assert tophat_mtf(32.0, 0.0) == pytest.approx(1.0)
    assert tophat_mtf(32.0, 32.0) == pytest.approx(0.0, abs=1e-12)
    lc = tophat_cutoff(16, 0.5)
    assert tophat_mtf(lc, 16) == pytest.approx(0.5, abs=1e-9)
    assert lc == pytest.approx(16 / 0.6034, rel=1e-3)


def test_power_law_fit_recovers_exponents():
    rng = np.random.default_rng(0)
    ppp = rng.uniform(0.005, 0.1, 60)
    snr = rng.uniform(5, 20, 60)
    y = 3.0 * ppp**-0.5 * snr**-0.2
    f = fit_power_law(y, dict(ppp=ppp, SNR=snr))
    assert f["a"] == pytest.approx(3.0, rel=1e-8)
    assert f["exponents"]["ppp"] == pytest.approx(-0.5, abs=1e-8)
    assert f["exponents"]["SNR"] == pytest.approx(-0.2, abs=1e-8)
    assert f["r2_log"] == pytest.approx(1.0)
    lp = particle_spacing(ppp)
    fl = fit_linear(2.0 * lp + 1.0, lp)
    assert fl["alpha"] == pytest.approx(2.0) and fl["beta"] == pytest.approx(1.0) and fl["r2"] == pytest.approx(1.0)


# ---------------------------------------------------------------- hiệu chỉnh
def _spec():
    k = np.linspace(1 / 256, 0.5, 128)
    return k, k ** (-5 / 3)


@pytest.mark.parametrize("mode", ["wiener", "inverse"])
def test_correction_identity_H_returns_original(mode):
    """H ≡ 1, N ≡ 0 ⇒ E_corr = E_meas (cả Wiener lẫn nghịch đảo, với mọi ε)."""
    k, E = _spec()
    for H in (TransferFunction.identity(), 1.0, np.ones_like(k), lambda kk: np.ones_like(kk)):
        out = correct_spectrum(k, E, H, 0.0, mode=mode, eps=0.3)
        np.testing.assert_allclose(out, E, rtol=1e-14)


def test_correction_inverts_known_filter_without_noise():
    k, Et = _spec()
    Hk = np.exp(-((k / 0.1) ** 2))
    Em = Hk**2 * Et
    Ec = correct_spectrum(k, Em, Hk, 0.0, mode="inverse", h_min=0.2)
    ok = Hk > 0.2
    np.testing.assert_allclose(Ec[ok], Et[ok], rtol=1e-12)
    assert np.all(np.isnan(Ec[~ok]))
    # Wiener: chính xác khi |H|→1, và không bao giờ vượt nghịch đảo
    Ew = correct_spectrum(k, Em, Hk, 0.0, mode="wiener", eps=1e-2, h_min=0.2)
    assert np.all(Ew[ok] <= Ec[ok] * (1 + 1e-12))
    assert Ew[0] == pytest.approx(Et[0], rel=1e-6)


def test_correction_removes_noise_floor():
    k, Et = _spec()
    Hk = np.exp(-((k / 0.15) ** 2))
    N = np.full_like(k, 1e-2)
    Em = Hk**2 * Et + N
    Ec = correct_spectrum(k, Em, Hk, N, mode="inverse", h_min=0.3)
    ok = Hk > 0.3
    np.testing.assert_allclose(Ec[ok], Et[ok], rtol=1e-10)
    # (E − N) âm được chặn về 0
    assert np.all(correct_spectrum(k, 0.5 * N, 1.0, N) == 0)


def test_noise_floor_tail_and_wiener_stability():
    k, Et = _spec()
    Em = 0.0 * Et + 3e-3
    assert np.allclose(noise_floor_tail(k, Em), 3e-3)
    Hk = np.full_like(k, 0.25)
    w = correct_spectrum(k, Et, Hk, 0.0, eps=1e-2, h_min=0.2)
    inv = correct_spectrum(k, Et, Hk, 0.0, mode="inverse", h_min=0.2)
    assert np.all(w < inv)  # Wiener giảm khuếch đại khi |H| nhỏ


def test_transfer_function_interpolation():
    tf = TransferFunction.from_mtf([4, 8, 16, 32, 64], [0.0, 0.3, 0.7, 0.95, 1.0])
    assert tf(1 / 16) == pytest.approx(0.7)
    assert tf(1 / 64) == pytest.approx(1.0)
    assert tf(1 / 512) == pytest.approx(1.0)        # k nhỏ hơn nút: giữ giá trị λ lớn nhất
    assert tf(0.4) == 0.0                           # k lớn hơn nút cuối (λ < 4): 0
    mid = tf(np.sqrt(1 / 16 * 1 / 32))              # nội suy tuyến tính theo log k
    assert mid == pytest.approx(0.825)


def test_cross_spectrum_identity_and_transfer():
    f = TurbulenceField((128, 128), rms=1.0, seed=3)
    u, v = f.on_grid((128, 128))
    k, E1 = radial_auto_spectrum(u, v, border=8)
    k2, E2 = radial_spectrum_2d(u, v, border=8)
    np.testing.assert_allclose(E1, E2, rtol=1e-12)
    _, S = radial_cross_spectrum(0.5 * u, 0.5 * v, u, v, border=8)
    tf = TransferFunction.from_cross_spectrum(k, S, E1)
    np.testing.assert_allclose(tf.H_nodes, 0.5, rtol=1e-10)


def test_valid_wavenumber():
    k, Et = _spec()
    assert valid_wavenumber(k, Et, Et) == pytest.approx(k[-1])
    E = Et.copy()
    E[k > 0.2] *= 0.5
    kv = valid_wavenumber(k, E, Et)
    assert kv < 0.2 and kv == pytest.approx(k[k <= 0.2][-1])
    assert valid_wavenumber(k, 0.5 * Et, Et) == 0.0
    E[10] = np.nan
    assert valid_wavenumber(k, E, Et) == pytest.approx(k[9])
