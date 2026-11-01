import numpy as np
import pytest

from pivbench.synth import (LambOseenField, SineField, generate_pair, load_dns_field,
                            render_particles, rk4_advect, synthetic_turbulence)


def test_turbulence_divergence_free_and_spectrum_slope():
    u, v = synthetic_turbulence((256, 256), rms=2.0, k_min=2, k_max=64, seed=0)
    fx = 2j * np.pi * np.fft.fftfreq(256)[None, :]
    fy = 2j * np.pi * np.fft.fftfreq(256)[:, None]
    div = np.real(np.fft.ifft2(fx * np.fft.fft2(u) + fy * np.fft.fft2(v)))
    assert np.abs(div).max() < 1e-10
    assert np.sqrt(np.mean(u**2 + v**2)) == pytest.approx(2.0)
    # phổ năng lượng vòng ∝ k^-5/3 trong dải quán tính
    P = np.abs(np.fft.fft2(u)) ** 2 + np.abs(np.fft.fft2(v)) ** 2
    K = np.hypot(*np.meshgrid(np.fft.fftfreq(256) * 256, np.fft.fftfreq(256) * 256))
    ks = np.arange(4, 60)
    E = np.array([P[(K >= k - 0.5) & (K < k + 0.5)].sum() for k in ks])
    slope = np.polyfit(np.log(ks), np.log(E), 1)[0]
    assert slope == pytest.approx(-5 / 3, abs=0.15)


def test_particle_mass_and_centroid():
    img = render_particles((32, 32), np.array([15.3]), np.array([16.7]), np.array([1.0]), 2.5)
    sigma = 2.5 / 4
    assert img.sum() == pytest.approx(2 * np.pi * sigma**2, rel=1e-6)
    Y, X = np.mgrid[0:32, 0:32]
    # tích phân trên pixel làm trọng tâm rời rạc lệch rất nhỏ (< 1e-3 px)
    assert (img * X).sum() / img.sum() == pytest.approx(15.3, abs=1e-3)
    assert (img * Y).sum() / img.sum() == pytest.approx(16.7, abs=1e-3)


def test_rk4_solid_body_rotation_is_accurate():
    class Rot:
        def __call__(self, x, y):
            return -0.1 * y, 0.1 * x

    x, y = rk4_advect(Rot(), np.array([10.0]), np.array([0.0]), 1.0)
    assert x[0] == pytest.approx(10 * np.cos(0.1), abs=1e-6)
    assert y[0] == pytest.approx(10 * np.sin(0.1), abs=1e-6)


def test_pair_density_noise_and_sheet():
    f = SineField(1.0, 32)
    clean = generate_pair(f, (128, 128), ppp=0.05, snr=None, seed=0)
    n_exp = 0.05 * 128 * 128
    assert clean["meta"]["n_particles"] > n_exp  # miền mở rộng có lề
    noisy = generate_pair(f, (128, 128), ppp=0.05, snr=5, seed=0)
    assert np.std(noisy["img1"] - clean["img1"]) > 0.05
    sheet = generate_pair(f, (128, 128), ppp=0.05, sheet_thickness=8, out_of_plane_w=4, seed=0)
    assert sheet["meta"]["n_lost_pairs"] > 0


def test_lamb_oseen_peak():
    f = LambOseenField(center=(0, 0), core_radius=10, u_max=3)
    r = np.linspace(0.1, 50, 5000)
    u, v = f(r, np.zeros_like(r))
    assert np.abs(v).max() == pytest.approx(3.0, rel=1e-4)


def test_dns_loader_npz(tmp_path):
    u = np.random.default_rng(0).normal(size=(40, 50))
    np.savez(tmp_path / "f.npz", u=u, v=-u)
    fld = load_dns_field(str(tmp_path / "f.npz"), scale=2.0)
    uu, vv = fld.on_grid((40, 50))
    np.testing.assert_allclose(uu, 2 * u)
    # nội suy bậc 3 đi qua các nút lưới
    a, _ = fld(np.array([7.0]), np.array([5.0]))
    assert a[0] == pytest.approx(2 * u[5, 7], abs=1e-9)
