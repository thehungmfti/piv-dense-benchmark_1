import numpy as np
import pytest

from pivbench.synth import generate_pair, make_field
from pivbench.synth.wake import AxisymmetricWakeField, WakeParams, random_wake_params


def test_wake_discrete_divergence_free_and_recirculation():
    f = AxisymmetricWakeField((192, 256), WakeParams(D=100, U_inf=5, x_base=20, U_rev=0.3, turb_rms=0.0,
                                                     kh_strength=0.0, street_strength=0.0, axisymmetric=False))
    # div rời rạc (sai phân trung tâm) của lưới nội bộ = 0 (ngoài mép lưới và sát thân)
    div = np.gradient(f.u, axis=1) + np.gradient(f.v, axis=0)
    from scipy import ndimage

    body = f.body_mask_grid

    near_body = ndimage.binary_dilation(body, iterations=3)
    near_body[:2], near_body[-2:], near_body[:, :2], near_body[:, -2:] = True, True, True, True  # sai phân 1 phía ở mép
    assert np.abs(div[~near_body]).max() < 1e-9
    u, v = f.on_grid((192, 256))
    # dòng ngược trên trục trong vùng tuần hoàn, dòng tự do ngoài wake
    axis = int(round((192 - 1) / 2))
    assert u[axis, 20 + 50] < -0.5  # ≈ −U_rev·U∞ = −1.5
    assert u[2, 200] == pytest.approx(5.0, rel=0.05)
    assert np.abs(u[f.body_mask]).max() < 1e-9 and f.body_mask[axis, 5]


def test_random_wake_and_pair():
    rng = np.random.default_rng(0)
    p = random_wake_params(rng, (128, 128))
    f = make_field("wake", (128, 128), **p.__dict__)
    pair = generate_pair(f, (128, 128), ppp=0.05)
    assert pair["img1"].shape == (128, 128) and np.isfinite(pair["u"]).all()


def test_wake_axisymmetric_continuity():
    """Chế độ trục đối xứng: ∂(r u)/∂x + ∂(r v)/∂r ≈ 0 còn div₂D ≠ 0."""
    from scipy import ndimage

    f = AxisymmetricWakeField((192, 256), WakeParams(D=100, U_inf=5, x_base=20, U_rev=0.3, turb_rms=0.0,
                                                     kh_strength=0.0, street_strength=0.0, y_axis=95.5))
    m = 24
    r = np.abs(np.arange(f.u.shape[0])[:, None] - m - 95.5)
    res = np.gradient(r * f.u, axis=1) + np.gradient(r * f.v, axis=0)
    ok = (r > 3) & ~ndimage.binary_dilation(f.body_mask_grid, iterations=4)
    ok[:3], ok[-3:], ok[:, :3], ok[:, -3:] = False, False, False, False
    scale = np.abs(np.gradient(r * f.u, axis=1)[ok]).max()
    assert np.abs(res[ok]).max() < 0.15 * scale
    assert np.percentile(np.abs(res[ok]), 95) < 0.02 * scale
    div2d = np.gradient(f.u, axis=1) + np.gradient(f.v, axis=0)
    assert np.abs(div2d[ok]).max() > 0.01
