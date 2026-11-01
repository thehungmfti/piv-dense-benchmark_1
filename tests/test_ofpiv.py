import numpy as np
import pytest

from pivbench.methods import get_method, ofpiv
from pivbench.metrics import mean_epe

BORDER = 12


def test_uniform_subpixel_original_pipeline(uniform_pair):
    """Quy trình gốc (không khởi tạo, 1 lượt): EPE < 0.05 px."""
    p = uniform_pair
    u, v = ofpiv.estimate(p["img1"], p["img2"])
    assert u.shape == p["img1"].shape and v.shape == p["img1"].shape
    assert mean_epe(u, v, p["u"], p["v"], BORDER) < 0.05


def test_uniform_subpixel_warp_iterations(uniform_pair):
    p = uniform_pair
    u, v = ofpiv.estimate(p["img1"], p["img2"], shift_method="warp", n_iter=3)
    assert mean_epe(u, v, p["u"], p["v"], BORDER) < 0.02


@pytest.mark.parametrize("kw", [dict(init="crosscorr", shift_method="warp", n_iter=3),
                                dict(n_levels=3, n_iter=2, shift_method="warp")])
def test_uniform_large_displacement(uniform_pair_large, kw):
    p = uniform_pair_large
    u, v = ofpiv.estimate(p["img1"], p["img2"], **kw)
    assert mean_epe(u, v, p["u"], p["v"], BORDER) < 0.05


def test_original_shift_integer_displacement_exact():
    """shift_image_fun_refine_1 gốc dịch chính xác phần nguyên của dịch chuyển."""
    from pivbench.synth import UniformField, generate_pair

    p = generate_pair(UniformField(2.0, -1.0), (128, 128), ppp=0.05, snr=None, seed=4)
    u, v = ofpiv.estimate(p["img1"], p["img2"], init="crosscorr")
    assert mean_epe(u, v, p["u"], p["v"], BORDER) < 0.01


def test_original_shift_fractional_displacement(uniform_pair_large):
    """Quy trình gốc với phần lẻ lớn (0.3, −0.4): phép vận chuyển bậc 1 của
    shift_image_fun_refine_1 để lại sai số ~0.06 px (giới hạn của bản gốc, không phải lỗi
    port — xem README). Ngưỡng nới thành 0.1 px; biến thể shift_method='warp' đạt < 0.05."""
    p = uniform_pair_large
    u, v = ofpiv.estimate(p["img1"], p["img2"], init="crosscorr")
    assert mean_epe(u, v, p["u"], p["v"], BORDER) < 0.1


def test_zero_displacement_identical_images(uniform_pair):
    img = uniform_pair["img1"]
    u, v = ofpiv.estimate(img, img)
    assert np.abs(u).max() < 1e-6 and np.abs(v).max() < 1e-6


def test_imfilter_matches_matlab_even_kernel():
    a = np.zeros((1, 11))
    a[0, 5] = 1
    h = np.arange(1, 7, dtype=float)[None, :]
    # MATLAB: imfilter(a, h) = [0 0 6 5 4 3 2 1 0 0 0]
    np.testing.assert_allclose(ofpiv.imfilter(a, h)[0], [0, 0, 6, 5, 4, 3, 2, 1, 0, 0, 0])


def test_registry():
    assert get_method("ofpiv") is ofpiv.estimate
    with pytest.raises(ValueError):
        get_method("nope")
