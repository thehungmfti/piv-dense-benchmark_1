from pivbench.methods import crosscorr
from pivbench.metrics import mean_epe


def test_uniform_large(uniform_pair_large):
    p = uniform_pair_large
    u, v = crosscorr.estimate(p["img1"], p["img2"], windows=(32, 16), overlap=0.5)
    assert u.shape == p["img1"].shape
    assert mean_epe(u, v, p["u"], p["v"], 12) < 0.05
