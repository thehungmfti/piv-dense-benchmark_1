import numpy as np
import pytest

from pivbench.synth import UniformField, generate_pair


@pytest.fixture(scope="session")
def uniform_pair():
    """Dịch chuyển đều dưới pixel, ảnh 128×128 không nhiễu."""
    return generate_pair(UniformField(0.5, 0.25), (128, 128), ppp=0.05, d_p=2.5, snr=None, seed=3)


@pytest.fixture(scope="session")
def uniform_pair_large():
    """Dịch chuyển đều lớn (> 2 px) — cần khởi tạo tương quan chéo."""
    return generate_pair(UniformField(2.3, -1.4), (128, 128), ppp=0.05, d_p=2.5, snr=None, seed=4)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
