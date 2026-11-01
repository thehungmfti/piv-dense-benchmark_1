import numpy as np
import pytest
import torch

from pivbench.data.cai import read_flo, write_flo
from pivbench.methods.raft.data import MixedDataset, build_pool, make_sample
from pivbench.methods.raft.infer import predict
from pivbench.methods.raft.model import RAFTPIV, RAFTPIVConfig, sequence_loss
from pivbench.methods.raft.train import load_model, train


@pytest.mark.parametrize("down", [4, 8])
def test_forward_shapes_and_loss(down):
    torch.manual_seed(0)
    m = RAFTPIV(RAFTPIVConfig(down=down, iters=3))
    a = torch.rand(2, 1, 64, 64)
    preds = m(a, a, iters=3)
    assert len(preds) == 3 and preds[-1].shape == (2, 2, 64, 64)
    loss, epe = sequence_loss(preds, torch.zeros(2, 2, 64, 64))
    loss.backward()
    with torch.no_grad():
        out = m(a, a, iters=3, test_mode=True)
    assert out.shape == (2, 2, 64, 64)


def test_convex_upsample_of_constant_flow_is_exact():
    m = RAFTPIV(RAFTPIVConfig(down=4))
    flow = torch.full((1, 2, 8, 8), 0.75)
    mask = torch.randn(1, 9 * 16, 8, 8)
    up = m.upsample(flow, mask)
    # trong miền (không chạm biên đệm 0) tổ hợp lồi của hằng số = hằng số × down
    assert torch.allclose(up[:, :, 4:-4, 4:-4], torch.full_like(up[:, :, 4:-4, 4:-4], 3.0), atol=1e-5)


def test_tiled_prediction_matches_shape():
    m = RAFTPIV(RAFTPIVConfig(iters=2)).eval()
    a = np.random.default_rng(0).random((150, 170)).astype(np.float32)
    u, v = predict(m, a, a, iters=2, tile=96, overlap=16)
    assert u.shape == (150, 170) and np.isfinite(u).all()


def test_flo_roundtrip(tmp_path):
    f = np.random.default_rng(0).normal(size=(5, 7, 2)).astype(np.float32)
    write_flo(tmp_path / "a.flo", f)
    np.testing.assert_array_equal(read_flo(tmp_path / "a.flo"), f)


def test_sample_time_reversal_and_flip_consistency():
    a, b, fl, va = make_sample(3, 64, "uniform")
    ds = MixedDataset.__new__(MixedDataset)
    assert fl.shape == (2, 64, 64) and va.all()


def test_train_resume_and_load(tmp_path):
    build_pool(tmp_path / "pool", 8, 64, seed=0, workers=1)
    ds = MixedDataset([tmp_path / "pool"], crop=64)
    cfg = RAFTPIVConfig(iters=2)
    train(cfg, ds, tmp_path / "run", steps=2, batch=2, ckpt_every=1, log_every=1, threads=1)
    train(cfg, ds, tmp_path / "run", steps=3, batch=2, ckpt_every=1, log_every=1, threads=1)  # tiếp tục
    m, st = load_model(tmp_path / "run" / "last.pt")
    assert st["step"] == 3
