import numpy as np
import pytest

from pivbench.methods import raftpiv


def test_unavailable_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("RAFTPIV_CKPT", raising=False)
    img = np.zeros((32, 32), np.float32)
    with pytest.raises(raftpiv.RaftPIVUnavailable, match="RAFT-PIV|torch"):
        raftpiv.estimate(img, img, repo=tmp_path / "missing")
