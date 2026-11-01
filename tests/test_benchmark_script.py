import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_run_benchmark_smoke(tmp_path):
    cfg = dict(name="smoke", shape=[64, 64], border=8, seeds=[0],
               synth=dict(ppp=0.05, snr=[None]),
               methods=[dict(name="cc", method="crosscorr", params=dict(windows=[32, 16])),
                        dict(name="of", method="ofpiv", params=dict(hs_maxnum=50, ls_maxnum=10)),
                        dict(name="raft_official", method="raftpiv", params=dict(backend="official",
                                                                                 repo="/nonexistent")),
                        dict(name="raft_local", method="raftpiv", params=dict(backend="local", iters=2))],
               fields=[dict(name="lo", kind="lamb_oseen", params=dict(core_radius=8, u_max=2)),
                       dict(name="sine", kind="sine", params=dict(amplitude=0.5), wavelengths=[16, 32])])
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_benchmark.py"), str(p), "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "smoke"
    for f in ["summary.csv", "mtf.csv", "cutoffs.csv", "unavailable.txt", "config.yaml"]:
        assert (out / f).is_file(), f
    assert list((out / "figures").glob("spectra_*.png")) and list((out / "figures").glob("mtf_*.png"))
    assert list((out / "fields").glob("*.npz"))
    rows = (out / "summary.csv").read_text()
    if (ROOT / "checkpoints" / "raftpiv_wake_v2" / "best.pt").is_file():
        assert "raft_local" in rows
