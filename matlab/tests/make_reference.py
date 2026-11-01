"""Sinh dữ liệu tham chiếu (Python) để kiểm tra các hàm MATLAB: python matlab/tests/make_reference.py out.mat"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pivbench.metrics import error_stats, mtf_from_sine, spectra_report, strong_gradient_errors  # noqa: E402
from pivbench.synth import LambOseenField, SineField  # noqa: E402

rng = np.random.default_rng(0)
ut, vt = LambOseenField(core_radius=14, u_max=3).on_grid((96, 128))
u = ut + 0.1 * rng.normal(size=ut.shape)
v = vt + 0.1 * rng.normal(size=ut.shape)
st = error_stats(u, v, ut, vt, 8)
sg = strong_gradient_errors(u, v, ut, vt, "vorticity", 10, 8)
sp = spectra_report(u, v, ut, vt, 8)
su, sv = SineField(0.7, 24, phase=0.3).on_grid((64, 96))
su_est = 0.8 * np.roll(su, 1, axis=1) + 0.01 * rng.normal(size=su.shape)
mt = mtf_from_sine(su_est, sv, 0.7, 24, "u", "x", 0.3, 4)
out = dict(u=u, v=v, ut=ut, vt=vt, su_est=su_est, sv=sv, mtf=mt["mtf"], phase_error=mt["phase_error"],
           epe_in=sg["epe_in"], epe_out=sg["epe_out"],
           **{k: st[k] for k in ["epe_mean", "epe_median", "epe_p95", "rmse_u", "rmse_v", "bias_u", "bias_v"]},
           **{f"sp_{k}": np.asarray(val) for k, val in sp.items()})
savemat(sys.argv[1], out)
print("ok")
