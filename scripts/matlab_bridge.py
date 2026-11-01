#!/usr/bin/env python
"""Cầu nối MATLAB ↔ Python cho pivbench (gọi từ matlab/*.m qua system()).

    python scripts/matlab_bridge.py job.mat result.mat

``job.mat`` (MATLAB ghi bằng save(..., '-v7')) chứa:
  * action      : 'estimate' | 'synth' | 'info'
  * params_json : chuỗi JSON tham số
  * estimate    : img1, img2 (double, H×W), method ('raftpiv'|'raftpiv_ls'|'ofpiv'|'ofpiv_warp'|'crosscorr')
  * synth       : field_json (JSON {kind, params}), shape [H W]; tham số ảnh trong params_json

``result.mat`` chứa u, v (H×W, px/khoảng thời gian; u theo cột/x, v theo hàng/y hướng xuống),
runtime_s, ok (1/0), message; với 'synth' thêm img1, img2, u_true, v_true, valid.
Quy ước ma trận MATLAB (hàng, cột) trùng với numpy nên không cần chuyển vị.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PRESETS = {
    "raftpiv": ("raftpiv", dict(backend="local", iters=16)),
    "raftpiv_v1": ("raftpiv", dict(backend="local", iters=16, checkpoint="checkpoints/raftpiv_wake/best.pt")),
    "raftpiv_ls": ("ofpiv", dict(lambda_HS=20, lambda_LS=2000, init="raftpiv", shift_method="warp", n_iter=3,
                                 raft_params=dict(iters=16))),
    "ofpiv": ("ofpiv", dict(lambda_HS=20, lambda_LS=2000, init="crosscorr")),
    "ofpiv_warp": ("ofpiv", dict(lambda_HS=20, lambda_LS=2000, init="crosscorr", shift_method="warp", n_iter=3)),
    "crosscorr": ("crosscorr", dict(windows=[64, 32, 16], overlap=0.75, iters_per_window=[1, 1, 2])),
}


def _str(x) -> str:
    if isinstance(x, str):
        return x
    a = np.asarray(x)
    if a.dtype.kind in "US":
        return "".join(a.ravel().tolist())
    return str(a.squeeze())


def _json(job, key):
    if key not in job:
        return {}
    s = _str(job[key]).strip()
    return json.loads(s) if s else {}


def run(job: dict) -> dict:
    action = _str(job.get("action", "estimate"))
    params = _json(job, "params_json")
    if action == "info":
        import torch

        from pivbench.methods.raftpiv import _local_ckpt

        return dict(ok=1, message="ok", python=sys.version.split()[0], torch=torch.__version__,
                    cuda=int(torch.cuda.is_available()), checkpoint=str(_local_ckpt()),
                    checkpoint_exists=int(_local_ckpt().is_file()), methods=list(PRESETS))
    if action == "estimate":
        from pivbench.methods import get_method

        name = _str(job["method"])
        if name not in PRESETS:
            raise ValueError(f"Phương pháp không biết: {name}; có: {list(PRESETS)}")
        base, defaults = PRESETS[name]
        kw = {**defaults, **params}
        if name == "raftpiv_ls" and "raft_params" in params:
            kw["raft_params"] = {**defaults["raft_params"], **params["raft_params"]}
        i1 = np.asarray(job["img1"], np.float64)
        i2 = np.asarray(job["img2"], np.float64)
        # chuẩn hoá về [0, 1] với cùng hệ số (các phương pháp đều bất biến với thang cường độ chung)
        s = max(i1.max(), i2.max(), 1e-12)
        t = time.time()
        u, v = get_method(base)((i1 / s).astype(np.float32), (i2 / s).astype(np.float32), **kw)
        return dict(ok=1, message="ok", u=np.asarray(u, float), v=np.asarray(v, float), runtime_s=time.time() - t,
                    method=name, params_used=json.dumps(kw, default=str))
    if action == "synth":
        from pivbench.synth import generate_pair, make_field

        fld = _json(job, "field_json")
        shape = tuple(int(x) for x in np.asarray(job.get("shape", [256, 256])).ravel())
        field = make_field(fld.get("kind", "wake"), shape, **fld.get("params", {}))
        pair = generate_pair(field, shape, **params)
        valid = ~getattr(field, "body_mask", np.zeros(shape, bool))
        i1, i2 = pair["img1"].astype(float), pair["img2"].astype(float)
        if not valid.all():  # thân mô hình: nền tối như khi huấn luyện
            rng = np.random.default_rng(int(params.get("seed", 0)) + 99)
            i1 = np.where(valid, i1, 0.1 + rng.normal(0, 0.02, shape))
            i2 = np.where(valid, i2, 0.1 + rng.normal(0, 0.02, shape))
        return dict(ok=1, message="ok", img1=i1, img2=i2, u_true=pair["u"], v_true=pair["v"],
                    valid=valid.astype(float), meta_json=json.dumps(pair["meta"], default=str))
    raise ValueError(f"action không biết: {action}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    job_path, out_path = argv
    try:
        job = loadmat(job_path, squeeze_me=True, chars_as_strings=True)
        res = run(job)
    except Exception as e:  # noqa: BLE001 — báo lỗi về MATLAB thay vì crash
        res = dict(ok=0, message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    savemat(out_path, res, do_compression=True)
    print(res.get("message", ""))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
