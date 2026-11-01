"""Wrapper cho RAFT-PIV (Lagemann, Lagemann, Mukherjee & Schröder, Nat. Mach. Intell. 3, 2021).

Hai backend:

* ``backend="local"`` (mặc định khi có checkpoint cục bộ): bản TÁI HIỆN trong
  ``pivbench.methods.raft`` (kiến trúc viết lại, huấn luyện lại cho wake đáy tù), checkpoint
  mặc định ``checkpoints/raftpiv_wake_v2/best.pt`` hoặc biến môi trường ``RAFTPIV_LOCAL_CKPT``.
* ``backend="official"``: mã nguồn + trọng số chính thức (KHÔNG có sẵn — không tìm được bản
  phát hành truy cập được khi xây dựng, xem README). Chỉ chạy khi người dùng tự cung cấp:

  * ``repo``: đường dẫn tới mã nguồn RAFT-PIV (biến môi trường ``RAFTPIV_REPO``,
    mặc định ``third_party/RAFT-PIV``), thêm vào ``sys.path``;
  * ``checkpoint``: file trọng số PyTorch (``RAFTPIV_CKPT``);
  * ``model_module`` / ``model_class``: nơi định nghĩa mạng (mặc định ``core.raft`` / ``RAFT``
    theo cấu trúc repo RAFT gốc của princeton-vl mà RAFT-PIV kế thừa) — chỉnh cho đúng repo
    thực tế;
  * ``model_args``: dict tham số khởi tạo (ví dụ ``{"small": False, "upsample": "convex"}``).

Nếu thiếu bất kỳ thành phần nào, ``estimate`` ném ``RaftPIVUnavailable`` với thông báo rõ ràng.
Trên CPU, ảnh lớn hơn ``max_pixels_cpu`` (mặc định 512×512) bị từ chối để tránh treo.
"""

from __future__ import annotations

import importlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO = _REPO_ROOT / "third_party" / "RAFT-PIV"
DEFAULT_LOCAL_CKPT = _REPO_ROOT / "checkpoints" / "raftpiv_wake_v2" / "best.pt"


class RaftPIVUnavailable(RuntimeError):
    """RAFT-PIV không chạy được (thiếu torch / mã nguồn / trọng số)."""


def _resolve(repo, checkpoint):
    repo = Path(repo or os.environ.get("RAFTPIV_REPO", DEFAULT_REPO))
    ckpt = checkpoint or os.environ.get("RAFTPIV_CKPT")
    if ckpt is None:
        cands = sorted((repo / "checkpoints").glob("*.pt*")) if (repo / "checkpoints").is_dir() else []
        ckpt = cands[0] if cands else None
    return repo, (Path(ckpt) if ckpt else None)


def check_available(repo=None, checkpoint=None) -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "Chưa cài torch (pip install torch)."
    repo, ckpt = _resolve(repo, checkpoint)
    if not repo.is_dir() or not any(repo.iterdir()):
        return False, (f"Không thấy mã nguồn RAFT-PIV tại {repo}. Đặt RAFTPIV_REPO hoặc clone repo "
                       "chính thức vào third_party/RAFT-PIV (xem README, mục RAFT-PIV).")
    if ckpt is None or not ckpt.is_file():
        return False, (f"Không thấy trọng số RAFT-PIV (checkpoint={ckpt}). Đặt RAFTPIV_CKPT tới file .pth "
                       "đã huấn luyện của tác giả.")
    return True, "ok"


_MODEL_CACHE: dict = {}


def _load_model(repo, ckpt, model_module, model_class, model_args, device):
    import torch

    key = (str(repo), str(ckpt), model_module, model_class, repr(model_args), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        core = repo / "core"
        if core.is_dir():
            sys.path.insert(0, str(core))
    try:
        mod = importlib.import_module(model_module)
        cls = getattr(mod, model_class)
    except (ImportError, AttributeError) as e:
        raise RaftPIVUnavailable(f"Không import được {model_module}.{model_class} từ {repo}: {e}") from e
    args = Namespace(**{"small": False, "mixed_precision": False, "alternate_corr": False,
                        "dropout": 0.0, **(model_args or {})})
    try:
        model = cls(args)
    except TypeError:
        model = cls(**(model_args or {}))
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RaftPIVUnavailable(f"Checkpoint không khớp kiến trúc: thiếu {len(missing)} tham số "
                                 f"(ví dụ {missing[:3]}); kiểm tra model_module/model_class/model_args.")
    model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _local_ckpt(checkpoint=None):
    c = Path(checkpoint or os.environ.get("RAFTPIV_LOCAL_CKPT") or DEFAULT_LOCAL_CKPT)
    if not c.is_absolute() and not c.exists() and (_REPO_ROOT / c).exists():
        c = _REPO_ROOT / c  # đường dẫn tương đối tính từ gốc repo
    return c


_LOCAL_CACHE: dict = {}


def estimate_local(img1, img2, checkpoint=None, iters: int = 16, device: str | None = None,
                   tile: int | None = 512, overlap: int = 64, **_ignored):
    """RAFT-PIV tái hiện (``pivbench.methods.raft``)."""
    try:
        import torch
    except ImportError as e:
        raise RaftPIVUnavailable("Chưa cài torch (pip install torch).") from e
    from .raft.infer import predict
    from .raft.train import load_model

    ck = _local_ckpt(checkpoint)
    if not ck.is_file():
        raise RaftPIVUnavailable(f"Không thấy checkpoint RAFT-PIV tái hiện tại {ck}. Huấn luyện bằng "
                                 "scripts/train_raftpiv.py hoặc đặt RAFTPIV_LOCAL_CKPT.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(ck), ck.stat().st_mtime, device)
    if key not in _LOCAL_CACHE:
        _LOCAL_CACHE.clear()
        _LOCAL_CACHE[key] = load_model(ck, device)[0]
    return predict(_LOCAL_CACHE[key], img1, img2, iters=iters, device=device, tile=tile, overlap=overlap)


def estimate(img1: np.ndarray, img2: np.ndarray, backend: str = "auto", **params):
    """``backend`` ∈ {'auto', 'local', 'official'}; 'auto' = local nếu có checkpoint cục bộ, ngược lại official."""
    if backend == "local" or (backend == "auto" and (_local_ckpt(params.get("checkpoint")).is_file()
                                                    and not params.get("repo"))):
        return estimate_local(img1, img2, **params)
    return estimate_official(img1, img2, **params)


def estimate_official(img1: np.ndarray, img2: np.ndarray, repo=None, checkpoint=None,
             model_module: str = "core.raft", model_class: str = "RAFT", model_args: dict | None = None,
             iters: int = 12, device: str | None = None, intensity_scale: float = 255.0,
             pad_multiple: int = 8, max_pixels_cpu: int = 512 * 512, **_ignored):
    """Ước lượng (u, v) cỡ ảnh bằng RAFT-PIV. Ném ``RaftPIVUnavailable`` nếu không chạy được."""
    ok, msg = check_available(repo, checkpoint)
    if not ok:
        raise RaftPIVUnavailable(msg)
    import torch
    import torch.nn.functional as F

    repo, ckpt = _resolve(repo, checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h, w = img1.shape
    if device == "cpu" and h * w > max_pixels_cpu:
        raise RaftPIVUnavailable(f"Ảnh {h}x{w} quá lớn để chạy RAFT-PIV trên CPU (giới hạn "
                                 f"{max_pixels_cpu} px; tăng max_pixels_cpu hoặc dùng GPU).")
    model = _load_model(repo, ckpt, model_module, model_class, model_args, device)

    def prep(a):
        t = torch.from_numpy(np.asarray(a, np.float32) * intensity_scale)[None, None]
        return t.to(device)

    t1, t2 = prep(img1), prep(img2)
    ph = (-h) % pad_multiple
    pw = (-w) % pad_multiple
    t1 = F.pad(t1, (0, pw, 0, ph), mode="replicate")
    t2 = F.pad(t2, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        try:
            out = model(t1, t2, iters=iters, test_mode=True)
        except TypeError:
            out = model(t1, t2)
    flow = out[-1] if isinstance(out, (list, tuple)) else out
    flow = flow[0, :, :h, :w].cpu().numpy()
    return flow[0].astype(np.float64), flow[1].astype(np.float64)
