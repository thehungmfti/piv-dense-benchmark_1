"""Bộ dữ liệu PIV của Cai et al. (Exp. Fluids 60:73, 2019) — https://github.com/shengzesnail/PIV_dataset

Mỗi mẫu: ``<cls>_<Re>_<id>_img1.tif``, ``..._img2.tif`` (256×256, uint8) và ``..._flow.flo``
(Middlebury .flo, kênh 0 = u theo x, kênh 1 = v theo y, px). Giấy phép: chỉ dùng cho nghiên cứu,
không phân phối lại — repo này KHÔNG chứa dữ liệu; tải theo ``scripts/download_data.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .loaders import read_image

FLO_MAGIC = 202021.25


def read_flo(path) -> np.ndarray:
    """Đọc file .flo → mảng (H, W, 2) float32."""
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, 1)
        if magic.size == 0 or magic[0] != FLO_MAGIC:
            raise ValueError(f"{path}: không phải file .flo")
        w, h = np.fromfile(f, np.int32, 2)
        data = np.fromfile(f, np.float32, 2 * int(w) * int(h))
    return data.reshape(int(h), int(w), 2)


def write_flo(path, flow: np.ndarray):
    h, w, _ = flow.shape
    with open(path, "wb") as f:
        np.array([FLO_MAGIC], np.float32).tofile(f)
        np.array([w, h], np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def list_cai_samples(root, classes=("cylinder", "backstep"), split: str | None = None,
                     list_dir=None) -> list[tuple[Path, Path, Path]]:
    """Liệt kê (img1, img2, flow). ``split`` ∈ {None, 'train', 'test'} dùng FlowData_<split>.list
    của tác giả (tìm trong ``list_dir`` hoặc ``root``); không có list → dùng toàn bộ."""
    root = Path(root)
    out = []
    for cls in classes:
        d = root / cls
        if not d.is_dir():
            continue
        for fl in sorted(d.glob("*_flow.flo")):
            stem = fl.name[: -len("_flow.flo")]
            out.append((d / f"{stem}_img1.tif", d / f"{stem}_img2.tif", fl))
    if split:
        lst = None
        for base in [list_dir, root, root.parent]:
            if base is not None and (Path(base) / f"FlowData_{split}.list").is_file():
                lst = Path(base) / f"FlowData_{split}.list"
                break
        if lst is not None:
            names = {Path(line.split()[2]).name for line in lst.read_text().splitlines() if line.strip()}
            out = [t for t in out if t[2].name in names]
    return out


def load_cai_sample(t):
    i1, i2, fl = t
    flow = read_flo(fl)
    return read_image(i1), read_image(i2), flow[..., 0].astype(np.float64), flow[..., 1].astype(np.float64)
