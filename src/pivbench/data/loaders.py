"""Đọc cặp ảnh PIV (PIV Challenge, ảnh thực nghiệm) và trường gốc nếu có.

Không tự tải dữ liệu — xem ``scripts/download_data.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".pgm")


def read_image(path: str | Path, as_float: bool = True) -> np.ndarray:
    """Đọc ảnh xám (tif/bmp/png...). Ảnh màu được đổi sang xám bằng trung bình kênh.

    ``as_float=True`` trả về float32 chuẩn hoá theo độ sâu bit của file
    (uint8 → /255, uint16 → /65535), giữ nguyên tỉ lệ cường độ giữa 2 ảnh.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        img = tifffile.imread(path)
    else:
        import imageio.v3 as iio

        img = iio.imread(path)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        dtype = img.dtype
        img = img[..., :3].mean(axis=-1).astype(dtype)
    if img.ndim != 2:
        raise ValueError(f"{path}: ảnh không phải 2D (shape={img.shape}); ảnh nhiều trang cần tách trước")
    if not as_float:
        return img
    if np.issubdtype(img.dtype, np.integer):
        return (img.astype(np.float32) / np.iinfo(img.dtype).max).astype(np.float32)
    return img.astype(np.float32)


# Các quy ước đặt tên cặp ảnh thường gặp: A001a/A001b, img_1/img_2, xxx_a/xxx_b, xxx1/xxx2
_PAIR_PATTERNS = [
    (re.compile(r"^(?P<stem>.*?)(?P<sep>[_\-.]?)(?P<tag>[aA])$"), {"a": "b", "A": "B"}),
    (re.compile(r"^(?P<stem>.*?)(?P<sep>[_\-.]?)(?P<tag>1)$"), {"1": "2"}),
]


def find_pairs(directory: str | Path, first_glob: str | None = None) -> list[tuple[Path, Path]]:
    """Tìm các cặp ảnh trong thư mục theo quy ước tên (a/b, 1/2).

    Nếu cho ``first_glob`` (ví dụ ``"*_a.tif"``) chỉ xét các file ảnh 1 khớp mẫu.
    Nếu không khớp quy ước nào, ghép các ảnh liên tiếp đã sắp xếp (0-1, 2-3, ...).
    """
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    candidates = sorted(directory.glob(first_glob)) if first_glob else files
    by_name = {p.name: p for p in files}
    pairs = []
    for p in candidates:
        for rx, mapping in _PAIR_PATTERNS:
            m = rx.match(p.stem)
            if m and m.group("tag") in mapping:
                other = m.group("stem") + m.group("sep") + mapping[m.group("tag")] + p.suffix
                if other in by_name:
                    pairs.append((p, by_name[other]))
                    break
        else:
            # Quy ước khác (ví dụ I1_xxx / I2_xxx): thử thay đúng một ký tự 1→2 hoặc a→b
            stem = p.stem
            for i in range(len(stem) - 1, -1, -1):
                swap = {"1": "2", "a": "b", "A": "B"}.get(stem[i])
                other = (stem[:i] + swap + stem[i + 1:] + p.suffix) if swap else None
                if other in by_name:
                    pairs.append((p, by_name[other]))
                    break
    if not pairs and not first_glob:
        pairs = [(files[i], files[i + 1]) for i in range(0, len(files) - 1, 2)]
    return pairs


def load_pair(path1: str | Path, path2: str | Path, roi: tuple[int, int, int, int] | None = None):
    """Đọc 1 cặp ảnh; ``roi`` = (y0, y1, x0, x1)."""
    i1, i2 = read_image(path1), read_image(path2)
    if i1.shape != i2.shape:
        raise ValueError(f"Kích thước 2 ảnh khác nhau: {i1.shape} vs {i2.shape}")
    if roi is not None:
        y0, y1, x0, x1 = roi
        i1, i2 = i1[y0:y1, x0:x1], i2[y0:y1, x0:x1]
    return i1, i2


def _resample_to_pixels(x, y, u, v, shape):
    """Đưa trường cho trên lưới vector (x, y tính bằng px) về lưới pixel."""
    from scipy.interpolate import griddata

    ny, nx = shape
    gy, gx = np.mgrid[0:ny, 0:nx]
    pts = np.column_stack([np.ravel(x), np.ravel(y)])
    ui = griddata(pts, np.ravel(u), (gx, gy), method="cubic")
    vi = griddata(pts, np.ravel(v), (gx, gy), method="cubic")
    return ui, vi


def load_ground_truth(path: str | Path, shape: tuple[int, int] | None = None,
                      u_key: str = "u", v_key: str = "v", x_key: str | None = None,
                      y_key: str | None = None, scale: float = 1.0, flip_v: bool = False):
    """Đọc trường gốc (px/khoảng thời gian).

    Hỗ trợ:
      * ``.npz`` / ``.h5`` / ``.mat`` (MATLAB ≤ v7.2 qua scipy; v7.3 là HDF5) với khoá ``u_key``, ``v_key``.
        Nếu có ``x_key``/``y_key`` (toạ độ px) và ``shape``, trường được nội suy về lưới pixel.
      * ``.dat`` / ``.txt`` / ``.csv``: bảng cột ``x y u v`` (bỏ qua dòng tiêu đề Tecplot
        không phải số) — nội suy về lưới pixel ``shape``.

    ``flip_v=True`` đổi dấu v khi dữ liệu dùng trục y hướng lên (nhiều file PIV Challenge).
    """
    path = Path(path)
    suf = path.suffix.lower()
    x = y = None
    if suf == ".npz":
        with np.load(path) as d:
            u, v = d[u_key], d[v_key]
            if x_key and y_key:
                x, y = d[x_key], d[y_key]
    elif suf in (".h5", ".hdf5", ".mat"):
        d = None
        if suf == ".mat":
            from scipy.io import loadmat

            try:
                d = loadmat(path)
            except NotImplementedError:  # MATLAB v7.3 = HDF5
                d = None
        if d is None:
            import h5py

            with h5py.File(path, "r") as f:
                d = {k: np.asarray(f[k]) for k in [u_key, v_key, x_key, y_key] if k and k in f}
            if suf == ".mat":  # MATLAB v7.3 lưu mảng chuyển vị
                d = {k: a.T for k, a in d.items()}
        u, v = np.asarray(d[u_key], float), np.asarray(d[v_key], float)
        if x_key and y_key:
            x, y = np.asarray(d[x_key], float), np.asarray(d[y_key], float)
    elif suf in (".dat", ".txt", ".csv"):
        rows = []
        with open(path) as fh:
            for line in fh:
                parts = re.split(r"[,\s;]+", line.strip())
                try:
                    vals = [float(s) for s in parts if s]
                except ValueError:
                    continue
                if len(vals) >= 4:
                    rows.append(vals[:4])
        arr = np.asarray(rows)
        x, y, u, v = arr.T
        if shape is None:
            raise ValueError("Bảng x y u v cần tham số shape để nội suy về lưới pixel")
    else:
        raise ValueError(f"Định dạng trường gốc không hỗ trợ: {suf}")
    u = np.asarray(u, float) * scale
    v = np.asarray(v, float) * scale * (-1 if flip_v else 1)
    if x is not None and shape is not None:
        u, v = _resample_to_pixels(x, y, u, v, shape)
    return u, v


@dataclass
class PIVCase:
    """Một ca dữ liệu: danh sách cặp ảnh + (tuỳ chọn) trường gốc tương ứng."""

    name: str
    pairs: list[tuple[Path, Path]]
    truth: list[Path | None] = field(default_factory=list)
    truth_kwargs: dict = field(default_factory=dict)
    roi: tuple[int, int, int, int] | None = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img1, img2 = load_pair(*self.pairs[i], roi=self.roi)
        out = dict(img1=img1, img2=img2, u=None, v=None, name=f"{self.name}[{i}]")
        t = self.truth[i] if i < len(self.truth) else None
        if t is not None:
            u, v = load_ground_truth(t, shape=img1.shape, **self.truth_kwargs)
            out["u"], out["v"] = u, v
        return out


def load_piv_challenge_case(directory: str | Path, name: str | None = None,
                            first_glob: str | None = None, truth_glob: str | None = None,
                            roi=None, **truth_kwargs) -> PIVCase:
    """Dựng ``PIVCase`` từ thư mục đã tải thủ công.

    ``truth_glob`` (ví dụ ``"*_truth.npz"``): các file trường gốc, ghép theo thứ tự
    sắp xếp với cặp ảnh; dùng một file duy nhất nếu chỉ có một (trường dừng).
    """
    directory = Path(directory)
    pairs = find_pairs(directory, first_glob)
    if not pairs:
        raise FileNotFoundError(f"Không tìm thấy cặp ảnh trong {directory}")
    truth: list[Path | None] = []
    if truth_glob:
        tf = sorted(directory.glob(truth_glob))
        truth = tf * len(pairs) if len(tf) == 1 else list(tf)
    return PIVCase(name or directory.name, pairs, truth, truth_kwargs, roi)
