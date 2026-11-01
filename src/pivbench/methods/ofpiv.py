"""Port Python của OpenOpticalFlow_PIV (Liu & Salazar, JORS 9(1), 2021, doi:10.5334/jors.326).

Mã gốc MATLAB: https://github.com/Tianshu-Liu/OpenOpticalFlow_PIV_v1 (MIT). Các hàm được
port 1–1 (tên hàm giữ như bản gốc):

  * ``correction_illumination``  ← correction_illumination.m
  * ``pre_processing_a``         ← pre_processing_a.m (lọc Gauss fspecial/imfilter)
  * ``shift_image_fun_refine_1`` ← shift_image_fun_refine_1.m
  * ``horn_schunk_estimator``    ← horn_schunk_estimator.m (vector hoá vòng lặp Jacobi)
  * ``generate_invmatrix``       ← generate_invmatrix.m
  * ``liu_shen_estimator``       ← liu_shen_estimator.m
  * ``optical_flow_physics``     ← OpticalFlowPhysics_fun.m
  * ``estimate``                 ← OpenOpticalFlow_PIV_Run.m

Ánh xạ MATLAB → scipy: ``imfilter(...)`` (tương quan, mặc định đệm 0) ↔
``ndimage.correlate(mode='constant')``; ``'replicate'`` ↔ ``mode='nearest'``;
``'symmetric'`` ↔ ``mode='reflect'``; ``imresize`` (bicubic) ↔ ``skimage.transform.resize(order=3)``.
Trong ``horn_schunk_estimator``/``liu_shen_estimator`` của bản gốc, "u" là thành phần theo
HÀNG (y) và "v" theo CỘT (x) vì kernel D lấy đạo hàm theo hàng; ``OpticalFlowPhysics_fun``
hoán đổi lại (``ux = v; uy = u``). Port giữ nguyên quy ước này bên trong.

KHÁC BẢN GỐC (có chủ đích, đều ghi chú tại chỗ):
  1. Chuẩn hoá cường độ: bản gốc giả định ảnh uint8 (0–255) và λ được chỉnh cho thang đó.
     Ở đây ảnh vào (thường float 0–1) được đưa về thang 0–``max_intensity`` (mặc định 255)
     bằng cùng một hệ số cho cả 2 ảnh (dựa trên ``rescaling_intensity.m`` nhưng dùng
     min/max chung thay vì riêng từng ảnh, để không làm sai lệch tỉ lệ cường độ).
  2. ``shift_image_fun_refine_1`` bản gốc ép ``uint8`` ảnh đã dịch; ở đây giữ float
     (tuỳ chọn ``quantize_shift=True`` để giống hệt bản gốc).
  3. Bản gốc cố định 1 lượt lặp và 1 mức (``scale_im``); ở đây thêm kim tự tháp
     ``n_levels`` mức (coarse-to-fine, hệ số 2) và ``n_iter`` lượt mỗi mức, dùng lại đúng
     ``shift_image_fun_refine_1`` để biến dạng ảnh theo trường mức thô. ``n_levels=1,
     n_iter=1`` khớp với bản gốc (với ``scale_im=1``).
  4. Khởi tạo tương quan chéo: bản gốc dùng PIVsuite (iaSize [64 16 8], step [32 8 4]);
     ở đây dùng ``pivbench.methods.crosscorr`` với cùng cỡ cửa sổ/bước. Mặc định
     ``init='none'`` (dịch chuyển nhỏ / kim tự tháp), ``init='crosscorr'`` để bật.
  5. ``horn_schunk_estimator`` bản gốc là vòng lặp Jacobi theo từng pixel; ở đây vector hoá
     (cùng công thức, cùng ma trận đếm lân cận ``cmtx`` — kể cả điểm lạ của bản gốc là các
     pixel cạnh (không phải góc) vẫn dùng hệ số 8 thay vì 5).
  6. Vòng gán dịch nguyên pixel trong ``shift_image_fun_refine_1`` được vector hoá; khi nhiều
     pixel đích trùng nhau, numpy giữ giá trị gán sau cùng theo thứ tự quét hàng — giống
     thứ tự vòng lặp ``for i (hàng), for j (cột)`` của MATLAB.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Tiện ích MATLAB
# ---------------------------------------------------------------------------
def fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    X, Y = np.meshgrid(ax, ax)
    h = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    h[h < np.finfo(float).eps * h.max()] = 0
    return h / h.sum()


def imfilter(a: np.ndarray, h: np.ndarray, boundary: str | float = 0.0) -> np.ndarray:
    """``imfilter(a, h, boundary, 'same')`` của MATLAB (tương quan, tâm floor((n+1)/2))."""
    mode = {"replicate": "nearest", "symmetric": "reflect", "circular": "wrap"}
    origin = [(-1 if s % 2 == 0 else 0) for s in h.shape]  # tâm MATLAB cho kernel cỡ chẵn
    if isinstance(boundary, str):
        return ndimage.correlate(a, h, mode=mode[boundary], origin=origin)
    return ndimage.correlate(a, h, mode="constant", cval=float(boundary), origin=origin)


def imresize(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize(a, shape, order=3, mode="edge", anti_aliasing=False, preserve_range=True)


# ---------------------------------------------------------------------------
# Tiền xử lý
# ---------------------------------------------------------------------------
def rescale_pair(im1, im2, max_intensity: float = 255.0):
    """Đưa 2 ảnh về thang [0, max_intensity] với cùng min/max (xem ghi chú 1)."""
    lo = min(im1.min(), im2.min())
    hi = max(im1.max(), im2.max())
    s = max_intensity / (hi - lo) if hi > lo else 1.0
    return (im1 - lo) * s, (im2 - lo) * s


def correction_illumination(im1, im2, window_shifting=None, size_average: int = 0):
    """Hiệu chỉnh cường độ toàn cục (tỉ số trung bình) và cục bộ (lọc trung bình N×N)."""
    I1 = np.asarray(im1, float)
    I2 = np.asarray(im2, float)
    if window_shifting is None:
        y3, y4, x3, x4 = 0, I1.shape[0], 0, I1.shape[1]
    else:  # [x1, x2, y1, y2] (1-based, bao gồm) như bản gốc
        x3, x4, y3, y4 = window_shifting
        x3, y3 = x3 - 1, y3 - 1
    R12 = I1[y3:y4, x3:x4].mean() / I2[y3:y4, x3:x4].mean()
    I2 = R12 * I2
    if size_average > 0:
        h = np.ones((size_average, size_average)) / size_average**2
        # filter2(h, I) = imfilter(I, h) với đệm 0
        I12F = imfilter(I1, h) - imfilter(I2, h)
        I2 = I2 + I12F
    return I1, I2


def pre_processing_a(im1, im2, scale_im: float = 1.0, size_filter: int = 6):
    I1 = np.asarray(im1, float)
    I2 = np.asarray(im2, float)
    if scale_im != 1:
        shp = (int(round(I1.shape[0] * scale_im)), int(round(I1.shape[1] * scale_im)))
        I1 = resize(I1, shp, order=3, anti_aliasing=scale_im < 1, preserve_range=True)
        I2 = resize(I2, shp, order=3, anti_aliasing=scale_im < 1, preserve_range=True)
    if size_filter and size_filter > 0:
        H1 = fspecial_gaussian(int(size_filter), size_filter * 0.62)
        # bản gốc: (imfilter(I,H1)+imfilter(I,H2))/2 với H1 == H2 — tương đương 1 lần lọc
        I1 = imfilter(I1, H1)
        I2 = imfilter(I2, H1)
    return I1, I2


# ---------------------------------------------------------------------------
# Dịch ảnh theo trường thô
# ---------------------------------------------------------------------------
def shift_image_fun_refine_1(ux, uy, Im1, Im2, quantize_shift: bool = False):
    """Tạo ảnh Im1 dịch theo (ux, uy) đã nội suy lên lưới ảnh mịn.

    Trả về (Im1_shift, uxI, uyI). Phần nguyên của dịch chuyển: chép pixel (ảnh nền là Im2
    như bản gốc); phần lẻ: hiệu chỉnh theo phương trình vận chuyển
    I ← I − ∂(I·δx)/∂x − ∂(I·δy)/∂y (sai phân tiến) với δ đã làm trơn Gauss 10×10.
    """
    Im1 = np.asarray(Im1, float)
    Im2 = np.asarray(Im2, float)
    m0, n0 = ux.shape
    m1, n1 = Im1.shape
    uxI = (n1 / n0) * (imresize(ux, (m1, n1)) if (m0, n0) != (m1, n1) else np.asarray(ux, float))
    uyI = (m1 / m0) * (imresize(uy, (m1, n1)) if (m0, n0) != (m1, n1) else np.asarray(uy, float))

    rx = np.round(uxI).astype(np.int64)
    ry = np.round(uyI).astype(np.int64)
    I, J = np.mgrid[0:m1, 0:n1]
    ti, tj = I + ry, J + rx
    inside = (ti >= 0) & (ti < m1) & (tj >= 0) & (tj < n1)
    ti = np.where(inside, ti, I)
    tj = np.where(inside, tj, J)
    Im3 = Im2.copy()
    Im3[ti.ravel(), tj.ravel()] = Im1.ravel()  # gán theo thứ tự quét hàng (ghi chú 6)

    duxI = uxI - np.round(uxI)
    duyI = uyI - np.round(uyI)
    H1 = fspecial_gaussian(10, 0.6 * 10)
    duxI = imfilter(duxI, H1)
    duyI = imfilter(duyI, H1)

    out = Im3.copy()
    p = Im3 * duxI
    q = Im3 * duyI
    term1 = p[:-1, 1:] - p[:-1, :-1]
    term2 = q[1:, :-1] - q[:-1, :-1]
    out[:-1, :-1] = Im3[:-1, :-1] - (term1 + term2)
    if quantize_shift:
        out = np.clip(np.round(out), 0, 255)  # uint8() của MATLAB: làm tròn + bão hoà
    return out, uxI, uyI


def warp_image_bicubic(ux, uy, Im1, Im2):
    """Thay thế (mở rộng, KHÔNG có trong bản gốc) cho ``shift_image_fun_refine_1``:
    Im1_w(x) = Im1(x − u(x)) bằng nội suy B-spline bậc 3 (tìm điểm ngược 1 bước).
    Chính xác hơn phép vận chuyển bậc 1 cho phần lẻ của bản gốc, cho phép lặp hội tụ."""
    Im1 = np.asarray(Im1, float)
    m0, n0 = ux.shape
    m1, n1 = Im1.shape
    uxI = (n1 / n0) * (imresize(ux, (m1, n1)) if (m0, n0) != (m1, n1) else np.asarray(ux, float))
    uyI = (m1 / m0) * (imresize(uy, (m1, n1)) if (m0, n0) != (m1, n1) else np.asarray(uy, float))
    Y, X = np.mgrid[0:m1, 0:n1].astype(float)
    # điểm xuất phát x0 thoả x0 + u(x0) = x  (1 bước điểm cố định)
    x0, y0 = X - uxI, Y - uyI
    ux0 = ndimage.map_coordinates(uxI, [y0, x0], order=1, mode="nearest")
    uy0 = ndimage.map_coordinates(uyI, [y0, x0], order=1, mode="nearest")
    out = ndimage.map_coordinates(Im1, [Y - uy0, X - ux0], order=3, mode="mirror")
    return out, uxI, uyI


# ---------------------------------------------------------------------------
# Horn–Schunck
# ---------------------------------------------------------------------------
_NB8 = np.array([[1.0, 1, 1], [1, 0, 1], [1, 1, 1]])


def horn_schunk_estimator(Ix, Iy, It, lam, tol=1e-12, maxnum=500, exact_counts=False):
    r, c = Ix.shape
    if exact_counts:
        cmtx = ndimage.correlate(np.ones((r, c)), _NB8, mode="constant")
    else:  # đúng như bản gốc: góc = 3, còn lại = 8
        cmtx = 8.0 * np.ones((r, c))
        cmtx[0, 0] = cmtx[0, -1] = cmtx[-1, 0] = cmtx[-1, -1] = 3.0
    den = cmtx * (Ix**2 + Iy**2) + lam * cmtx**2
    uv = (Ix * Iy) / den
    u1 = (Iy**2 + lam * cmtx) / den
    u2 = (Ix * It) / (Ix**2 + Iy**2 + lam * cmtx)
    v1 = (Ix**2 + lam * cmtx) / den
    v2 = (Iy * It) / (Ix**2 + Iy**2 + lam * cmtx)
    u = np.zeros((r, c))
    v = np.zeros((r, c))
    k = 0
    err = np.inf
    while err > tol and k < maxnum:
        tu = ndimage.correlate(u, _NB8, mode="constant")
        tv = ndimage.correlate(v, _NB8, mode="constant")
        un = u1 * tu - uv * tv - u2
        vn = v1 * tv - uv * tu - v2
        err = (np.linalg.norm(un - u) + np.linalg.norm(vn - v)) / (r * c)
        u, v = un, vn
        k += 1
    return u, v


# ---------------------------------------------------------------------------
# Liu–Shen (physics-based optical flow)
# ---------------------------------------------------------------------------
_D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2.0
_M = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4.0
_F = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], float)
_D2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], float)
_H = _NB8


def generate_invmatrix(I, alpha, h=1.0):
    cmtx = imfilter(np.ones_like(I), _H / (h * h))
    A11 = I * (imfilter(I, _D2 / (h * h), "replicate") - 2 * I / (h * h)) - alpha * cmtx
    A22 = I * (imfilter(I, _D2.T / (h * h), "replicate") - 2 * I / (h * h)) - alpha * cmtx
    A12 = I * imfilter(I, _M / (h * h), "replicate")
    det = A11 * A22 - A12 * A12
    return A22 / det, -A12 / det, A11 / det


def liu_shen_estimator(I0, I1, f, dx, dt, lam, tol, maxnum, u0, v0):
    """Trả về (u, v, error); u theo hàng (y), v theo cột (x) — như bản gốc."""
    rep = "replicate"
    IIx = I0 * imfilter(I0, _D / dx, rep)
    IIy = I0 * imfilter(I0, _D.T / dx, rep)
    II = I0 * I0
    g = (I1 - I0) / dt - f
    Ixt = I0 * imfilter(g, _D / dx, rep)
    Iyt = I0 * imfilter(g, _D.T / dx, rep)
    u = np.asarray(u0, float).copy()
    v = np.asarray(v0, float).copy()
    r, c = I1.shape
    B11, B12, B22 = generate_invmatrix(I0, lam, dx)
    errors = []
    k = 0
    err = np.inf
    while err > tol and k < maxnum:
        bu = (2 * IIx * imfilter(u, _D / dx, rep) + IIx * imfilter(v, _D.T / dx, rep)
              + IIy * imfilter(v, _D / dx, rep) + II * imfilter(u, _F / dx**2, rep)
              + II * imfilter(v, _M / dx**2, rep) + lam * imfilter(u, _H / dx**2) + Ixt)
        bv = (IIy * imfilter(u, _D / dx, rep) + IIx * imfilter(u, _D.T / dx, rep)
              + 2 * IIy * imfilter(v, _D.T / dx, rep) + II * imfilter(u, _M / dx**2, rep)
              + II * imfilter(v, _F.T / dx**2, rep) + lam * imfilter(v, _H / dx**2) + Iyt)
        un = -(B11 * bu + B12 * bv)
        vn = -(B12 * bu + B22 * bv)
        err = (np.linalg.norm(un - u) + np.linalg.norm(vn - v)) / (r * c)
        u, v = un, vn
        errors.append(err)
        k += 1
    return u, v, np.asarray(errors)


def laplacian(u, h=1.0):
    return -u * imfilter(np.ones_like(u), _H / (h * h)) + imfilter(u, _H / (h * h))


def optical_flow_physics(I1, I2, lambda_1, lambda_2, hs_maxnum=500, hs_tol=1e-12,
                         ls_maxnum=60, ls_tol=1e-8, Dm=0.0, hs_exact_counts=False):
    """``OpticalFlowPhysics_fun.m``: HS khởi tạo → Liu–Shen tinh chỉnh. Trả về (ux, uy, ux_hs, uy_hs)."""
    D1 = np.array([[0, 0, 0], [0, -1, -1], [0, 1, 1]]) / 2.0
    F1 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]) / 4.0
    Ia = (I1 + I2) / 2
    Ix = imfilter(Ia, D1, "symmetric")      # thực chất ∂/∂hàng
    Iy = imfilter(Ia, D1.T, "symmetric")    # ∂/∂cột
    It = imfilter(I2 - I1, F1, "symmetric")
    u, v = horn_schunk_estimator(Ix, Iy, It, lambda_1, hs_tol, hs_maxnum, hs_exact_counts)
    ux_horn, uy_horn = v, u
    f = Dm * laplacian(I1, 1.0)
    u, v, _ = liu_shen_estimator(I1, I2, f, 1.0, 1.0, lambda_2, ls_tol, ls_maxnum, uy_horn, ux_horn)
    return v, u, ux_horn, uy_horn


# ---------------------------------------------------------------------------
# Giao diện chung
# ---------------------------------------------------------------------------
def estimate(img1: np.ndarray, img2: np.ndarray, lambda_HS: float = 20.0, lambda_LS: float = 2000.0,
             n_levels: int = 1, n_iter: int = 1, init: str = "none", size_filter: int = 6,
             size_average: int = 0, max_intensity: float = 255.0, edge_width: int = 1,
             cc_windows=(64, 16, 8), cc_overlap: float = 0.5, hs_maxnum: int = 500,
             ls_maxnum: int = 60, quantize_shift: bool = False, shift_method: str = "original",
             return_hs: bool = False, raft_params: dict | None = None, **_ignored):
    """OpenOpticalFlow_PIV: trả về (u, v) cỡ ảnh, đơn vị px/khoảng thời gian.

    Tham số chính: ``lambda_HS`` (λ₁, Horn–Schunck), ``lambda_LS`` (λ₂, Liu–Shen),
    ``n_levels`` (số mức kim tự tháp), ``n_iter`` (lượt lặp mỗi mức),
    ``init`` ∈ {'none', 'crosscorr', 'raftpiv'} ('raftpiv' = mở rộng: khởi tạo bằng RAFT-PIV tái hiện,
    tham số qua ``raft_params``), ``shift_method`` ∈ {'original' (shift_image_fun_refine_1),
    'warp' (nội suy bậc 3, mở rộng — xem ``warp_image_bicubic``)}.
    """
    I1 = np.asarray(img1, float)
    I2 = np.asarray(img2, float)
    I1, I2 = rescale_pair(I1, I2, max_intensity)
    I1, I2 = correction_illumination(I1, I2, None, size_average)
    I1, I2 = pre_processing_a(I1, I2, 1.0, size_filter)
    shape = I1.shape

    # Kim tự tháp (mức 0 = độ phân giải gốc). Ảnh đã lọc Gauss ở mức gốc nên chỉ cần
    # thu nhỏ có chống răng cưa cho các mức thô.
    pyr = [(I1, I2)]
    for _ in range(1, int(n_levels)):
        a, b = pyr[-1]
        shp = (max(8, a.shape[0] // 2), max(8, a.shape[1] // 2))
        pyr.append((resize(a, shp, order=3, anti_aliasing=True, preserve_range=True),
                    resize(b, shp, order=3, anti_aliasing=True, preserve_range=True)))

    coarse = pyr[-1][0].shape
    if init in ("crosscorr", "raftpiv"):
        if init == "crosscorr":
            from .crosscorr import estimate as cc_estimate

            u0, v0 = cc_estimate(np.asarray(img1, float), np.asarray(img2, float),
                                 windows=cc_windows, overlap=cc_overlap)
        else:  # mở rộng: RAFT-PIV (tái hiện) làm trường khởi tạo, Liu–Shen tinh chỉnh
            from .raftpiv import estimate_local

            u0, v0 = estimate_local(img1, img2, **(raft_params or {}))
        ux = imresize(u0, coarse) * coarse[1] / shape[1]
        uy = imresize(v0, coarse) * coarse[0] / shape[0]
    elif init == "none":
        ux = np.zeros(coarse)
        uy = np.zeros(coarse)
    else:
        raise ValueError(f"init không hợp lệ: {init}")

    ux_hs = uy_hs = None
    for lev in range(len(pyr) - 1, -1, -1):
        a, b = pyr[lev]
        for _ in range(int(n_iter)):
            if shift_method == "original":
                Im1_shift, uxI, uyI = shift_image_fun_refine_1(ux, uy, a, b, quantize_shift)
            elif shift_method == "warp":
                Im1_shift, uxI, uyI = warp_image_bicubic(ux, uy, a, b)
            else:
                raise ValueError(f"shift_method không hợp lệ: {shift_method}")
            dux, duy, ux_hs, uy_hs = optical_flow_physics(Im1_shift, b, lambda_HS, lambda_LS,
                                                          hs_maxnum=hs_maxnum, ls_maxnum=ls_maxnum)
            ux = uxI + dux
            uy = uyI + duy

    if edge_width and edge_width > 0:  # "clean up the edges" — chỉ cạnh trái và trên như bản gốc
        e = int(edge_width)
        ux[:, :e] = ux[:, e:2 * e]
        uy[:, :e] = uy[:, e:2 * e]
        ux[:e, :] = ux[e:2 * e, :]
        uy[:e, :] = uy[e:2 * e, :]
    if return_hs:
        return ux, uy, (ux_hs, uy_hs)
    return ux, uy
