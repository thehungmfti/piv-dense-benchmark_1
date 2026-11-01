# pivbench trong MATLAB

Thư mục này cho phép dùng toàn bộ pivbench từ **MATLAB** (giao diện đồ hoạ hoặc lệnh/script):
RAFT-PIV tái hiện (đã huấn luyện cho wake đáy tù), OpenOpticalFlow_PIV, PIV tương quan chéo, bộ sinh ảnh
hạt tổng hợp có trường gốc, và các phép đo sai số / phổ / MTF.

## Cách hoạt động

- **Tính toán PIV** (RAFT-PIV là mạng PyTorch) chạy bằng Python của repo. Hàm MATLAB ghi một file `.mat`
  tạm, gọi `python scripts/matlab_bridge.py` bằng `system()`, rồi đọc kết quả `.mat`. Không cần
  `pyenv`, không phụ thuộc phiên bản MATLAB–Python tương thích.
- **Phép đo** (`pivbench_metrics`, `pivbench_spectra`, `pivbench_mtf`, `pivbench_vorticity`) viết **thuần
  MATLAB**, không cần toolbox, cho kết quả trùng bản Python tới sai số máy (đã kiểm tra:
  `tests/test_pivbench_matlab.m`, sai lệch tương đối ≤ 1e-14).
- **Giao diện** `pivbench_gui` dùng `uifigure` — cần **MATLAB R2020b trở lên**. Các hàm còn lại chạy được
  trên MATLAB cũ hơn (R2016b+, cần `jsonencode`) và cả GNU Octave ≥ 7.

## Cài đặt (một lần)

1. Cài **Python 3.11** (python.org; trên Windows tích "Add python.exe to PATH" / có lệnh `py`).
2. Trong thư mục gốc repo chạy:
   - Windows: `setup_windows.bat`
   - Linux/macOS: `./setup_unix.sh`

   Script tạo môi trường `.venv`, cài numpy/scipy/scikit-image/torch… và chạy test.
   Dùng Python khác (Anaconda…): đặt trong MATLAB
   `setenv('PIVBENCH_PYTHON', 'C:\...\python.exe')` hoặc ghi đường dẫn vào `matlab/pivbench_python.txt`.
3. Trong MATLAB:

```matlab
cd <repo>/matlab
pivbench_check          % in phiên bản Python/torch, checkpoint RAFT-PIV
pivbench_gui            % mở giao diện
```

## Giao diện `pivbench_gui`

1. **Dữ liệu**: *Ảnh 1… / Ảnh 2…* (tif/bmp/png/jpg), *Mặt nạ thân…* (ảnh: pixel ≠ 0 = vùng thân mô hình,
   kết quả đặt NaN), *Trường gốc (.mat)…* (biến `u`, `v` [px] nếu có — để tính sai số), hoặc
   *Sinh ảnh thử* (wake sát đáy / vùng tuần hoàn / wake xa / sin / xoáy) với ppp, SNR.
2. **Phương pháp**: `raftpiv` (v2, mặc định), `raftpiv_ls` (RAFT + Liu–Shen), `ofpiv`, `ofpiv_warp`,
   `crosscorr`, `raftpiv_v1`; *So sánh tất cả* chạy 4 phương pháp. Tham số: số vòng lặp & cỡ ô RAFT,
   CPU/CUDA, λ_HS/λ_LS, cửa sổ CC, dx [m/px] / dt [s] để quy đổi sang m/s.
3. **Kết quả**: tab Ảnh / |u| + vector / Xoáy ω / Sai số (EPE) / Phổ; bảng sai số (EPE TB, trung vị, p95,
   RMSE, EPE vùng |ω| mạnh) khi có trường gốc. *Xuất .mat* (u, v, x, y, ω, …), *Xuất Tecplot .dat*,
   *Xử lý cả thư mục* (ghép cặp theo tên a/b hoặc 1/2, lưu vào `pivbench_out/`).

## Dùng bằng lệnh

```matlab
addpath('<repo>/matlab')
[u, v, info] = pivbench_estimate('run_a.tif', 'run_b.tif');                 % RAFT-PIV v2
[u, v] = pivbench_estimate(I1, I2, 'crosscorr', struct('windows', [48 24 12]));
[u, v] = pivbench_estimate(I1, I2, 'raftpiv', struct('iters', 24, 'device', 'cuda'));

S = pivbench_synth('wake', struct('D', 140, 'U_inf', 6, 'x_base', 30), [256 384], struct('ppp', 0.05, 'snr', 10));
[u, v] = pivbench_estimate(S.img1, S.img2, 'raftpiv');
M = pivbench_metrics(u, v, S.u_true, S.v_true, 16, S.valid)               % EPE, RMSE, vùng xoáy mạnh…
P = pivbench_spectra(u, v, S.u_true, S.v_true, 16);                       % phổ gốc / ước lượng / sai số
loglog(P.kx(2:end), P.true.E_x(2:end), P.kx(2:end), P.err.E_x(2:end))
```

`demo_pivbench.m` chạy toàn bộ ví dụ (≈ 1–2 phút trên CPU): wake + 4 phương pháp + bảng sai số + MTF + hình.

**Quy ước**: `u` theo cột (x, sang phải), `v` theo hàng (y **hướng xuống** như toạ độ ảnh), đơn vị
pixel/khoảng thời gian giữa 2 ảnh. Vẽ với `imagesc` (trục ij) dùng `quiver(X, Y, u, v)`; với trục y hướng
lên dùng `quiver(X, Y, u, -v)`. ω tính theo toạ độ ảnh nên ngược dấu so với hệ y hướng lên.

## Danh sách file

| file | chức năng |
|---|---|
| `pivbench_gui.m` | giao diện đồ hoạ |
| `pivbench_estimate.m` | ước lượng (u, v) từ cặp ảnh — mọi phương pháp |
| `pivbench_synth.m` | sinh cặp ảnh hạt tổng hợp + trường gốc (wake đáy tù, sin, xoáy, rối, đều) |
| `pivbench_metrics.m` | EPE/RMSE/bias, EPE trong/ngoài vùng |ω| mạnh, sai số ω tại tâm xoáy |
| `pivbench_spectra.m` | phổ 1D (Welch, Hann) theo x, y và phổ 2D trung bình vòng: gốc, ước lượng, sai số |
| `pivbench_mtf.m` | MTF & độ lệch pha tại bước sóng của trường sin |
| `pivbench_vorticity.m` | ω = ∂v/∂x − ∂u/∂y |
| `pivbench_check.m`, `pivbench_config.m`, `pivbench_call.m` | cấu hình & cầu nối Python |
| `demo_pivbench.m` | ví dụ đầy đủ |
| `tests/test_pivbench_matlab.m` | đối chiếu số với Python (chạy được trong MATLAB/Octave) |

## Kiểm tra đã làm

- `tests/test_pivbench_matlab.m` và `demo_pivbench.m` chạy thành công trên **GNU Octave 8.4** (Linux).
- `pivbench_gui.m` đã được kiểm tra cú pháp (Octave đọc toàn bộ file), nhưng **chưa chạy được giao diện** vì
  môi trường xây dựng không có MATLAB (Octave không hỗ trợ `uifigure`). Nếu gặp lỗi khi mở giao diện, hãy gửi
  thông báo lỗi để sửa.
