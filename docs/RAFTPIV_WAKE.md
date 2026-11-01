# RAFT-PIV tái hiện cho dòng wake sau mô hình trục đối xứng đáy tù

Tài liệu này mô tả mạng RAFT-PIV **viết lại từ đầu** trong `src/pivbench/methods/raft/`, dữ liệu huấn
luyện, tham số đã chỉnh cho bài toán wake đáy tù, và cách huấn luyện lại / chạy trên ảnh thí nghiệm của bạn.

> Đây **không** phải mã hay trọng số của Lagemann et al. (2021) — bản gốc không truy cập được khi xây dựng.
> Kiến trúc theo RAFT (Teed & Deng 2020) với các điều chỉnh kiểu RAFT-PIV; chi tiết có thể khác bản gốc.
> Mọi số liệu đánh giá dưới đây đến từ dữ liệu tổng hợp hoặc dữ liệu Cai et al. có trường gốc;
> chưa kiểm chứng trên ảnh thí nghiệm thật của mô hình đáy tù.

## 1. Kiến trúc (`model.py`)

| khối | cấu hình mặc định (`configs/raftpiv_wake.yaml`) | lý do cho wake đáy tù |
|---|---|---|
| Đầu vào | ảnh xám 1 kênh, chuẩn hoá từng ảnh (trung bình 0, độ lệch chuẩn 1) | bền với chênh lệch độ sáng 2 xung laser, phản xạ gần đáy |
| Bộ mã hoá đặc trưng / ngữ cảnh | conv 7×7 s2 → 3 tầng residual (32, 48, 64 kênh), **giảm mẫu /4** (RAFT gốc /8) | hạt 1.5–4 px và lớp cắt mỏng sát mép đáy cần lưới đặc trưng mịn; /2 quá chậm trên CPU (≈ 5×) |
| Đặc trưng tương quan | 128 kênh, InstanceNorm | |
| Kim tự tháp tương quan | toàn cặp, 4 mức, bán kính tra cứu 4 | tầm ±4·4·2³ = ±128 px ≫ dịch chuyển dòng tự do (thường ≤ 12–16 px) |
| Cập nhật | SepConvGRU (ẩn 96, ngữ cảnh 64, chuyển động 96); 8 vòng khi huấn luyện, 16 khi suy luận | |
| Nâng mẫu | lồi (convex) 3×3 từ lưới /4 về từng pixel | giữ gradient sắc ở lớp cắt (không nội suy song tuyến) |
| Hàm mất mát | Σ 0.8^(N−i−1)·‖f_i − f_gt‖₁, bỏ điểm trong thân mô hình | |
| Số tham số | ≈ 1.8 M | |

Suy luận ảnh lớn: chia ô 512×512 chồng lấn 64 px, ghép trọng số cosin (`infer.py`).

## 2. Dữ liệu huấn luyện

**Bài học từ v1 → v2.** v1 chỉ thấy các trường có div₂D = 0 (hàm dòng phẳng, rối 2D, CFD 2D của Cai) và học
một prior "không nén được": với u = 0.5·sin(2πx/λ) (thuần nén) MTF ≈ 0 tới λ = 64 px. Trên mặt kinh tuyến của
dòng trục đối xứng, div₂D = ∂u/∂x + ∂v/∂r = −v/r ≠ 0 (đặc biệt gần trục, vùng tuần hoàn) nên prior này sai cho
bài toán. v2 (`configs/raftpiv_wake_v2.yaml`) dùng dòng trung bình theo **hàm dòng Stokes** (liên tục trục đối
xứng) + 20 % trường nén được (u, v ngẫu nhiên độc lập) + 12 % sin — λ_c(0.5) cải thiện 97 → 45 px; vẫn kém
optical flow / tương quan chéo do huấn luyện ngắn.


1. **Wake đáy tù tổng hợp** (`synth/wake.py`; v1: 75 % của 16 000 mẫu, v2: 45 % của 12 000 mẫu 128×128):
   vùng tuần hoàn dài L_r = 0.8–1.6 D với dòng ngược 0.1–0.4 U∞, profile thiếu hụt từ dạng "mũ"
   (lớp cắt mỏng) sang Gauss, xoáy cuộn của lớp cắt, mode xoáy so le phía sau L_r, rối k^-5/3 tập trung
   ở lớp cắt; thân mô hình che (không hạt, nền tối). v2: dòng trung bình theo hàm dòng Stokes (trục đối xứng);
   xoáy/rối qua hàm dòng phẳng.
   Khung nhìn ngẫu nhiên: sát đáy / vùng tuần hoàn / wake xa; D = 0.3–2 lần cạnh ảnh; U∞ = 1–12 px.
2. **Trường tổng quát**: xoáy Lamb–Oseen, rối, sin, đều, (v2) trường nén được — tránh mạng chỉ "học thuộc" wake.
3. **Cai et al. 2019** (cylinder Re 40–400 = wake trụ tròn, backstep Re 800–1500 = dòng tách),
   tập train theo `FlowData_train.list` (5 070 cặp), xác suất 30 % mỗi mẫu. Giấy phép: chỉ nghiên cứu,
   KHÔNG phân phối lại — không có trong repo/zip; tải theo `scripts/download_data.md`.

Ảnh hạt: ppp 0.01–0.12 (log-uniform), d_p 1.5–4 px, SNR 4–40 hoặc không nhiễu, tấm laser 6–16 px với
dịch chuyển ngoài mặt phẳng σ_w ≤ 0.3 bề dày (wake trục đối xứng có thành phần vuông góc mạnh), lượng tử
8 bit, gradient chiếu sáng. Tăng cường: lật x/y (đổi dấu thành phần), đảo thời gian (chính xác với dữ liệu
tổng hợp vì trường gốc ở thời điểm giữa), gamma/độ lợi.

## 3. Chỉnh cho thí nghiệm của bạn

Sửa `configs/raftpiv_wake.yaml` và khoảng tham số ảnh `ImagingRanges` (`methods/raft/data.py`) cho khớp:

- **U∞ theo px** = U∞[m/s]·Δt / (kích thước pixel trong mặt vật). Đặt `u_inf` bao trùm giá trị này (±50 %).
- **D theo px**: `random_wake_params` lấy D = 0.3–2 lần cạnh ảnh huấn luyện (128 px) — với ảnh thật
  D = 300–600 px, cửa sổ 128 px tương ứng 0.2–0.4 D (đã bao gồm).
- **Hạt**: đo d_p và ppp trên ảnh thật (ví dụ bằng `skimage.feature.blob_log`), chỉnh `d_p`, `ppp`.
- **Đáy/thân mô hình**: nếu có phản xạ sáng ở mép đáy, tăng `level` trong `make_sample` hoặc thêm vệt sáng;
  dùng `--mask` khi suy luận để đặt NaN trong thân.
- **Mạng**: `down: 8` nhanh ≈ 3× (phân giải kém hơn), `down: 2` chính xác hơn nhưng chậm ≈ 5× — chỉ nên dùng
  với GPU. Tăng `hidden_dim`/`enc_channels` nếu có GPU và nhiều bước huấn luyện.
- **Tinh chỉnh (fine-tune)**: nếu có trường gốc (CFD/LES của chính mô hình), xuất u, v (px) ra `.npz`
  và dùng `kind: dns` trong `pivbench.synth.load_dns_field` để sinh kho dữ liệu riêng
  (`build_pool` với hàm `make_sample` sửa lại), rồi chạy tiếp từ `best.pt`.

## 4. Lệnh

```bash
# huấn luyện từ đầu (CPU 4 nhân: ≈ 1.2–2 s/bước, 6 000 bước ≈ 2–3.5 h; GPU: --device cuda)
python scripts/train_raftpiv.py configs/raftpiv_wake.yaml
# tinh chỉnh v2 từ v1 (wake trục đối xứng + trường nén được)
python scripts/train_raftpiv.py configs/raftpiv_wake_v2.yaml
# tiếp tục: chạy lại cùng lệnh (đọc last.pt); tăng train.steps để huấn luyện thêm

# suy luận trên ảnh của bạn
python scripts/infer_raftpiv.py run_a.tif run_b.tif -o out/ --compare
python scripts/infer_raftpiv.py --dir data/my_wake -o out/ --mask body_mask.png

# benchmark so sánh với ofpiv / crosscorr (wake tổng hợp + bộ test Cai)
python scripts/run_benchmark.py configs/wake_benchmark.yaml
```

## 5. Kết quả & hạn chế

Xem mục RAFT-PIV trong README (bảng EPE, MTF, đánh giá với điều kiện ảnh đa dạng). Tóm tắt: RAFT tái hiện
bền hơn tương quan chéo khi ảnh xấu (không thất bại, p90 thấp hơn ~40 %) nhưng làm trơn mạnh hơn
(λ_c(0.5) = 45 px so với 16–23 px) và chưa chính xác hơn ở ảnh "dễ". Cần huấn luyện dài hơn (GPU) và kiểm chứng
trên ảnh thí nghiệm thật.
