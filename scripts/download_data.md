# Tải dữ liệu thủ công

Repo **không tự tải** dữ liệu: phần lớn nguồn dưới đây cần đăng nhập, token hoặc chấp nhận
điều khoản sử dụng. Hãy tải bằng trình duyệt rồi đặt vào `data/raw/<tên_ca>/`
(thư mục này nằm trong `.gitignore`).

## 1. PIV Challenge (ảnh + trường gốc tổng hợp)

- Trang chính: <https://www.pivchallenge.org/> — các kỳ 2001, 2003, 2005, 2014.
- Một số ca có ảnh tổng hợp kèm trường gốc DNS (ví dụ các ca ảnh tổng hợp của kỳ 2003/2005/2014).
  Một số ca yêu cầu đăng ký trước khi tải.
- Sau khi giải nén, ví dụ:

```
data/raw/pivc2005_caseB/
    B001_1.tif  B001_2.tif  ...
    truth.npz            # nếu tự chuyển trường gốc sang npz (khoá u, v theo px/khoảng thời gian)
```

- Nạp trong Python:

```python
from pivbench.data import load_piv_challenge_case
case = load_piv_challenge_case("data/raw/pivc2005_caseB", truth_glob="truth.npz")
sample = case[0]   # dict img1, img2, u, v
```

Lưu ý quy ước: pivbench dùng `x` = cột, `y` = hàng (hướng xuống), đơn vị px/khoảng thời gian giữa
2 ảnh. Nếu trường gốc dùng trục y hướng lên, dùng `flip_v=True`; nếu đơn vị là m/s thì đặt
`scale = Δt / (kích thước pixel trong mặt vật)`.

## 2. JHTDB (Johns Hopkins Turbulence Databases)

- <https://turbulence.pha.jhu.edu/> — cần **token cá nhân** (đăng ký miễn phí).
- Lấy một lát 2D (ví dụ `isotropic1024coarse`, thành phần u, v trên mặt z = const) bằng
  công cụ chính thức (`givernylib` / `pyJHTDB`) hoặc cutout service, lưu thành HDF5/npz với
  các khoá `u`, `v`.
- Nạp: `pivbench.synth.load_dns_field("data/raw/jhtdb_slice.h5", u_key="u", v_key="v", scale=...)`
  rồi dùng làm trường đầu vào cho bộ sinh ảnh (`kind: dns` trong YAML).
- `pivbench.synth.fields.JHTDBSource` là giao diện để trống cho việc truy cập trực tuyến.

## 3. Bộ dữ liệu RAFT-PIV (Lagemann et al., Nature Machine Intelligence 2021)

- Zenodo DOI 10.5281/zenodo.4432495 (TFRecord). Chỉ cần nếu muốn huấn luyện/kiểm tra RAFT-PIV
  trên dữ liệu gốc của tác giả.
- Trọng số mô hình: xem mục RAFT-PIV trong README (chưa tìm được bản phát hành truy cập được
  từ môi trường xây dựng repo này).

## 4. Bộ dữ liệu PIV của Cai et al. (2019) — dùng huấn luyện/kiểm tra RAFT-PIV

- <https://github.com/shengzesnail/PIV_dataset> (chỉ dùng cho nghiên cứu; không phân phối lại). Các file zip
  lưu bằng Git LFS — tải trực tiếp (cylinder 0.7 GB, backstep 1.8 GB):

```bash
mkdir -p ../data_cai && cd ../data_cai
for f in cylinder backstep; do
  curl -L -o $f.zip "https://media.githubusercontent.com/media/shengzesnail/PIV_dataset/master/PIV-genImages/data/$f.zip"
  unzip -q $f.zip && rm $f.zip
done
curl -L -O https://raw.githubusercontent.com/shengzesnail/PIV_dataset/master/FlowData_train.list
curl -L -O https://raw.githubusercontent.com/shengzesnail/PIV_dataset/master/FlowData_test.list
```

- Đường dẫn mặc định trong `configs/raftpiv_wake*.yaml` và `configs/wake_benchmark.yaml` là `../data_cai`
  (cạnh thư mục repo). Không có dữ liệu này, huấn luyện chỉ dùng dữ liệu tổng hợp.

## 5. Ảnh mẫu của OpenOpticalFlow_PIV

- <https://github.com/Tianshu-Liu/OpenOpticalFlow_PIV_v1> có các cặp ảnh `*.tif`
  (vortex pair, wall jet, twin jet, White Oval) — không có trường gốc, dùng để so sánh định tính.

```bash
git clone --depth 1 https://github.com/Tianshu-Liu/OpenOpticalFlow_PIV_v1 data/raw/oof_samples
```
