# P1 trong MATLAB — độ phân giải hiệu dụng & hiệu chỉnh phổ

```matlab
cd matlab/p1
p1_resolution_gui                                   % mở với results/p1_mtf_quick, results/p1_spectral
p1_resolution_gui('../../results/p1_mtf_full', '../../results/p1_spectral')
```

Yêu cầu MATLAB R2020b+ (uifigure). Việc **xem/phân tích/hiệu chỉnh** chạy thuần MATLAB trên CSV do Python sinh ra;
chỉ hai nút "Chạy … (Python)" mới gọi Python (đường dẫn Python như `pivbench_config`, xem `../README_MATLAB.md`).

| khu vực | chức năng |
|---|---|
| 1) Dữ liệu | chọn thư mục kết quả MTF (`mtf_long.csv`) và phổ (`transfer.csv`, `spectra_long.csv`, `k10.csv`); chạy lại sweep/hiệu chỉnh `--quick` |
| 2) MTF / λ_c | chọn A, ppp, SNR, ngưỡng, các cấu hình → tab **MTF(λ)** (kèm sinc top-hat W = 16, 32) và **Bản đồ nhiệt λ_c**; nút **Khớp mô hình λ_c** → λ_c = a·ppp^b·SNR^c(·s^d), R², λ_c/ℓ_p |
| 3) Hiệu chỉnh phổ | trường kiểm tra (rối k^-5/3 / DNS 2D), phương pháp, (ppp, SNR), nguồn H (`H_xs_turb_train`, `H_sine_T`, `H_sine_L`, `identity`), sàn nhiễu (`N_xs`, `N_zero`, `tail`, `none`), Wiener/nghịch đảo, ε / ngưỡng \|H\| / tol → tab **Hàm truyền H(k)**, **Hiệu chỉnh phổ**, **E/E_true** với k₁₀ trước/sau |
| Áp lên trường của tôi | nạp `.mat` có `u`, `v` (px/Δt; tuỳ chọn `ut`, `vt` là trường gốc), tính phổ 2D (`pivbench_spectra`) và hiệu chỉnh bằng H đã chọn. **H phải được đo ở cùng phương pháp, ppp, SNR và biên độ dịch chuyển tương tự ảnh của bạn.** |
| Xuất | phổ đang xem → `.mat`/`.csv`; lưu PNG mọi tab |

Hàm dùng độc lập (cùng công thức với Python, xem `docs/P1_SPECTRAL_CORRECTION.md`):

| hàm | mô tả |
|---|---|
| `p1_correct_spectrum(k, Em, H, N, mode, eps, h_min)` | E_corr (Wiener / nghịch đảo) |
| `p1_transfer(k, k_nodes, H_nodes)` | H(k) nội suy log k; `p1_transfer(k, 1./lambda, mtf)` |
| `p1_noise_tail(k, Em)` | sàn nhiễu đuôi phẳng |
| `p1_valid_k(k, E, Et, tol, k_max, k_min)` | k₁₀ |
| `p1_cutoff(lambda, mtf, thr)` | λ_c |
| `p1_tophat_mtf(lambda, W)` | \|sinc(W/λ)\| |
| `p1_fit_powerlaw(y, X, names)` | hồi quy log |

Kiểm tra: `cd tests; test_p1_matlab` (đã chạy đạt trên GNU Octave 8.4; giao diện uifigure **chưa được chạy thử
trong MATLAB thật** — môi trường xây dựng không có MATLAB).
