# P1 — Hàm truyền và hiệu chỉnh phổ: công thức

Tài liệu này ghi chính xác các công thức được cài trong `src/pivbench/resolution/` (Python) và
`matlab/p1/` (MATLAB, cùng công thức, có test đối chiếu).

## 1. Quy ước

* Số sóng `k` tính bằng **chu kỳ/px** (như `pivbench.metrics.spectra`); bước sóng λ = 1/k.
  Với số sóng góc κ = 2πk thì H(κ) = MTF(2π/κ) — cùng một hàm, chỉ khác trục.
* Phổ 2D: `E(k)` = ½(|û|² + |v̂|²) với cửa sổ Hann 2D trên vùng đã bỏ biên `border` px, trung bình theo
  vòng |k| (bin rộng 1/N), chuẩn hoá mật độ: Σ E·Δk ≈ ½⟨u² + v²⟩.
* Phổ chéo: `S_ab(k)` = ½ Re⟨â_u b̂_u* + â_v b̂_v*⟩, cùng cửa sổ/chuẩn hoá (S_aa = E_a).

## 2. MTF từ trường sin

Trường gốc `u = A sin(2πx/λ)` (mode **dọc/nén**, đúng đặc tả) hoặc `v = A sin(2πx/λ)` (mode **ngang/cắt**,
div = 0). Khớp bình phương tối thiểu tại đúng λ trên vùng bỏ biên:
`f_est ≈ a sin θ + b cos θ + c`, θ = 2πx/λ  ⇒  **MTF(λ) = √(a² + b²)/A**, lệch pha Δφ = atan2(b, a)
(đổi ra px: Δφ·λ/2π). λ_c(t): λ nhỏ nhất mà từ đó trở lên MTF ≥ t, nội suy tuyến tính theo log λ.

Lý thuyết top-hat (tương quan chéo 1 lượt, cửa sổ W): **MTF(λ) = |sinc(W/λ)| = |sin(πW/λ)/(πW/λ)|**,
λ_c(0.9) = 3.85·W, λ_c(0.5) = 1.66·W.

## 3. Hàm truyền H(k)

1. **Từ sin**: `H(k) = MTF(λ = 1/k)` tại các nút λ ∈ {4, …, 128} px, nội suy tuyến tính theo log k;
   k < 1/128: giữ giá trị nút đầu (≈ 1); k > 1/4: H = 0 (không tin vùng chưa đo).
   `H_L` = mode dọc, `H_T` = mode ngang.
2. **Từ phổ chéo** trên tập huấn luyện có trường gốc:
   `H_xs(k) = ⟨S_est,true(k)⟩ / ⟨S_true,true(k)⟩` (trung bình tập hợp trước khi chia).
   Với phương pháp tuyến tính, dịch bất biến: û_est = H û_true + n̂ với n không tương quan với u_true
   ⇒ H_xs là ước lượng không chệch của H (đây là bộ lọc Wiener "tốt nhất" theo nghĩa bình phương tối thiểu).

## 4. Mô hình phổ đo và sàn nhiễu

    E_meas(k) = |H(k)|² · E_true(k) + N(k)

* (a) **đuôi phẳng**: `N_tail` = trung vị E_meas trên k ∈ [0.35, 0.5] chu kỳ/px (hằng số).
* (b) **dịch chuyển 0**: `N_zero(k)` = phổ của trường ước lượng trên cặp ảnh có cùng vị trí hạt, nhiễu ảnh
  độc lập (cùng ppp, SNR); trung bình trên `n_zero` cặp. Với SNR = ∞, hai ảnh trùng nhau ⇒ N_zero ≡ 0.
* (c) **tập huấn luyện**: `N_xs(k) = max(E_meas − H_xs² E_true, 0)` (trung bình tập hợp).

## 5. Hiệu chỉnh

    nghịch đảo :  E_corr(k) = max(E_meas − N, 0) / |H|²
    Wiener     :  E_corr(k) = max(E_meas − N, 0) · |H|² / ( |H|⁴ + ε (1 − |H|²)² )

chỉ áp tại các k có **|H(k)| > h_min**; nơi khác E_corr = NaN (không báo cáo).

* Số hạng ổn định `ε(1 − |H|²)²` bằng 0 khi |H| = 1 ⇒ **H ≡ 1, N ≡ 0 trả về đúng phổ gốc** với mọi ε
  (được kiểm tra trong `tests/test_resolution.py` và `matlab/p1/tests/test_p1_matlab.m`).
* Khi |H| → 0: E_corr → (E_meas − N)|H|²/ε → 0, tức không khuếch đại nhiễu (khác nghịch đảo: ∝ 1/|H|²).
* Wiener ≤ nghịch đảo tại mọi k (thiên về đánh giá thấp ở vùng |H| nhỏ).
* Mặc định: ε = 0.01, h_min = 0.2 (khuếch đại tối đa của nghịch đảo 1/h_min² = 25).

## 6. Tiêu chí đánh giá

`k₁₀` = số sóng lớn nhất k* sao cho |E(k)/E_true(k) − 1| < 10 % với **mọi** bin k ≤ k* (liên tục từ k nhỏ
nhất; bin NaN coi như không đạt). Giới hạn trên: dải có năng lượng của trường gốc (rối tổng hợp:
k ≤ 0.25 chu kỳ/px; DNS 2D: k ≤ 1/3). Tỉ lệ cải thiện = k₁₀(E_corr)/k₁₀(E_meas). Báo cáo kèm λ₁₀ = 1/k₁₀.

## 7. Giới hạn của mô hình

* Mô hình tuyến tính–dịch bất biến. Optical flow có điều chuẩn, tương quan chéo nhiều lượt có biến dạng
  cửa sổ và mạng nơ-ron **không** tuyến tính: H phụ thuộc biên độ, gradient và nội dung phổ của trường.
  Vì vậy H đo trên sin (một mode, biên độ cố định) có thể không chuyển sang trường rối — xem mục kiểm tra chéo
  trong `reports/P1_report.md`.
* Trường không nén 2D chỉ chứa mode ngang (û(k) ⊥ k), nên H_L (sin nén) là mode sai cho rối không nén;
  H_T phù hợp hơn về mặt vật lý.
* N(k) phụ thuộc trường (gradient làm tăng sai số ngẫu nhiên); N_zero đánh giá thấp nhiễu khi có gradient.
