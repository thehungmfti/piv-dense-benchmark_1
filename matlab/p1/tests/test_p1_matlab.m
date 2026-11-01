function test_p1_matlab()
%TEST_P1_MATLAB  Kiểm tra các hàm thuần MATLAB của P1 (chạy được cả trên GNU Octave ≥ 6).
%   cd matlab/p1/tests; test_p1_matlab
here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
n = 0;

% 1) H ≡ 1, N ≡ 0 ⇒ phổ gốc (Wiener với mọi ε, và nghịch đảo)
k = linspace(1/256, 0.5, 128).';
E = k.^(-5/3);
for eps_ = [0 1e-3 0.3]
    assert(max(abs(p1_correct_spectrum(k, E, 1, 0, 'wiener', eps_, 0.2) - E) ./ E) < 1e-14);
end
assert(max(abs(p1_correct_spectrum(k, E, ones(size(k)), 0, 'inverse') - E) ./ E) < 1e-14); n = n + 1;

% 2) Nghịch đảo đúng bộ lọc đã biết + trừ sàn nhiễu; NaN khi |H| ≤ ngưỡng
Hk = exp(-(k / 0.15).^2); N = 1e-2;
Ec = p1_correct_spectrum(k, Hk.^2 .* E + N, Hk, N, 'inverse', 0, 0.3);
ok = Hk > 0.3;
assert(max(abs(Ec(ok) - E(ok)) ./ E(ok)) < 1e-10 && all(isnan(Ec(~ok)))); n = n + 1;

% 3) MTF lý thuyết và λ_c
assert(abs(p1_tophat_mtf(32, 32)) < 1e-12 && abs(p1_tophat_mtf(32, 0) - 1) < 1e-12);
lam = [4 8 16 32 64]; m = [0 0.2 0.6 0.95 1];
t = (0.95 - 0.9) / (0.95 - 0.6);
assert(abs(p1_cutoff(lam, m, 0.9) - exp(log(32) + t * (log(16) - log(32)))) < 1e-12);
assert(isnan(p1_cutoff(lam, 0.1 * m, 0.5)) && p1_cutoff(lam, ones(1, 5), 0.9) == 4); n = n + 1;

% 4) Hàm truyền: nội suy log k, giữ đầu, 0 ở k cao
Hn = p1_transfer([1/16, 1/64, 1/512, 0.4, sqrt(1/16 * 1/32)], 1 ./ lam, [0 0.3 0.7 0.95 1]);
assert(all(abs(Hn - [0.7 1 1 0 0.825]) < 1e-12)); n = n + 1;

% 5) Khớp luỹ thừa
ppp = linspace(0.005, 0.1, 40).'; snr = linspace(5, 20, 40).'; snr = snr(randperm(40));
F = p1_fit_powerlaw(3 * ppp.^-0.5 .* snr.^-0.2, [ppp snr], {'ppp', 'SNR'});
assert(abs(F.a - 3) < 1e-8 && all(abs(F.b - [-0.5 -0.2]) < 1e-8) && abs(F.r2_log - 1) < 1e-12); n = n + 1;

% 6) k hợp lệ
E2 = E; E2(k > 0.2) = 0.5 * E2(k > 0.2);
kk = k(k <= 0.2);
assert(p1_valid_k(k, E, E, 0.1) == k(end) && p1_valid_k(k, E2, E, 0.1) == kk(end) && p1_valid_k(k, 0.5 * E, E) == 0); n = n + 1;

% 7) Sàn nhiễu đuôi
assert(all(abs(p1_noise_tail(k, 3e-3 * ones(size(k))) - 3e-3) < 1e-15)); n = n + 1;
fprintf('test_p1_matlab: %d nhóm kiểm tra đạt.\n', n);
end
