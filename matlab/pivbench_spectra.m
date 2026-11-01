function S = pivbench_spectra(u, v, ut, vt, border, nperseg)
%PIVBENCH_SPECTRA  Phổ năng lượng của trường gốc, trường ước lượng và sai số (thuần MATLAB).
%   S = pivbench_spectra(u, v, ut, vt)
%   S = pivbench_spectra(u, v, ut, vt, border, nperseg)
%
%   Phổ 1D theo x (dọc hàng) và theo y (dọc cột): Welch, cửa sổ Hann (tuần hoàn), chồng 50 %,
%   trừ trung bình từng đoạn, mật độ một phía, trung bình trên mọi hàng/cột; E = ½(E_uu + E_vv).
%   Phổ 2D: FFT 2D với cửa sổ Hann 2D, trung bình theo vòng |k| (mật độ theo |k|).
%   Số sóng k tính bằng chu kỳ/px; bước sóng λ = 1/k.
%   Kết quả giống src/pivbench/metrics/spectra.py (đã đối chiếu số).
%
%   S.kx, S.ky, S.k2d và với tag ∈ {true, est, err}: S.<tag>.E_x, E_y, E2d, Euu_x, Evv_x, Euu_y, Evv_y
if nargin < 5 || isempty(border), border = 0; end
if nargin < 6, nperseg = []; end
c = @(a) a(border+1:end-border, border+1:end-border);
F = {c(ut), c(vt); c(u), c(v); c(u - ut), c(v - vt)};
tags = {'true', 'est', 'err'};
for t = 1:3
    a = F{t, 1}; b = F{t, 2};
    [kx, Euu_x] = welch_rows(a, nperseg);
    [~, Evv_x] = welch_rows(b, nperseg);
    [ky, Euu_y] = welch_rows(a.', nperseg);
    [~, Evv_y] = welch_rows(b.', nperseg);
    [k2, E2] = radial2d(a, b);
    R = struct('Euu_x', Euu_x, 'Evv_x', Evv_x, 'E_x', 0.5 * (Euu_x + Evv_x), ...
               'Euu_y', Euu_y, 'Evv_y', Evv_y, 'E_y', 0.5 * (Euu_y + Evv_y), 'E2d', E2);
    S.(tags{t}) = R;
end
S.kx = kx; S.ky = ky; S.k2d = k2;
end

function [k, P] = welch_rows(f, nperseg)
% Welch dọc mỗi hàng, trung bình trên các hàng
[nr, n] = size(f);
if isempty(nperseg), nperseg = n; end
L = min(n, nperseg);
step = L - floor(L / 2);
w = 0.5 - 0.5 * cos(2 * pi * (0:L-1) / L);          % Hann tuần hoàn (scipy 'hann')
starts = 1:step:(n - L + 1);
nf = floor(L / 2) + 1;
acc = zeros(1, nf);
for s = starts
    seg = f(:, s:s+L-1);
    seg = seg - mean(seg, 2);
    X = fft(seg .* w, [], 2);
    p = abs(X(:, 1:nf)).^2 / sum(w.^2);
    if mod(L, 2) == 0
        p(:, 2:end-1) = 2 * p(:, 2:end-1);
    else
        p(:, 2:end) = 2 * p(:, 2:end);
    end
    acc = acc + mean(p, 1);
end
P = acc / numel(starts);
k = (0:nf-1) / L;
if nr == 0, P = []; end
end

function [k, E] = radial2d(a, b)
[ny, nx] = size(a);
wy = 0.5 - 0.5 * cos(2 * pi * (0:ny-1)' / ny);
wx = 0.5 - 0.5 * cos(2 * pi * (0:nx-1) / nx);
win = wy * wx;
wn = mean(win(:).^2);
P = abs(fft2((a - mean(a(:))) .* win)).^2 / (nx * ny * wn) + ...
    abs(fft2((b - mean(b(:))) .* win)).^2 / (nx * ny * wn);
P = 0.5 * P;
fx = [0:ceil(nx/2)-1, -floor(nx/2):-1] / nx;
fy = [0:ceil(ny/2)-1, -floor(ny/2):-1]' / ny;
K = sqrt(fx.^2 + fy.^2);
dk = 1 / min(nx, ny);
nb = floor(min(nx, ny) / 2);
edges = ((0:nb) + 0.5) * dk;
idx = discretize_nt(K(:), edges);
ok = idx >= 1 & idx <= nb;
s = accumarray(idx(ok), P(ok), [nb 1]).';
E = s / (nx * ny) / dk;
k = 0.5 * (edges(1:end-1) + edges(2:end));
end

function idx = discretize_nt(x, edges)
% chỉ số khoảng [edges(i), edges(i+1)) như numpy.digitize − 1; ngoài khoảng → 0
idx = zeros(size(x));
for i = 1:numel(edges) - 1
    idx(x >= edges(i) & x < edges(i+1)) = i;
end
end
