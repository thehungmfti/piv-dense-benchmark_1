function M = pivbench_metrics(u, v, ut, vt, border, valid, topPercent)
%PIVBENCH_METRICS  Sai số so với trường gốc (thuần MATLAB, không cần toolbox).
%   M = pivbench_metrics(u, v, ut, vt)
%   M = pivbench_metrics(u, v, ut, vt, border, valid, topPercent)
%
%   border     : bỏ border pixel mỗi cạnh (mặc định 0)
%   valid      : mặt nạ logic điểm được tính (mặc định tất cả; ví dụ ~thân mô hình)
%   topPercent : mặt nạ vùng xoáy mạnh = top topPercent % của |ω| trường gốc (mặc định 10)
%
%   M: epe_mean, epe_median, epe_p95, epe_max, rmse_u, rmse_v, rmse, bias_u, bias_v, n,
%      epe_in, epe_out (trong/ngoài vùng |ω| mạnh), omega_peak_true, omega_center_rel_error
%   EPE = sqrt((u-ut)^2 + (v-vt)^2) [px]. Điểm NaN bị bỏ qua.
if nargin < 5 || isempty(border), border = 0; end
if nargin < 6 || isempty(valid), valid = true(size(u)); end
if nargin < 7 || isempty(topPercent), topPercent = 10; end
[ny, nx] = size(u);
inner = false(ny, nx);
inner(border+1:ny-border, border+1:nx-border) = true;
du = u - ut;
dv = v - vt;
ok = inner & logical(valid) & isfinite(du) & isfinite(dv);
e = sqrt(du.^2 + dv.^2);
M = basic(e, du, dv, ok);
w = pivbench_vorticity(ut, vt);
aw = abs(w(inner));
thr = prctile_nt(aw(:), 100 - topPercent);
strong = abs(w) >= thr;
M.epe_in = mean(e(ok & strong));
M.epe_out = mean(e(ok & ~strong));
% đỉnh xoáy: ω ước lượng trung bình trên đĩa bán kính 2 px tại vị trí |ω_gốc| cực đại
we = pivbench_vorticity(u, v);
wi = abs(w);
wi(~inner) = -Inf;
[~, idx] = max(wi(:));
[iy, ix] = ind2sub(size(w), idx);
[X, Y] = meshgrid(1:nx, 1:ny);
disk = (X - ix).^2 + (Y - iy).^2 <= 4;
M.omega_peak_true = w(iy, ix);
M.omega_center_rel_error = mean(we(disk)) / mean(w(disk)) - 1;
end

function M = basic(e, du, dv, ok)
ee = e(ok); a = du(ok); b = dv(ok);
M.epe_mean = mean(ee);
M.epe_median = median(ee);
M.epe_p95 = prctile_nt(ee, 95);
M.epe_max = max(ee);
M.rmse_u = sqrt(mean(a.^2));
M.rmse_v = sqrt(mean(b.^2));
M.rmse = sqrt(mean(a.^2 + b.^2));
M.bias_u = mean(a);
M.bias_v = mean(b);
M.n = numel(ee);
end

function p = prctile_nt(x, q)
% phân vị kiểu nội suy tuyến tính (giống numpy.percentile mặc định), không cần Statistics Toolbox
x = sort(x(:));
n = numel(x);
if n == 0, p = NaN; return; end
pos = 1 + (n - 1) * q / 100;
lo = floor(pos); hi = min(lo + 1, n);
p = x(lo) + (pos - lo) * (x(hi) - x(lo));
end
