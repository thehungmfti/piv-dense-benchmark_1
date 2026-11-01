function kv = p1_valid_k(k, E, Et, tol, k_max, k_min)
%P1_VALID_K  Số sóng lớn nhất k* mà |E/E_true − 1| < tol với mọi k ≤ k* (liên tục từ k nhỏ nhất).
%   kv = p1_valid_k(k, E, Et, tol, k_max, k_min) — giống pivbench.resolution.valid_wavenumber.
if nargin < 4 || isempty(tol), tol = 0.1; end
if nargin < 5 || isempty(k_max), k_max = Inf; end
if nargin < 6 || isempty(k_min), k_min = -Inf; end
sel = isfinite(Et) & Et > 0 & k <= k_max & k >= k_min;
kk = k(sel); r = E(sel) ./ Et(sel);
good = isfinite(r) & abs(r - 1) < tol;
if isempty(good) || ~good(1), kv = 0; return; end
b = find(~good, 1);
if isempty(b), kv = kk(end); else, kv = kk(b - 1); end
end
