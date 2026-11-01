function F = p1_fit_powerlaw(y, X, names)
%P1_FIT_POWERLAW  Khớp y = a·Π X_i^b_i bằng hồi quy log (bình phương tối thiểu).
%   F = p1_fit_powerlaw(y, X, names) — y (n×1), X (n×p), names (1×p cell).
%   F.a, F.b (1×p), F.names, F.r2_log, F.r2_lin, F.n, F.pred (dự đoán tại các hàng hợp lệ), F.ok
y = y(:);
ok = isfinite(y) & y > 0 & all(isfinite(X) & X > 0, 2);
F.names = names; F.ok = ok; F.n = nnz(ok);
p = size(X, 2);
if F.n < p + 2
    F.a = NaN; F.b = nan(1, p); F.r2_log = NaN; F.r2_lin = NaN; F.pred = []; return
end
A = [ones(F.n, 1), log(X(ok, :))];
ly = log(y(ok));
c = A \ ly;
pl = A * c;
F.a = exp(c(1)); F.b = c(2:end).';
F.r2_log = 1 - sum((ly - pl).^2) / sum((ly - mean(ly)).^2);
yl = y(ok); pe = exp(pl);
F.r2_lin = 1 - sum((yl - pe).^2) / sum((yl - mean(yl)).^2);
F.pred = pe;
end
