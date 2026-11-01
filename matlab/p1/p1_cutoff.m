function lc = p1_cutoff(lambda, mtf, thr)
%P1_CUTOFF  Bước sóng cắt λ_c tại MTF = thr (giống pivbench.metrics.mtf.cutoff_wavelength).
%   Đi từ λ lớn xuống λ nhỏ, lấy điểm cắt đầu tiên, nội suy tuyến tính theo log λ.
%   NaN nếu MTF chưa bao giờ ≥ thr; λ_min nếu luôn ≥ thr.
lambda = lambda(:); mtf = mtf(:);
[lambda, o] = sort(lambda, 'descend'); mtf = mtf(o);
lc = NaN;
i0 = find(mtf >= thr, 1);
if isempty(i0), return; end
for i = i0:numel(lambda) - 1
    if mtf(i) >= thr && thr > mtf(i + 1)
        t = (mtf(i) - thr) / (mtf(i) - mtf(i + 1));
        lc = exp(log(lambda(i)) + t * (log(lambda(i + 1)) - log(lambda(i))));
        return
    end
end
lc = lambda(end);
end
