function H = p1_transfer(k, k_nodes, H_nodes, high_k_value)
%P1_TRANSFER  Đánh giá H(k) từ các nút (nội suy tuyến tính theo log k), như TransferFunction (Python).
%   H = p1_transfer(k, 1./lambda, mtf)       % H(k) = MTF(λ = 1/k), k [chu kỳ/px]
%   k < min(k_nodes): giữ giá trị nút đầu; k > max(k_nodes): high_k_value (mặc định 0).
if nargin < 4, high_k_value = 0; end
ok = isfinite(k_nodes(:)) & isfinite(H_nodes(:)) & k_nodes(:) > 0;
kn = k_nodes(ok); Hn = H_nodes(ok);
[kn, o] = sort(kn(:)); Hn = Hn(o);
H = interp1(log(kn), Hn(:), log(max(k, 1e-12)), 'linear', NaN);
H(k < kn(1)) = Hn(1);
H(k > kn(end)) = high_k_value;
if numel(kn) == 1, H(:) = Hn(1); end
end
