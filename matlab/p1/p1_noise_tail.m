function N = p1_noise_tail(k, Em, k_lo, k_hi)
%P1_NOISE_TAIL  Sàn nhiễu (a): hằng số = trung vị Em trên dải k ∈ [k_lo, k_hi] (mặc định 0.35–0.5).
if nargin < 3, k_lo = 0.35; end
if nargin < 4, k_hi = 0.5; end
sel = k >= k_lo & k <= k_hi & isfinite(Em);
if any(sel), N = median(Em(sel)) * ones(size(Em)); else, N = zeros(size(Em)); end
end
