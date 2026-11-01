function m = p1_tophat_mtf(lambda, W)
%P1_TOPHAT_MTF  MTF lý thuyết của trung bình top-hat cỡ W (tương quan chéo 1 lượt): |sinc(W/λ)|.
%   m = p1_tophat_mtf(lambda, W)   — lambda [px] (mảng), W [px].
x = W ./ lambda;
m = ones(size(x));
nz = x ~= 0;
m(nz) = abs(sin(pi * x(nz)) ./ (pi * x(nz)));
end
