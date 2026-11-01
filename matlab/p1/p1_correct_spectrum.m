function Ec = p1_correct_spectrum(k, Em, H, N, mode, eps_, h_min)
%P1_CORRECT_SPECTRUM  Hiệu chỉnh phổ đo bằng hàm truyền (giống pivbench.resolution.correct_spectrum).
%   Ec = p1_correct_spectrum(k, Em, H, N, mode, eps, h_min)
%     H     : mảng cùng cỡ k (hoặc vô hướng), N : sàn nhiễu (vô hướng hoặc mảng)
%     mode  : 'wiener' (mặc định) hoặc 'inverse'
%   inverse: Ec = max(Em − N, 0) / |H|²
%   wiener : Ec = max(Em − N, 0) · |H|² / (|H|⁴ + ε(1 − |H|²)²)
%   Ec = NaN tại các k có |H| ≤ h_min.  H ≡ 1, N ≡ 0 ⇒ Ec = Em.
if nargin < 4 || isempty(N), N = 0; end
if nargin < 5 || isempty(mode), mode = 'wiener'; end
if nargin < 6 || isempty(eps_), eps_ = 1e-2; end
if nargin < 7 || isempty(h_min), h_min = 0.2; end
Hk = abs(H .* ones(size(k)));
S = max(Em - N, 0);
H2 = Hk.^2;
switch lower(mode)
    case 'inverse'
        Ec = S ./ H2;
    case 'wiener'
        Ec = S .* H2 ./ (H2.^2 + eps_ * (1 - H2).^2);
    otherwise
        error('p1:mode', 'mode không biết: %s', mode);
end
Ec(~(Hk > h_min)) = NaN;
end
