function R = pivbench_mtf(u, v, amplitude, wavelength, component, axisName, phase, border)
%PIVBENCH_MTF  MTF tại đúng bước sóng của trường sin gốc A·sin(2π s/λ + φ).
%   R = pivbench_mtf(u, v, A, lambda)                         % u = A sin(2πx/λ)
%   R = pivbench_mtf(u, v, A, lambda, component, axisName, phase, border)
%   component 'u'|'v', axisName 'x'|'y'. Khớp bình phương tối thiểu a·sin + b·cos + c;
%   R.mtf = sqrt(a²+b²)/A, R.phase_error = atan2(b, a) [rad].
if nargin < 5 || isempty(component), component = 'u'; end
if nargin < 6 || isempty(axisName), axisName = 'x'; end
if nargin < 7 || isempty(phase), phase = 0; end
if nargin < 8 || isempty(border), border = 0; end
if component == 'u', f = u; else, f = v; end
[ny, nx] = size(f);
[X, Y] = meshgrid(0:nx-1, 0:ny-1);   % toạ độ 0-based như Python
if axisName == 'x', s = X; else, s = Y; end
th = 2 * pi * s / wavelength + phase;
c = @(a) a(border+1:end-border, border+1:end-border);
fc = c(f); tc = c(th);
ok = isfinite(fc);
A = [sin(tc(ok)), cos(tc(ok)), ones(nnz(ok), 1)];
coef = A \ fc(ok);
R.wavelength = wavelength;
R.mtf = hypot(coef(1), coef(2)) / amplitude;
R.phase_error = atan2(coef(2), coef(1));
end
