function test_pivbench_matlab(refFile, withPython)
%TEST_PIVBENCH_MATLAB  Đối chiếu các hàm MATLAB với bản Python (chạy được cả trên GNU Octave).
%   test_pivbench_matlab()                    % tự sinh tham chiếu bằng Python
%   test_pivbench_matlab(refFile, withPython)
here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
cfg = pivbench_config();
if nargin < 1 || isempty(refFile)
    refFile = [tempname() '_ref.mat'];
    [st, out] = system(sprintf('"%s" "%s" "%s"', cfg.python, fullfile(here, 'make_reference.py'), refFile));
    assert(st == 0, out);
end
if nargin < 2, withPython = true; end
R = load(refFile);
nfail = 0;
M = pivbench_metrics(R.u, R.v, R.ut, R.vt, 8);
for f = {'epe_mean', 'epe_median', 'epe_p95', 'rmse_u', 'rmse_v', 'bias_u', 'bias_v', 'epe_in', 'epe_out'}
    nfail = nfail + check(f{1}, M.(f{1}), R.(f{1}), 1e-9);
end
S = pivbench_spectra(R.u, R.v, R.ut, R.vt, 8);
nfail = nfail + check('kx', S.kx, R.sp_kx, 1e-12);
nfail = nfail + check('k2d', S.k2d, R.sp_k2d, 1e-12);
tags = {'true', 'est', 'err'};
for t = 1:3
    for f = {'E_x', 'E_y', 'E2d'}
        nfail = nfail + check(sprintf('%s.%s', tags{t}, f{1}), S.(tags{t}).(f{1}), R.(sprintf('sp_%s_%s', tags{t}, f{1})), 1e-8);
    end
end
T = pivbench_mtf(R.su_est, R.sv, 0.7, 24, 'u', 'x', 0.3, 4);
nfail = nfail + check('mtf', T.mtf, R.mtf, 1e-9);
nfail = nfail + check('phase_error', T.phase_error, R.phase_error, 1e-9);
if withPython
    % cầu nối Python: sinh ảnh + ước lượng bằng tương quan chéo (nhanh)
    Sy = pivbench_synth('uniform', struct('u0', 1.3, 'v0', -0.6), [128 128], struct('ppp', 0.05, 'seed', 2));
    [u, v] = pivbench_estimate(uint8(255 * Sy.img1 / max(Sy.img1(:))), uint8(255 * Sy.img2 / max(Sy.img1(:))), ...
                               'crosscorr', struct('windows', [32 16], 'overlap', 0.5));
    Mb = pivbench_metrics(u, v, Sy.u_true, Sy.v_true, 12);
    fprintf('cầu nối crosscorr: EPE = %.4f px\n', Mb.epe_mean);
    nfail = nfail + (Mb.epe_mean > 0.08);
end
if nfail == 0
    fprintf('TẤT CẢ KIỂM TRA ĐỀU ĐẠT\n');
else
    error('%d kiểm tra thất bại', nfail);
end
end

function bad = check(name, a, b, tol)
a = a(:); b = b(:);
if numel(a) ~= numel(b)
    fprintf('FAIL %-14s kích thước %d vs %d\n', name, numel(a), numel(b)); bad = 1; return;
end
err = max(abs(a - b) ./ max(1, abs(b)));
bad = ~(err <= tol);
fprintf('%s %-14s sai lệch tương đối %.2e\n', ifelse(bad, 'FAIL', 'ok  '), name, err);
end

function s = ifelse(c, a, b)
if c, s = a; else, s = b; end
end
