%DEMO_PIVBENCH  Ví dụ dùng pivbench từ MATLAB (không cần giao diện).
%   Sinh một cặp ảnh wake đáy tù có trường gốc, chạy 4 phương pháp, in bảng sai số,
%   vẽ trường vận tốc/sai số/phổ, đo MTF trên trường sin, lưu demo_pivbench_result.mat.
%   Chạy:  cd matlab; demo_pivbench
%
%   Ảnh của bạn: [u, v] = pivbench_estimate('anh_a.tif', 'anh_b.tif', 'raftpiv');

try
    doPlot = ~isempty(available_graphics_toolkits());   % GNU Octave
catch
    doPlot = usejava('desktop');                       % MATLAB (false khi chạy -batch/-nodisplay)
end
fprintf('== Kiểm tra môi trường ==\n');
pivbench_check();

%% 1) Cặp ảnh thử: wake sát đáy (D = 140 px, U∞ = 6 px), ppp 0.05, SNR 10
fp = struct('D', 140, 'U_inf', 6, 'x_base', 30, 'L_r', 1.3, 'U_rev', 0.3, ...
            'kh_strength', 0.6, 'street_strength', 0.3, 'turb_rms', 0.06, 'seed', 11);
T = pivbench_synth('wake', fp, [256 384], struct('ppp', 0.05, 'snr', 10, 'seed', 0));
fprintf('Ảnh %d×%d, %d hạt\n', size(T.img1, 1), size(T.img1, 2), T.meta.n_particles);

%% 2) Các phương pháp
methods = {'crosscorr', 'ofpiv', 'raftpiv', 'raftpiv_ls'};
res = struct('method', {}, 'u', {}, 'v', {}, 'runtime', {}, 'M', {});
fprintf('\n%-11s %8s %8s %8s %8s %8s %8s\n', 'phương pháp', 'EPE', 'trung vị', 'p95', 'EPE in', 'EPE out', 't [s]');
for i = 1:numel(methods)
    [u, v, info] = pivbench_estimate(T.img1, T.img2, methods{i});
    M = pivbench_metrics(u, v, T.u_true, T.v_true, 16, T.valid);
    res(end + 1) = struct('method', methods{i}, 'u', u, 'v', v, 'runtime', info.runtime_s, 'M', M); %#ok<SAGROW>
    fprintf('%-11s %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f\n', methods{i}, M.epe_mean, M.epe_median, ...
            M.epe_p95, M.epe_in, M.epe_out, info.runtime_s);
end

%% 3) MTF: u = 0.5 sin(2πx/λ)
lams = [16 32 64 128];
mtf = zeros(numel(methods), numel(lams));
for j = 1:numel(lams)
    Ts = pivbench_synth('sine', struct('amplitude', 0.5, 'wavelength', lams(j)), [256 256], struct('ppp', 0.05, 'seed', 1));
    for i = 1:numel(methods)
        [u, v] = pivbench_estimate(Ts.img1, Ts.img2, methods{i});
        R = pivbench_mtf(u, v, 0.5, lams(j), 'u', 'x', 0, 16);
        mtf(i, j) = R.mtf;
    end
end
fprintf('\nMTF (A = 0.5 px)   λ = %s px\n', mat2str(lams));
for i = 1:numel(methods)
    fprintf('%-11s %s\n', methods{i}, sprintf('%6.2f', mtf(i, :)));
end

save('demo_pivbench_result.mat', 'T', 'res', 'lams', 'mtf', 'methods');
fprintf('\nĐã lưu demo_pivbench_result.mat\n');

%% 4) Hình
if doPlot
    figure('Name', 'pivbench demo', 'Position', [50 50 1400 700]);
    n = numel(methods);
    subplot(2, n + 1, 1); imagesc(hypot(T.u_true, T.v_true)); axis image; colorbar; title('|u| gốc [px]');
    for i = 1:n
        subplot(2, n + 1, i + 1); imagesc(hypot(res(i).u, res(i).v)); axis image; colorbar;
        title(sprintf('|u| %s', res(i).method), 'Interpreter', 'none');
        subplot(2, n + 1, n + 2 + i); imagesc(hypot(res(i).u - T.u_true, res(i).v - T.v_true), [0 0.6]);
        axis image; colorbar; title(sprintf('EPE %s: %.3f px', res(i).method, res(i).M.epe_mean), 'Interpreter', 'none');
    end
    subplot(2, n + 1, n + 2);
    semilogx(lams, mtf', '-o', 'LineWidth', 1.5); grid on; ylim([0 1.15]);
    xlabel('λ [px]'); ylabel('MTF'); legend(methods, 'Interpreter', 'none', 'Location', 'southeast');
    figure('Name', 'Phổ năng lượng');
    for i = 1:n
        u = res(i).u; v = res(i).v; u(~T.valid) = T.u_true(~T.valid); v(~T.valid) = T.v_true(~T.valid);
        P = pivbench_spectra(u, v, T.u_true, T.v_true, 16);
        loglog(P.kx(2:end), P.est.E_x(2:end), 'LineWidth', 1.5); hold on;
        loglog(P.kx(2:end), P.err.E_x(2:end), '--', 'LineWidth', 1);
    end
    loglog(P.kx(2:end), P.true.E_x(2:end), 'k', 'LineWidth', 2); hold off; grid on;
    xlabel('k_x [chu kỳ/px]'); ylabel('E(k_x)');
    legend([reshape([methods; strcat(methods, ' sai số')], 1, []), {'gốc'}], 'Interpreter', 'none', 'Location', 'southwest');
end

