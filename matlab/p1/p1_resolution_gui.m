function fig = p1_resolution_gui(mtfDir, specDir)
%P1_RESOLUTION_GUI  Giao diện MATLAB cho nghiên cứu P1: độ phân giải hiệu dụng (MTF) và hiệu chỉnh phổ.
%   p1_resolution_gui                       mở với results/p1_mtf_quick và results/p1_spectral_quick
%   p1_resolution_gui(mtfDir, specDir)      chỉ định thư mục kết quả
%
%   Yêu cầu MATLAB R2020b+ (uifigure/uigridlayout). Phần phân tích/hiệu chỉnh chạy THUẦN MATLAB trên các
%   file CSV do Python sinh ra (scripts/mtf_sweep.py, scripts/spectral_correction.py); chỉ các nút
%   "Chạy … (Python)" mới gọi Python (đường dẫn Python lấy từ pivbench_config, xem README_MATLAB.md).
%
%   Tab:
%     MTF(λ)            đường MTF theo phương pháp, chọn A / ppp / SNR, chồng sinc top-hat lý thuyết
%     Bản đồ nhiệt λ_c  λ_c(ppp, SNR) tại MTF = 0.9 hoặc 0.5
%     Mô hình λ_c       khớp λ_c = a·ppp^b·SNR^c·s^d (hồi quy log) và λ_c/ℓ_p, ℓ_p = 1/√ppp
%     Hàm truyền H(k)   H_L, H_T (sin), H_xs (phổ chéo trên rối) và sàn nhiễu
%     Hiệu chỉnh phổ    E_true, E_meas, E_corr với nguồn H, N, ε, ngưỡng |H| tuỳ chọn; k₁₀ trước/sau
%   Nút "Áp lên trường của tôi" hiệu chỉnh phổ của một trường đo (.mat chứa u, v) bằng H đã chọn.

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));                 % matlab/ (pivbench_config, pivbench_spectra)
root = fileparts(fileparts(here));
if nargin < 1 || isempty(mtfDir), mtfDir = fullfile(root, 'results', 'p1_mtf_quick'); end
if nargin < 2 || isempty(specDir), specDir = fullfile(root, 'results', 'p1_spectral_quick'); end

fig = uifigure('Name', 'P1 — Độ phân giải hiệu dụng & hiệu chỉnh phổ (PIV dày đặc)', ...
               'Position', [50 50 1400 840]);
S = struct('root', root, 'mtfDir', mtfDir, 'specDir', specDir, 'mtf', [], 'cut', [], ...
           'tr', [], 'sp', [], 'k10', [], 'user', []);
setappdata(fig, 'S', S);

g = uigridlayout(fig, [1 2]);
g.ColumnWidth = {350, '1x'};
c = uigridlayout(g, [30 2]);
c.RowHeight = repmat({23}, 1, 30);
c.RowHeight{30} = '1x';
c.ColumnWidth = {150, '1x'};
c.Scrollable = 'on';

r = 1;
hdr(c, r, '1) Dữ liệu kết quả'); r = r + 1;
H.btnMtfDir = uibutton(c, 'Text', 'Thư mục MTF…', 'ButtonPushedFcn', @(~, ~) onPickDir(fig, 'mtf')); place(H.btnMtfDir, r, 1);
H.btnSpecDir = uibutton(c, 'Text', 'Thư mục phổ…', 'ButtonPushedFcn', @(~, ~) onPickDir(fig, 'spec')); place(H.btnSpecDir, r, 2); r = r + 1;
H.btnRunMtf = uibutton(c, 'Text', 'Chạy sweep --quick (Python)', 'ButtonPushedFcn', @(~, ~) onRunPython(fig, 'mtf')); place(H.btnRunMtf, r, [1 2]); r = r + 1;
H.btnRunSpec = uibutton(c, 'Text', 'Chạy hiệu chỉnh phổ --quick (Python)', 'ButtonPushedFcn', @(~, ~) onRunPython(fig, 'spec')); place(H.btnRunSpec, r, [1 2]); r = r + 1;

hdr(c, r, '2) MTF / λ_c'); r = r + 1;
lbl(c, r, 'Biên độ A [px]'); H.A = uidropdown(c, 'Items', {'-'}, 'ValueChangedFcn', @(~, ~) refreshMTF(fig)); place(H.A, r, 2); r = r + 1;
lbl(c, r, 'ppp'); H.ppp = uidropdown(c, 'Items', {'-'}, 'ValueChangedFcn', @(~, ~) refreshMTF(fig)); place(H.ppp, r, 2); r = r + 1;
lbl(c, r, 'SNR'); H.snr = uidropdown(c, 'Items', {'-'}, 'ValueChangedFcn', @(~, ~) refreshMTF(fig)); place(H.snr, r, 2); r = r + 1;
lbl(c, r, 'Ngưỡng MTF'); H.thr = uidropdown(c, 'Items', {'0.5', '0.9'}, 'ValueChangedFcn', @(~, ~) refreshMTF(fig)); place(H.thr, r, 2); r = r + 1;
lbl(c, r, 'Phương pháp (chọn nhiều)');
H.cfgs = uilistbox(c, 'Items', {'-'}, 'Multiselect', 'on', 'ValueChangedFcn', @(~, ~) refreshMTF(fig));
place(H.cfgs, [r r + 2], 2); r = r + 3;
H.sinc = uicheckbox(c, 'Text', 'Chồng sinc top-hat (W = 16, 32)', 'Value', true, 'ValueChangedFcn', @(~, ~) refreshMTF(fig)); place(H.sinc, r, [1 2]); r = r + 1;
H.btnFit = uibutton(c, 'Text', 'Khớp mô hình λ_c', 'ButtonPushedFcn', @(~, ~) onFit(fig)); place(H.btnFit, r, [1 2]); r = r + 1;

hdr(c, r, '3) Hiệu chỉnh phổ'); r = r + 1;
lbl(c, r, 'Trường kiểm tra'); H.field = uidropdown(c, 'Items', {'turb', 'dns'}, 'ValueChangedFcn', @(~, ~) refreshSpec(fig)); place(H.field, r, 2); r = r + 1;
lbl(c, r, 'Phương pháp'); H.scfg = uidropdown(c, 'Items', {'-'}, 'ValueChangedFcn', @(~, ~) refreshSpec(fig)); place(H.scfg, r, 2); r = r + 1;
lbl(c, r, 'ppp / SNR'); H.scase = uidropdown(c, 'Items', {'-'}, 'ValueChangedFcn', @(~, ~) refreshSpec(fig)); place(H.scase, r, 2); r = r + 1;
lbl(c, r, 'Nguồn H(k)'); H.hsrc = uidropdown(c, 'Items', {'H_xs_turb_train', 'H_sine_T', 'H_sine_L', 'identity'}, ...
    'ValueChangedFcn', @(~, ~) refreshSpec(fig)); place(H.hsrc, r, 2); r = r + 1;
lbl(c, r, 'Sàn nhiễu N(k)'); H.nsrc = uidropdown(c, 'Items', {'N_xs', 'N_zero', 'tail', 'none'}, ...
    'ValueChangedFcn', @(~, ~) refreshSpec(fig)); place(H.nsrc, r, 2); r = r + 1;
lbl(c, r, 'Kiểu'); H.mode = uidropdown(c, 'Items', {'wiener', 'inverse'}, 'ValueChangedFcn', @(~, ~) refreshSpec(fig)); place(H.mode, r, 2); r = r + 1;
lbl(c, r, 'ε / ngưỡng |H| / tol'); H.par = uieditfield(c, 'text', 'Value', '0.01 / 0.2 / 0.1', ...
    'ValueChangedFcn', @(~, ~) refreshSpec(fig)); place(H.par, r, 2); r = r + 1;
H.btnUser = uibutton(c, 'Text', 'Áp lên trường của tôi (.mat u, v)…', 'ButtonPushedFcn', @(~, ~) onUserField(fig)); place(H.btnUser, r, [1 2]); r = r + 1;
H.btnExport = uibutton(c, 'Text', 'Xuất phổ đang xem (.mat/.csv)…', 'ButtonPushedFcn', @(~, ~) onExport(fig)); place(H.btnExport, r, [1 2]); r = r + 1;
H.btnFigs = uibutton(c, 'Text', 'Lưu hình PNG các tab…', 'ButtonPushedFcn', @(~, ~) onSaveFigs(fig)); place(H.btnFigs, r, [1 2]); r = r + 1;
H.log = uitextarea(c, 'Editable', 'off', 'Value', {'Sẵn sàng.'}); place(H.log, r, [1 2]);

d = uigridlayout(g, [2 1]);
d.RowHeight = {'2x', '1x'};
tg = uitabgroup(d);
names = {'MTF(λ)', 'Bản đồ nhiệt λ_c', 'Mô hình λ_c', 'Hàm truyền H(k)', 'Hiệu chỉnh phổ', 'E/E_true'};
for i = 1:numel(names)
    t = uitab(tg, 'Title', names{i});
    gl = uigridlayout(t, [1 1]);
    H.ax(i) = uiaxes(gl); %#ok<AGROW>
end
H.tabs = tg;
H.table = uitable(d);
setappdata(fig, 'H', H);
loadAll(fig);
end

% =====================================================================
% Tiện ích giao diện
% =====================================================================
function place(h, row, col)
h.Layout.Row = row;
h.Layout.Column = col;
end

function lbl(c, row, txt)
place(uilabel(c, 'Text', txt), row, 1);
end

function hdr(c, row, txt)
l = uilabel(c, 'Text', txt, 'FontWeight', 'bold');
place(l, row, [1 2]);
end

function logmsg(fig, msg)
H = getappdata(fig, 'H');
H.log.Value = [H.log.Value(:); {[datestr(now, 'HH:MM:SS') '  ' msg]}];
try, scroll(H.log, 'bottom'); catch, end
drawnow;
end

function s = snrstr(x)
if isinf(x), s = 'Inf'; else, s = sprintf('%g', x); end
end

function x = str2snr(s)
if strcmpi(s, 'inf'), x = Inf; else, x = str2double(s); end
end

function T = readcsv(f)
opts = detectImportOptions(f);
T = readtable(f, opts);
% cột chữ "inf" → số
for v = {'snr'}
    if ismember(v{1}, T.Properties.VariableNames) && ~isnumeric(T.(v{1}))
        T.(v{1}) = str2double(string(T.(v{1})));
    end
end
end

% =====================================================================
% Nạp dữ liệu
% =====================================================================
function loadAll(fig)
S = getappdata(fig, 'S');
H = getappdata(fig, 'H');
f = fullfile(S.mtfDir, 'mtf_long.csv');
if exist(f, 'file')
    S.mtf = readcsv(f);
    S.mtf = S.mtf(strcmp(S.mtf.status, 'ok'), :);
    S.cut = buildCutoffs(S.mtf);
    H.A.Items = arrayfun(@(x) sprintf('%g', x), unique(S.mtf.A), 'UniformOutput', false);
    pp = unique(S.mtf.ppp);
    H.ppp.Items = arrayfun(@(x) sprintf('%g', x), pp, 'UniformOutput', false);
    H.ppp.Value = H.ppp.Items{min(numel(pp), max(1, find(pp >= 0.05, 1)))};
    H.snr.Items = arrayfun(@snrstr, sort(unique(S.mtf.snr), 'descend'), 'UniformOutput', false);
    cf = unique(S.mtf.method_cfg);
    H.cfgs.Items = cf(:).';
    H.cfgs.Value = cf(:).';
    logmsg(fig, sprintf('MTF: %d hàng từ %s', height(S.mtf), f));
else
    logmsg(fig, ['Chưa có ' f ' — chạy sweep (nút Python) hoặc chọn thư mục khác.']);
end
f1 = fullfile(S.specDir, 'transfer.csv');
f2 = fullfile(S.specDir, 'spectra_long.csv');
if exist(f1, 'file') && exist(f2, 'file')
    S.tr = readcsv(f1);
    S.sp = readcsv(f2);
    if exist(fullfile(S.specDir, 'k10.csv'), 'file'), S.k10 = readcsv(fullfile(S.specDir, 'k10.csv')); end
    H.scfg.Items = unique(S.tr.cfg).';
    cs = unique(S.tr(:, {'ppp', 'snr'}), 'rows');
    H.scase.Items = arrayfun(@(i) sprintf('%g / %s', cs.ppp(i), snrstr(cs.snr(i))), 1:height(cs), 'UniformOutput', false);
    logmsg(fig, sprintf('Phổ: %d hàng H(k), %d hàng phổ từ %s', height(S.tr), height(S.sp), S.specDir));
else
    logmsg(fig, ['Chưa có transfer.csv/spectra_long.csv trong ' S.specDir]);
end
setappdata(fig, 'S', S);
setappdata(fig, 'H', H);
refreshMTF(fig);
refreshSpec(fig);
end

function C = buildCutoffs(T)
% λ_c (MTF trung bình theo seed) cho mỗi (cfg, A, ppp, snr) — thuần MATLAB
[G, cfg, A, ppp, snr] = findgroups(T.method_cfg, T.A, T.ppp, T.snr);
n = max(G);
lc9 = nan(n, 1); lc5 = nan(n, 1); ls = nan(n, 1);
for i = 1:n
    sub = T(G == i, :);
    [lam, ~, j] = unique(sub.wavelength);
    m = accumarray(j, sub.mtf, [], @mean);
    lc9(i) = p1_cutoff(lam, m, 0.9);
    lc5(i) = p1_cutoff(lam, m, 0.5);
    if ismember('lambda_scale', sub.Properties.VariableNames), ls(i) = sub.lambda_scale(1); end
end
C = table(cfg, A, ppp, snr, ls, lc9, lc5, 'VariableNames', {'cfg', 'A', 'ppp', 'snr', 'lambda_scale', 'lc_0_9', 'lc_0_5'});
C.l_p = 1 ./ sqrt(C.ppp);
end

% =====================================================================
% MTF, bản đồ nhiệt
% =====================================================================
function refreshMTF(fig)
S = getappdata(fig, 'S');
H = getappdata(fig, 'H');
if isempty(S.mtf), return; end
A = str2double(H.A.Value); ppp = str2double(H.ppp.Value); snr = str2snr(H.snr.Value);
thr = str2double(H.thr.Value);
cfgs = H.cfgs.Value;
if ischar(cfgs), cfgs = {cfgs}; end
ax = H.ax(1); cla(ax); hold(ax, 'on');
cols = lines(max(1, numel(cfgs)));
rows = {};
for i = 1:numel(cfgs)
    sub = S.mtf(strcmp(S.mtf.method_cfg, cfgs{i}) & S.mtf.A == A & S.mtf.ppp == ppp & S.mtf.snr == snr, :);
    if isempty(sub), continue; end
    [lam, ~, j] = unique(sub.wavelength);
    m = accumarray(j, sub.mtf, [], @mean);
    sd = accumarray(j, sub.mtf, [], @std);
    errorbar(ax, lam, m, sd, '-o', 'Color', cols(i, :), 'MarkerSize', 4, 'DisplayName', cfgs{i});
    ph = accumarray(j, sub.phase_err_px, [], @mean);
    rows(end + 1, :) = {cfgs{i}, p1_cutoff(lam, m, 0.9), p1_cutoff(lam, m, 0.5), mean(sub.time_s), ...
                        ph(end)}; %#ok<AGROW>
end
if H.sinc.Value
    lf = logspace(log10(4), log10(128), 200);
    plot(ax, lf, p1_tophat_mtf(lf, 32), ':', 'Color', [0.8 0 0], 'DisplayName', 'sinc W=32');
    plot(ax, lf, p1_tophat_mtf(lf, 16), ':', 'Color', [1 0.5 0], 'DisplayName', 'sinc W=16');
end
yline(ax, 0.9, 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
yline(ax, 0.5, 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
set(ax, 'XScale', 'log', 'XTick', [4 8 16 32 64 128]);
ylim(ax, [-0.05 1.25]); grid(ax, 'on'); box(ax, 'on');
xlabel(ax, '\lambda [px]'); ylabel(ax, 'MTF');
title(ax, sprintf('A = %g px, ppp = %g, SNR = %s', A, ppp, snrstr(snr)));
legend(ax, 'Location', 'southeast');
hold(ax, 'off');
H.table.ColumnName = {'cấu hình', 'λ_c(0.9) [px]', 'λ_c(0.5) [px]', 'thời gian TB [s]', 'lệch pha @λmax [px]'};
H.table.Data = rows;

% Bản đồ nhiệt: cấu hình đầu tiên được chọn
ax = H.ax(2); cla(ax);
if isempty(cfgs), return; end
C = S.cut(strcmp(S.cut.cfg, cfgs{1}) & S.cut.A == A, :);
pp = unique(C.ppp); ss = sort(unique(C.snr), 'descend');
M = nan(numel(ss), numel(pp));
for i = 1:numel(ss)
    for j = 1:numel(pp)
        x = C(C.snr == ss(i) & C.ppp == pp(j), :);
        if ~isempty(x)
            if thr == 0.9, M(i, j) = x.lc_0_9(1); else, M(i, j) = x.lc_0_5(1); end
        end
    end
end
imagesc(ax, M, 'AlphaData', ~isnan(M));
colormap(ax, flipud(hot)); cb = colorbar(ax); cb.Label.String = sprintf('\\lambda_c(MTF=%g) [px]', thr);
for i = 1:numel(ss)
    for j = 1:numel(pp)
        if isnan(M(i, j)), s = '–'; else, s = sprintf('%.1f', M(i, j)); end
        text(ax, j, i, s, 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end
ax.XTick = 1:numel(pp); ax.XTickLabel = arrayfun(@(x) sprintf('%g', x), pp, 'UniformOutput', false);
ax.YTick = 1:numel(ss); ax.YTickLabel = arrayfun(@snrstr, ss, 'UniformOutput', false);
xlabel(ax, 'ppp'); ylabel(ax, 'SNR'); axis(ax, 'tight');
title(ax, sprintf('%s, A = %g px — λ_c tại MTF = %g ("–" = không đạt)', cfgs{1}, A, thr), 'Interpreter', 'none');
end

function onFit(fig)
S = getappdata(fig, 'S');
H = getappdata(fig, 'H');
if isempty(S.cut), return; end
thr = str2double(H.thr.Value);
A = str2double(H.A.Value);
C = S.cut(S.cut.A == A, :);
if thr == 0.9, y = C.lc_0_9; else, y = C.lc_0_5; end
fams = {'ofpiv', 'cc32', 'cc16', 'raftpiv'};
ax = H.ax(3); cla(ax); hold(ax, 'on');
rows = {};
cols = lines(numel(fams));
for i = 1:numel(fams)
    sel = startsWith(C.cfg, fams{i}) & isfinite(C.snr);
    if ~any(sel), continue; end
    if strcmp(fams{i}, 'ofpiv')
        X = [C.ppp(sel), C.snr(sel), C.lambda_scale(sel)]; nm = {'ppp', 'SNR', 's'};
    else
        X = [C.ppp(sel), C.snr(sel)]; nm = {'ppp', 'SNR'};
    end
    F = p1_fit_powerlaw(y(sel), X, nm);
    ys = y(sel);
    if F.n > 0
        scatter(ax, ys(F.ok), F.pred, 25, cols(i, :), 'filled', 'DisplayName', fams{i});
    end
    b = [F.b, nan(1, 3 - numel(F.b))];
    % λ_c/ℓ_p (mọi SNR, kể cả ∞)
    sel2 = startsWith(C.cfg, fams{i}) & isfinite(y);
    ratio = y(sel2) ./ C.l_p(sel2);
    rows(end + 1, :) = {fams{i}, F.a, b(1), b(2), b(3), F.r2_log, F.r2_lin, F.n, mean(ratio), std(ratio) / mean(ratio)}; %#ok<AGROW>
end
mx = max(y(isfinite(y)));
if ~isempty(mx), plot(ax, [1 mx], [1 mx], 'k--', 'HandleVisibility', 'off'); end
set(ax, 'XScale', 'log', 'YScale', 'log'); grid(ax, 'on'); box(ax, 'on');
xlabel(ax, '\lambda_c đo [px]'); ylabel(ax, '\lambda_c mô hình [px]');
title(ax, sprintf('λ_c = a·ppp^b·SNR^c(·s^d), MTF = %g, A = %g (SNR hữu hạn)', thr, A));
legend(ax, 'Location', 'northwest'); hold(ax, 'off');
H.table.ColumnName = {'họ', 'a', 'b (ppp)', 'c (SNR)', 'd (s)', 'R² log', 'R² tuyến tính', 'n', 'TB λ_c/ℓ_p', 'CV λ_c/ℓ_p'};
H.table.Data = rows;
H.tabs.SelectedTab = H.tabs.Children(3);
logmsg(fig, 'Đã khớp mô hình λ_c (thuần MATLAB, p1_fit_powerlaw).');
end

% =====================================================================
% Hàm truyền & hiệu chỉnh phổ
% =====================================================================
function [cfg, ppp, snr] = specSel(H)
cfg = H.scfg.Value;
p = strsplit(H.scase.Value, '/');
ppp = str2double(strtrim(p{1}));
snr = str2snr(strtrim(p{min(2, numel(p))}));
end

function [Hk, N] = pickHN(S, H, cfg, ppp, snr, k, Em)
T = S.tr(strcmp(S.tr.cfg, cfg) & S.tr.ppp == ppp & S.tr.snr == snr, :);
T = sortrows(T, 'k');
if strcmp(H.hsrc.Value, 'identity')
    Hk = ones(size(k));
else
    Hk = p1_transfer(k, T.k, T.(H.hsrc.Value), 0);
end
switch H.nsrc.Value
    case 'none', N = zeros(size(k));
    case 'tail', N = p1_noise_tail(k, Em);
    otherwise
        N = interp1(T.k, T.(H.nsrc.Value), k, 'linear', 'extrap');
        N = max(N, 0);
end
end

function refreshSpec(fig)
S = getappdata(fig, 'S');
H = getappdata(fig, 'H');
if isempty(S.tr) || isempty(S.sp), return; end
[cfg, ppp, snr] = specSel(H);
par = str2num(strrep(H.par.Value, '/', ' ')); %#ok<ST2NM>
par = [par, 0.01, 0.2, 0.1]; eps_ = par(1); hmin = par(2); tol = par(3);

% --- Hàm truyền
T = sortrows(S.tr(strcmp(S.tr.cfg, cfg) & S.tr.ppp == ppp & S.tr.snr == snr, :), 'k');
ax = H.ax(4); cla(ax); hold(ax, 'on');
nm = {'H_sine_L', 'H_sine_T', 'H_xs_turb_train', 'H_xs_turb_test', 'H_xs_dns_test'};
st = {'-', '-', '-', '--', '--'};
for i = 1:numel(nm)
    if ismember(nm{i}, T.Properties.VariableNames)
        plot(ax, T.k, T.(nm{i}), st{i}, 'LineWidth', 1.2, 'DisplayName', strrep(nm{i}, '_', '\_'));
    end
end
yline(ax, hmin, ':', 'ngưỡng |H|', 'HandleVisibility', 'off');
set(ax, 'XScale', 'log'); ylim(ax, [-0.2 1.3]); grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'k [chu kỳ/px]'); ylabel(ax, 'H(k)');
title(ax, sprintf('%s, ppp = %g, SNR = %s', cfg, ppp, snrstr(snr)), 'Interpreter', 'none');
legend(ax, 'Location', 'southwest'); hold(ax, 'off');

% --- Phổ
P = S.sp(strcmp(S.sp.kind, H.field.Value) & strcmp(S.sp.role, 'test') & strcmp(S.sp.cfg, cfg) & ...
         S.sp.ppp == ppp & S.sp.snr == snr, :);
P = sortrows(P, 'k');
if isempty(P), logmsg(fig, 'Không có phổ cho lựa chọn này.'); return; end
k = P.k; Et = P.E_true; Em = P.E_meas;
[Hk, N] = pickHN(S, H, cfg, ppp, snr, k, Em);
Ec = p1_correct_spectrum(k, Em, Hk, N, H.mode.Value, eps_, hmin);
if strcmp(H.field.Value, 'turb'), kmax = 0.25; else, kmax = 1 / 3; end
kmax = min(kmax, max(k(Et > 1e-12 * max(Et))));
k0 = p1_valid_k(k, Em, Et, tol, kmax);
k1 = p1_valid_k(k, Ec, Et, tol, kmax);
ax = H.ax(5); cla(ax);
loglog(ax, k, Et, 'k-', 'LineWidth', 2, 'DisplayName', 'E_{true}'); hold(ax, 'on');
loglog(ax, k, Em, '-', 'Color', [0.85 0.2 0.1], 'LineWidth', 1.5, 'DisplayName', 'E_{meas}');
loglog(ax, k, Ec, '-', 'Color', [0 0.45 0.75], 'LineWidth', 1.5, 'DisplayName', 'E_{corr}');
loglog(ax, k, max(N, eps), ':', 'Color', [0.5 0.5 0.5], 'DisplayName', 'N(k)');
if k0 > 0, xline(ax, k0, '--', 'Color', [0.85 0.2 0.1], 'HandleVisibility', 'off'); end
if k1 > 0, xline(ax, k1, '--', 'Color', [0 0.45 0.75], 'HandleVisibility', 'off'); end
ylim(ax, [max(Et) * 1e-7, max(Et) * 3]); grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'k [chu kỳ/px]'); ylabel(ax, 'E(k)');
title(ax, sprintf('%s | %s + %s | k_{10}: %.3g → %.3g (λ: %.1f → %.1f px), ×%.2f', H.field.Value, ...
    strrep(H.hsrc.Value, '_', '\_'), strrep(H.nsrc.Value, '_', '\_'), k0, k1, 1 / k0, 1 / k1, k1 / k0));
legend(ax, 'Location', 'southwest'); hold(ax, 'off');
ax = H.ax(6); cla(ax);
semilogx(ax, k, Em ./ Et, '-', 'Color', [0.85 0.2 0.1], 'DisplayName', 'E_{meas}/E_{true}'); hold(ax, 'on');
semilogx(ax, k, Ec ./ Et, '-', 'Color', [0 0.45 0.75], 'DisplayName', 'E_{corr}/E_{true}');
patch(ax, [k(1) kmax kmax k(1)], [1 - tol 1 - tol 1 + tol 1 + tol], [0.5 0.5 0.5], 'FaceAlpha', 0.2, ...
      'EdgeColor', 'none', 'HandleVisibility', 'off');
xlim(ax, [k(1) kmax * 1.2]); ylim(ax, [0 2]); grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'k [chu kỳ/px]'); ylabel(ax, 'E / E_{true}'); legend(ax, 'Location', 'southwest'); hold(ax, 'off');
S.cur = struct('k', k, 'E_true', Et, 'E_meas', Em, 'E_corr', Ec, 'H', Hk, 'N', N, 'k10_meas', k0, 'k10_corr', k1);
setappdata(fig, 'S', S);
if ~isempty(S.k10)
    K = S.k10(strcmp(S.k10.cfg, cfg) & S.k10.ppp == ppp & S.k10.snr == snr & strcmp(S.k10.field, H.field.Value), :);
    H.table.ColumnName = {'trường', 'biến thể (Python)', 'k₁₀', 'λ₁₀ [px]', 'k₁₀ đo', 'cải thiện ×'};
    H.table.Data = [K.field, K.variant, num2cell([K.k10, K.lambda10, K.k10_meas, K.improvement])];
end
end

% =====================================================================
% Nút
% =====================================================================
function onPickDir(fig, which)
S = getappdata(fig, 'S');
if strcmp(which, 'mtf'), d0 = S.mtfDir; else, d0 = S.specDir; end
if ~exist(d0, 'dir'), d0 = fullfile(S.root, 'results'); end
d = uigetdir(d0, 'Chọn thư mục kết quả');
figure(fig);
if isequal(d, 0), return; end
if strcmp(which, 'mtf'), S.mtfDir = d; else, S.specDir = d; end
setappdata(fig, 'S', S);
loadAll(fig);
end

function onRunPython(fig, which)
S = getappdata(fig, 'S');
cfg = pivbench_config();
if strcmp(which, 'mtf')
    script = fullfile(S.root, 'scripts', 'mtf_sweep.py'); out = S.mtfDir;
else
    script = fullfile(S.root, 'scripts', 'spectral_correction.py'); out = S.specDir;
end
sel = uiconfirm(fig, sprintf(['Chạy %s --quick trên CPU (có thể mất 10–40 phút; MATLAB sẽ chờ).\n' ...
    'Kết quả ghi vào %s. Tiếp tục?'], script, out), 'Chạy Python', 'Options', {'Chạy', 'Huỷ'});
if ~strcmp(sel, 'Chạy'), return; end
cmd = sprintf('"%s" "%s" --quick --out "%s"', cfg.python, script, out);
if ispc, cmd = ['"' cmd '"']; end
logmsg(fig, ['Đang chạy: ' cmd]);
d = uiprogressdlg(fig, 'Title', 'Python', 'Message', 'Đang chạy…', 'Indeterminate', 'on');
[st, txt] = system(cmd);
close(d);
lines_ = splitlines(string(txt));
logmsg(fig, sprintf('Python kết thúc (mã %d). %s', st, strjoin(lines_(max(1, end - 3):end), ' | ')));
loadAll(fig);
end

function onUserField(fig)
S = getappdata(fig, 'S');
H = getappdata(fig, 'H');
[f, p] = uigetfile('*.mat', 'Trường đo (.mat chứa u, v [px/Δt])');
figure(fig);
if isequal(f, 0), return; end
D = load(fullfile(p, f));
if ~isfield(D, 'u') || ~isfield(D, 'v')
    uialert(fig, 'File cần biến u và v (cùng kích thước, px/khoảng thời gian).', 'Thiếu dữ liệu'); return
end
u = double(D.u); v = double(D.v);
u(~isfinite(u)) = 0; v(~isfinite(v)) = 0;
b = 16;
Sp = pivbench_spectra(u, v, u, v, b);
k = Sp.k2d(:); Em = Sp.est.E2d(:);
[cfg, ppp, snr] = specSel(H);
par = str2num(strrep(H.par.Value, '/', ' ')); %#ok<ST2NM>
par = [par, 0.01, 0.2];
[Hk, N] = pickHN(S, H, cfg, ppp, snr, k, Em);
Ec = p1_correct_spectrum(k, Em, Hk, N, H.mode.Value, par(1), par(2));
ax = H.ax(5); cla(ax);
loglog(ax, k, Em, '-', 'Color', [0.85 0.2 0.1], 'LineWidth', 1.5, 'DisplayName', 'E_{meas} (trường của bạn)'); hold(ax, 'on');
loglog(ax, k, Ec, '-', 'Color', [0 0.45 0.75], 'LineWidth', 1.5, 'DisplayName', 'E_{corr}');
loglog(ax, k, max(N, eps), ':', 'Color', [0.5 0.5 0.5], 'DisplayName', 'N(k)');
if isfield(D, 'ut') && isfield(D, 'vt')
    St = pivbench_spectra(D.ut, D.vt, D.ut, D.vt, b);
    loglog(ax, St.k2d, St.est.E2d, 'k-', 'LineWidth', 2, 'DisplayName', 'E_{true}');
end
grid(ax, 'on'); xlabel(ax, 'k [chu kỳ/px]'); ylabel(ax, 'E(k)');
title(ax, sprintf('%s — H từ %s (%s, ppp %g, SNR %s)', f, strrep(H.hsrc.Value, '_', '\_'), cfg, ppp, snrstr(snr)), ...
      'Interpreter', 'tex');
legend(ax, 'Location', 'southwest'); hold(ax, 'off');
H.tabs.SelectedTab = H.tabs.Children(5);
S.cur = struct('k', k, 'E_meas', Em, 'E_corr', Ec, 'H', Hk, 'N', N, 'source', fullfile(p, f));
setappdata(fig, 'S', S);
logmsg(fig, ['Đã hiệu chỉnh phổ của ' f '. Lưu ý: H phải đo ở cùng phương pháp/ppp/SNR với ảnh của bạn.']);
end

function onExport(fig)
S = getappdata(fig, 'S');
if ~isfield(S, 'cur') || isempty(S.cur), uialert(fig, 'Chưa có phổ để xuất.', 'Xuất'); return; end
[f, p] = uiputfile({'*.mat'; '*.csv'}, 'Lưu phổ', 'p1_spectrum.mat');
figure(fig);
if isequal(f, 0), return; end
cur = S.cur;
[~, ~, ext] = fileparts(f);
if strcmpi(ext, '.csv')
    fn = fieldnames(cur); keep = {};
    for i = 1:numel(fn)
        if isnumeric(cur.(fn{i})) && numel(cur.(fn{i})) == numel(cur.k), keep{end + 1} = fn{i}; end %#ok<AGROW>
    end
    T = table();
    for i = 1:numel(keep), T.(keep{i}) = cur.(keep{i})(:); end
    writetable(T, fullfile(p, f));
else
    save(fullfile(p, f), '-struct', 'cur');
end
logmsg(fig, ['Đã lưu ' fullfile(p, f)]);
end

function onSaveFigs(fig)
H = getappdata(fig, 'H');
d = uigetdir(pwd, 'Thư mục lưu PNG');
figure(fig);
if isequal(d, 0), return; end
names = {'mtf', 'heatmap_lc', 'fit_lc', 'transfer', 'spectrum_corr', 'spectrum_ratio'};
for i = 1:numel(H.ax)
    try
        exportgraphics(H.ax(i), fullfile(d, ['p1_' names{i} '.png']), 'Resolution', 150);
    catch err
        logmsg(fig, ['Không lưu được ' names{i} ': ' err.message]);
    end
end
logmsg(fig, ['Đã lưu hình vào ' d]);
end
