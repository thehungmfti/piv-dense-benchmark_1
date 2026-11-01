function fig = pivbench_gui()
%PIVBENCH_GUI  Giao diện MATLAB cho PIV dày đặc: RAFT-PIV (tái hiện), OpenOpticalFlow_PIV, tương quan chéo.
%   pivbench_gui    mở cửa sổ. Yêu cầu MATLAB R2020b+ (uifigure/uigridlayout) và Python của repo
%                   (xem pivbench_check, README_MATLAB.md).
%
%   Chức năng:
%     • Mở cặp ảnh (tif/bmp/png/jpg) hoặc sinh cặp ảnh thử có trường gốc (wake đáy tù, sin, xoáy).
%     • Chọn phương pháp + tham số, chạy một phương pháp hoặc so sánh tất cả.
%     • Xem ảnh, |u| + vector, xoáy ω, bản đồ sai số (nếu có trường gốc), phổ năng lượng & bảng sai số.
%     • Mặt nạ thân mô hình (ảnh trắng/đen, pixel ≠ 0 = che), quy đổi px → m/s (dx, dt).
%     • Xuất .mat (u, v, x, y, ω, …) và Tecplot .dat; xử lý cả thư mục cặp ảnh.

fig = uifigure('Name', 'pivbench — PIV dày đặc (RAFT-PIV / OpenOpticalFlow / CC)', ...
               'Position', [60 60 1380 820]);
S = struct('I1', [], 'I2', [], 'names', {{'', ''}}, 'ut', [], 'vt', [], 'valid', [], ...
           'mask', [], 'res', struct('label', {}, 'u', {}, 'v', {}, 'runtime', {}), 'dir', pwd);
setappdata(fig, 'S', S);

g = uigridlayout(fig, [1 2]);
g.ColumnWidth = {340, '1x'};

% ---------------- cột điều khiển ----------------
c = uigridlayout(g, [24 2]);
c.RowHeight = repmat({24}, 1, 24);
c.RowHeight{22} = '1x';
c.ColumnWidth = {150, '1x'};
c.Scrollable = 'on';

r = 1;
lab(c, r, '1) Dữ liệu', true); r = r + 1;
H.btnI1 = uibutton(c, 'Text', 'Ảnh 1…', 'ButtonPushedFcn', @(~, ~) onLoad(fig, 1)); place(H.btnI1, r, 1);
H.btnI2 = uibutton(c, 'Text', 'Ảnh 2…', 'ButtonPushedFcn', @(~, ~) onLoad(fig, 2)); place(H.btnI2, r, 2); r = r + 1;
H.btnMask = uibutton(c, 'Text', 'Mặt nạ thân…', 'ButtonPushedFcn', @(~, ~) onMask(fig)); place(H.btnMask, r, 1);
H.btnGT = uibutton(c, 'Text', 'Trường gốc (.mat)…', 'ButtonPushedFcn', @(~, ~) onGT(fig)); place(H.btnGT, r, 2); r = r + 1;
H.synthKind = uidropdown(c, 'Items', {'wake sát đáy', 'wake vùng tuần hoàn', 'wake xa', ...
    'sin u=A sin(2πx/λ)', 'xoáy Lamb–Oseen'}); place(H.synthKind, r, 1);
H.btnSynth = uibutton(c, 'Text', 'Sinh ảnh thử', 'ButtonPushedFcn', @(~, ~) onSynth(fig)); place(H.btnSynth, r, 2); r = r + 1;
lbl(c, r, 'ppp / SNR (0 = không nhiễu)');
H.pppSnr = uieditfield(c, 'text', 'Value', '0.05 / 10'); place(H.pppSnr, r, 2); r = r + 1;
lbl(c, r, 'λ sin [px] / A [px]');
H.sinePar = uieditfield(c, 'text', 'Value', '32 / 0.5'); place(H.sinePar, r, 2); r = r + 1;

lab(c, r, '2) Phương pháp', true); r = r + 1;
H.method = uidropdown(c, 'Items', {'raftpiv', 'raftpiv_ls', 'ofpiv', 'ofpiv_warp', 'crosscorr', 'raftpiv_v1'}, ...
    'Value', 'raftpiv'); place(H.method, r, 1);
H.compare = uicheckbox(c, 'Text', 'So sánh tất cả', 'Value', false); place(H.compare, r, 2); r = r + 1;
lbl(c, r, 'RAFT: vòng lặp / ô (px)');
H.raftPar = uieditfield(c, 'text', 'Value', '16 / 512'); place(H.raftPar, r, 2); r = r + 1;
lbl(c, r, 'Thiết bị');
H.device = uidropdown(c, 'Items', {'cpu', 'cuda'}, 'Value', 'cpu'); place(H.device, r, 2); r = r + 1;
lbl(c, r, 'OF: λ_HS / λ_LS');
H.ofPar = uieditfield(c, 'text', 'Value', '20 / 2000'); place(H.ofPar, r, 2); r = r + 1;
lbl(c, r, 'CC: cửa sổ (px)');
H.ccWin = uieditfield(c, 'text', 'Value', '64 32 16'); place(H.ccWin, r, 2); r = r + 1;
lbl(c, r, 'dx [m/px] / dt [s]');
H.scale = uieditfield(c, 'text', 'Value', '1 / 1'); place(H.scale, r, 2); r = r + 1;

lab(c, r, '3) Chạy & xuất', true); r = r + 1;
H.btnRun = uibutton(c, 'Text', '▶ Chạy', 'FontWeight', 'bold', 'ButtonPushedFcn', @(~, ~) onRun(fig)); place(H.btnRun, r, 1);
H.btnSpec = uibutton(c, 'Text', 'Phổ năng lượng', 'ButtonPushedFcn', @(~, ~) onSpectra(fig)); place(H.btnSpec, r, 2); r = r + 1;
H.btnMat = uibutton(c, 'Text', 'Xuất .mat', 'ButtonPushedFcn', @(~, ~) onExport(fig, 'mat')); place(H.btnMat, r, 1);
H.btnDat = uibutton(c, 'Text', 'Xuất Tecplot .dat', 'ButtonPushedFcn', @(~, ~) onExport(fig, 'dat')); place(H.btnDat, r, 2); r = r + 1;
H.btnBatch = uibutton(c, 'Text', 'Xử lý cả thư mục…', 'ButtonPushedFcn', @(~, ~) onBatch(fig)); place(H.btnBatch, r, [1 2]); r = r + 1;
H.btnCheck = uibutton(c, 'Text', 'Kiểm tra Python', 'ButtonPushedFcn', @(~, ~) onCheck(fig)); place(H.btnCheck, r, [1 2]); r = r + 1;
H.log = uitextarea(c, 'Editable', 'off', 'Value', {'Sẵn sàng. Mở ảnh hoặc sinh ảnh thử.'}); place(H.log, r, [1 2]);

% ---------------- cột hiển thị ----------------
d = uigridlayout(g, [2 1]);
d.RowHeight = {'2x', '1x'};
tg = uitabgroup(d);
names = {'Ảnh', 'Vận tốc |u|', 'Xoáy ω', 'Sai số (EPE)', 'Phổ'};
for i = 1:numel(names)
    t = uitab(tg, 'Title', names{i});
    H.tab(i) = t; %#ok<AGROW>
    gl = uigridlayout(t, [1 1]);
    H.ax(i) = uiaxes(gl); %#ok<AGROW>
end
H.tabs = tg;
H.table = uitable(d, 'ColumnName', {'phương pháp', 'EPE TB', 'EPE trung vị', 'EPE p95', 'RMSE u', 'RMSE v', ...
    'EPE vùng |ω| mạnh', 'EPE ngoài', 'thời gian [s]'});
setappdata(fig, 'H', H);
end

% =====================================================================
function place(h, row, col)
h.Layout.Row = row;
h.Layout.Column = col;
end

function lbl(c, row, txt)
l = uilabel(c, 'Text', txt);
place(l, row, 1);
end

function lab(c, row, txt, bold)
l = uilabel(c, 'Text', txt);
if bold, l.FontWeight = 'bold'; end
place(l, row, [1 2]);
end

function logmsg(fig, msg)
H = getappdata(fig, 'H');
H.log.Value = [H.log.Value(:); {[datestr(now, 'HH:MM:SS') '  ' msg]}];
try, scroll(H.log, 'bottom'); catch, end  % scroll có từ R2020b
drawnow;
end

function nums = parseNums(str)
nums = str2num(strrep(str, '/', ' ')); %#ok<ST2NM>
end

% ---------------- dữ liệu ----------------
function onLoad(fig, which)
S = getappdata(fig, 'S');
[f, p] = uigetfile({'*.tif;*.tiff;*.bmp;*.png;*.jpg', 'Ảnh PIV'}, sprintf('Chọn ảnh %d', which), S.dir);
figure(fig);
if isequal(f, 0), return; end
I = double(imread(fullfile(p, f)));
if ndims(I) == 3, I = mean(I(:, :, 1:3), 3); end
S.dir = p;
S.names{which} = fullfile(p, f);
if which == 1, S.I1 = I; else, S.I2 = I; end
S.ut = []; S.vt = []; S.valid = []; S.res = S.res([]);
setappdata(fig, 'S', S);
logmsg(fig, sprintf('Ảnh %d: %s (%d×%d)', which, f, size(I, 1), size(I, 2)));
showImage(fig);
end

function onMask(fig)
S = getappdata(fig, 'S');
[f, p] = uigetfile({'*.png;*.tif;*.bmp;*.mat', 'Mặt nạ'}, 'Mặt nạ thân mô hình (≠0 = che)', S.dir);
figure(fig);
if isequal(f, 0), return; end
[~, ~, ext] = fileparts(f);
if strcmpi(ext, '.mat')
    m = load(fullfile(p, f)); fn = fieldnames(m); M = m.(fn{1});
else
    M = imread(fullfile(p, f)); if ndims(M) == 3, M = M(:, :, 1); end
end
S.mask = M ~= 0;
setappdata(fig, 'S', S);
logmsg(fig, sprintf('Mặt nạ: %s (%.1f %% bị che)', f, 100 * mean(S.mask(:))));
end

function onGT(fig)
S = getappdata(fig, 'S');
[f, p] = uigetfile('*.mat', 'Trường gốc: biến u, v (px/khoảng thời gian, cùng cỡ ảnh)', S.dir);
figure(fig);
if isequal(f, 0), return; end
m = load(fullfile(p, f));
if ~isfield(m, 'u') || ~isfield(m, 'v')
    uialert(fig, 'File cần có biến u và v.', 'Trường gốc'); return;
end
S.ut = double(m.u); S.vt = double(m.v); S.valid = true(size(S.ut));
setappdata(fig, 'S', S);
logmsg(fig, ['Trường gốc: ' f]);
end

function onSynth(fig)
H = getappdata(fig, 'H');
S = getappdata(fig, 'S');
ps = parseNums(H.pppSnr.Value);
ip = struct('ppp', ps(1), 'seed', 0);
if numel(ps) > 1 && ps(2) > 0, ip.snr = ps(2); end
sp = parseNums(H.sinePar.Value);
shape = [256 384];
switch H.synthKind.Value
    case 'wake sát đáy'
        kind = 'wake'; fp = struct('D', 140, 'U_inf', 6, 'x_base', 30, 'L_r', 1.3, 'U_rev', 0.3, ...
            'kh_strength', 0.6, 'street_strength', 0.3, 'turb_rms', 0.06, 'seed', 11);
    case 'wake vùng tuần hoàn'
        kind = 'wake'; fp = struct('D', 140, 'U_inf', 8, 'x_base', -120, 'L_r', 1.2, 'U_rev', 0.25, ...
            'kh_strength', 0.8, 'street_strength', 0.5, 'turb_rms', 0.08, 'seed', 12);
    case 'wake xa'
        kind = 'wake'; fp = struct('D', 110, 'U_inf', 8, 'x_base', -380, 'L_r', 1.3, 'U_rev', 0.2, ...
            'kh_strength', 0.3, 'street_strength', 0.8, 'turb_rms', 0.1, 'seed', 13);
    case 'sin u=A sin(2πx/λ)'
        kind = 'sine'; fp = struct('wavelength', sp(1), 'amplitude', sp(2)); shape = [256 256];
    otherwise
        kind = 'lamb_oseen'; fp = struct('core_radius', 16, 'u_max', 4); shape = [256 256];
end
logmsg(fig, sprintf('Sinh ảnh thử: %s …', H.synthKind.Value));
try
    T = pivbench_synth(kind, fp, shape, ip);
catch err
    uialert(fig, err.message, 'Lỗi sinh ảnh'); return;
end
S.I1 = T.img1; S.I2 = T.img2; S.ut = T.u_true; S.vt = T.v_true; S.valid = T.valid;
S.mask = ~T.valid; S.names = {'(tổng hợp)', '(tổng hợp)'}; S.res = S.res([]);
S.synth = struct('kind', kind, 'params', fp);
setappdata(fig, 'S', S);
logmsg(fig, sprintf('Xong: %d hạt, |u|max = %.2f px', T.meta.n_particles, max(hypot(T.u_true(:), T.v_true(:)))));
showImage(fig);
end

% ---------------- chạy ----------------
function params = methodParams(fig, m)
H = getappdata(fig, 'H');
rp = parseNums(H.raftPar.Value);
op = parseNums(H.ofPar.Value);
params = struct();
switch m
    case {'raftpiv', 'raftpiv_v1'}
        params = struct('iters', rp(1), 'tile', rp(2), 'device', H.device.Value);
    case 'raftpiv_ls'
        params = struct('lambda_HS', op(1), 'lambda_LS', op(2), ...
            'raft_params', struct('iters', rp(1), 'tile', rp(2), 'device', H.device.Value));
    case {'ofpiv', 'ofpiv_warp'}
        params = struct('lambda_HS', op(1), 'lambda_LS', op(2));
    case 'crosscorr'
        params = struct('windows', parseNums(H.ccWin.Value));
end
end

function onRun(fig)
H = getappdata(fig, 'H');
S = getappdata(fig, 'S');
if isempty(S.I1) || isempty(S.I2)
    uialert(fig, 'Chưa có cặp ảnh.', 'Thiếu dữ liệu'); return;
end
if H.compare.Value
    methods = {'crosscorr', 'ofpiv', 'raftpiv', 'raftpiv_ls'};
else
    methods = {H.method.Value};
end
S.res = S.res([]);
for i = 1:numel(methods)
    m = methods{i};
    logmsg(fig, sprintf('Chạy %s …', m));
    try
        [u, v, info] = pivbench_estimate(S.I1, S.I2, m, methodParams(fig, m));
    catch err
        logmsg(fig, ['LỖI: ' err.message]);
        uialert(fig, err.message, ['Lỗi ' m]);
        continue;
    end
    if ~isempty(S.mask) && isequal(size(S.mask), size(u))
        u(S.mask) = NaN; v(S.mask) = NaN;
    end
    S.res(end + 1) = struct('label', m, 'u', u, 'v', v, 'runtime', info.runtime_s);
    logmsg(fig, sprintf('  %s: %.2f s', m, info.runtime_s));
end
setappdata(fig, 'S', S);
showResults(fig);
end

% ---------------- hiển thị ----------------
function showImage(fig)
H = getappdata(fig, 'H');
S = getappdata(fig, 'S');
ax = H.ax(1);
cla(ax);
imagesc(ax, S.I1); colormap(ax, gray); axis(ax, 'image'); ax.YDir = 'reverse';
title(ax, S.names{1}, 'Interpreter', 'none');
H.tabs.SelectedTab = H.tab(1);
end

function [u, v, unit] = scaled(fig, u, v)
H = getappdata(fig, 'H');
sc = parseNums(H.scale.Value);
if numel(sc) >= 2 && ~(sc(1) == 1 && sc(2) == 1)
    u = u * sc(1) / sc(2); v = v * sc(1) / sc(2); unit = 'm/s';
else
    unit = 'px';
end
end

function showResults(fig)
H = getappdata(fig, 'H');
S = getappdata(fig, 'S');
if isempty(S.res), return; end
R = S.res(end);
[u, v, unit] = scaled(fig, R.u, R.v);
[ny, nx] = size(u);
st = max(1, round(min(ny, nx) / 32));
[X, Y] = meshgrid(1:nx, 1:ny);
ax = H.ax(2); cla(ax);
imagesc(ax, hypot(u, v)); axis(ax, 'image'); colormap(ax, parula); colorbar(ax);
hold(ax, 'on');
quiver(ax, X(1:st:end, 1:st:end), Y(1:st:end, 1:st:end), u(1:st:end, 1:st:end), v(1:st:end, 1:st:end), ...
    1.5, 'Color', [0.92 0.41 0.20]);
hold(ax, 'off');
title(ax, sprintf('|u| [%s] — %s', unit, R.label), 'Interpreter', 'none');
ax = H.ax(3); cla(ax);
w = pivbench_vorticity(R.u, R.v);
lim = prctile99(abs(w(isfinite(w))));
imagesc(ax, w, [-lim lim]); axis(ax, 'image'); colormap(ax, bluered()); colorbar(ax);
title(ax, sprintf('ω [1/khoảng thời gian] — %s (toạ độ ảnh, y hướng xuống)', R.label), 'Interpreter', 'none');
rows = {};
if ~isempty(S.ut)
    ax = H.ax(4); cla(ax);
    e = hypot(R.u - S.ut, R.v - S.vt);
    imagesc(ax, e); axis(ax, 'image'); colormap(ax, hot); colorbar(ax);
    title(ax, sprintf('EPE [px] — %s', R.label), 'Interpreter', 'none');
    valid = S.valid;
    if isempty(valid), valid = true(size(e)); end
    for i = 1:numel(S.res)
        M = pivbench_metrics(S.res(i).u, S.res(i).v, S.ut, S.vt, 16, valid & isfinite(S.res(i).u));
        rows(end + 1, :) = {S.res(i).label, M.epe_mean, M.epe_median, M.epe_p95, M.rmse_u, M.rmse_v, ...
            M.epe_in, M.epe_out, S.res(i).runtime}; %#ok<AGROW>
    end
else
    for i = 1:numel(S.res)
        rows(end + 1, :) = {S.res(i).label, NaN, NaN, NaN, NaN, NaN, NaN, NaN, S.res(i).runtime}; %#ok<AGROW>
    end
end
H.table.Data = rows;
H.tabs.SelectedTab = H.tab(2);
end

function onSpectra(fig)
H = getappdata(fig, 'H');
S = getappdata(fig, 'S');
if isempty(S.res), uialert(fig, 'Chưa có kết quả.', 'Phổ'); return; end
ax = H.ax(5); cla(ax); hold(ax, 'on');
col = [0.16 0.47 0.84; 0.92 0.41 0.20; 0.11 0.69 0.48; 0.93 0.63 0; 0.91 0.48 0.64; 0 0.51 0];
leg = {};
for i = 1:numel(S.res)
    u = S.res(i).u; v = S.res(i).v;
    u(~isfinite(u)) = 0; v(~isfinite(v)) = 0;
    if isempty(S.ut)
        ut = zeros(size(u)); vt = ut;
    else
        ut = S.ut; vt = S.vt;
        if ~isempty(S.valid)  % trong thân mô hình dùng giá trị gốc (không tính)
            u(~S.valid) = ut(~S.valid); v(~S.valid) = vt(~S.valid);
        end
    end
    P = pivbench_spectra(u, v, ut, vt, 16);
    k = P.kx(2:end);
    if i == 1 && ~isempty(S.ut)
        loglog(ax, k, P.true.E_x(2:end), 'k', 'LineWidth', 2); leg{end + 1} = 'gốc'; %#ok<AGROW>
    end
    loglog(ax, k, P.est.E_x(2:end), '-', 'Color', col(i, :), 'LineWidth', 2); leg{end + 1} = S.res(i).label; %#ok<AGROW>
    if ~isempty(S.ut)
        loglog(ax, k, P.err.E_x(2:end), '--', 'Color', col(i, :), 'LineWidth', 1.5);
        leg{end + 1} = [S.res(i).label ' — sai số']; %#ok<AGROW>
    end
end
hold(ax, 'off');
ax.XScale = 'log'; ax.YScale = 'log'; grid(ax, 'on');
xlabel(ax, 'k_x [chu kỳ/px]  (λ = 1/k)'); ylabel(ax, 'E(k_x) [px²·px]');
legend(ax, leg, 'Location', 'southwest', 'Interpreter', 'none');
title(ax, 'Phổ năng lượng 1D theo x (Welch, Hann)');
H.tabs.SelectedTab = H.tab(5);
end

% ---------------- xuất ----------------
function onExport(fig, fmt)
S = getappdata(fig, 'S');
if isempty(S.res), uialert(fig, 'Chưa có kết quả.', 'Xuất'); return; end
R = S.res(end);
[u, v, unit] = scaled(fig, R.u, R.v);
if strcmp(fmt, 'mat')
    [f, p] = uiputfile('*.mat', 'Lưu kết quả', fullfile(S.dir, ['piv_' R.label '.mat']));
else
    [f, p] = uiputfile('*.dat', 'Lưu Tecplot', fullfile(S.dir, ['piv_' R.label '.dat']));
end
figure(fig);
if isequal(f, 0), return; end
[ny, nx] = size(u);
[x, y] = meshgrid(0:nx-1, 0:ny-1);
w = pivbench_vorticity(R.u, R.v);
out = fullfile(p, f);
if strcmp(fmt, 'mat')
    method = R.label; images = S.names; all_results = S.res; %#ok<NASGU>
    u_px = R.u; v_px = R.v; %#ok<NASGU>
    save(out, 'u', 'v', 'u_px', 'v_px', 'x', 'y', 'w', 'unit', 'method', 'images', 'all_results');
else
    fid = fopen(out, 'w');
    fprintf(fid, 'TITLE = "pivbench %s"\nVARIABLES = "x" "y" "u" "v" "omega"\nZONE I=%d, J=%d, F=POINT\n', R.label, nx, ny);
    D = [x(:) y(:) u(:) v(:) w(:)]';
    fprintf(fid, '%g %g %.6g %.6g %.6g\n', D);
    fclose(fid);
end
logmsg(fig, ['Đã lưu ' out]);
end

function onBatch(fig)
S = getappdata(fig, 'S');
H = getappdata(fig, 'H');
p = uigetdir(S.dir, 'Thư mục chứa cặp ảnh (a/b hoặc 1/2)');
figure(fig);
if isequal(p, 0), return; end
pairs = find_pairs(p);
if isempty(pairs), uialert(fig, 'Không tìm thấy cặp ảnh.', 'Thư mục'); return; end
outd = fullfile(p, 'pivbench_out');
if ~exist(outd, 'dir'), mkdir(outd); end
m = H.method.Value;
for i = 1:size(pairs, 1)
    logmsg(fig, sprintf('[%d/%d] %s', i, size(pairs, 1), pairs{i, 1}));
    try
        [u, v, info] = pivbench_estimate(fullfile(p, pairs{i, 1}), fullfile(p, pairs{i, 2}), m, methodParams(fig, m));
        if ~isempty(S.mask) && isequal(size(S.mask), size(u)), u(S.mask) = NaN; v(S.mask) = NaN; end
        [~, stem] = fileparts(pairs{i, 1});
        runtime = info.runtime_s; method = m; %#ok<NASGU>
        save(fullfile(outd, [stem '_' m '.mat']), 'u', 'v', 'runtime', 'method');
    catch err
        logmsg(fig, ['  LỖI: ' err.message]);
    end
end
logmsg(fig, ['Xong, kết quả trong ' outd]);
end

function pairs = find_pairs(p)
L = [dir(fullfile(p, '*.tif')); dir(fullfile(p, '*.tiff')); dir(fullfile(p, '*.bmp')); dir(fullfile(p, '*.png'))];
names = sort({L.name});
pairs = cell(0, 2);
used = false(size(names));
for i = 1:numel(names)
    if used(i), continue; end
    [~, st, ex] = fileparts(names{i});
    for k = numel(st):-1:1
        sw = '';
        switch st(k)
            case '1', sw = '2';
            case 'a', sw = 'b';
            case 'A', sw = 'B';
        end
        if isempty(sw), continue; end
        other = [st(1:k-1) sw st(k+1:end) ex];
        j = find(strcmp(names, other), 1);
        if ~isempty(j)
            pairs(end + 1, :) = {names{i}, other}; %#ok<AGROW>
            used([i j]) = true;
            break;
        end
    end
end
end

function onCheck(fig)
try
    info = pivbench_check();
    logmsg(fig, sprintf('Python %s, torch %s, CUDA %d, checkpoint: %d', info.python, info.torch, ...
        info.cuda, info.checkpoint_exists));
catch err
    logmsg(fig, ['LỖI: ' err.message]);
    uialert(fig, err.message, 'Python');
end
end

function p = prctile99(x)
x = sort(x(:));
if isempty(x), p = 1; return; end
p = x(max(1, round(0.99 * numel(x))));
if p == 0, p = 1; end
end

function c = bluered()
n = 128;
t = linspace(0, 1, n)';
c = [[t; ones(n, 1)], [t; flipud(t)], [ones(n, 1); flipud(t)]];
end
