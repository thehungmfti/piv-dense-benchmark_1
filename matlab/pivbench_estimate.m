function [u, v, info] = pivbench_estimate(I1, I2, method, params)
%PIVBENCH_ESTIMATE  Trường dịch chuyển dày đặc (1 vector/pixel) từ một cặp ảnh PIV.
%   [u, v, info] = pivbench_estimate(I1, I2)                 % RAFT-PIV v2 (mặc định)
%   [u, v, info] = pivbench_estimate(I1, I2, method)
%   [u, v, info] = pivbench_estimate(I1, I2, method, params)
%
%   I1, I2  : ảnh xám (ma trận số, uint8/uint16/double), hoặc tên file ảnh.
%   method  : 'raftpiv'    - RAFT-PIV tái hiện, trọng số v2 (wake đáy tù)      [mặc định]
%             'raftpiv_v1' - RAFT-PIV v1
%             'raftpiv_ls' - lai: RAFT-PIV khởi tạo + Liu–Shen tinh chỉnh
%             'ofpiv'      - OpenOpticalFlow_PIV (port Python, CC khởi tạo)
%             'ofpiv_warp' - như ofpiv, dịch ảnh bậc 3, 3 lượt
%             'crosscorr'  - PIV tương quan chéo nhiều lượt (64-32-16, chồng 75 %)
%   params  : struct ghi đè tham số, ví dụ
%             struct('iters', 24, 'tile', 512, 'device', 'cuda')            (raftpiv)
%             struct('lambda_LS', 4000, 'n_iter', 2)                         (ofpiv)
%             struct('windows', [48 24 12], 'overlap', 0.5)                  (crosscorr)
%
%   u, v    : cùng kích thước ảnh, đơn vị pixel/khoảng thời gian giữa 2 ảnh;
%             u theo cột (x, sang phải), v theo hàng (y, HƯỚNG XUỐNG như toạ độ ảnh).
%             Để vẽ với trục y hướng lên: quiver(x, y, u, -v) hoặc dùng axis ij.
%   info    : runtime_s, method, params_used.
if nargin < 3 || isempty(method), method = 'raftpiv'; end
if nargin < 4, params = struct(); end
if ischar(I1), I1 = imread(I1); end
if ischar(I2), I2 = imread(I2); end
I1 = to_gray(I1);
I2 = to_gray(I2);
if ~isequal(size(I1), size(I2))
    error('pivbench:size', 'Hai ảnh phải cùng kích thước.');
end
job.action = 'estimate';
job.method = method;
job.img1 = I1;
job.img2 = I2;
if isempty(fieldnames(params))
    job.params_json = '';
else
    job.params_json = jsonencode(params);
end
res = pivbench_call(job);
u = res.u;
v = res.v;
info = struct('runtime_s', res.runtime_s, 'method', res.method, 'params_used', res.params_used);
end

function g = to_gray(I)
I = double(I);
if ndims(I) == 3
    I = mean(I(:, :, 1:min(3, size(I, 3))), 3);
end
g = I;
end
