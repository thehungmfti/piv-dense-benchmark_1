function res = pivbench_call(job)
%PIVBENCH_CALL  (nội bộ) Gửi một công việc sang Python qua scripts/matlab_bridge.py.
%   res = pivbench_call(job) — job là struct (action, ... ); trả về struct kết quả.
%   Ghi job ra file .mat tạm, gọi  python matlab_bridge.py job.mat res.mat  bằng system(),
%   rồi đọc res.mat. Báo lỗi MATLAB nếu Python thất bại.

cfg = pivbench_config();
tmp = tempname();
jobf = [tmp '_job.mat'];
resf = [tmp '_res.mat'];
cleanup = onCleanup(@() delete_if_exists({jobf, resf}));
save(jobf, '-struct', 'job', '-v7');
cmd = sprintf('"%s" "%s" "%s" "%s"', cfg.python, cfg.bridge, jobf, resf);
if ispc
    cmd = ['"' cmd '"'];  % cmd.exe bóc cặp ngoặc kép ngoài cùng — bọc thêm để giữ đường dẫn có dấu cách
end
[status, out] = system(cmd);
if ~exist(resf, 'file')
    error('pivbench:python', ['Không chạy được Python.\nLệnh: %s\nKết quả:\n%s\n' ...
          'Kiểm tra cài đặt bằng pivbench_check() hoặc đặt PIVBENCH_PYTHON.'], cmd, out);
end
res = load(resf);
if ~isfield(res, 'ok') || ~res.ok
    msg = '';
    if isfield(res, 'message'), msg = res.message; end
    tb = '';
    if isfield(res, 'traceback'), tb = res.traceback; end
    error('pivbench:python', 'Python báo lỗi (mã %d): %s\n%s', status, msg, tb);
end
end

function delete_if_exists(files)
for i = 1:numel(files)
    if exist(files{i}, 'file'), delete(files{i}); end
end
end
