function info = pivbench_check()
%PIVBENCH_CHECK  Kiểm tra Python, torch và checkpoint RAFT-PIV có sẵn sàng không.
%   info = pivbench_check()   in ra và trả về struct thông tin.
cfg = pivbench_config();
fprintf('Repo   : %s\nPython : %s\n', cfg.root, cfg.python);
job.action = 'info';
info = pivbench_call(job);
fprintf('Python %s, torch %s, CUDA %d\nCheckpoint RAFT-PIV: %s (tồn tại: %d)\n', ...
    info.python, info.torch, info.cuda, info.checkpoint, info.checkpoint_exists);
end
