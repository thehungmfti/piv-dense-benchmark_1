function cfg = pivbench_config()
%PIVBENCH_CONFIG  Đường dẫn repo và trình thông dịch Python dùng bởi các hàm pivbench_*.
%   cfg = pivbench_config() trả về struct với các trường:
%     root    - thư mục gốc repo (cha của thư mục matlab/)
%     python  - đường dẫn python. Thứ tự ưu tiên:
%               1) biến môi trường PIVBENCH_PYTHON
%               2) file matlab/pivbench_python.txt (một dòng: đường dẫn python)
%               3) <root>/.venv/bin/python (Linux/macOS) hoặc <root>\.venv\Scripts\python.exe (Windows)
%               4) 'python' trong PATH
%     bridge  - scripts/matlab_bridge.py
%
%   Ví dụ (Windows, Anaconda):  setenv('PIVBENCH_PYTHON', 'C:\Users\ban\anaconda3\envs\piv\python.exe')

here = fileparts(mfilename('fullpath'));
cfg.root = fileparts(here);
cfg.bridge = fullfile(cfg.root, 'scripts', 'matlab_bridge.py');
py = getenv('PIVBENCH_PYTHON');
if isempty(py)
    f = fullfile(here, 'pivbench_python.txt');
    if exist(f, 'file')
        py = strtrim(fileread(f));
    end
end
if isempty(py)
    if ispc
        cand = fullfile(cfg.root, '.venv', 'Scripts', 'python.exe');
    else
        cand = fullfile(cfg.root, '.venv', 'bin', 'python');
    end
    if exist(cand, 'file')
        py = cand;
    else
        py = 'python';
    end
end
cfg.python = py;
end
