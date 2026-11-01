@echo off
REM Cài môi trường Python cho pivbench (Windows). Chạy trong thư mục gốc repo.
REM Cần Python 3.11 (https://www.python.org/downloads/) — lệnh "py -3.11" phải chạy được.
py -3.11 -m venv .venv || goto :err
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[all]" || goto :err
REM GPU NVIDIA (tuỳ chọn): .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python -m pytest -q
echo.
echo Xong. Trong MATLAB:  cd matlab ^&^& pivbench_check ^&^& pivbench_gui
goto :eof
:err
echo LOI cai dat — xem thong bao o tren.
exit /b 1
