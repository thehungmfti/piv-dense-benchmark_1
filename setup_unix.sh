#!/usr/bin/env bash
# Cài môi trường Python cho pivbench (Linux/macOS). Chạy trong thư mục gốc repo. Cần python3.11.
set -euo pipefail
cd "$(dirname "$0")"
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[all]"
.venv/bin/python -m pytest -q
echo "Xong. Trong MATLAB:  cd matlab; pivbench_check; pivbench_gui"
