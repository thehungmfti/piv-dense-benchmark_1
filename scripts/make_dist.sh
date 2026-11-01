#!/usr/bin/env bash
# Đóng gói chương trình + trọng số + kết quả thành dist/piv-dense-benchmark_<ngày>.zip
# (không gồm .venv, dữ liệu tải về, kho dữ liệu sinh, checkpoint last.pt).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
name="piv-dense-benchmark_$(date +%Y%m%d)"
out="dist/${name}.zip"
rm -f "$out"
git ls-files -z | xargs -0 zip -q -9 "$out"
# kết quả & checkpoint đã sinh (một số bị .gitignore loại khỏi git)
for p in checkpoints/raftpiv_wake*/best.pt checkpoints/raftpiv_wake*/best.json; do
  [ -f "$p" ] && zip -q -9 "$out" "$p"
done
echo "→ $out ($(du -h "$out" | cut -f1))"
