#!/usr/bin/env python
"""Tổng hợp results/<name>/summary.csv + cutoffs.csv thành bảng markdown và biểu đồ EPE.

    python scripts/summarize_results.py results/wake_benchmark

Ghi ``summary.md`` và ``figures/epe_by_field.png`` vào thư mục kết quả.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pivbench import plotting  # noqa: E402


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main(argv=None):
    argv = argv or sys.argv[1:]
    d = Path(argv[0])
    rows = [r for r in csv.DictReader(open(d / "summary.csv")) if r["status"] == "ok"]
    labels = list(dict.fromkeys(r["label"] for r in rows))
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["field"] == "sine":
            continue
        groups[r["field"]][r["label"]].append(r)
    lines = [f"# Tổng hợp `{d.name}`", "",
             "EPE trung bình [px] (trung bình trên mọi ppp × SNR × seed của trường; với `cai_*`: trên các cặp ảnh test). "
             "In = vùng top-10 % |ω| của trường gốc; thời gian = s/cặp trên CPU.", "",
             "| trường | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    best_fmt = {}
    for fld, by in groups.items():
        vals = {lab: np.nanmean([_f(x["epe_mean"]) for x in by.get(lab, [])]) if by.get(lab) else np.nan for lab in labels}
        best = min((v for v in vals.values() if np.isfinite(v)), default=np.nan)
        best_fmt[fld] = vals
        cells = []
        for lab in labels:
            v = vals[lab]
            vin = np.nanmean([_f(x.get("epe_in")) for x in by.get(lab, [])]) if by.get(lab) else np.nan
            s = f"{v:.3f}" + (f" (in {vin:.3f})" if np.isfinite(vin) else "")
            cells.append(f"**{s}**" if v == best else s)
        lines.append(f"| {fld} | " + " | ".join(cells) + " |")
    rt = {lab: np.nanmean([_f(r["runtime_s"]) for r in rows if r["label"] == lab]) for lab in labels}
    lines.append("| thời gian [s] | " + " | ".join(f"{rt[lab]:.2f}" for lab in labels) + " |")
    if (d / "cutoffs.csv").is_file():
        cut = list(csv.DictReader(open(d / "cutoffs.csv")))
        keys = [k for k in cut[0] if k.startswith("lambda_c")]
        lines += ["", "Bước sóng cắt MTF [px] (trường sin u = 0.5·sin(2πx/λ), trung bình trên ppp × SNR; NaN = không đạt):",
                  "", "| phương pháp | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for lab in labels:
            cs = [r for r in cut if r["label"] == lab]
            if cs:
                lines.append(f"| {lab} | " + " | ".join(f"{np.nanmean([_f(r[k]) for r in cs]):.1f}" for k in keys) + " |")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib.pyplot as plt

    flds = list(best_fmt)
    cols = plotting.method_colors(labels)
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(flds) * len(labels) / 3), 4), constrained_layout=True)
    w = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        ax.bar(np.arange(len(flds)) + (i - (len(labels) - 1) / 2) * w, [best_fmt[f][lab] for f in flds], w * 0.92,
               color=cols[lab], label=lab)
    ax.set_xticks(np.arange(len(flds)))
    ax.set_xticklabels(flds)
    ax.set_ylabel("EPE trung bình [px]")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(len(labels), 6), fontsize=8, loc="upper left")
    ax.set_title(f"EPE theo trường — {d.name}", fontsize=10, loc="left")
    fig.savefig(d / "figures" / "epe_by_field.png", dpi=120)


if __name__ == "__main__":
    main()
