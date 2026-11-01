#!/usr/bin/env python
"""P1-A — Bản đồ MTF cho PIV dày đặc.

Trường sin u = A·sin(2πx/λ) (mặc định, mode dọc/nén; ``--component v`` cho mode ngang/cắt
v = A·sin(2πx/λ)), quét λ × A × ppp × SNR × seed và các cấu hình phương pháp:

* ``ofpiv_s{s}``: OpenOpticalFlow_PIV (khởi tạo tương quan chéo, 1 lượt) với
  λ_HS = 20·s, λ_LS = 2000·s; ``--lambda-mode paired`` (mặc định) quét s ∈ lưới log 5 giá trị
  {0.1, 0.316, 1, 3.16, 10}; ``--lambda-mode grid`` quét tích 5×5 (λ_HS × λ_LS).
* ``cc32``, ``cc16``: tương quan chéo nhiều lượt, cửa sổ cuối 32 / 16 px (bước 25 %).
* ``raftpiv``: RAFT-PIV tái hiện (checkpoints/raftpiv_wake_v2) nếu có torch + trọng số.

Mỗi cấu hình: MTF(λ), lệch pha, λ_c tại MTF = 0.9 và 0.5. Đầu ra trong ``--out``:
    mtf_long.csv     (tidy: 1 hàng / cấu hình × λ × A × ppp × SNR × seed)
    cutoffs.csv      (λ_c theo từng seed và theo MTF trung bình seed — seed = "mean")
    fits.csv / fits.json  (mô hình λ_c = a·ppp^b·SNR^c·s^d và λ_c = α·ℓ_p + β, ℓ_p = 1/√ppp)
    figures/*.png
    run_info.json    (lưới đã chạy, có cờ quick)

Chạy:
    python scripts/mtf_sweep.py --quick                 # lưới thu nhỏ, CPU (~15–30 phút, 4 nhân)
    python scripts/mtf_sweep.py --out results/p1_mtf_full --workers 16   # lưới đầy đủ
    python scripts/mtf_sweep.py --analyze-only --out results/p1_mtf_quick
Kết quả được ghi nối tiếp; chạy lại cùng ``--out`` sẽ bỏ qua các ca đã có (tiếp tục được).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FULL = dict(
    wavelengths=[4, 6, 8, 12, 16, 24, 32, 48, 64, 128],
    amplitudes=[0.5, 2.0, 5.0],
    ppp=[0.005, 0.01, 0.02, 0.05, 0.1],
    snr=[math.inf, 20.0, 10.0, 5.0],
    lambda_scales=[0.1, 0.316, 1.0, 3.16, 10.0],
    seeds=[0, 1, 2],
    methods=["ofpiv", "cc32", "cc16", "raftpiv"],
)
QUICK = dict(
    wavelengths=[4, 6, 8, 12, 16, 24, 32, 48, 64, 128],
    amplitudes=[0.5, 2.0],
    ppp=[0.01, 0.02, 0.05, 0.1],
    snr=[math.inf, 10.0, 5.0],
    lambda_scales=[0.1, 1.0, 10.0],
    seeds=[0],
    methods=["ofpiv", "cc32", "cc16", "raftpiv"],
)
LAMBDA_HS0, LAMBDA_LS0 = 20.0, 2000.0
THRESHOLDS = (0.9, 0.5)

COLS = ["method_cfg", "method", "lambda_scale", "lambda_HS", "lambda_LS", "window", "component",
        "A", "wavelength", "ppp", "snr", "seed", "mtf", "phase_err_rad", "phase_err_px",
        "residual_rms", "time_s", "status"]


# ---------------------------------------------------------------------------
# Cấu hình phương pháp
# ---------------------------------------------------------------------------
def method_configs(methods, lambda_scales, lambda_mode="paired"):
    cfgs = []
    for m in methods:
        if m == "ofpiv":
            if lambda_mode == "grid":
                pairs = [(LAMBDA_HS0 * a, LAMBDA_LS0 * b, f"ofpiv_hs{a:g}_ls{b:g}", np.sqrt(a * b))
                         for a in lambda_scales for b in lambda_scales]
            else:
                pairs = [(LAMBDA_HS0 * s, LAMBDA_LS0 * s, f"ofpiv_s{s:g}", s) for s in lambda_scales]
            for hs, ls, lab, s in pairs:
                cfgs.append(dict(label=lab, method="ofpiv", lambda_scale=s, lambda_HS=hs, lambda_LS=ls,
                                 window=np.nan,
                                 params=dict(lambda_HS=hs, lambda_LS=ls, init="crosscorr", n_levels=1,
                                             n_iter=1)))
        elif m == "cc32":
            cfgs.append(dict(label="cc32", method="crosscorr", lambda_scale=np.nan, lambda_HS=np.nan,
                             lambda_LS=np.nan, window=32,
                             params=dict(windows=(64, 32), overlap=0.75, iters_per_window=(1, 2))))
        elif m == "cc16":
            cfgs.append(dict(label="cc16", method="crosscorr", lambda_scale=np.nan, lambda_HS=np.nan,
                             lambda_LS=np.nan, window=16,
                             params=dict(windows=(64, 32, 16), overlap=0.75, iters_per_window=(1, 1, 2))))
        elif m == "raftpiv":
            cfgs.append(dict(label="raftpiv", method="raftpiv", lambda_scale=np.nan, lambda_HS=np.nan,
                             lambda_LS=np.nan, window=np.nan,
                             params=dict(backend="local", iters=16, device="cpu", tile=None)))
        else:
            raise ValueError(m)
    return cfgs


def raft_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch chưa cài"
    from pivbench.methods.raftpiv import _local_ckpt

    ck = _local_ckpt()
    return (ck.is_file(), f"checkpoint {ck}" + ("" if ck.is_file() else " không tồn tại"))


# ---------------------------------------------------------------------------
# Một ca (một cặp ảnh) — chạy mọi cấu hình phương pháp
# ---------------------------------------------------------------------------
def run_case(job):
    from pivbench.methods import estimate
    from pivbench.metrics.mtf import mtf_from_sine
    from pivbench.synth import SineField, generate_pair

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    A, lam, ppp, snr, seed = job["A"], job["wavelength"], job["ppp"], job["snr"], job["seed"]
    comp, shape, border = job["component"], tuple(job["shape"]), job["border"]
    field = SineField(amplitude=A, wavelength=lam, component=comp, axis="x")
    # seed ảnh phụ thuộc mọi tham số trừ phương pháp ⇒ mọi phương pháp thấy cùng cặp ảnh
    img_seed = int(seed * 1_000_003 + round(lam * 1000) + round(A * 7919) + round(ppp * 1e5) * 13
                   + (0 if not np.isfinite(snr) else int(snr) * 101))
    s = generate_pair(field, shape, ppp=ppp, snr=None if not np.isfinite(snr) else snr, d_p=2.5,
                      seed=img_seed)
    rows = []
    for cfg in job["cfgs"]:
        t0 = time.time()
        status = "ok"
        try:
            u, v = estimate(cfg["method"], s["img1"], s["img2"], **cfg["params"])
            r = mtf_from_sine(u, v, A, lam, component=comp, axis="x", border=border)
            mtf, ph, res = r["mtf"], r["phase_error"], r["residual_rms"]
        except Exception as e:  # noqa: BLE001
            mtf = ph = res = np.nan
            status = f"error: {type(e).__name__}: {e}"[:200]
        rows.append(dict(method_cfg=cfg["label"], method=cfg["method"], lambda_scale=cfg["lambda_scale"],
                         lambda_HS=cfg["lambda_HS"], lambda_LS=cfg["lambda_LS"], window=cfg["window"],
                         component=comp, A=A, wavelength=lam, ppp=ppp, snr=snr, seed=seed, mtf=mtf,
                         phase_err_rad=ph, phase_err_px=ph * lam / (2 * np.pi) if np.isfinite(ph) else np.nan,
                         residual_rms=res, time_s=time.time() - t0, status=status))
    return rows


# ---------------------------------------------------------------------------
# Phân tích
# ---------------------------------------------------------------------------
def compute_cutoffs(df):
    import pandas as pd

    from pivbench.metrics.mtf import cutoff_wavelength

    keys = ["method_cfg", "method", "lambda_scale", "window", "component", "A", "ppp", "snr"]
    out = []
    ok = df[df.status == "ok"]
    for kv, g in ok.groupby(keys + ["seed"], dropna=False):
        g = g.sort_values("wavelength")
        for t in THRESHOLDS:
            out.append(dict(zip(keys + ["seed"], kv), threshold=t,
                            lambda_c=cutoff_wavelength(g.wavelength, g.mtf, t)))
    for kv, g in ok.groupby(keys, dropna=False):
        m = g.groupby("wavelength").mtf.mean()
        for t in THRESHOLDS:
            out.append(dict(zip(keys, kv), seed="mean", threshold=t,
                            lambda_c=cutoff_wavelength(m.index.values, m.values, t),
                            n_seeds=g.seed.nunique()))
    c = pd.DataFrame(out)
    c["l_p"] = 1.0 / np.sqrt(c.ppp)
    c["lc_over_lp"] = c.lambda_c / c.l_p
    return c


def compute_fits(cut):
    from pivbench.resolution.mtf_model import fit_linear, fit_power_law

    rows = []
    cm = cut[cut.seed.astype(str) == "mean"]
    for (meth, comp, A, t), g in cm.groupby(["method", "component", "A", "threshold"]):
        groups = [("all", g)] if meth == "ofpiv" else [(lab, gg) for lab, gg in g.groupby("method_cfg")]
        for lab, gg in groups:
            fin = gg[np.isfinite(gg.snr)]
            X = dict(ppp=fin.ppp.values, SNR=fin.snr.values)
            if meth == "ofpiv":
                X["lambda_scale"] = fin.lambda_scale.values
            f = fit_power_law(fin.lambda_c.values, X)
            rows.append(dict(method=meth, method_cfg=lab, component=comp, A=A, threshold=t,
                             model="power: a*ppp^b*SNR^c" + ("*s^d" if meth == "ofpiv" else ""),
                             a=f["a"], b_ppp=f["exponents"].get("ppp"), c_snr=f["exponents"].get("SNR"),
                             d_lambda=f["exponents"].get("lambda_scale", np.nan), r2_log=f["r2_log"],
                             r2_lin=f["r2_lin"], rmse_log=f["rmse_log"], n=f["n"], snr_set="finite"))
            # ℓ_p: tất cả SNR (kể cả ∞)
            for snr_set, sub in [("all", gg), ("inf", gg[~np.isfinite(gg.snr)])]:
                fl = fit_linear(sub.lambda_c.values, sub.l_p.values)
                rows.append(dict(method=meth, method_cfg=lab, component=comp, A=A, threshold=t,
                                 model="linear: alpha*l_p+beta", a=fl["alpha"], b_ppp=fl["beta"],
                                 c_snr=np.nan, d_lambda=np.nan, r2_log=np.nan, r2_lin=fl["r2"],
                                 rmse_log=np.nan, n=fl["n"], snr_set=snr_set,
                                 ratio_mean=fl["ratio_mean"], ratio_cv=fl["ratio_cv"],
                                 c_origin=fl.get("c_origin", np.nan), r2_origin=fl["r2_origin"]))
            # λ_c chỉ theo ppp (b) cho từng SNR hữu hạn — kiểm tra b ≈ −0.5 (tỉ lệ với ℓ_p)
            fp = fit_power_law(gg.lambda_c.values, dict(ppp=gg.ppp.values,
                                                         **({"lambda_scale": gg.lambda_scale.values}
                                                            if meth == "ofpiv" else {})))
            rows.append(dict(method=meth, method_cfg=lab, component=comp, A=A, threshold=t,
                             model="power: a*ppp^b" + ("*s^d" if meth == "ofpiv" else "") + " (all SNR)",
                             a=fp["a"], b_ppp=fp["exponents"].get("ppp"), c_snr=np.nan,
                             d_lambda=fp["exponents"].get("lambda_scale", np.nan), r2_log=fp["r2_log"],
                             r2_lin=fp["r2_lin"], rmse_log=fp["rmse_log"], n=fp["n"], snr_set="all"))
    import pandas as pd

    return pd.DataFrame(rows)


def snr_label(s):
    return "∞" if not np.isfinite(s) else f"{s:g}"


def make_figures(df, cut, fits, outdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pivbench.resolution.mtf_model import tophat_mtf

    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok = df[df.status == "ok"]
    mean = ok.groupby(["method_cfg", "method", "window", "A", "ppp", "snr", "wavelength"],
                      dropna=False).mtf.agg(["mean", "std"]).reset_index()
    cfgs = sorted(ok.method_cfg.unique(), key=lambda s: (not s.startswith("ofpiv"), s))
    lam_f = np.geomspace(4, 128, 200)
    cmap = plt.get_cmap("viridis")

    # (1) MTF(λ) từng phương pháp, tại ppp/SNR "chuẩn" và theo A
    ppp_ref = 0.05 if 0.05 in ok.ppp.values else sorted(ok.ppp.unique())[-2]
    for A in sorted(ok.A.unique()):
        snrs = sorted(ok.snr.unique(), key=lambda s: -s)
        fig, axes = plt.subplots(1, len(snrs), figsize=(4.2 * len(snrs), 3.8), sharey=True, squeeze=False)
        for ax, snr in zip(axes[0], snrs):
            sub = mean[(mean.A == A) & (mean.ppp == ppp_ref) & (mean.snr == snr)]
            ofs = [c for c in cfgs if c.startswith("ofpiv")]
            for i, c in enumerate(cfgs):
                g = sub[sub.method_cfg == c].sort_values("wavelength")
                if g.empty:
                    continue
                if c.startswith("ofpiv"):
                    col, ls = cmap(i / max(1, len(ofs) - 1) * 0.85), "-"
                else:
                    col, ls = {"cc32": "tab:red", "cc16": "tab:orange", "raftpiv": "k"}.get(c, "gray"), "--"
                ax.errorbar(g.wavelength, g["mean"], yerr=g["std"].fillna(0), color=col, ls=ls, marker="o",
                            ms=3, lw=1.2, capsize=2, label=c)
            for W, col in [(32, "tab:red"), (16, "tab:orange")]:
                ax.plot(lam_f, tophat_mtf(lam_f, W), color=col, ls=":", lw=1, label=f"sinc W={W}")
            ax.axhline(0.9, color="gray", lw=0.5)
            ax.axhline(0.5, color="gray", lw=0.5)
            ax.set_xscale("log")
            ax.set_xticks([4, 8, 16, 32, 64, 128])
            ax.set_xticklabels([4, 8, 16, 32, 64, 128])
            ax.set_ylim(-0.05, 1.25)
            ax.set_xlabel("λ [px]")
            ax.set_title(f"A={A:g} px, ppp={ppp_ref:g}, SNR={snr_label(snr)}")
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("MTF")
        axes[0, -1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"mtf_by_method_A{A:g}.png", dpi=130)
        plt.close(fig)

    # (2) MTF(λ) theo ppp cho 1 cấu hình mỗi họ phương pháp
    for c in cfgs:
        sub = mean[(mean.method_cfg == c)]
        As = sorted(sub.A.unique())
        fig, axes = plt.subplots(1, len(As), figsize=(4.2 * len(As), 3.6), sharey=True, squeeze=False)
        for ax, A in zip(axes[0], As):
            pp = sorted(sub.ppp.unique())
            for j, p in enumerate(pp):
                for snr, ls in [(sub.snr.max(), "-"), (sub.snr.min(), "--")]:
                    g = sub[(sub.A == A) & (sub.ppp == p) & (sub.snr == snr)].sort_values("wavelength")
                    ax.plot(g.wavelength, g["mean"], ls=ls, marker=".", color=cmap(j / max(1, len(pp) - 1)),
                            label=f"ppp={p:g}, SNR={snr_label(snr)}")
            if c.startswith("cc"):
                ax.plot(lam_f, tophat_mtf(lam_f, int(c[2:])), "r:", label="sinc (top-hat)")
            ax.set_xscale("log")
            ax.set_ylim(-0.05, 1.25)
            ax.set_title(f"{c}, A={A:g}")
            ax.set_xlabel("λ [px]")
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("MTF")
        axes[0, -1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(fig_dir / f"mtf_vs_ppp_{c}.png", dpi=120)
        plt.close(fig)

    # (3) Bản đồ nhiệt λ_c(ppp, SNR)
    cm = cut[cut.seed.astype(str) == "mean"]
    for t in THRESHOLDS:
        for A in sorted(cm.A.unique()):
            sub = cm[(cm.threshold == t) & (cm.A == A)]
            fig, axes = plt.subplots(1, len(cfgs), figsize=(2.6 * len(cfgs), 2.9), squeeze=False)
            pp = sorted(sub.ppp.unique())
            ss = sorted(sub.snr.unique(), key=lambda s: -s)
            vmax = np.nanpercentile(sub.lambda_c, 95) if np.isfinite(sub.lambda_c).any() else 128
            for ax, c in zip(axes[0], cfgs):
                M = np.full((len(ss), len(pp)), np.nan)
                for i, s_ in enumerate(ss):
                    for j, p in enumerate(pp):
                        r = sub[(sub.method_cfg == c) & (sub.snr == s_) & (sub.ppp == p)]
                        if len(r):
                            M[i, j] = r.lambda_c.values[0]
                im = ax.imshow(M, cmap="magma_r", vmin=4, vmax=vmax, aspect="auto")
                for i in range(len(ss)):
                    for j in range(len(pp)):
                        ax.text(j, i, "–" if not np.isfinite(M[i, j]) else f"{M[i, j]:.0f}", ha="center",
                                va="center", fontsize=7, color="w" if (np.isfinite(M[i, j]) and
                                                                        M[i, j] > 0.6 * vmax) else "k")
                ax.set_xticks(range(len(pp)))
                ax.set_xticklabels([f"{p:g}" for p in pp], fontsize=7, rotation=45)
                ax.set_yticks(range(len(ss)))
                ax.set_yticklabels([snr_label(s_) for s_ in ss], fontsize=7)
                ax.set_title(c, fontsize=8)
                ax.set_xlabel("ppp", fontsize=7)
            axes[0, 0].set_ylabel("SNR")
            fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label=f"λ_c(MTF={t:g}) [px]")
            fig.suptitle(f"λ_c tại MTF={t:g}, A={A:g} px  ('–' = không bao giờ đạt ngưỡng)", fontsize=9)
            fig.savefig(fig_dir / f"heatmap_lc{t:g}_A{A:g}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

    # (4) λ_c/ℓ_p
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axes, THRESHOLDS):
        sub = cm[(cm.threshold == t)]
        for i, c in enumerate(cfgs):
            g = sub[sub.method_cfg == c]
            g = g[np.isfinite(g.lambda_c)]
            if g.empty:
                continue
            col = cmap(i / len(cfgs))
            for snr, mk in zip(sorted(g.snr.unique(), key=lambda s: -s), "osd^v"):
                gg = g[g.snr == snr].groupby("l_p").lc_over_lp.mean()
                ax.plot(gg.index, gg.values, marker=mk, color=col, lw=0.8, ms=4,
                        label=c if snr == g.snr.max() else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ℓ_p = 1/√ppp [px]")
        ax.set_ylabel(f"λ_c(MTF={t:g}) / ℓ_p")
        ax.set_title(f"MTF={t:g}  (ký hiệu: o ∞, s 20/10, d 10/5 …; trung bình theo A)")
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "lc_over_lp.png", dpi=130)
    plt.close(fig)

    # (5) lệch pha
    A0 = sorted(ok.A.unique())[0]
    sub = ok[(ok.A == A0) & (ok.ppp == ppp_ref)].groupby(["method_cfg", "snr", "wavelength"]).phase_err_px.mean()
    sub = sub.reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, c in enumerate(cfgs):
        g = sub[(sub.method_cfg == c) & (sub.snr == sub.snr.max())]
        ax.plot(g.wavelength, g.phase_err_px, marker="o", ms=3, label=c, color=cmap(i / len(cfgs)))
    ax.set_xscale("log")
    ax.set_ylim(-3, 3)
    ax.set_xlabel("λ [px]")
    ax.set_ylabel("lệch pha [px] = Δφ·λ/2π")
    ax.set_title(f"A={A0:g}, ppp={ppp_ref:g}, SNR={snr_label(sub.snr.max())} (vô nghĩa khi MTF≈0)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "phase_error.png", dpi=130)
    plt.close(fig)


def analyze(outdir: Path):
    import pandas as pd

    df = pd.read_csv(outdir / "mtf_long.csv")
    cut = compute_cutoffs(df)
    cut.to_csv(outdir / "cutoffs.csv", index=False)
    fits = compute_fits(cut)
    fits.to_csv(outdir / "fits.csv", index=False)
    make_figures(df, cut, fits, outdir)
    # Bảng tóm tắt λ_c (seed = mean) cho báo cáo
    cm = cut[cut.seed.astype(str) == "mean"]
    piv = cm.pivot_table(index=["method_cfg", "A", "snr"], columns=["threshold", "ppp"], values="lambda_c")
    piv.to_csv(outdir / "cutoffs_table.csv")
    n_err = int((df.status != "ok").sum())
    print(f"[analyze] {len(df)} hàng, {n_err} lỗi; cutoffs.csv, fits.csv, figures/ → {outdir}")
    return df, cut, fits


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="lưới thu nhỏ chạy CPU")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--shape", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--border", type=int, default=16)
    ap.add_argument("--component", choices=["u", "v"], default="u",
                    help="u: u=A sin(2πx/λ) (nén, mặc định đặc tả); v: v=A sin(2πx/λ) (cắt, div=0)")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--lambda-mode", choices=["paired", "grid"], default="paired")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--amplitudes", type=float, nargs="+", default=None)
    ap.add_argument("--ppp", type=float, nargs="+", default=None)
    ap.add_argument("--snr", type=float, nargs="+", default=None, help="dùng inf cho không nhiễu")
    ap.add_argument("--wavelengths", type=float, nargs="+", default=None)
    ap.add_argument("--lambda-scales", type=float, nargs="+", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)

    grid = dict(QUICK if a.quick else FULL)
    for k in ("methods", "seeds", "amplitudes", "ppp", "snr", "wavelengths", "lambda_scales"):
        val = getattr(a, k)
        if val is not None:
            grid[k] = val
    out = Path(a.out or (ROOT / "results" / ("p1_mtf_quick" if a.quick else "p1_mtf_full")))
    if a.component == "v" and a.out is None:
        out = out.with_name(out.name + "_transverse")
    out.mkdir(parents=True, exist_ok=True)
    if a.analyze_only:
        analyze(out)
        return

    methods = list(grid["methods"])
    unavailable = {}
    if "raftpiv" in methods:
        ok, msg = raft_available()
        if not ok:
            methods.remove("raftpiv")
            unavailable["raftpiv"] = msg
            print(f"[mtf_sweep] bỏ raftpiv: {msg}")
    cfgs = method_configs(methods, grid["lambda_scales"], a.lambda_mode)

    csv = out / "mtf_long.csv"
    done = set()
    if csv.exists():
        import pandas as pd

        old = pd.read_csv(csv)
        done = set(zip(old.A, old.wavelength, old.ppp, old.snr, old.seed, old.method_cfg))
    jobs = []
    for A, lam, ppp, snr, seed in itertools.product(grid["amplitudes"], grid["wavelengths"], grid["ppp"],
                                                    grid["snr"], grid["seeds"]):
        todo = [c for c in cfgs if (A, lam, ppp, snr, seed, c["label"]) not in done]
        if todo:
            jobs.append(dict(A=A, wavelength=lam, ppp=ppp, snr=snr, seed=seed, cfgs=todo,
                             component=a.component, shape=a.shape, border=a.border))
    info = dict(quick=a.quick, grid={k: [str(x) for x in v] for k, v in grid.items()}, shape=a.shape,
                border=a.border, component=a.component, lambda_mode=a.lambda_mode,
                method_configs=[{k: (str(v) if k == "params" else v) for k, v in c.items()} for c in cfgs],
                unavailable=unavailable, lambda_HS0=LAMBDA_HS0, lambda_LS0=LAMBDA_LS0, d_p=2.5,
                n_jobs=len(jobs))
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=float, ensure_ascii=False))
    print(f"[mtf_sweep] {len(jobs)} cặp ảnh × ≤{len(cfgs)} cấu hình, {a.workers} tiến trình → {out}")

    t0 = time.time()
    header = not csv.exists()
    import pandas as pd

    def write(rows):
        nonlocal header
        pd.DataFrame(rows, columns=COLS).to_csv(csv, mode="a", header=header, index=False)
        header = False

    if a.workers > 1:
        from multiprocessing import get_context

        with get_context("fork").Pool(a.workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(run_case, jobs, chunksize=1)):
                write(rows)
                if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(jobs)}  {el / 60:.1f} phút, còn ~{el / (i + 1) * (len(jobs) - i - 1) / 60:.1f}",
                          flush=True)
    else:
        for i, j in enumerate(jobs):
            write(run_case(j))
            print(f"  {i + 1}/{len(jobs)}", flush=True)
    analyze(out)


if __name__ == "__main__":
    main()
