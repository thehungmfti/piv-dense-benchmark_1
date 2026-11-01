#!/usr/bin/env python
"""P1-B — Hiệu chỉnh phổ năng lượng bằng hàm truyền đo được.

Với mỗi cấu hình phương pháp × (ppp, SNR):

1. Hàm truyền từ trường sin (cùng thủ tục với ``mtf_sweep.py``), biên độ ``--A-sine``:
     H_L(k) = MTF(1/k) của mode dọc/nén  u = A sin(2πx/λ)   (đúng đặc tả mục 5)
     H_T(k) = MTF(1/k) của mode ngang/cắt v = A sin(2πx/λ)   (mode duy nhất có trong trường div = 0)
2. Hàm truyền phổ chéo trên tập HUẤN LUYỆN trường rối tổng hợp k^-5/3 (seed khác tập kiểm tra):
     H_xs(k) = S_est,true(k) / S_true,true(k),   N_xs(k) = E_meas − H_xs² E_true.
3. Sàn nhiễu: (a) đuôi phẳng N_tail (trung vị E_meas, k ∈ [0.35, 0.5] chu kỳ/px);
   (b) cặp ảnh dịch chuyển 0 N_zero(k); (c) N_xs từ tập huấn luyện.
4. Kiểm chứng trên tập KIỂM TRA: rối k^-5/3 (seed mới) và DNS 2D (``pivbench.resolution.dns2d``,
   hoặc ``--dns-path`` nếu có file DNS thật trong data/). Báo cáo k₁₀ — số sóng lớn nhất mà
   |E/E_true − 1| < 10 % liên tục từ k nhỏ nhất — trước/sau hiệu chỉnh và tỉ lệ cải thiện.

Hiệu chỉnh (Wiener, xem docs/P1_SPECTRAL_CORRECTION.md):
    E_corr = max(E_meas − N, 0) · |H|² / (|H|⁴ + ε(1 − |H|²)²),  chỉ tại |H| > h_min.

Đầu ra (``--out``): spectra.npz, transfer.csv, k10.csv, figures/*.png, run_info.json.

    python scripts/spectral_correction.py --quick
    python scripts/spectral_correction.py --out results/p1_spectral_full --n-train 16 --n-test 16 --seeds-sine 3
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))

from mtf_sweep import method_configs, raft_available  # noqa: E402

WAVELENGTHS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128]


def _snr_arg(snr):
    return None if not np.isfinite(snr) else snr


def _set_threads():
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Jobs (một cặp ảnh, chạy mọi cấu hình)
# ---------------------------------------------------------------------------
def job_sine(j):
    from pivbench.methods import estimate
    from pivbench.metrics.mtf import mtf_from_sine
    from pivbench.synth import SineField, generate_pair

    _set_threads()
    f = SineField(amplitude=j["A"], wavelength=j["lam"], component=j["comp"], axis="x")
    s = generate_pair(f, tuple(j["shape"]), ppp=j["ppp"], snr=_snr_arg(j["snr"]), d_p=2.5, seed=j["seed"])
    out = []
    for c in j["cfgs"]:
        try:
            u, v = estimate(c["method"], s["img1"], s["img2"], **c["params"])
            m = mtf_from_sine(u, v, j["A"], j["lam"], component=j["comp"], border=j["border"])["mtf"]
        except Exception as e:  # noqa: BLE001
            print("  lỗi", c["label"], e, flush=True)
            m = np.nan
        out.append(dict(kind="sine", comp=j["comp"], cfg=c["label"], ppp=j["ppp"], snr=j["snr"],
                        lam=j["lam"], seed=j["seed"], mtf=m))
    return out


def make_truth(kind, seed, shape, rms, dns_path=None):
    from pivbench.synth import GriddedField, TurbulenceField

    if kind == "turb":
        return TurbulenceField(shape, rms=rms, seed=seed)
    if kind == "dns":
        if dns_path:
            from pivbench.synth import load_dns_field

            return load_dns_field(dns_path)
        from pivbench.resolution.dns2d import run_dns2d

        u, v = run_dns2d(shape[0], seed=seed, rms_out=rms)
        return GriddedField(u, v, periodic=True, name="dns2d")
    if kind == "zero":
        from pivbench.synth.fields import UniformField

        return UniformField(u0=0.0, v0=0.0)
    raise ValueError(kind)


def job_field(j):
    from pivbench.methods import estimate
    from pivbench.resolution import radial_auto_spectrum, radial_cross_spectrum
    from pivbench.synth import generate_pair

    _set_threads()
    shape = tuple(j["shape"])
    field = make_truth(j["kind"], j["seed"], shape, j["rms"], j.get("dns_path"))
    s = generate_pair(field, shape, ppp=j["ppp"], snr=_snr_arg(j["snr"]), d_p=2.5, seed=10_000 + j["seed"])
    b = j["border"]
    k, Ett = radial_auto_spectrum(s["u"], s["v"], b)
    out = []
    for c in j["cfgs"]:
        try:
            u, v = estimate(c["method"], s["img1"], s["img2"], **c["params"])
            _, Eee = radial_auto_spectrum(u, v, b)
            _, Set = radial_cross_spectrum(u, v, s["u"], s["v"], b)
            eu, ev = u - s["u"], v - s["v"]
            epe = float(np.mean(np.hypot(eu, ev)[b:-b, b:-b]))
        except Exception as e:  # noqa: BLE001
            print("  lỗi", c["label"], e, flush=True)
            Eee = Set = np.full_like(k, np.nan)
            epe = np.nan
        out.append(dict(kind=j["kind"], role=j["role"], cfg=c["label"], ppp=j["ppp"], snr=j["snr"],
                        seed=j["seed"], k=k, E_true=Ett, E_meas=Eee, S_et=Set, epe=epe))
    return out


def run_pool(fn, jobs, workers):
    res = []
    t0 = time.time()
    if workers > 1:
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
                res.extend(r)
                if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                    print(f"  {fn.__name__}: {i + 1}/{len(jobs)}  {(time.time() - t0) / 60:.1f} phút", flush=True)
    else:
        for j in jobs:
            res.extend(fn(j))
    return res


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--shape", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--border", type=int, default=16)
    ap.add_argument("--methods", nargs="+", default=["ofpiv", "cc32", "cc16", "raftpiv"])
    ap.add_argument("--lambda-scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--ppp", type=float, nargs="+", default=[0.05])
    ap.add_argument("--snr", type=float, nargs="+", default=[math.inf, 10.0])
    ap.add_argument("--rms", type=float, default=2.0, help="rms của |u| trường rối/DNS (px)")
    ap.add_argument("--A-sine", type=float, default=2.0, help="biên độ sin để đo H (≈ √2·rms thành phần)")
    ap.add_argument("--seeds-sine", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--n-zero", type=int, default=None)
    ap.add_argument("--dns-path", default=None, help="file DNS thật (npz/h5, khoá u, v) nếu có")
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--h-min", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)
    q = a.quick
    a.seeds_sine = a.seeds_sine or (1 if q else 3)
    a.n_train = a.n_train or (4 if q else 16)
    a.n_test = a.n_test or (4 if q else 16)
    a.n_zero = a.n_zero or (2 if q else 4)
    out = Path(a.out or (ROOT / "results" / ("p1_spectral_quick" if q else "p1_spectral_full")))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if a.dns_path is None:
        for cand in ["data/raw/pivc2005_caseB/truth.npz", "data/raw/jhtdb_slice.h5", "data/raw/jhtdb_slice.npz"]:
            if (ROOT / cand).is_file():
                a.dns_path = str(ROOT / cand)
    if a.analyze_only:
        analyze(out, a)
        return

    methods = list(a.methods)
    unavailable = {}
    if "raftpiv" in methods:
        ok, msg = raft_available()
        if not ok:
            methods.remove("raftpiv")
            unavailable["raftpiv"] = msg
            print("[spectral] bỏ raftpiv:", msg)
    cfgs = method_configs(methods, a.lambda_scales)
    base = dict(shape=a.shape, border=a.border, cfgs=cfgs)

    t0 = time.time()
    sine_jobs = [dict(base, A=a.A_sine, lam=lam, comp=comp, ppp=p, snr=s, seed=sd)
                 for comp in ("u", "v") for lam in WAVELENGTHS for p in a.ppp for s in a.snr
                 for sd in range(a.seeds_sine)]
    field_jobs = []
    for p in a.ppp:
        for s in a.snr:
            field_jobs += [dict(base, kind="turb", role="train", seed=100 + i, ppp=p, snr=s, rms=a.rms)
                           for i in range(a.n_train)]
            field_jobs += [dict(base, kind="turb", role="test", seed=500 + i, ppp=p, snr=s, rms=a.rms)
                           for i in range(a.n_test)]
            field_jobs += [dict(base, kind="dns", role="test", seed=900 + i, ppp=p, snr=s, rms=a.rms,
                                dns_path=a.dns_path) for i in range(1 if a.dns_path else a.n_test)]
            field_jobs += [dict(base, kind="zero", role="zero", seed=700 + i, ppp=p, snr=s, rms=a.rms)
                           for i in range(a.n_zero)]
    print(f"[spectral] {len(sine_jobs)} cặp sin + {len(field_jobs)} cặp trường, {len(cfgs)} cấu hình → {out}")
    sine = run_pool(job_sine, sine_jobs, a.workers)
    fields = run_pool(job_field, field_jobs, a.workers)

    import pandas as pd

    pd.DataFrame(sine).to_csv(out / "sine_mtf.csv", index=False)
    # Gộp phổ: trung bình tập hợp theo (kind, role, cfg, ppp, snr)
    keys = sorted({(r["kind"], r["role"], r["cfg"], r["ppp"], r["snr"]) for r in fields}, key=str)
    store = {}
    rows = []
    for kk in keys:
        g = [r for r in fields if (r["kind"], r["role"], r["cfg"], r["ppp"], r["snr"]) == kk]
        tag = "|".join(map(str, kk))
        store["k"] = g[0]["k"]
        for q_ in ("E_true", "E_meas", "S_et"):
            store[f"{tag}|{q_}"] = np.nanmean([r[q_] for r in g], axis=0)
        rows.append(dict(kind=kk[0], role=kk[1], cfg=kk[2], ppp=kk[3], snr=kk[4], n=len(g),
                         epe=float(np.nanmean([r["epe"] for r in g]))))
    np.savez_compressed(out / "spectra.npz", **store)
    pd.DataFrame(rows).to_csv(out / "fields_summary.csv", index=False)
    info = dict(vars(a), quick=q, cfgs=[c["label"] for c in cfgs], unavailable=unavailable,
                dns_source=a.dns_path or "dns2d (tự giải, pivbench.resolution.dns2d)",
                minutes=(time.time() - t0) / 60)
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str, ensure_ascii=False))
    analyze(out, a)


# ---------------------------------------------------------------------------
def analyze(out: Path, a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from pivbench.resolution import (TransferFunction, correct_spectrum, noise_floor_tail,
                                     noise_from_ensemble, valid_wavenumber)

    sine = pd.read_csv(out / "sine_mtf.csv")
    Z = dict(np.load(out / "spectra.npz"))
    k = Z["k"]
    fs = pd.read_csv(out / "fields_summary.csv")
    trows, krows = [], []
    for (cfg, ppp, snr), _ in fs.groupby(["cfg", "ppp", "snr"]):
        def get(kind, role, q):
            return Z.get(f"{kind}|{role}|{cfg}|{ppp}|{snr}|{q}")

        sm = sine[(sine.cfg == cfg) & (sine.ppp == ppp) & (sine.snr == snr)]
        mL = sm[sm.comp == "u"].groupby("lam").mtf.mean()
        mT = sm[sm.comp == "v"].groupby("lam").mtf.mean()
        H = {"sine_L": TransferFunction.from_mtf(mL.index, mL.values, "sine_L"),
             "sine_T": TransferFunction.from_mtf(mT.index, mT.values, "sine_T")}
        Et_tr, Em_tr, Se_tr = get("turb", "train", "E_true"), get("turb", "train", "E_meas"), get("turb", "train", "S_et")
        H["xs_turb"] = TransferFunction.from_cross_spectrum(k, Se_tr, Et_tr, "xs_turb")
        N_xs = noise_from_ensemble(Em_tr, Et_tr, H["xs_turb"](k))
        N_zero = get("zero", "zero", "E_meas")
        # Hàm truyền "thật" trên tập kiểm tra (chỉ để chẩn đoán)
        diag = {}
        for kind in ("turb", "dns"):
            Et, Se = get(kind, "test", "E_true"), get(kind, "test", "S_et")
            if Et is not None:
                diag[kind] = TransferFunction.from_cross_spectrum(k, Se, Et, f"xs_{kind}_test")
        for i, kv in enumerate(k):
            trows.append(dict(cfg=cfg, ppp=ppp, snr=snr, k=kv, wavelength=1 / kv,
                              H_sine_L=H["sine_L"](kv), H_sine_T=H["sine_T"](kv), H_xs_turb_train=H["xs_turb"](kv),
                              H_xs_turb_test=diag["turb"](kv) if "turb" in diag else np.nan,
                              H_xs_dns_test=diag["dns"](kv) if "dns" in diag else np.nan,
                              N_zero=N_zero[i] if N_zero is not None else np.nan, N_xs=N_xs[i]))

        for kind in ("turb", "dns"):
            Et, Em = get(kind, "test", "E_true"), get(kind, "test", "E_meas")
            if Et is None:
                continue
            kmax = 0.25 if kind == "turb" else 1.0 / 3.0
            band = Et > 1e-12 * np.nanmax(Et)
            kmax = min(kmax, k[band].max())
            N_tail = noise_floor_tail(k, Em)
            variants = {
                "meas (không hiệu chỉnh)": Em,
                "H_L sin + N_tail": correct_spectrum(k, Em, H["sine_L"], N_tail, eps=a.eps, h_min=a.h_min),
                "H_L sin + N_zero": correct_spectrum(k, Em, H["sine_L"], N_zero, eps=a.eps, h_min=a.h_min),
                "H_T sin + N_zero": correct_spectrum(k, Em, H["sine_T"], N_zero, eps=a.eps, h_min=a.h_min),
                "H_xs rối + N_xs": correct_spectrum(k, Em, H["xs_turb"], N_xs, eps=a.eps, h_min=a.h_min),
                "H_xs rối + N_xs (inverse)": correct_spectrum(k, Em, H["xs_turb"], N_xs, mode="inverse",
                                                             h_min=a.h_min),
            }
            k0 = valid_wavenumber(k, Em, Et, a.tol, k_max=kmax)
            for name, E in variants.items():
                kc = valid_wavenumber(k, E, Et, a.tol, k_max=kmax)
                sel = (k <= kmax) & np.isfinite(E) & (Et > 0)
                krows.append(dict(field=kind, cfg=cfg, ppp=ppp, snr=snr, variant=name, k10=kc,
                                  lambda10=1 / kc if kc > 0 else np.inf, k10_meas=k0,
                                  improvement=kc / k0 if k0 > 0 else np.nan,
                                  mean_abs_log_err=float(np.mean(np.abs(np.log(E[sel] / Et[sel]))))
                                  if sel.any() else np.nan))
            # Hình phổ
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            ax = axes[0]
            ax.loglog(k, Et, "k-", lw=2, label="E_true")
            ax.loglog(k, Em, "C3-", lw=1.5, label="E_meas")
            for (name, E), col in zip(list(variants.items())[1:], ["C0", "C1", "C2", "C4", "C5"]):
                ax.loglog(k, E, color=col, lw=1, ls="--" if "inverse" in name else "-", label="E_corr: " + name)
            if N_zero is not None:
                ax.loglog(k, np.maximum(N_zero, 1e-12), "C7:", lw=1, label="N_zero")
            ax.axvline(kmax, color="gray", lw=0.5)
            ax.set_ylim(np.nanmax(Et) * 1e-7, np.nanmax(Et) * 3)
            ax.set_xlabel("k [chu kỳ/px]")
            ax.set_ylabel("E(k)")
            ax.set_title(f"{kind} — {cfg}, ppp={ppp:g}, SNR={'∞' if not np.isfinite(snr) else f'{snr:g}'}")
            ax.legend(fontsize=6.5)
            ax.grid(alpha=0.3, which="both")
            ax = axes[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ax.semilogx(k, Em / Et, "C3-", lw=1.5, label="E_meas/E_true")
                for (name, E), col in zip(list(variants.items())[1:], ["C0", "C1", "C2", "C4", "C5"]):
                    ax.semilogx(k, E / Et, color=col, lw=1, ls="--" if "inverse" in name else "-", label=name)
            ax.axhspan(0.9, 1.1, color="gray", alpha=0.2)
            ax.set_ylim(0, 2)
            ax.set_xlim(k[0], kmax * 1.2)
            ax.set_xlabel("k [chu kỳ/px]")
            ax.set_ylabel("E / E_true")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=6.5)
            fig.tight_layout()
            fig.savefig(out / "figures" / f"spectra_{kind}_{cfg}_ppp{ppp:g}_snr{snr:g}.png", dpi=120)
            plt.close(fig)

        # Hình hàm truyền
        fig, ax = plt.subplots(figsize=(6, 4))
        kf = np.geomspace(k[0], 0.5, 300)
        ax.semilogx(kf, H["sine_L"](kf), "C0-", label="H_L (sin nén)")
        ax.semilogx(kf, H["sine_T"](kf), "C2-", label="H_T (sin cắt)")
        ax.semilogx(k, Se_tr / Et_tr, "C4.", ms=3, label="H_xs rối (train)")
        for kind, tf in diag.items():
            ax.semilogx(k, tf(k), "--", lw=0.8, label=f"H_xs {kind} (test, chẩn đoán)")
        ax.axhline(1, color="gray", lw=0.5)
        ax.set_ylim(-0.2, 1.3)
        ax.set_xlabel("k [chu kỳ/px]")
        ax.set_ylabel("H(k)")
        ax.set_title(f"{cfg}, ppp={ppp:g}, SNR={'∞' if not np.isfinite(snr) else f'{snr:g}'}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"transfer_{cfg}_ppp{ppp:g}_snr{snr:g}.png", dpi=120)
        plt.close(fig)

    pd.DataFrame(trows).to_csv(out / "transfer.csv", index=False)
    kd = pd.DataFrame(krows)
    kd.to_csv(out / "k10.csv", index=False)
    print(kd.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
