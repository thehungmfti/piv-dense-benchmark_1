#!/usr/bin/env python
"""Chạy benchmark theo file YAML: phương pháp × trường × ppp × SNR × tham số.

    python scripts/run_benchmark.py configs/example_small.yaml

Kết quả ghi vào ``results/<name>/``:
  * ``summary.csv``      — 1 dòng / (trường, ppp, SNR, seed, biến thể phương pháp)
  * ``mtf.csv``          — MTF(λ) và độ lệch pha cho trường sin
  * ``cutoffs.csv``      — λ_c tại các ngưỡng MTF (trung bình theo seed)
  * ``spectra/*.npz``    — phổ 1D/2D của trường gốc, ước lượng và sai số
  * ``fields/*.npz``     — trường gốc + ước lượng (+ ảnh nếu ``save_images``)
  * ``figures/*.png``    — phổ, MTF, bản đồ EPE
  * ``config.yaml``      — bản sao cấu hình đã dùng
"""

from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pivbench import metrics as M  # noqa: E402
from pivbench import plotting  # noqa: E402
from pivbench.methods import get_method  # noqa: E402
from pivbench.synth import generate_pair, make_field  # noqa: E402


def _as_list(x):
    if x is None:
        return [None]
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt(x) for x in v)
    return str(v)


def expand_methods(entries):
    out = []
    for e in entries:
        base = dict(e.get("params") or {})
        sweep = e.get("sweep") or {}
        keys = list(sweep)
        for combo in itertools.product(*[_as_list(sweep[k]) for k in keys]) if keys else [()]:
            params = {**base, **dict(zip(keys, combo))}
            label = e.get("name", e["method"])
            if keys:
                label += "[" + ",".join(f"{k}={_fmt(v)}" for k, v in zip(keys, combo)) + "]"
            out.append(dict(label=label, method=e["method"], params=params))
    return out


def expand_fields(entries):
    out = []
    for e in entries:
        kind = e["kind"]
        params = dict(e.get("params") or {})
        name = e.get("name", kind)
        if kind == "cai":  # mẫu thật từ bộ dữ liệu Cai et al. (ảnh + trường gốc có sẵn)
            from pivbench.data.cai import list_cai_samples

            root = Path(params["root"])
            root = root if root.is_absolute() else ROOT / root
            for cls in params.get("classes", ["cylinder"]):
                items = list_cai_samples(root, (cls,), params.get("split", "test"))[: params.get("n", 8)]
                for t in items:
                    out.append(dict(name=f"{name}_{cls}", case=t[2].name[: -len("_flow.flo")], kind=kind,
                                    params=dict(sample=t), mtf=False))
            continue
        if kind == "sine" and e.get("wavelengths"):
            for lam in e["wavelengths"]:
                out.append(dict(name=name, case=f"{name}_lam{_fmt(float(lam))}", kind=kind,
                                params={**params, "wavelength": float(lam)}, mtf=True))
        else:
            out.append(dict(name=name, case=name, kind=kind, params=params, mtf=False))
    return out


def run(cfg: dict, out_root: Path, quick: bool = False):
    name = cfg.get("name", "experiment")
    out = out_root / name
    for sub in ("spectra", "fields", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shape = tuple(cfg.get("shape", [256, 256]))
    border = int(cfg.get("border", 16))
    seeds = _as_list(cfg.get("seeds", [0]))
    synth = dict(cfg.get("synth") or {})
    ppps = _as_list(synth.pop("ppp", 0.05))
    snrs = _as_list(synth.pop("snr", None))
    mcfg = cfg.get("metrics") or {}
    thresholds = tuple(mcfg.get("mtf_thresholds", [0.9, 0.5]))
    gmask = mcfg.get("gradient_mask", {"quantity": "vorticity", "top_percent": 10})
    save_images = bool(cfg.get("save_images", False))
    methods = expand_methods(cfg["methods"])
    fields = expand_fields(cfg["fields"])
    if quick:
        seeds, ppps, snrs = seeds[:1], ppps[:1], snrs[:1]

    rows, mtf_rows = [], []
    unavailable = {}
    spec_groups: dict = {}
    t_start = time.time()
    done_dataset = set()
    for fc, ppp, snr, seed in itertools.product(fields, ppps, snrs, seeds):
        if fc["kind"] == "cai":
            if fc["case"] in done_dataset:  # ảnh có sẵn: không lặp theo ppp/SNR/seed
                continue
            done_dataset.add(fc["case"])
            from pivbench.data.cai import load_cai_sample

            i1, i2, uu, vv = load_cai_sample(fc["params"]["sample"])
            ppp, snr, seed = "dataset", "dataset", 0
            pair = dict(img1=i1, img2=i2, u=uu, v=vv, meta=dict(n_particles=-1))
            # Chỉ đánh giá nơi có hạt: một số ảnh Cai (cylinder) có dải đen hoàn toàn (cường độ 0, trường
            # gốc 0) — ở đó không có thông tin, mạng học từ chính bộ dữ liệu sẽ được lợi không công bằng.
            from scipy import ndimage

            valid = ndimage.maximum_filter(((i1 > 0) | (i2 > 0)).astype(np.uint8), size=9) > 0
        else:
            field = make_field(fc["kind"], shape, **fc["params"])
            pair = generate_pair(field, shape, ppp=ppp, snr=snr, seed=seed, **synth)
            valid = ~getattr(field, "body_mask", np.zeros(shape, bool))
            if not valid.all():  # thân mô hình: không có hạt, cường độ nền tối như khi huấn luyện
                rng = np.random.default_rng(seed + 99)
                for k in ("img1", "img2"):
                    pair[k] = np.where(valid, pair[k], 0.1 + rng.normal(0, 0.02, shape)).astype(np.float32)
        ut, vt = pair["u"], pair["v"]
        tag = f"{fc['case']}_ppp{_fmt(ppp)}_snr{_fmt(snr)}_s{seed}"
        print(f"[{time.time() - t_start:7.1f}s] {tag}: {pair['meta']['n_particles']} hạt", flush=True)
        save = dict(u_true=ut.astype(np.float32), v_true=vt.astype(np.float32))
        if save_images:
            save.update(img1=pair["img1"], img2=pair["img2"])
        est_plot = {}
        for m in methods:
            if m["label"] in unavailable:
                continue
            row = dict(field=fc["name"], case=fc["case"], ppp=ppp, snr=snr, seed=seed,
                       method=m["method"], label=m["label"], n_particles=pair["meta"]["n_particles"])
            try:
                t0 = time.time()
                u, v = get_method(m["method"])(pair["img1"], pair["img2"], **m["params"])
                row["runtime_s"] = time.time() - t0
            except Exception as e:  # noqa: BLE001 — ghi lại lỗi, không dừng cả benchmark
                msg = f"{type(e).__name__}: {e}"
                print(f"    {m['label']}: KHÔNG CHẠY ĐƯỢC — {msg}", flush=True)
                if type(e).__name__ == "RaftPIVUnavailable":
                    unavailable[m["label"]] = msg
                else:
                    traceback.print_exc()
                rows.append({**row, "status": "error", "error": msg})
                continue
            row["status"] = "ok"
            if not valid.all():  # trong thân mô hình: không đánh giá; phổ dùng giá trị gốc tại đó
                u = np.where(valid, u, ut)
                v = np.where(valid, v, vt)
            row.update(M.error_stats(u, v, ut, vt, border, valid))
            if fc["mtf"]:
                p = fc["params"]
                r = M.mtf_from_sine(u, v, p.get("amplitude", 1.0), p["wavelength"], p.get("component", "u"),
                                    p.get("axis", "x"), p.get("phase", 0.0), border)
                row.update(mtf=r["mtf"], phase_error=r["phase_error"])
                mtf_rows.append(dict(label=m["label"], method=m["method"], ppp=ppp, snr=snr, seed=seed, **r))
            elif fc["kind"] not in ("uniform",):
                row.update(M.strong_gradient_errors(u, v, ut, vt, gmask.get("quantity", "vorticity"),
                                                    gmask.get("top_percent", 10), border, valid))
                if fc["kind"] in ("lamb_oseen", "vortex"):
                    row.update(M.vortex_peak_error(u, v, ut, vt, border=border))
                spec = M.spectra_report(u, v, ut, vt, border)
                np.savez_compressed(out / "spectra" / f"{tag}__{_safe(m['label'])}.npz", **spec)
                if seed == seeds[0]:
                    spec_groups.setdefault((fc["case"], ppp, snr), {})[m["label"]] = spec
            print(f"    {m['label']:<40s} EPE={row['epe_mean']:.4f}  t={row['runtime_s']:.2f}s"
                  + (f"  MTF={row['mtf']:.3f}" if "mtf" in row else ""), flush=True)
            rows.append(row)
            save[f"u__{_safe(m['label'])}"] = u.astype(np.float32)
            save[f"v__{_safe(m['label'])}"] = v.astype(np.float32)
            est_plot[m["label"]] = (u, v)
        np.savez_compressed(out / "fields" / f"{tag}.npz", **save)
        if est_plot and seed == seeds[0] and not fc["mtf"]:
            plotting.plot_fields(ut, vt, est_plot, f"{fc['case']} — ppp={_fmt(ppp)}, SNR={_fmt(snr)}",
                                 out / "figures" / f"fields_{fc['case']}_ppp{_fmt(ppp)}_snr{_fmt(snr)}.png", border)

    _write_csv(out / "summary.csv", rows)
    if mtf_rows:
        _write_csv(out / "mtf.csv", mtf_rows)
        cut_rows = []
        for (lab, ppp, snr), grp in _group(mtf_rows, ("label", "ppp", "snr")).items():
            lam = sorted({r["wavelength"] for r in grp})
            mean_mtf = [np.mean([r["mtf"] for r in grp if r["wavelength"] == l]) for l in lam]
            cut_rows.append(dict(label=lab, ppp=ppp, snr=snr, **M.cutoffs(lam, mean_mtf, thresholds)))
        _write_csv(out / "cutoffs.csv", cut_rows)
        for (ppp, snr), grp in _group(mtf_rows, ("ppp", "snr")).items():
            avg = []
            for (lab, lam), g2 in _group(grp, ("label", "wavelength")).items():
                avg.append(dict(label=lab, wavelength=lam, mtf=np.mean([r["mtf"] for r in g2]),
                                phase_error=np.mean([r["phase_error"] for r in g2])))
            plotting.plot_mtf(avg, f"MTF trường sin — ppp={_fmt(ppp)}, SNR={_fmt(snr)}",
                              out / "figures" / f"mtf_ppp{_fmt(ppp)}_snr{_fmt(snr)}.png", thresholds)
    for (case, ppp, snr), specs in spec_groups.items():
        plotting.plot_spectra(specs, f"Phổ năng lượng — {case}, ppp={_fmt(ppp)}, SNR={_fmt(snr)}",
                              out / "figures" / f"spectra_{case}_ppp{_fmt(ppp)}_snr{_fmt(snr)}.png")
    if unavailable:
        (out / "unavailable.txt").write_text("\n".join(f"{k}: {v}" for k, v in unavailable.items()) + "\n")
    print(f"Xong sau {time.time() - t_start:.1f}s → {out}")
    return rows, mtf_rows


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.=" else "_" for c in s)


def _group(rows, keys):
    g: dict = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--name", help="ghi đè tên thí nghiệm")
    ap.add_argument("--quick", action="store_true", help="chỉ seed/ppp/SNR đầu tiên")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    if a.name:
        cfg["name"] = a.name
    out = a.out / cfg.get("name", "experiment")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "config.yaml")
    run(cfg, a.out, a.quick)


if __name__ == "__main__":
    main()
