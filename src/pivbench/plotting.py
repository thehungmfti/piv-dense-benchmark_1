"""Hình PNG cho kết quả benchmark (matplotlib, nền sáng)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Bảng màu phân loại cố định (thứ tự không đổi), mực chữ trung tính
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRUE_INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "font.size": 9, "legend.frameon": False,
})


def method_colors(labels):
    return {lab: SERIES[i % len(SERIES)] for i, lab in enumerate(labels)}


def plot_spectra(spec_by_method: dict, title: str, path):
    """spec_by_method: {label: spectra_report dict}. Ba ô: E(k_x), E(k_y), E(|k|) 2D.

    Đường đen = trường gốc; màu liền = ước lượng; màu đứt = sai số (u_est − u_true).
    """
    labels = list(spec_by_method)
    cols = method_colors(labels)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    keys = [("kx", "E_x", "E(k_x) — phổ 1D theo x"), ("ky", "E_y", "E(k_y) — phổ 1D theo y"),
            ("k2d", "E2d", "E(|k|) — phổ 2D trung bình vòng")]
    for ax, (kk, ek, ttl) in zip(axs, keys):
        first = spec_by_method[labels[0]]
        k = first[kk]
        sl = k > 0
        ax.loglog(k[sl], first[f"true_{ek}"][sl], color=TRUE_INK, lw=2, label="gốc (truth)")
        for lab in labels:
            s = spec_by_method[lab]
            ax.loglog(k[sl], s[f"est_{ek}"][sl], color=cols[lab], lw=2, label=f"{lab}")
            ax.loglog(k[sl], s[f"err_{ek}"][sl], color=cols[lab], lw=1.5, ls="--", label=f"{lab} — sai số")
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.set_xlabel("k [chu kỳ/px]")
        sec = ax.secondary_xaxis("top", functions=(lambda x: 1 / np.maximum(x, 1e-9),
                                                   lambda x: 1 / np.maximum(x, 1e-9)))
        sec.set_xlabel("λ [px]", color=MUTED)
    axs[0].set_ylabel("mật độ phổ [px²·px]")
    axs[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mtf(mtf_rows: list[dict], title: str, path, thresholds=(0.9, 0.5)):
    """mtf_rows: dict có khoá label, wavelength, mtf, phase_error."""
    labels = list(dict.fromkeys(r["label"] for r in mtf_rows))
    cols = method_colors(labels)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for lab in labels:
        rs = sorted((r for r in mtf_rows if r["label"] == lab), key=lambda r: r["wavelength"])
        lam = [r["wavelength"] for r in rs]
        a1.semilogx(lam, [r["mtf"] for r in rs], "-o", color=cols[lab], ms=5, label=lab)
        a2.semilogx(lam, np.degrees([r["phase_error"] for r in rs]), "-o", color=cols[lab], ms=5, label=lab)
    for t in thresholds:
        a1.axhline(t, color=MUTED, lw=1, ls=":")
        a1.text(4.2, t + 0.02, f"MTF = {t:g}", color=MUTED, fontsize=8)
    a1.axhline(1.0, color=TRUE_INK, lw=1)
    a1.set_ylim(-0.05, 1.15)
    a1.set_xlabel("bước sóng λ [px]")
    a1.set_ylabel("MTF = A_est / A_true")
    a1.set_title("MTF", fontsize=9, loc="left")
    a2.axhline(0, color=TRUE_INK, lw=1)
    a2.set_xlabel("bước sóng λ [px]")
    a2.set_ylabel("độ lệch pha [°]")
    a2.set_title("Độ lệch pha", fontsize=9, loc="left")
    a1.legend(loc="lower right")
    for a in (a1, a2):
        a.set_xticks([4, 8, 16, 32, 64, 128, 256])
        a.set_xticklabels(["4", "8", "16", "32", "64", "128", "256"])
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fields(u_true, v_true, est: dict, title: str, path, border: int = 0):
    """Hàng 1: |u| gốc và ước lượng; hàng 2: EPE từng phương pháp."""
    labels = list(est)
    n = len(labels) + 1
    fig, axs = plt.subplots(2, n, figsize=(3.0 * n, 6.0), constrained_layout=True)
    sp = np.hypot(u_true, v_true)
    vmax = float(np.nanmax(sp))
    epes = {lab: np.hypot(u - u_true, v - v_true) for lab, (u, v) in est.items()}
    emax = float(np.nanpercentile(np.concatenate([e.ravel() for e in epes.values()]), 99)) or 1.0
    im = axs[0, 0].imshow(sp, cmap="Blues", vmin=0, vmax=vmax)
    axs[0, 0].set_title("|u| gốc", fontsize=9)
    for i, lab in enumerate(labels, start=1):
        u, v = est[lab]
        axs[0, i].imshow(np.hypot(u, v), cmap="Blues", vmin=0, vmax=vmax)
        axs[0, i].set_title(f"|u| {lab}", fontsize=9)
        ie = axs[1, i].imshow(epes[lab], cmap="Oranges", vmin=0, vmax=emax)
        axs[1, i].set_title(f"EPE {lab}", fontsize=9)
    fig.colorbar(im, ax=axs[0, :], shrink=0.8, label="px")
    fig.colorbar(ie, ax=axs[1, :], shrink=0.8, label="EPE [px]")
    axs[1, 0].axis("off")
    for a in axs.ravel():
        a.grid(False)
        a.set_xticks([])
        a.set_yticks([])
        if border:
            ny, nx = u_true.shape
            a.add_patch(plt.Rectangle((border - 0.5, border - 0.5), nx - 2 * border, ny - 2 * border,
                                      fill=False, ec=MUTED, lw=0.8, ls="--"))
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=110)
    plt.close(fig)
