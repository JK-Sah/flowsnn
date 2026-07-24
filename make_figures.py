"""
Regenerate every figure from results/*.json and results/*.npz.

    python make_figures.py

Colours are the Okabe-Ito subset blue / vermillion / green / charcoal, checked
for colour-vision-deficient separation (OKLab dE >= 8 under simulated
deuteranopia and protanopia for every pair). Categorical hues are assigned in
fixed order and never cycled.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flowsnn import config as C
from flowsnn import data as D

ROOT = Path(__file__).parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, VERM, GREEN, CHAR = "#0072B2", "#D55E00", "#009E73", "#333333"
SERIES = [BLUE, VERM, GREEN, CHAR]
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ── Fig 1: framework schematic ────────────────────────────────────────────────

def fig_framework():
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    boxes = [
        ("Flow source\n(uniform / wake / dipole)", "#eaf2fb"),
        ("Reduced-order FSI\ncompliant-pillar array\n(2N signals)", "#eaf6f0"),
        ("Phasic + tonic\nevent encoder\n(spike trains)", "#fdf6e3"),
        ("Spiking network\n(LIF, BPTT,\nsurrogate gradient)", "#fbeee8"),
        ("Structure class +\nbulk-flow vector\n(energy estimate)", "#f2eefb"),
    ]
    w, gap = 16.0, 4.0
    for i, (label, fc) in enumerate(boxes):
        x = 1 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 10), w, 12, facecolor=fc,
                                   edgecolor=MUTED, linewidth=0.9,
                                   zorder=2))
        ax.text(x + w / 2, 16, label, ha="center", va="center", fontsize=8,
                zorder=3)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.6, 16), xytext=(x + w + 0.6, 16),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.add_patch(plt.Rectangle((18, 3), 62, 5, facecolor="none",
                               edgecolor=GREEN, linestyle="--", linewidth=1.1))
    ax.text(49, 5.5, "lattice-Boltzmann CFD calibrates the source domain "
                     "and supplies the transfer test set",
            ha="center", va="center", fontsize=8, color=GREEN)
    ax.text(49, 26, "Training domain (reduced-order)  →  "
                    "test domain (reduced-order and CFD-derived)",
            ha="center", va="center", fontsize=8.5, color=MUTED)
    _save(fig, "Figure_1_Framework.png")


# ── Fig 2: example signals and spike raster ───────────────────────────────────

def fig_signals():
    sig, lab, _ = D.make_dataset(12, 0)
    std, _ = D.standardize(sig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.6),
                             gridspec_kw={"height_ratios": [1, 1.25]})
    tv = np.arange(C.T_STEPS) / C.FS
    for j, name in enumerate(D.CLASS_NAMES):
        idx = np.where(lab == j)[0][0]
        ax = axes[0, j]
        for ch in range(C.N_PIL, C.N_PIL + 4):
            ax.plot(tv, std[idx, :, ch], lw=0.8, color=SERIES[j], alpha=0.55)
        ax.set_title(name, color=INK)
        ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel("transverse deflection\n(standardised)")

        enc = D.encode(std[idx:idx + 1], "combined")[0]
        t_idx, c_idx = np.nonzero(enc)
        ax2 = axes[1, j]
        ax2.scatter(t_idx / C.FS, c_idx, s=0.7, color=SERIES[j], linewidths=0)
        ax2.set_ylim(0, C.N_CHAN_ENC)
        ax2.set_xlabel("time (s)")
        ax2.axhline(2 * C.N_CHAN_RAW, color=MUTED, lw=0.7, ls="--")
        if j == 0:
            ax2.set_ylabel("spike channel")
            ax2.text(0.02, 0.30 * C.N_CHAN_ENC, "phasic", fontsize=7.5,
                     color=MUTED)
            ax2.text(0.02, 0.80 * C.N_CHAN_ENC, "tonic", fontsize=7.5,
                     color=MUTED)
        ax2.set_title(f"activity {enc.mean() * 100:.1f}%", fontsize=8,
                      color=MUTED)
    fig.tight_layout()
    _save(fig, "Figure_2_Example_Signals_and_Spikes.png")


# ── Fig 3: confusion matrix and regression ────────────────────────────────────

def fig_confusion(results):
    entry = results["main_snn"]
    cm = np.array(entry["confusion_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), D.CLASS_NAMES)
    ax.set_yticks(range(3), D.CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("structure classification (3 seeds)")
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if cm[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")

    ax = axes[1]
    # the transfer panel is the ROM row of the generalization matrix, so this
    # figure and Section 3.3 cite the same run
    gmat = _load("generalize.json")
    if gmat is not None:
        classes = list(D.CLASS_NAMES)
        domains = [("ROM", "in-domain", BLUE),
                   ("2d", "CFD 2-D", GREEN),
                   ("re220", "CFD 3-D", VERM)]
        x = np.arange(len(classes))
        w = 0.26
        for i, (key, label, colour) in enumerate(domains):
            cell = gmat["ROM"][key]
            vals = [cell[c] * 100 for c in classes]
            ax.bar(x + (i - 1) * w, vals, w, color=colour, label=label)
        ax.axhline(100 / 3, color=MUTED, ls="--", lw=0.9)
        ax.set_xticks(x, classes)
        ax.set_ylabel("per-class accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("transfer to CFD flow fields, by class")
        ax.legend(fontsize=8, ncol=3, loc="lower center")
    fig.tight_layout()
    _save(fig, "Figure_3_Confusion_and_Transfer.png")


# ── Fig 4: accuracy versus energy ─────────────────────────────────────────────

def fig_accuracy_energy(results):
    order = [("main_snn", "SNN (events)", BLUE),
             ("baseline_mlp", "MLP", VERM),
             ("baseline_cnn", "CNN", GREEN),
             ("baseline_gru", "GRU", CHAR)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for key, label, colour in order:
        if key not in results:
            continue
        a = results[key]["agg"]
        e = a["energy_uJ_mean"]
        for ax, metric, err in ((axes[0], "acc_mean", "acc_std"),
                                (axes[1], "hmae_mean", "hmae_std")):
            y = a[metric] * (100 if metric == "acc_mean" else 1)
            ye = a[err] * (100 if metric == "acc_mean" else 1)
            ax.errorbar(e, y, yerr=ye, fmt="o", ms=9, color=colour,
                        ecolor=MUTED, elinewidth=1, capsize=3,
                        markeredgecolor="white", markeredgewidth=1.2,
                        label=label, zorder=3)
            ax.annotate(label, (e, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8.5, color=INK)
    axes[0].set_ylabel("structure accuracy (%)")
    axes[1].set_ylabel("heading MAE (deg)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("estimated compute energy per inference (µJ, log)")
        ax.margins(y=0.22)
    axes[1].set_yscale("log")
    fig.suptitle("Accuracy and motion error against estimated compute energy",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "Figure_4_Accuracy_Energy.png")


# ── Fig 5: encoding ablation ──────────────────────────────────────────────────

def fig_ablation(results):
    keys = [("abl_phasic", "phasic only"), ("abl_tonic", "tonic only"),
            ("abl_combined", "combined")]
    keys = [(k, n) for k, n in keys if k in results]
    if not keys:
        return
    names = [n for _, n in keys]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    panels = [("acc_mean", "acc_std", "structure accuracy (%)", 100, BLUE),
              ("smae_mean", "smae_std", "speed MAE (mm/s)", 1, VERM),
              ("hmae_mean", "hmae_std", "heading MAE (deg)", 1, GREEN)]
    for ax, (m, s, title, scale, colour) in zip(axes, panels):
        vals = [results[k]["agg"][m] * scale for k, _ in keys]
        errs = [results[k]["agg"][s] * scale for k, _ in keys]
        bars = ax.bar(names, vals, yerr=errs, color=colour, width=0.55,
                      capsize=4, error_kw=dict(ecolor=MUTED, lw=1))
        top = max(v + e for v, e in zip(vals, errs))
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + top * 0.04,
                    f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.set_title(title)
        ax.set_ylim(0, top * 1.22)
        ax.tick_params(axis="x", rotation=12)
        if m == "acc_mean":
            ax.axhline(100 / 3, color=MUTED, ls="--", lw=0.9)
            ax.text(-0.42, 100 / 3 + top * 0.03, "chance", fontsize=8,
                    color=MUTED, ha="left")
    fig.suptitle("Encoding ablation: the two pathways carry different tasks",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "Figure_5_Encoding_Ablation.png")


# ── Fig 6: CFD wake validation ────────────────────────────────────────────────

def fig_wake(cfd):
    cache = np.load(RES / "wake_cache.npz")
    w = cfd["wake"]
    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, :])
    field = cache["field_uy"].T
    lim = np.abs(field).max() * 0.6
    ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim,
              aspect="auto")
    mask = cache["mask"].T
    ax.contour(mask.astype(float), levels=[0.5], colors=[INK], linewidths=1.0)
    px = cache["px"]
    ax.plot(px, np.full_like(px, cache["field_uy"].shape[1] // 2 + 8), "o",
            ms=3.5, color=GREEN, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_title("transverse velocity: Karman street with the sensor-probe array")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y")
    ax.grid(False)

    ax = fig.add_subplot(gs[1, 0])
    sig = cache["uy"][:, 0] - cache["uy"][:, 0].mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=1.0) * float(cache["diam"]) / float(cache["u_lb"])
    ax.plot(freqs, spec / spec.max(), color=BLUE, lw=1.2)
    ax.axvline(w["strouhal"], color=VERM, ls="--", lw=1.1)
    ax.text(w["strouhal"], 1.02, f" St = {w['strouhal']:.3f}", color=VERM,
            fontsize=8.5)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Strouhal number")
    ax.set_ylabel("normalised spectrum")
    ax.set_title("shedding spectrum")

    ax = fig.add_subplot(gs[1, 1])
    d = np.array(w["conv_profile_x_over_d"])
    r = np.array(w["conv_profile_ratio"])
    ax.plot(d, r, "o-", color=BLUE, ms=5, lw=1.4,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.axhline(0.85, color=VERM, ls="--", lw=1.1)
    ax.text(d[-1], 0.855, "reduced-model value 0.85", color=VERM, fontsize=8,
            ha="right")
    ax.set_xlabel("downstream distance x/D")
    ax.set_ylabel("convection ratio $U_c/U$")
    ax.set_title("wake convection ratio recovers downstream")
    ax.set_ylim(0.8, 1.02)
    fig.tight_layout()
    _save(fig, "Figure_6_CFD_Wake_Validation.png")


# ── Fig 7: dipole validation ──────────────────────────────────────────────────

def fig_dipole(cfd):
    cache = np.load(RES / "dipole_cache.npz")
    dp = cfd["dipole"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    ax = axes[0]
    field = cache["field_uy"].T
    lim = np.abs(field).max() * 0.4
    ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim,
              aspect="auto")
    ax.set_title("dipole transverse velocity field")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y")
    ax.grid(False)

    ax = axes[1]
    radii = cache["radii"].astype(float)
    amp = cache["amp"]
    ax.loglog(radii, amp / amp.max(), "o", color=BLUE, ms=5,
              markeredgecolor="white", markeredgewidth=0.8, label="LBM")
    for expo, colour, ls in ((2.0, GREEN, "-"), (3.0, VERM, "--")):
        model = radii ** (-expo)
        ax.loglog(radii, model / model.max(), ls, color=colour, lw=1.3,
                  label=f"$1/r^{int(expo)}$")
    ax.set_xlabel("radius (lattice units)")
    ax.set_ylabel("normalised amplitude")
    ax.set_title(f"decay exponent {dp['decay_exponent']:.2f}")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(["$1/r^2$", "$1/r^3$"],
           [dp["rmse_inv_r2"], dp["rmse_inv_r3"]],
           color=[GREEN, VERM], width=0.5)
    ax.set_ylabel("profile RMSE")
    ax.set_title("2-D law fits the measured profile")
    for i, v in enumerate([dp["rmse_inv_r2"], dp["rmse_inv_r3"]]):
        ax.text(i, v * 1.03, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.margins(y=0.2)
    fig.tight_layout()
    _save(fig, "Figure_7_Dipole_Validation.png")


# ── Fig 8: VIV sweep ──────────────────────────────────────────────────────────

def fig_viv(cfd):
    v = cfd.get("viv")
    if not v:
        return
    sweep = v["sweep"]
    ur = np.array([r["ur"] for r in sweep])
    y = np.array([r["y_rms_over_d"] for r in sweep])
    fw = np.array([r["f_wake"] for r in sweep])
    fn = np.array([r["f_nat"] for r in sweep])
    st = cfd["wake"]["strouhal"]
    f_shed = st * 0.05 / 20.0

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    ax.plot(ur, fw / f_shed, "o-", color=BLUE, ms=5, lw=1.4,
            markeredgecolor="white", markeredgewidth=0.8, label="wake")
    ax.plot(ur, fn / f_shed, "s--", color=CHAR, ms=5, lw=1.2,
            markeredgecolor="white", markeredgewidth=0.8, label="structure natural")
    ax.axhline(1.0, color=VERM, ls=":", lw=1.1)
    ax.set_xlabel("reduced velocity $U_r$")
    ax.set_ylabel("frequency / rigid shedding frequency")
    ax.set_title("wake frequency is independent of $U_r$: no lock-in")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ur, y, "o-", color=BLUE, ms=5, lw=1.4,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(1.0 / st, color=VERM, ls="--", lw=1.1)
    ax.text(1.0 / st, ax.get_ylim()[1] * 0.95, "  $1/St$", color=VERM,
            fontsize=8.5)
    ax.set_xlabel("reduced velocity $U_r$")
    ax.set_ylabel("$Y_{rms}/D$")
    ax.set_title("response amplitude")
    fig.tight_layout()
    _save(fig, "Figure_8_VIV_Sweep.png")


# ── Fig 9: dimensionality of the dipole decay law ─────────────────────────────

def fig_dipole_dimensionality(cfd, cfd3d):
    """The 2-D and 3-D decay laws side by side.

    This is the figure that settles whether 1/r^3 was wrong or merely applied to
    a simulation of the wrong dimensionality.
    """
    path = RES / "dipole3d_cache.npz"
    if not path.exists() or cfd3d is None:
        return
    c3 = np.load(path)
    c2 = np.load(RES / "dipole_cache.npz")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    panels = [
        (c2["radii"].astype(float), c2["amp"], None, None,
         cfd["dipole"]["decay_exponent"], "2-D line dipole", 2.0),
        (c3["radii"].astype(float), c3["amp"],
         cfd3d["fit_r_min"], cfd3d["fit_r_max"],
         cfd3d["decay_exponent"], "3-D oscillating sphere", 3.0),
    ]
    for ax, (radii, amp, r_lo, r_hi, expo, title, theory) in zip(axes, panels):
        fitted = np.ones_like(radii, dtype=bool)
        if r_lo is not None:
            fitted = (radii >= r_lo) & (radii <= r_hi)
        norm = amp / amp[fitted].max()
        ax.loglog(radii[fitted], norm[fitted], "o", color=BLUE, ms=5,
                  markeredgecolor="white", markeredgewidth=0.8, label="LBM")
        if (~fitted).any():
            ax.loglog(radii[~fitted], norm[~fitted], "o", color=MUTED, ms=4,
                      alpha=0.45, markeredgecolor="none",
                      label="excluded from fit")
        rr = radii[fitted]
        for e, colour, ls in ((2.0, GREEN, "-"), (3.0, VERM, "--")):
            model = rr ** (-e)
            ax.loglog(rr, model / model.max(), ls, color=colour, lw=1.3,
                      label=f"$1/r^{int(e)}$")
        ax.set_xlabel("radius (lattice units)")
        ax.set_ylabel("normalised amplitude")
        ax.set_title(f"{title}: fitted exponent {expo:.2f}"
                     f"  (theory {theory:.0f})")
        ax.legend(fontsize=8)
    fig.suptitle("The decay exponent follows the dimensionality of the source",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "Figure_9_Dipole_Dimensionality.png")


# ── Fig 10: spanwise structure of the 3-D wake ────────────────────────────────

def fig_wake_spanwise(cfd3d_wake):
    path = RES / "wake3d_re220_cache.npz"
    if not path.exists() or cfd3d_wake is None:
        return
    c = np.load(path)
    uy = c["uy"]                      # (time, span, probe)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    t = np.arange(min(3000, uy.shape[0]))
    for k in range(uy.shape[1]):
        ax.plot(t, uy[:len(t), k, 0] + k * 4 * uy.std(), lw=0.7,
                color=BLUE, alpha=0.8)
    ax.set_xlabel("lattice step")
    ax.set_ylabel("transverse velocity (offset by station)")
    ax.set_title("probe signal at eight spanwise stations")
    ax.set_yticks([])

    ax = axes[1]
    coh = cfd3d_wake["spanwise_coherence_per_probe"]
    xs = (c["px"] - int(c["cx"])) / float(c["diam"])
    ax.plot(xs, coh, "o-", color=BLUE, ms=5, lw=1.4,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.axhline(1.0, color=VERM, ls="--", lw=1.1)
    ax.text(xs[-1], 1.002, "spanwise-uniform (2-D)", color=VERM, fontsize=8,
            ha="right")
    ax.set_xlabel("downstream distance x/D")
    ax.set_ylabel("mean inter-station correlation")
    ax.set_title(f"spanwise coherence, Re = {cfd3d_wake['re']}")
    fig.tight_layout()
    _save(fig, "Figure_10_Wake_Spanwise.png")


# ── Fig 11: train-on-CFD generalization matrix ────────────────────────────────

def fig_generalization():
    g = _load("generalize.json")
    if g is None:
        return
    doms = ["ROM", "2d", "re220", "re300", "re380"]
    labels = ["ROM", "2-D", "Re220", "Re300", "Re380"]
    mat = np.array([[g[tr][te]["wake"] * 100 for te in doms] for tr in doms])

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(mat, cmap="viridis", vmin=20, vmax=90, aspect="equal")
    ax.set_xticks(range(len(doms)), labels)
    ax.set_yticks(range(len(doms)), labels)
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.set_title("wake-class accuracy: train-on-A, test-on-B (%)")
    ax.grid(False)
    for i in range(len(doms)):
        for j in range(len(doms)):
            v = mat[i, j]
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                    fontweight=weight,
                    color="white" if v < 55 else INK)
    # frame the diagonal (on-distribution) cells
    for i in range(len(doms)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=VERM, lw=1.8))
    fig.colorbar(im, ax=ax, fraction=0.046, label="wake accuracy (%)")
    fig.tight_layout()
    _save(fig, "Figure_11_Generalization_Matrix.png")


def main():
    results = _load("results.json")
    cfd = _load("cfd_summary.json")
    cfd3d_dip = _load("cfd3d_dipole.json")
    cfd3d_wake = _load("cfd3d_wake_re220.json")
    print("generating figures")
    fig_framework()
    fig_signals()
    if results:
        fig_confusion(results)
        fig_accuracy_energy(results)
        fig_ablation(results)
    if cfd:
        fig_wake(cfd)
        fig_dipole(cfd)
        fig_viv(cfd)
        fig_dipole_dimensionality(cfd, cfd3d_dip)
    fig_wake_spanwise(cfd3d_wake)
    fig_generalization()


if __name__ == "__main__":
    main()
