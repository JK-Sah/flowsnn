"""
Run the lattice-Boltzmann calibration suite.

Writes results/cfd_summary.json (scalar findings for the paper's tables) and
results/cfd_cache.npz (probe time series reused as the CFD test domain).

    python run_cfd.py            # everything
    python run_cfd.py wake       # one configuration
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from flowsnn import config as C
from flowsnn import lbm

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def run_wake():
    print("== cylinder wake ==", flush=True)
    t0 = time.time()
    r = lbm.cylinder_wake(nx=C.LBM_WAKE["nx"], ny=C.LBM_WAKE["ny"],
                          diam=C.LBM_WAKE["diam"], re=C.LBM_WAKE["re"],
                          u_lb=C.LBM_WAKE["u_lb"],
                          n_steps=44000, record_from=14000)
    st, f_peak = lbm.measure_strouhal(r["uy"], r["diam"], r["u_lb"])
    dist, ratios = lbm.convection_ratio_profile(r["row"], r["cx"], r["diam"],
                                                r["u_lb"])
    # inside about two diameters the vortices have not yet formed, so a
    # convection speed is not defined there; those stations are excluded.
    valid = dist >= 2.0
    summary = dict(
        strouhal=st,
        f_peak_lu=f_peak,
        blockage=r["blockage"],
        omega=r["omega"],
        reynolds=C.LBM_WAKE["re"],
        conv_ratio_near=float(ratios[valid][0]),
        conv_ratio_far=float(ratios[valid][-1]),
        conv_ratio_mean=float(ratios[valid].mean()),
        conv_profile_x_over_d=dist[valid].tolist(),
        conv_profile_ratio=ratios[valid].tolist(),
        formation_length_excluded_d=2.0,
        elapsed_s=time.time() - t0,
    )
    print(json.dumps(summary, indent=2)[:600], flush=True)
    np.savez_compressed(OUT / "wake_cache.npz", row=r["row"], uy=r["uy"],
                        px=r["px"], cx=r["cx"], diam=r["diam"],
                        u_lb=r["u_lb"], f_peak=f_peak,
                        field_ux=r["field_ux"], field_uy=r["field_uy"],
                        mask=r["mask"])
    return summary


def run_dipole():
    print("== dipole ==", flush=True)
    t0 = time.time()
    r = lbm.dipole(nx=C.LBM_DIPOLE["nx"], ny=C.LBM_DIPOLE["ny"],
                   omega=C.LBM_DIPOLE["omega"],
                   force_amp=C.LBM_DIPOLE["force_amp"],
                   freq=C.LBM_DIPOLE["freq"],
                   n_steps=C.LBM_DIPOLE["n_steps"],
                   record_from=C.LBM_DIPOLE["record_from"])
    exponent, amp, _ = lbm.measure_decay_exponent(r["radii"], r["rad_amp"])
    sig = r["uy"][:, r["uy"].shape[1] // 2]
    spec = np.abs(np.fft.rfft(sig - sig.mean()))
    k = int(np.argmax(spec[1:])) + 1
    f_meas = lbm._parabolic_peak(spec, k) / len(sig)
    summary = dict(
        decay_exponent=exponent,
        rmse_inv_r2=lbm.profile_rmse(r["radii"], amp, 2.0),
        rmse_inv_r3=lbm.profile_rmse(r["radii"], amp, 3.0),
        source_freq_lu=r["freq"],
        measured_freq_lu=float(f_meas),
        freq_error_pct=float(100 * abs(f_meas - r["freq"]) / r["freq"]),
        elapsed_s=time.time() - t0,
    )
    print(json.dumps(summary, indent=2), flush=True)
    np.savez_compressed(OUT / "dipole_cache.npz", uy=r["uy"], px=r["px"],
                        radii=r["radii"], rad_amp=r["rad_amp"],
                        amp=amp, freq=r["freq"], field_uy=r["field_uy"])
    return summary


def run_viv():
    print("== VIV sweep ==", flush=True)
    t0 = time.time()
    rows = []
    for ur in C.LBM_VIV["ur_values"]:
        r = lbm.viv(ur, nx=C.LBM_VIV["nx"], ny=C.LBM_VIV["ny"],
                    diam=C.LBM_VIV["diam"], re=C.LBM_VIV["re"],
                    u_lb=C.LBM_VIV["u_lb"],
                    mass_ratio=C.LBM_VIV["mass_ratio"],
                    zeta=C.LBM_VIV["zeta"],
                    n_steps=C.LBM_VIV["n_steps"],
                    record_from=C.LBM_VIV["record_from"])
        r.pop("y_history")
        rows.append(r)
        print(f"  Ur={ur:.1f}  Yrms/D={r['y_rms_over_d']:.4f}"
              f"  f_wake={r['f_wake']:.6f}  f_struct={r['f_struct']:.6f}",
              flush=True)
    peak = max(rows, key=lambda d: d["y_rms_over_d"])
    summary = dict(sweep=rows, peak_ur=peak["ur"],
                   peak_y_rms_over_d=peak["y_rms_over_d"],
                   elapsed_s=time.time() - t0)
    return summary


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    path = OUT / "cfd_summary.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if which in ("all", "wake"):
        out["wake"] = run_wake()
    if which in ("all", "dipole"):
        out["dipole"] = run_dipole()
    if which in ("all", "viv"):
        out["viv"] = run_viv()
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
