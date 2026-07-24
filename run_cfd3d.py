"""
Three-dimensional lattice-Boltzmann runs. Needs a GPU.

    python run_cfd3d.py dipole [--n 256] [--steps 24000]
    python run_cfd3d.py wake   [--steps 60000]
    python run_cfd3d.py smoke          # tiny run, checks the solver end to end

Writes results/cfd3d_<case>.json and results/<case>3d_cache.npz.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flowsnn import lbm3d as L

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def run_dipole(n, steps, record_from, radius, progress):
    t0 = time.time()
    r = L.dipole_3d(n=n, radius=radius, n_steps=steps,
                    record_from=record_from, progress=progress)
    # exclude the outer 20% of the box, where the open boundary distorts
    expo, amp, keep, r_sq = L.fit_decay(r["radii"], r["rake"],
                                       radius=radius, r_max=0.8 * (n // 2))
    sig = r["rake"][:, len(r["radii"]) // 3]
    spec = np.abs(np.fft.rfft(sig - sig.mean()))
    k = int(np.argmax(spec[1:])) + 1
    f_meas = k / len(sig)
    summary = dict(
        case="dipole_3d", n=n, steps=steps, source_radius=radius,
        decay_exponent=expo, fit_r_squared=r_sq,
        rmse_inv_r2=L.profile_rmse(r["radii"], amp, keep, 2.0),
        rmse_inv_r3=L.profile_rmse(r["radii"], amp, keep, 3.0),
        rmse_inv_r4=L.profile_rmse(r["radii"], amp, keep, 4.0),
        n_fit_points=int(keep.sum()),
        fit_r_min=float(r["radii"][keep].min()),
        fit_r_max=float(r["radii"][keep].max()),
        source_freq_lu=r["freq"], measured_freq_lu=float(f_meas),
        freq_error_pct=float(100 * abs(f_meas - r["freq"]) / r["freq"]),
        on_gpu=L.ON_GPU, elapsed_s=time.time() - t0,
    )
    np.savez_compressed(OUT / "dipole3d_cache.npz", radii=r["radii"],
                        rake=r["rake"].astype(np.float32), amp=amp, keep=keep)
    return summary


def wake_grid_for_re(re, u_lb=0.05, target_omega=1.94):
    """Pick cylinder diameter and domain so the relaxation rate stays safe.

    nu = u_lb * D / Re, and omega = 1/(3 nu + 0.5) climbs toward its stability
    limit of 2 as Re rises at fixed geometry. Holding omega near 1.94 by scaling
    D with Re keeps every run in the sweep equally stable, at the cost of a
    larger grid at higher Re. Domain proportions match the Re = 220 reference
    (nx/D = 16, ny/D = nz/D = 16/3).
    """
    nu = (1.0 / target_omega - 0.5) / 3.0
    diam = max(20, int(round(nu * re / u_lb)))
    nx = int(round(16 * diam))
    ny = nz = int(round(16 * diam / 3))
    # keep dimensions even for clean spanwise periodicity
    ny += ny % 2
    nz += nz % 2
    return dict(diam=diam, nx=nx, ny=ny, nz=nz)


def run_wake(steps, record_from, progress, re=220):
    t0 = time.time()
    g = wake_grid_for_re(re)
    print(f"  Re={re}: diam={g['diam']} grid={g['nx']}x{g['ny']}x{g['nz']}"
          f" ({g['nx']*g['ny']*g['nz']/1e6:.1f}M cells)", flush=True)
    r = L.wake_3d(re=re, n_steps=steps, record_from=record_from,
                  progress=progress, **g)
    coh_mean, coh_per_probe = L.spanwise_coherence(r["uy"])
    ref = r["uy"][:, r["uy"].shape[1] // 2, 0]
    spec = np.abs(np.fft.rfft((ref - ref.mean()) * np.hanning(len(ref))))
    k = int(np.argmax(spec[1:])) + 1
    f_peak = k / len(ref)
    summary = dict(
        case="wake_3d", re=r["re"], nx=r["nx"], ny=r["ny"], nz=r["nz"],
        strouhal=float(f_peak * r["diam"] / r["u_lb"]), f_peak_lu=float(f_peak),
        spanwise_coherence=coh_mean,
        spanwise_coherence_per_probe=coh_per_probe,
        probe_rms=float(r["uy"].std()),
        on_gpu=L.ON_GPU, elapsed_s=time.time() - t0,
    )
    # tag the cache by Reynolds number so the sweep produces distinct domains.
    # This deliberately does not overwrite wake3d_cache.npz, which is the
    # finalised Re = 220 reference behind the transfer figure in the paper.
    np.savez_compressed(OUT / f"wake3d_re{re}_cache.npz", uy=r["uy"],
                        row=r["row"], row_every=r["row_every"], px=r["px"],
                        pz=r["pz"], cx=r["cx"], diam=r["diam"], u_lb=r["u_lb"],
                        f_peak=f_peak, re=re)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["dipole", "wake", "smoke"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--re", type=int, default=220,
                    help="Reynolds number for the wake case")
    ap.add_argument("--progress", type=int, default=2000)
    args = ap.parse_args()

    print(f"GPU backend: {L.ON_GPU}", flush=True)
    if args.case == "smoke":
        s = run_dipole(64, 1200, 600, 4.0, 400)
        print(json.dumps(s, indent=2), flush=True)
        w = run_wake(1500, 900, 500)
        print(json.dumps(w, indent=2), flush=True)
        return

    if args.case == "dipole":
        steps = args.steps or 24000
        s = run_dipole(args.n, steps, steps // 2, args.radius, args.progress)
        path = OUT / "cfd3d_dipole.json"
    else:
        steps = args.steps or 60000
        s = run_wake(steps, steps // 2, args.progress, re=args.re)
        path = OUT / f"cfd3d_wake_re{args.re}.json"
    path.write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
