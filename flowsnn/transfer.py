"""
CFD-driven test domain.

The decoder is trained on the reduced-order model and tested on signals whose
fluctuation structure comes from the lattice-Boltzmann solver instead. This is
the sim-to-sim step: the task definition, the sensor model, the noise model and
the decoder are all unchanged, and only the origin of the class-carrying
fluctuation is swapped.

What actually changes, by class:

    uniform   nothing. The class is defined by the absence of a coherent
              fluctuation, so there is nothing to substitute. Reported for
              completeness, not as evidence of transfer.
    wake      the analytic travelling wave is replaced by transverse velocity
              sampled from the simulated Karman street at eight stations spaced
              like the pillar array, retimed so that its shedding frequency
              matches St*U/D for the sample's bulk speed. This carries the real
              phase structure, harmonic content and cycle-to-cycle jitter that
              the reduced model idealises away.
    dipole    the analytic 1/r^2 profile is replaced by the measured radial
              amplitude profile of the simulated force-dipole, interpolated to
              a randomised source position.

Bulk advection, turbulence, sensor noise and the pillar transfer function are
applied exactly as in the reduced-order model, so the comparison isolates the
fluctuation field.
"""

import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from . import config as C
from . import data as D

CACHE_DIR = Path(__file__).resolve().parent.parent / "results"
_px = np.asarray(C.PILLAR_X)


def _load(name):
    path = CACHE_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python run_cfd.py` first")
    return np.load(path)


def _antialias(series, stride):
    """Low-pass a series before decimating it by `stride`.

    Sampling every `stride`-th point without this folds everything above the new
    Nyquist back down into the band of interest. For a near-sinusoidal 2-D wake
    there is little energy up there and it hardly matters; for a broadband 3-D
    wake it manufactures harmonics that are not in the flow, which would look
    exactly like a physical difference between the two domains.
    """
    if stride <= 1.0:
        return series
    # cutoff as a fraction of the original Nyquist, with a margin below the
    # post-decimation Nyquist of 1/stride
    wn = min(0.9 / stride, 0.99)
    b, a = butter(4, wn, btype="low")
    return filtfilt(b, a, series, axis=0)


def _resample(series, cycles_per_step, target_cycles_per_sample, n_out, rng):
    """Resample a CFD time series so its dominant frequency hits a target.

    series: (n_steps, n_probe). Returns (n_out, n_probe).
    """
    stride = target_cycles_per_sample / cycles_per_step
    series = _antialias(series, stride)
    n_steps = series.shape[0]
    span = stride * (n_out - 1)
    if span >= n_steps - 1:
        # not enough recorded signal at this stride; wrap around instead of
        # extrapolating, the series is statistically stationary
        start = rng.uniform(0, n_steps - 1)
        idx = (start + stride * np.arange(n_out)) % (n_steps - 1)
    else:
        start = rng.uniform(0, n_steps - 1 - span)
        idx = start + stride * np.arange(n_out)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n_steps - 1)
    frac = (idx - lo)[:, None]
    return series[lo] * (1 - frac) + series[hi] * frac


def physical_spacing_lu(diam):
    """Probe spacing that maps the physical sensor geometry onto the lattice.

    The pillar array has a fixed spacing-to-body-diameter ratio; mapping that
    ratio onto the simulated cylinder is what a real sensor placed in the wake
    would sample. The resulting phase ramp is then whatever the flow produces at
    that geometry, and it is deliberately *not* tuned to reproduce the
    reduced-order model's ramp: if the simulated wake sheds at a different
    Strouhal number, the ramp differs, and that difference is a genuine part of
    the domain shift the transfer test is meant to expose.
    """
    return max(1, int(round(C.SPACING / C.D_BODY * diam)))


def wake_phase_ramp(cache, wake_source, n_probe=C.N_PIL):
    """Report the phase ramp the array spans, in shedding wavelengths.

    Determined by the shedding wavelength and the physical probe span, not by a
    cross-correlation lag, which wraps modulo one wavelength and understates a
    ramp above a full wavelength.
    """
    diam = float(cache["diam"])
    u_lb = float(cache["u_lb"])
    every = int(cache["row_every"]) if "row_every" in cache else 1
    f_peak = float(cache["f_peak"]) * (every if wake_source == "3d" else 1)
    wavelength = C.CONV_RATIO * u_lb / f_peak
    span = physical_spacing_lu(diam) * (n_probe - 1)
    return span / wavelength


def _wake_bank(cache, n_probe=C.N_PIL):
    """Extract probe windows from the recorded 2-D wake row.

    Stations are taken downstream of the vortex formation region at the physical
    sensor spacing.
    """
    row = cache["row"]
    cx = int(cache["cx"])
    diam = float(cache["diam"])
    spacing_lu = physical_spacing_lu(diam)
    banks = []
    for offset_d in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
        x0 = int(cx + offset_d * diam)
        px = x0 + np.arange(n_probe) * spacing_lu
        if px[-1] >= row.shape[1] - 1:
            continue
        win = row[:, px].astype(np.float64)
        win = win - win.mean(axis=0, keepdims=True)
        rms = win.std() + 1e-12
        banks.append(win / rms)          # unit RMS, spatial structure retained
    if not banks:
        raise RuntimeError("no valid probe stations in the recorded wake row")
    return banks, float(cache["f_peak"])


def _dipole_profile(cache):
    """Measured radial amplitude profile, normalised, as a callable."""
    radii = cache["radii"].astype(float)
    amp = cache["amp"].astype(float)
    amp = amp / amp.max()
    # radii are in lattice units where the source stand-off sweep is defined;
    # express as a function of r / r_min so it can be rescaled to SI
    r_norm = radii / radii[0]

    def profile(r_over_rmin):
        return np.interp(np.clip(r_over_rmin, r_norm[0], r_norm[-1]),
                         r_norm, amp)

    return profile


def _wake_bank_3d(cache, n_probe=C.N_PIL):
    """Probe windows from the three-dimensional wake.

    Uses the recorded streamwise row, (time, spanwise station, x), so probe
    spacing can be phase-matched to the modelled array exactly as for the 2-D
    bank. Each spanwise station and streamwise offset gives one window, and
    because the wake is above the onset of spanwise instability those windows
    are not copies of each other.
    """
    if "row" not in cache:
        raise KeyError(
            "wake3d_cache.npz has no 'row' array - it was written by an older "
            "version of wake_3d that fixed the probe rake at run time. Re-run "
            "`python run_cfd3d.py wake`.")
    row = cache["row"]
    cx = int(cache["cx"])
    diam = float(cache["diam"])
    # physical sensor spacing, same as the 2-D bank; the row was subsampled in
    # time, so the per-recorded-sample frequency is scaled and returned for the
    # resampler
    every = int(cache["row_every"]) if "row_every" in cache else 1
    spacing = physical_spacing_lu(diam)
    f_peak_eff = float(cache["f_peak"]) * every
    banks = []
    for k in range(row.shape[1]):
        for offset_d in (2.0, 2.5, 3.0, 3.5, 4.0):
            x0 = int(cx + offset_d * diam)
            px = x0 + np.arange(n_probe) * spacing
            if px[-1] >= row.shape[2] - 1:
                continue
            win = row[:, k, :][:, px].astype(np.float64)
            win = win - win.mean(axis=0, keepdims=True)
            rms = win.std() + 1e-12
            banks.append(win / rms)
    if not banks:
        raise RuntimeError("no valid probe stations in the 3-D wake row")
    return banks, f_peak_eff


def _wake_banks_for(wake_source):
    """Resolve a wake_source string to (probe banks, per-sample peak frequency).

        "2d"        the 2-D Karman street
        "3d"        the finite-span 3-D wake at Re = 220 (canonical cache)
        "re<N>"     the swept 3-D wake at Reynolds number N
    """
    if wake_source == "2d":
        return _wake_bank(_load("wake_cache.npz"))
    if wake_source == "3d":
        # the canonical 3-D wake is the Re = 220 run from the Reynolds sweep,
        # so the transfer test (Section 3.3) and the generalisation study
        # (Section 3.4) cite one run at one resolution
        return _wake_bank_3d(_load("wake3d_re220_cache.npz"))
    if wake_source.startswith("re"):
        return _wake_bank_3d(_load(f"wake3d_{wake_source}_cache.npz"))
    raise ValueError(f"unknown wake_source: {wake_source}")


def make_cfd_dataset(per_class, seed, wake_source="2d"):
    """Build a CFD-driven dataset with the same task definition as the ROM.

    wake_source selects which simulation supplies the wake fluctuation (see
    _wake_banks_for). Uniform and dipole classes are unchanged across sources,
    so a dataset built from one Reynolds number differs from another only in the
    wake class, which is what the train-on-CFD generalisation test relies on.
    """
    dip_cache = _load("dipole_cache.npz")
    banks, f_peak = _wake_banks_for(wake_source)
    profile = _dipole_profile(dip_cache)

    rng = np.random.default_rng(seed + 9000)
    sigs, labs, tgts = [], [], []

    # ── uniform: unchanged from the reduced-order model ───────────────────────
    sigs_u, labs_u, tgts_u = D.gen_uniform(per_class, rng)
    sigs.append(sigs_u)
    labs.append(labs_u)
    tgts.append(tgts_u)

    # ── wake: CFD probe series, retimed per sample ────────────────────────────
    speed = rng.uniform(C.U_MIN, C.U_MAX, per_class)
    heading = rng.uniform(-math.pi, math.pi, per_class)
    ux = np.empty((per_class, C.T_STEPS, C.N_PIL))
    uy = np.empty((per_class, C.T_STEPS, C.N_PIL))
    for i in range(per_class):
        bank = banks[rng.integers(len(banks))]
        f_shed = C.STROUHAL * speed[i] / C.D_BODY          # Hz
        fluct = _resample(bank, f_peak, f_shed / C.FS, C.T_STEPS, rng)
        # match the reduced model's amplitude convention: a sinusoid of
        # amplitude WAKE_FLUC_FRAC*U has RMS WAKE_FLUC_FRAC*U/sqrt(2)
        fluct = fluct * (C.WAKE_FLUC_FRAC * speed[i] / math.sqrt(2.0))
        ux[i] = speed[i] * np.cos(heading[i])
        uy[i] = fluct + speed[i] * np.sin(heading[i])
    sigs_w, labs_w, tgts_w = D._assemble(ux, uy, 1, speed, heading, rng)
    sigs.append(sigs_w)
    labs.append(labs_w)
    tgts.append(tgts_w)

    # ── dipole: measured radial profile, randomised source ────────────────────
    speed = rng.uniform(C.U_MIN, C.U_MAX, per_class)
    heading = rng.uniform(-math.pi, math.pi, per_class)
    tv = np.arange(C.T_STEPS) / C.FS
    ux = np.empty((per_class, C.T_STEPS, C.N_PIL))
    uy = np.empty((per_class, C.T_STEPS, C.N_PIL))
    for i in range(per_class):
        f_src = rng.uniform(C.F_DIP_MIN, C.F_DIP_MAX)
        x_src = rng.uniform(_px[0], _px[-1])
        y_src = rng.uniform(C.STANDOFF_MIN, C.STANDOFF_MAX)
        radius = np.sqrt((_px - x_src) ** 2 + y_src ** 2)
        shape = profile(radius / radius.min())
        shape = shape / shape.max()
        amp = rng.uniform(C.DIP_AMP_MIN, C.DIP_AMP_MAX) * speed[i]
        ux[i] = speed[i] * np.cos(heading[i])
        uy[i] = (amp * shape * np.sin(2 * math.pi * f_src * tv[:, None])
                 + speed[i] * np.sin(heading[i]))
    sigs_d, labs_d, tgts_d = D._assemble(ux, uy, 2, speed, heading, rng)
    sigs.append(sigs_d)
    labs.append(labs_d)
    tgts.append(tgts_d)

    return (np.concatenate(sigs), np.concatenate(labs), np.concatenate(tgts))
