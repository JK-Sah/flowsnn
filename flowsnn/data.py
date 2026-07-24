"""
Reduced-order source domain: canonical flows -> pillar deflections -> spikes.

The three flow classes share a common construction. Every sample carries a bulk
advection (U cos t, U sin t) which is the regression target, plus a
class-specific fluctuation which is what the classifier has to recognise:

    uniform   no fluctuation
    wake      transverse travelling wave at f_s = St U / D, phase-ramped along
              the array by the convection speed
    dipole    localised oscillation decaying as 1/r^DIPOLE_DECAY from a
              randomised source position

Local velocity is then passed through the pillar transfer function (a damped
second-order resonator) to give deflections.
"""

import math
import numpy as np
from scipy.signal import bilinear, lfilter

from . import config as C

_px = np.asarray(C.PILLAR_X)
_b_cont = np.array([1.0])
_a_cont = np.array([1.0 / C.OMEGA_NAT ** 2, 2 * C.ZETA / C.OMEGA_NAT, 1.0])
B_PILLAR, A_PILLAR = bilinear(_b_cont, _a_cont, fs=C.FS)


def pillar_response(v):
    """Apply the pillar transfer function. v: (n, T, N_PIL) -> same shape."""
    n, t, npil = v.shape
    flat = v.transpose(0, 2, 1).reshape(n * npil, t)
    out = lfilter(B_PILLAR, A_PILLAR, flat, axis=-1)
    return out.reshape(n, npil, t).transpose(0, 2, 1)


def _ar1_turbulence(shape, rng, speed):
    """Temporally correlated free-stream turbulence, independent per pillar.

    TURB_INTENSITY is a turbulence intensity, u_rms/U, so the fluctuation scales
    with each sample's own bulk speed. Scaling it against U_MAX instead (as an
    earlier version did) leaves the slowest samples carrying 48% intensity
    rather than 6%, which buries the class-carrying fluctuation in noise at the
    bottom of the speed range.
    """
    n, t, npil = shape
    sigma = (C.TURB_INTENSITY * np.asarray(speed))[:, None]
    scale = math.sqrt(1.0 - C.AR1_ALPHA ** 2)
    noise = np.empty(shape, dtype=np.float64)
    noise[:, 0, :] = rng.standard_normal((n, npil)) * sigma
    for k in range(1, t):
        noise[:, k, :] = (C.AR1_ALPHA * noise[:, k - 1, :]
                          + scale * rng.standard_normal((n, npil)) * sigma)
    return noise


def _bulk(n, rng):
    speed = rng.uniform(C.U_MIN, C.U_MAX, n)
    heading = rng.uniform(-math.pi, math.pi, n)
    return speed, heading


def _assemble(ux, uy, label, speed, heading, rng):
    """Add turbulence and sensor noise, transduce, and pack into a sample set."""
    ux = ux + _ar1_turbulence(ux.shape, rng, speed)
    uy = uy + _ar1_turbulence(uy.shape, rng, speed)
    signal = np.concatenate([pillar_response(ux), pillar_response(uy)],
                            axis=-1).astype(np.float32)
    sigma = signal.std(axis=(0, 1), keepdims=True) + 1e-8
    signal = signal + (C.SENSOR_NOISE * sigma
                       * rng.standard_normal(signal.shape).astype(np.float32))
    target = np.stack([speed * np.cos(heading),
                       speed * np.sin(heading)], axis=-1).astype(np.float32)
    labels = np.full(len(speed), label, dtype=np.int64)
    return signal, labels, target


def gen_uniform(n, rng):
    speed, heading = _bulk(n, rng)
    ones = np.ones((n, C.T_STEPS, C.N_PIL))
    ux = speed[:, None, None] * np.cos(heading[:, None, None]) * ones
    uy = speed[:, None, None] * np.sin(heading[:, None, None]) * ones
    return _assemble(ux, uy, 0, speed, heading, rng)


def gen_wake(n, rng):
    speed, heading = _bulk(n, rng)
    tv = np.arange(C.T_STEPS) / C.FS
    ux = np.empty((n, C.T_STEPS, C.N_PIL))
    uy = np.empty((n, C.T_STEPS, C.N_PIL))
    for i in range(n):
        f_shed = C.STROUHAL * speed[i] / C.D_BODY
        phase = 2 * math.pi * f_shed * _px / (C.CONV_RATIO * speed[i])
        ux[i] = speed[i] * np.cos(heading[i])
        uy[i] = (C.WAKE_FLUC_FRAC * speed[i]
                 * np.sin(2 * math.pi * f_shed * tv[:, None] - phase)
                 + speed[i] * np.sin(heading[i]))
    return _assemble(ux, uy, 1, speed, heading, rng)


def gen_dipole(n, rng):
    speed, heading = _bulk(n, rng)
    tv = np.arange(C.T_STEPS) / C.FS
    ux = np.empty((n, C.T_STEPS, C.N_PIL))
    uy = np.empty((n, C.T_STEPS, C.N_PIL))
    for i in range(n):
        f_src = rng.uniform(C.F_DIP_MIN, C.F_DIP_MAX)
        x_src = rng.uniform(_px[0], _px[-1])
        y_src = rng.uniform(C.STANDOFF_MIN, C.STANDOFF_MAX)
        radius = np.sqrt((_px - x_src) ** 2 + y_src ** 2)
        decay = radius ** C.DIPOLE_DECAY + 1e-9
        amp = rng.uniform(C.DIP_AMP_MIN, C.DIP_AMP_MAX) * speed[i]
        # normalise so amplitude at the closest pillar is `amp`, independent of
        # stand-off; the shape of the along-array profile is what carries class
        # information, not its absolute scale.
        profile = (1.0 / decay)
        profile = profile / profile.max()
        ux[i] = speed[i] * np.cos(heading[i])
        uy[i] = (amp * profile * np.sin(2 * math.pi * f_src * tv[:, None])
                 + speed[i] * np.sin(heading[i]))
    return _assemble(ux, uy, 2, speed, heading, rng)


GENERATORS = (gen_uniform, gen_wake, gen_dipole)
CLASS_NAMES = ("uniform", "wake", "dipole")


def make_dataset(per_class, seed):
    """Balanced dataset over the three flow classes."""
    rng = np.random.default_rng(seed)
    sigs, labs, tgts = [], [], []
    for gen in GENERATORS:
        s, l, t = gen(per_class, rng)
        sigs.append(s)
        labs.append(l)
        tgts.append(t)
    return (np.concatenate(sigs), np.concatenate(labs), np.concatenate(tgts))


def standardize(x, stats=None):
    """Per-channel standardisation. Statistics are reusable on a target domain."""
    if stats is None:
        mu = x.mean(axis=(0, 1), keepdims=True)
        sd = x.std(axis=(0, 1), keepdims=True) + 1e-8
        return (x - mu) / sd, (mu, sd)
    mu, sd = stats
    return (x - mu) / sd


def split(x, y, r, seed):
    """Shuffled train / validation / test split per config.SPLIT."""
    rng = np.random.default_rng(seed + 100)
    idx = rng.permutation(len(y))
    n_tr = int(C.SPLIT[0] * len(y))
    n_va = int(C.SPLIT[1] * len(y))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return ((x[tr], y[tr], r[tr]), (x[va], y[va], r[va]), (x[te], y[te], r[te]))


# ── phasic / tonic spike encoder ──────────────────────────────────────────────

def encode_phasic(x):
    """Level-crossing encoder: ON/OFF spike when a channel moves by a threshold."""
    n, t, c = x.shape
    on = np.zeros((n, t, c), dtype=np.float32)
    off = np.zeros((n, t, c), dtype=np.float32)
    ref = x[:, 0, :].copy()
    for k in range(1, t):
        delta = x[:, k, :] - ref
        on[:, k, :] = (delta >= C.PHASIC_THRESHOLD)
        off[:, k, :] = (delta <= -C.PHASIC_THRESHOLD)
        ref = x[:, k, :].copy()
    return on, off


def encode_tonic(x):
    """Integrate-and-fire rate encoder on the low-passed level, sign-split."""
    n, t, c = x.shape
    fire_at = 1.0 / (C.TONIC_GAIN + 1e-9)
    lp = np.empty_like(x)
    lp[:, 0, :] = x[:, 0, :]
    for k in range(1, t):
        lp[:, k, :] = (C.TONIC_LP_ALPHA * lp[:, k - 1, :]
                       + (1 - C.TONIC_LP_ALPHA) * x[:, k, :])
    acc_p = np.zeros((n, c), dtype=np.float32)
    acc_m = np.zeros((n, c), dtype=np.float32)
    plus = np.zeros((n, t, c), dtype=np.float32)
    minus = np.zeros((n, t, c), dtype=np.float32)
    for k in range(t):
        acc_p += np.maximum(lp[:, k, :], 0)
        acc_m += np.maximum(-lp[:, k, :], 0)
        fp = acc_p >= fire_at
        fm = acc_m >= fire_at
        plus[:, k, :] = fp
        minus[:, k, :] = fm
        acc_p -= fp * fire_at
        acc_m -= fm * fire_at
    return plus, minus


def encode(x, mode="combined"):
    """Pack the two pathways into a 4*(2N)-channel binary spike tensor.

    Ablated pathways are zeroed rather than removed so that input dimensionality
    (and therefore network capacity) is identical across the three conditions.
    """
    on, off = encode_phasic(x)
    plus, minus = encode_tonic(x)
    zero = np.zeros_like(on)
    if mode == "phasic":
        parts = [on, off, zero, zero]
    elif mode == "tonic":
        parts = [zero, zero, plus, minus]
    elif mode == "combined":
        parts = [on, off, plus, minus]
    else:
        raise ValueError(f"unknown encoding mode: {mode}")
    return np.concatenate(parts, axis=-1).astype(np.float32)
