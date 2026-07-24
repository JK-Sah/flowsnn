"""
D2Q9 BGK lattice-Boltzmann solver used to calibrate and to replace the
reduced-order source domain.

Three configurations:

    cylinder_wake  flow past a fixed circular cylinder, giving a Karman street
                   from which the Strouhal number and the wake convection ratio
                   are measured
    dipole         a localised oscillating body force in quiescent fluid, from
                   which the radial decay exponent is measured
    viv            the cylinder mounted on a transverse spring, swept across
                   reduced velocity, bounding the one-way-coupling regime

The probe time series these produce are also used directly as a CFD-derived
test domain for the decoder (see flowsnn.transfer).
"""

import numpy as np

# ── D2Q9 lattice ──────────────────────────────────────────────────────────────
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0


def equilibrium(rho, ux, uy):
    """f_eq for all nine directions. rho, ux, uy: (nx, ny) -> (9, nx, ny)."""
    usq = ux ** 2 + uy ** 2
    feq = np.empty((9, ) + rho.shape)
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        feq[i] = W[i] * rho * (1.0 + cu / CS2
                               + 0.5 * cu ** 2 / CS2 ** 2
                               - 0.5 * usq / CS2)
    return feq


def macroscopic(f):
    rho = f.sum(axis=0)
    ux = (CX[:, None, None] * f).sum(axis=0) / rho
    uy = (CY[:, None, None] * f).sum(axis=0) / rho
    return rho, ux, uy


def stream(f):
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(np.roll(f[i], CX[i], axis=0), CY[i], axis=1)
    return out


def zou_he_west(f, u_in):
    """Velocity inlet on the west face."""
    f0, f2, f4 = f[0, 0], f[2, 0], f[4, 0]
    f3, f6, f7 = f[3, 0], f[6, 0], f[7, 0]
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)
    f[1, 0] = f3 + 2.0 / 3.0 * rho * u_in
    f[5, 0] = f7 - 0.5 * (f2 - f4) + rho * u_in / 6.0
    f[8, 0] = f6 + 0.5 * (f2 - f4) + rho * u_in / 6.0
    return f


def disc_mask(nx, ny, cx, cy, radius):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2


def _bounce_back(f_post, mask, wall_uy=0.0, rho_ref=1.0):
    """Half-way bounce-back on the masked nodes, optionally with a moving wall.

    The mask is held fixed; wall motion enters only through the momentum term,
    which is the small-displacement linearisation used for the VIV sweep.
    """
    f_bb = f_post.copy()
    for i in range(9):
        f_bb[i][mask] = f_post[OPP[i]][mask]
        if wall_uy != 0.0:
            f_bb[i][mask] -= (2.0 * W[OPP[i]] * rho_ref
                              * (CY[OPP[i]] * wall_uy) / CS2)
    return f_bb


def boundary_links(mask):
    """Fluid nodes whose neighbour in direction i is solid, per direction.

    roll(mask, -c_i) places mask[x + c_i] at x, so the conjunction below selects
    exactly the fluid-side end of every link that crosses the surface.
    """
    links = {}
    for i in range(1, 9):
        neigh = np.roll(np.roll(mask, -CX[i], axis=0), -CY[i], axis=1)
        links[i] = (~mask) & neigh
    return links


def _momentum_exchange_fy(f_post, links, wall_uy=0.0, rho_ref=1.0):
    """Transverse hydrodynamic force on the obstacle (Ladd momentum exchange).

    Across a boundary link the momentum delivered is

        c_i [ 2 f_i^post(x_f) - 2 w_i rho (c_i . u_wall) / c_s^2 ]

    The second term is the wall-motion contribution. Dropping it leaves a
    residual force in phase with the wall velocity, which acts as negative
    damping and makes the coupled system diverge.
    """
    fy = 0.0
    for i in range(1, 9):
        if CY[i] == 0:
            continue
        link = links[i]
        n_link = int(link.sum())
        if n_link == 0:
            continue
        fy += 2.0 * CY[i] * f_post[i][link].sum()
        if wall_uy != 0.0:
            fy -= (2.0 * CY[i] * n_link * W[i] * rho_ref
                   * (CY[i] * wall_uy) / CS2)
    return fy


# ── configuration 1: cylinder wake ────────────────────────────────────────────

def cylinder_wake(nx=320, ny=90, diam=20, re=120, u_lb=0.05,
                  n_steps=24000, record_from=10000, n_probes=8,
                  probe_offset=3.0, probe_spacing=0.5, probe_y_offset=8,
                  progress=None):
    """Flow past a fixed cylinder. Returns probe velocity histories.

    Probes sit downstream of the cylinder on a line parallel to the flow,
    mimicking the pillar array. probe_offset and probe_spacing are in diameters.
    """
    radius = diam / 2.0
    nu = u_lb * diam / re
    omega = 1.0 / (3.0 * nu + 0.5)
    cx, cy = nx // 5, ny // 2
    mask = disc_mask(nx, ny, cx, cy, radius)

    px = (cx + probe_offset * diam
          + np.arange(n_probes) * probe_spacing * diam).astype(int)
    py = int(cy + probe_y_offset)
    px = np.clip(px, 0, nx - 2)

    rho = np.ones((nx, ny))
    ux = np.full((nx, ny), u_lb)
    uy = np.zeros((nx, ny))
    # small transverse perturbation to break the symmetry and trigger shedding
    ux[:, :] *= 1.0 + 0.01 * np.sin(2.0 * np.pi * np.arange(ny) / ny)[None, :]
    ux[mask] = 0.0
    f = equilibrium(rho, ux, uy)

    west_going = [3, 6, 7]
    n_rec = n_steps - record_from
    rec_uy = np.zeros((n_rec, n_probes))
    rec_ux = np.zeros((n_rec, n_probes))
    # the whole probe row is kept as well, so the convection ratio can be
    # resolved as a function of downstream distance and so that arbitrary
    # probe windows can be extracted afterwards without re-running
    rec_row = np.zeros((n_rec, nx), dtype=np.float32)

    for step in range(n_steps):
        # zero-gradient outflow: the unknown (inward) populations on the east
        # face are copied from the neighbouring column before moments are taken
        f[west_going, -1] = f[west_going, -2]

        rho, ux, uy = macroscopic(f)
        ux[0] = u_lb
        uy[0] = 0.0
        rho[0] = ((f[[0, 2, 4], 0].sum(axis=0)
                   + 2.0 * f[west_going, 0].sum(axis=0)) / (1.0 - u_lb))
        ux[mask] = 0.0
        uy[mask] = 0.0

        f_post = f - omega * (f - equilibrium(rho, ux, uy))
        # Zou/He: reconstruct the inward populations at the inlet after collision
        f_post[[1, 5, 8], 0] = equilibrium(rho[0:1], ux[0:1], uy[0:1])[[1, 5, 8], 0]
        f_post = _bounce_back(f_post, mask)
        f = stream(f_post)

        if step >= record_from:
            k = step - record_from
            rec_uy[k] = uy[px, py]
            rec_ux[k] = ux[px, py]
            rec_row[k] = uy[:, py]
        if progress and step % progress == 0:
            print(f"    wake step {step}/{n_steps}", flush=True)

    return dict(uy=rec_uy, ux=rec_ux, row=rec_row, px=px, py=py, cx=cx, cy=cy,
                diam=diam, u_lb=u_lb, omega=omega, nu=nu, mask=mask,
                nx=nx, ny=ny, blockage=diam / ny,
                field_ux=ux, field_uy=uy)


def convection_ratio_profile(row, cx, diam, u_lb, n_win=8, spacing=10,
                             start_d=1.0, end_d=8.0, step_d=0.5):
    """Convection ratio measured in a sliding window down the wake.

    Returns (distance in diameters, ratio) so the near-wake deficit and its
    downstream recovery can be reported rather than a single number.
    """
    dists, ratios = [], []
    d = start_d
    while d <= end_d:
        x0 = int(cx + d * diam)
        px = x0 + np.arange(n_win) * spacing
        if px[-1] >= row.shape[1] - 1:
            break
        ratio, _ = measure_convection_ratio(row[:, px], px, u_lb)
        if np.isfinite(ratio):
            dists.append(d)
            ratios.append(ratio)
        d += step_d
    return np.array(dists), np.array(ratios)


def _parabolic_peak(spec, k):
    """Sub-bin peak location by parabolic interpolation of the spectrum."""
    if k <= 0 or k >= len(spec) - 1:
        return float(k)
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-30:
        return float(k)
    return k + 0.5 * (a - c) / denom


def measure_strouhal(uy_series, diam, u_lb):
    """Shedding frequency from the dominant spectral peak of probe 0.

    The peak is refined to sub-bin resolution: at these lattice parameters the
    record spans only a few shedding cycles, so the raw bin index is coarse.
    """
    sig = uy_series[:, 0] - uy_series[:, 0].mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    k = int(np.argmax(spec[1:])) + 1
    k_ref = _parabolic_peak(spec, k)
    peak = k_ref / len(sig)
    return float(peak * diam / u_lb), float(peak)


def measure_convection_ratio(uy_series, px, u_lb, max_lag=2000):
    """Convection speed from inter-probe cross-correlation lag.

    Robust where the spectral phase method is not: it needs only the time shift
    between neighbouring probes, not a finely resolved shedding frequency. The
    lag is fitted linearly against probe separation and inverted for the
    convection speed.
    """
    n_probe = uy_series.shape[1]
    ref = uy_series[:, 0] - uy_series[:, 0].mean()
    lags = [0.0]
    for j in range(1, n_probe):
        sig = uy_series[:, j] - uy_series[:, j].mean()
        corr = np.correlate(sig, ref, mode="full")
        centre = len(ref) - 1
        lo = centre
        hi = min(centre + max_lag, len(corr) - 1)
        k = int(np.argmax(corr[lo:hi + 1])) + lo
        lags.append(_parabolic_peak(corr, k) - centre)
    lags = np.array(lags)
    dist = np.asarray(px, dtype=float) - px[0]
    slope = np.polyfit(dist, lags, 1)[0]        # timesteps per lattice unit
    if abs(slope) < 1e-12:
        return float("nan"), lags
    return float((1.0 / slope) / u_lb), lags


# ── configuration 2: oscillating dipole ───────────────────────────────────────

def dipole(nx=200, ny=160, omega=1.786, force_amp=1.5e-4, freq=6.67e-4,
           n_steps=14000, record_from=8000, n_probes=8, probe_spacing=6,
           standoff=20, progress=None):
    """Localised oscillating body force in quiescent fluid."""
    sx, sy = nx // 2, ny // 2
    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    f = equilibrium(rho, ux, uy)

    px = (sx + (np.arange(n_probes) - n_probes // 2) * probe_spacing).astype(int)
    py = int(sy + standoff)

    src = disc_mask(nx, ny, sx, sy, 3.0)
    n_rec = n_steps - record_from
    rec_uy = np.zeros((n_rec, n_probes))

    # radial probe rake for the decay-exponent fit
    radii = np.arange(8, 60, 2)
    rad_amp = np.zeros((n_rec, len(radii)))

    for step in range(n_steps):
        rho, ux, uy = macroscopic(f)
        force = force_amp * np.sin(2.0 * np.pi * freq * step)
        uy_eq = uy.copy()
        uy_eq[src] += force / rho[src]           # Guo-style velocity shift
        f_post = f - omega * (f - equilibrium(rho, ux, uy_eq))
        f = stream(f_post)
        f[:, 0] = f[:, 1]                        # open boundaries
        f[:, -1] = f[:, -2]
        f[:, :, 0] = f[:, :, 1]
        f[:, :, -1] = f[:, :, -2]

        if step >= record_from:
            k = step - record_from
            rec_uy[k] = uy[px, py]
            rad_amp[k] = uy[sx + radii, sy]
        if progress and step % progress == 0:
            print(f"    dipole step {step}/{n_steps}", flush=True)

    return dict(uy=rec_uy, px=px, py=py, sx=sx, sy=sy, radii=radii,
                rad_amp=rad_amp, freq=freq, standoff=standoff,
                field_uy=uy, nx=nx, ny=ny)


def measure_decay_exponent(radii, rad_amp):
    """Power-law fit of oscillation amplitude against radius."""
    amp = rad_amp.std(axis=0)
    good = amp > 0
    slope, intercept = np.polyfit(np.log(radii[good]), np.log(amp[good]), 1)
    exponent = float(-slope)
    fit = np.exp(intercept) * radii ** slope
    return exponent, amp, fit


def profile_rmse(radii, amp, exponent):
    """Normalised RMSE of a 1/r^exponent profile against the measured one."""
    model = radii.astype(float) ** (-exponent)
    model = model / model.max()
    meas = amp / amp.max()
    return float(np.sqrt(np.mean((model - meas) ** 2)))


# ── configuration 3: vortex-induced vibration ─────────────────────────────────

def viv(ur, nx=240, ny=90, diam=20, re=120, u_lb=0.05, mass_ratio=10.0,
        zeta=0.02, n_steps=13000, record_from=7500, progress=None):
    """Cylinder on a transverse spring at one reduced velocity."""
    radius = diam / 2.0
    nu = u_lb * diam / re
    omega_lb = 1.0 / (3.0 * nu + 0.5)
    cx, cy = nx // 5, ny // 2
    mask = disc_mask(nx, ny, cx, cy, radius)
    links = boundary_links(mask)

    f_nat = u_lb / (ur * diam)
    m = mass_ratio * np.pi * radius ** 2
    k = m * (2.0 * np.pi * f_nat) ** 2
    c = 2.0 * zeta * np.sqrt(k * m)

    rho = np.ones((nx, ny))
    ux = np.full((nx, ny), u_lb)
    ux[:, :] *= 1.0 + 0.01 * np.sin(2.0 * np.pi * np.arange(ny) / ny)[None, :]
    uy = np.zeros((nx, ny))
    ux[mask] = 0.0
    f = equilibrium(rho, ux, uy)

    west_going = [3, 6, 7]
    y = v = 0.0
    n_rec = n_steps - record_from
    rec_y = np.zeros(n_rec)
    rec_uy = np.zeros(n_rec)
    probe_x, probe_y = int(cx + 3 * diam), int(cy + 8)

    for step in range(n_steps):
        f[west_going, -1] = f[west_going, -2]
        rho, ux, uy = macroscopic(f)
        ux[0] = u_lb
        uy[0] = 0.0
        rho[0] = ((f[[0, 2, 4], 0].sum(axis=0)
                   + 2.0 * f[west_going, 0].sum(axis=0)) / (1.0 - u_lb))
        ux[mask] = 0.0
        uy[mask] = v

        f_post = f - omega_lb * (f - equilibrium(rho, ux, uy))
        f_post[[1, 5, 8], 0] = equilibrium(rho[0:1], ux[0:1],
                                           uy[0:1])[[1, 5, 8], 0]
        fy = _momentum_exchange_fy(f_post, links, wall_uy=v)
        f_post = _bounce_back(f_post, mask, wall_uy=v)
        f = stream(f_post)

        # structural update: the cylinder responds only transversely
        acc = (fy - c * v - k * y) / m
        v += acc
        y += v

        if step >= record_from:
            rec_y[step - record_from] = y
            rec_uy[step - record_from] = uy[probe_x, probe_y]
        if progress and step % progress == 0:
            print(f"    viv ur={ur} step {step}/{n_steps}", flush=True)

    y_rms = float(np.std(rec_y) / diam)
    sig = rec_uy - rec_uy.mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=1.0)
    f_wake = float(freqs[1:][np.argmax(spec[1:])])
    ysig = rec_y - rec_y.mean()
    yspec = np.abs(np.fft.rfft(ysig * np.hanning(len(ysig))))
    f_struct = float(freqs[1:][np.argmax(yspec[1:])])
    return dict(ur=ur, y_rms_over_d=y_rms, f_wake=f_wake, f_struct=f_struct,
                f_nat=f_nat, y_history=rec_y)
