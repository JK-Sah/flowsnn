"""
Three-dimensional D3Q19 BGK lattice-Boltzmann solver, GPU-resident.

Two configurations, both of which the two-dimensional solver cannot settle:

    dipole_3d    an oscillating spherical source in quiescent fluid. The radial
                 decay of an oscillating sphere's velocity field is 1/r^3; the
                 line dipole of a two-dimensional simulation decays as 1/r^2.
                 The reduced-order sensor model has to use whichever matches the
                 geometry it is claiming to represent, and only a 3-D run can
                 measure the 3-D exponent.

    wake_3d      flow past a finite-span cylinder at a Reynolds number above the
                 onset of spanwise instability (about 190), so the wake carries
                 the spanwise structure that a 2-D simulation cannot produce.
                 Probe signals from this run give a transfer test domain that is
                 not spanwise-uniform.

Runs on CuPy if a GPU is present and falls back to NumPy otherwise, which is
only useful for small smoke tests: a 256^3 domain is roughly 250 times the work
of the 2-D cases.
"""

import numpy as np

try:
    import cupy as _cp
    xp = _cp
    ON_GPU = True
except Exception:                                  # pragma: no cover
    xp = np
    ON_GPU = False


def to_host(a):
    return _cp.asnumpy(a) if ON_GPU and isinstance(a, _cp.ndarray) else np.asarray(a)


# ── D3Q19 lattice ─────────────────────────────────────────────────────────────
_C = [(0, 0, 0),
      (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
      (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
      (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
      (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
C = np.array(_C, dtype=np.int32)
W = np.array([1/3] + [1/18] * 6 + [1/36] * 12)
OPP = np.array([_C.index((-cx, -cy, -cz)) for cx, cy, cz in _C], dtype=np.int32)
CS2 = 1.0 / 3.0
Q = 19


def equilibrium(rho, ux, uy, uz, out=None):
    usq = ux * ux + uy * uy + uz * uz
    shape = (Q,) + rho.shape
    feq = xp.empty(shape, dtype=rho.dtype) if out is None else out
    for i in range(Q):
        cu = C[i, 0] * ux + C[i, 1] * uy + C[i, 2] * uz
        feq[i] = W[i] * rho * (1.0 + cu / CS2
                               + 0.5 * cu * cu / (CS2 * CS2)
                               - 0.5 * usq / CS2)
    return feq


def macroscopic(f):
    rho = f.sum(axis=0)
    ux = xp.zeros_like(rho)
    uy = xp.zeros_like(rho)
    uz = xp.zeros_like(rho)
    for i in range(1, Q):
        if C[i, 0]:
            ux += C[i, 0] * f[i]
        if C[i, 1]:
            uy += C[i, 1] * f[i]
        if C[i, 2]:
            uz += C[i, 2] * f[i]
    inv = 1.0 / rho
    return rho, ux * inv, uy * inv, uz * inv


def stream(f, out):
    for i in range(Q):
        out[i] = xp.roll(f[i], (int(C[i, 0]), int(C[i, 1]), int(C[i, 2])),
                         axis=(0, 1, 2))
    return out


def bounce_back(f_post, mask):
    """In-place half-way bounce-back on solid nodes."""
    tmp = f_post[:, mask]
    f_post[:, mask] = tmp[OPP]
    return f_post


# ── configuration 1: oscillating sphere (3-D dipole) ──────────────────────────

def dipole_3d(n=256, radius=6.0, omega=1.8, force_amp=2.0e-4, freq=6.67e-4,
              n_steps=24000, record_from=12000, dtype=np.float32,
              progress=2000):
    """Oscillating spherical source in quiescent fluid.

    Returns the transverse velocity amplitude along a radial rake, from which
    the decay exponent is fitted.
    """
    cx = cy = cz = n // 2
    grid = xp.arange(n, dtype=dtype)
    X = grid[:, None, None]
    Y = grid[None, :, None]
    Z = grid[None, None, :]
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    src = r2 <= radius ** 2

    rho = xp.ones((n, n, n), dtype=dtype)
    zero = xp.zeros((n, n, n), dtype=dtype)
    f = equilibrium(rho, zero, zero, zero)
    f2 = xp.empty_like(f)

    # radial rake along +x at the source plane, outside the forcing sphere
    radii = np.arange(int(radius) + 4, n // 2 - 8, 2)
    n_rec = n_steps - record_from
    rake = np.zeros((n_rec, len(radii)), dtype=np.float64)
    rad_idx = xp.asarray(radii + cx)

    for step in range(n_steps):
        rho, ux, uy, uz = macroscopic(f)
        # velocity-shift forcing, applied along y so the source is a dipole
        drive = dtype(force_amp * np.sin(2.0 * np.pi * freq * step))
        uy_eff = uy.copy()
        uy_eff[src] += drive / rho[src]
        feq = equilibrium(rho, ux, uy_eff, uz)
        f *= (1.0 - omega)
        f += omega * feq
        stream(f, f2)
        f, f2 = f2, f
        # open boundaries: zero-gradient on all six faces
        f[:, 0] = f[:, 1]
        f[:, -1] = f[:, -2]
        f[:, :, 0] = f[:, :, 1]
        f[:, :, -1] = f[:, :, -2]
        f[:, :, :, 0] = f[:, :, :, 1]
        f[:, :, :, -1] = f[:, :, :, -2]

        if step >= record_from:
            rake[step - record_from] = to_host(uy[rad_idx, cy, cz])
        if progress and step % progress == 0:
            print(f"    dipole3d step {step}/{n_steps}", flush=True)

    return dict(radii=radii, rake=rake, freq=freq, n=n, radius=radius)


def fit_decay(radii, rake, r_min_factor=1.6, radius=6.0, r_max=None):
    """Power-law fit of oscillation amplitude against radius.

    Two exclusions. Inside about 1.6 source radii the solution is dominated by
    the forcing itself rather than by the radiated field. Beyond about 0.8 of
    the domain half-width the zero-gradient outer boundary distorts the field:
    including those points pulls the fitted exponent down by roughly 0.06 while
    lowering the fit quality, so they are dropped by default.
    """
    amp = rake.std(axis=0)
    keep = (radii > r_min_factor * radius) & (amp > 0)
    if r_max is not None:
        keep &= radii <= r_max
    log_r, log_a = np.log(radii[keep]), np.log(amp[keep])
    slope, intercept = np.polyfit(log_r, log_a, 1)
    pred = slope * log_r + intercept
    r_squared = float(1.0 - ((log_a - pred) ** 2).sum()
                      / ((log_a - log_a.mean()) ** 2).sum())
    return float(-slope), amp, keep, r_squared


def profile_rmse(radii, amp, keep, exponent):
    model = radii[keep].astype(float) ** (-exponent)
    model = model / model.max()
    meas = amp[keep] / amp[keep].max()
    return float(np.sqrt(np.mean((model - meas) ** 2)))


# ── configuration 2: finite-span cylinder wake ────────────────────────────────

def wake_3d(nx=384, ny=128, nz=128, diam=24, re=220, u_lb=0.05,
            n_steps=60000, record_from=30000, n_probes=8, probe_offset=3.0,
            probe_y_offset=10, n_span=8, dtype=np.float32, progress=2500):
    """Flow past a spanwise cylinder above the onset of 3-D wake instability.

    Probes are taken at n_span spanwise stations so the recorded field carries
    the spanwise variation that a two-dimensional simulation cannot produce.
    """
    radius = diam / 2.0
    nu = u_lb * diam / re
    omega = 1.0 / (3.0 * nu + 0.5)
    cx, cy = nx // 5, ny // 2

    grid_x = xp.arange(nx, dtype=dtype)[:, None, None]
    grid_y = xp.arange(ny, dtype=dtype)[None, :, None]
    mask = ((grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= radius ** 2)
    mask = xp.broadcast_to(mask, (nx, ny, nz)).copy()

    rho = xp.ones((nx, ny, nz), dtype=dtype)
    ux = xp.full((nx, ny, nz), u_lb, dtype=dtype)
    # spanwise-varying perturbation to seed the 3-D instability rather than
    # letting it grow from round-off
    kz = xp.arange(nz, dtype=dtype)[None, None, :]
    ux *= (1.0 + dtype(0.02) * xp.sin(2.0 * np.pi * kz / nz)
           * xp.sin(2.0 * np.pi * grid_y / ny))
    ux[mask] = 0.0
    zero = xp.zeros((nx, ny, nz), dtype=dtype)
    f = equilibrium(rho, ux, zero, zero)
    f2 = xp.empty_like(f)

    px = np.clip((cx + probe_offset * diam
                  + np.arange(n_probes) * 0.5 * diam).astype(int), 0, nx - 2)
    pz = np.linspace(nz // 8, nz - nz // 8, n_span).astype(int)
    py = int(cy + probe_y_offset)
    px_d, pz_d = xp.asarray(px), xp.asarray(pz)

    west = [i for i in range(Q) if C[i, 0] < 0]
    east = [i for i in range(Q) if C[i, 0] > 0]
    n_rec = n_steps - record_from
    rec = np.zeros((n_rec, n_span, n_probes), dtype=np.float32)
    # The whole streamwise row is kept at each spanwise station, subsampled in
    # time, so probe spacing can be chosen afterwards to match the modelled
    # array's phase ramp. Fixing the rake at run time bakes in a spacing that
    # will not correspond to a whole number of shedding wavelengths.
    row_every = 2
    n_row = n_rec // row_every
    rec_row = np.zeros((n_row, n_span, nx), dtype=np.float32)

    for step in range(n_steps):
        f[west, -1] = f[west, -2]
        rho, ux, uy, uz = macroscopic(f)
        ux[0] = u_lb
        uy[0] = 0.0
        uz[0] = 0.0
        ux[mask] = 0.0
        uy[mask] = 0.0
        uz[mask] = 0.0

        feq = equilibrium(rho, ux, uy, uz)
        f *= (1.0 - omega)
        f += omega * feq
        # reconstruct the inward-pointing populations at the inlet plane
        f[east, 0] = feq[east, 0]
        f = bounce_back(f, mask)
        stream(f, f2)
        f, f2 = f2, f
        # free-slip top/bottom, periodic spanwise (roll already handles z)
        f[:, :, 0] = f[:, :, 1]
        f[:, :, -1] = f[:, :, -2]

        if step >= record_from:
            k = step - record_from
            rec[k] = to_host(uy[px_d[None, :], py, pz_d[:, None]])
            if k % row_every == 0 and k // row_every < n_row:
                rec_row[k // row_every] = to_host(uy[:, py, pz_d].T)
        if progress and step % progress == 0:
            print(f"    wake3d step {step}/{n_steps}", flush=True)

    return dict(uy=rec, row=rec_row, row_every=row_every, px=px, py=py, pz=pz,
                cx=cx, diam=diam, u_lb=u_lb, re=re, omega=omega,
                nx=nx, ny=ny, nz=nz)


def spanwise_coherence(rec):
    """Correlation between spanwise stations at the same streamwise probe.

    1.0 means the wake is spanwise-uniform, i.e. effectively two-dimensional.
    """
    n_rec, n_span, n_probe = rec.shape
    out = []
    for j in range(n_probe):
        block = rec[:, :, j] - rec[:, :, j].mean(axis=0, keepdims=True)
        cm = np.corrcoef(block.T)
        iu = np.triu_indices(n_span, k=1)
        out.append(float(np.mean(cm[iu])))
    return float(np.mean(out)), out
