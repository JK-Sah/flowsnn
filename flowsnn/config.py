"""
Single source of truth for every parameter reported in the manuscript.

Any number that appears in a table or in the text is defined here and nowhere
else, so the paper and the code cannot drift apart.
"""

import math

# ── sensor array / reduced-order FSI ──────────────────────────────────────────
N_PIL = 8            # pillars
SPACING = 0.01       # pillar spacing [m]
F_NAT = 35.0         # pillar natural frequency [Hz]
ZETA = 0.12          # damping ratio
D_BODY = 0.012       # virtual shedding-body diameter [m]

# ── signal ────────────────────────────────────────────────────────────────────
T_STEPS = 256        # samples per window
FS = 500.0           # sampling rate [Hz]
N_CHAN_RAW = 2 * N_PIL   # 16 (streamwise + transverse per pillar)

# ── flow ──────────────────────────────────────────────────────────────────────
U_MIN, U_MAX = 0.05, 0.40   # bulk speed [m/s]
STROUHAL = 0.20   # LBM at 22% blockage gives 0.214; see results/cfd_summary.json
# Vortex convection speed as a fraction of the bulk speed. The LBM sweep gives
# 0.85 at two diameters downstream, rising to 0.99 by eight; the sensor array
# sits in the near wake, so the near-wake value is used. Stations inside two
# diameters lie in the vortex formation region where a convection speed is not
# defined.
CONV_RATIO = 0.85
DIPOLE_DECAY = 2.0   # LBM gives 2.05; the 3.0 of sphere theory is a 3-D law
F_DIP_MIN, F_DIP_MAX = 10.0, 45.0    # dipole source frequency [Hz]
STANDOFF_MIN, STANDOFF_MAX = 0.01, 0.05   # dipole stand-off [m]
WAKE_FLUC_FRAC = 0.30       # wake transverse fluctuation as fraction of U
# Dipole amplitude at the nearest pillar, as a fraction of U. Set to bracket the
# wake fluctuation so that the three classes carry comparable signal-to-noise
# and must be told apart by spatial and temporal structure rather than by
# amplitude alone. An earlier version divided by r^2 in SI units without
# normalising, which produced fluctuation velocities of order 20 m/s against a
# bulk flow of 0.05-0.40 m/s and made the dipole class separable by amplitude.
DIP_AMP_MIN, DIP_AMP_MAX = 0.15, 0.35

# ── noise ─────────────────────────────────────────────────────────────────────
TURB_INTENSITY = 0.06   # turbulence intensity u_rms/U, relative to each sample's own speed
AR1_ALPHA = 0.85        # AR(1) coefficient for temporally correlated turbulence
SENSOR_NOISE = 0.02     # fraction of per-channel sigma

# ── phasic / tonic encoder ────────────────────────────────────────────────────
# Chosen so that combined spike activity lands near 4% of time-channel slots
# (about 630 spikes per sample). The threshold trades sparsity against accuracy
# directly and therefore sets the energy result; run_sweep.py measures that
# trade-off, and Section 4.3 reports it.
PHASIC_THRESHOLD = 0.15   # normalised deflection units
TONIC_GAIN = 0.12
TONIC_LP_ALPHA = 0.90
N_CHAN_ENC = 4 * N_CHAN_RAW   # 64: [on, off, plus, minus]

# ── spiking network ───────────────────────────────────────────────────────────
BETA = 0.90          # membrane decay
THRESHOLD = 1.0      # spike threshold
SURR_SLOPE = 25.0    # fast-sigmoid surrogate gradient slope
HIDDEN = 128
# Recurrent LIF connections, off by default. Measured rather than assumed (see
# results/design_checks.json): recurrence buys about 3 points of structure
# accuracy and costs about 2 degrees of heading accuracy, for roughly twice the
# synapse count. The trade is not obviously worth it, and the feedforward form
# is the one the update rule in the paper describes, so it is the default.
RECURRENT = False

# Temporal bins in the spike-count readout; 1 is the conventional whole-window
# average. This is the single most consequential architecture choice here.
# Structure accuracy climbs monotonically with resolution (69% at 1 bin, 91% at
# 32) while heading error grows (2.2 to 7.9 degrees) and cross-domain
# robustness on the wake class falls. 16 is the compromise the paper reports.
READOUT_BINS = 16
N_CLASSES = 3
N_REG = 2

# ── training ──────────────────────────────────────────────────────────────────
LR = 2e-3
BATCH = 64
GRAD_CLIP = 1.0
LAMBDA_REG = 0.1     # weight on the MSE regression term
LAMBDA_RATE = 0.05   # spike-rate regularisation (SNN only)
PATIENCE = 5         # early-stopping patience on validation loss
EPOCHS_MAIN = 30
EPOCHS_ABLATION = 25
NPC_MAIN = 400       # samples per class, headline experiment
NPC_ABLATION = 300   # samples per class, ablation and baselines
SPLIT = (0.70, 0.10, 0.20)   # train / validation / test
SEEDS = [0, 1, 2]

# ── energy model (45 nm CMOS, Horowitz 2014) ──────────────────────────────────
E_MAC = 4.6e-12   # J
E_AC = 0.9e-12    # J
INFERENCE_RATE = FS   # one inference per sample window slide [Hz]

# ── baselines ─────────────────────────────────────────────────────────────────
MLP_HIDDEN = 128
CNN_CHANNELS = 32
CNN_K1, CNN_K2 = 7, 5
GRU_HIDDEN = 128

# ── LBM (lattice units) ───────────────────────────────────────────────────────
LBM_WAKE = dict(nx=320, ny=90, u_lb=0.05, re=120, diam=20,
                n_steps=24000, record_from=10000)
LBM_DIPOLE = dict(nx=200, ny=160, omega=1.786, force_amp=1.5e-4,
                  freq=6.67e-4, n_steps=14000, record_from=8000)
# A lightly damped oscillator needs of order Q cycles to reach steady state.
# At zeta = 0.02 that is about 25 shedding cycles, or roughly 50 000 lattice
# steps; the transient is discarded and the response measured over the second
# half of a much longer run. Sweeps short enough to cover only a few cycles
# show no resonance at all, because the response never builds.
LBM_VIV = dict(nx=240, ny=90, u_lb=0.05, re=120, diam=20,
               mass_ratio=10.0, zeta=0.02, n_steps=90000, record_from=50000,
               ur_values=[3.0, 3.5, 4.0, 4.3, 4.6, 5.0, 5.5, 6.0, 7.0])

PILLAR_X = [i * SPACING for i in range(N_PIL)]
OMEGA_NAT = 2 * math.pi * F_NAT
