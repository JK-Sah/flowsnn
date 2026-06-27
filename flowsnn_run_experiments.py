"""
flowsnn_run_experiments.py
==========================
Run the 3-seed FlowSNN experiments.
Produces flowsnn_results.json with mean ± std statistics used in Tables 2–3
and Section 4.1.

Requirements
------------
    pip install torch scipy numpy tqdm
    # snntorch is OPTIONAL – the script implements LIF manually via PyTorch autograd
    # pip install snntorch   (install if you want to verify against the snntorch path)

Usage
-----
    python flowsnn_run_experiments.py              # runs all experiments

Output
------
    flowsnn_results.json  – all metrics (accuracy, heading MAE, speed MAE, energy)

Runtime estimate (CPU-only, modern laptop)
------------------------------------------
    ~10–20 minutes total across all seeds and models.
    GPU (CUDA or MPS) will be ~10× faster.
"""

import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import bilinear, lfilter

# ── reproducibility ───────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True

# ── physical / model hyper-parameters ────────────────────────────────────────
N_PIL   = 8           # number of pillars
SP      = 0.01        # pillar spacing [m]
FN      = 35.0        # pillar natural frequency [Hz]
ZETA    = 0.12        # damping ratio
D       = 0.012       # pillar diameter [m]
T_STEPS = 256         # time steps per sample
FS      = 500.0       # sampling frequency [Hz]
U_MIN, U_MAX = 0.05, 0.40   # flow speed range [m/s]
ST      = 0.20        # Strouhal number
CONV    = 0.69        # convection speed ratio
DF_MIN, DF_MAX = 10.0, 45.0  # dipole source freq range [Hz]
DS_MIN, DS_MAX = 0.01, 0.05  # dipole stand-off distance range [m]
TURB    = 0.06        # turbulence intensity (fraction of U_MAX)
AR1     = 0.85        # AR(1) turbulence coefficient
NOISE   = 0.02        # measurement noise fraction

# Encoder parameters
PT      = 0.05        # phasic threshold (in normalised deflection units)
TG      = 0.12        # tonic gain [dimensionless]
LPA     = 0.90        # low-pass alpha for tonic encoder

# SNN / training
BETA    = 0.90        # LIF membrane decay
THRESH  = 1.0         # spike threshold
SLOPE   = 25.0        # surrogate gradient slope
H       = 128         # hidden size
LR      = 2e-3        # initial learning rate
BATCH   = 64          # batch size
L_REG   = 0.05        # spike-rate regularisation weight
NC      = 3           # number of flow classes
NR      = 2           # number of regression targets (Vx, Vy)
E_MAC   = 4.6e-12     # energy per MAC (45 nm CMOS) [J]
E_AC    = 0.9e-12     # energy per AC  (45 nm CMOS) [J]
SEEDS   = [0, 1, 2]

DEVICE  = torch.device("cuda" if torch.cuda.is_available()
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))
CKPT_FILE = Path(__file__).parent / "flowsnn_results.json"

px    = np.arange(N_PIL) * SP
wn    = 2 * math.pi * FN
b_c   = np.array([1.0])
a_c   = np.array([1.0 / wn**2, 2*ZETA / wn, 1.0])
b_d, a_d = bilinear(b_c, a_c, fs=FS)


# ── data generation ───────────────────────────────────────────────────────────

def _filt(v: np.ndarray) -> np.ndarray:
    """2nd-order LTI filter (structural dynamics). v: (n, T, N_PIL)"""
    n_, T_, N_ = v.shape
    return (lfilter(b_d, a_d,
                    v.transpose(0, 2, 1).reshape(n_ * N_, T_),
                    axis=-1)
            .reshape(n_, N_, T_)
            .transpose(0, 2, 1))


def _add_turbulence(ux, uy, rng):
    n, T, N = ux.shape
    s_t = TURB * U_MAX
    scale = math.sqrt(1 - AR1**2)
    for arr in (ux, uy):
        noise = np.zeros((n, T, N))
        noise[:, 0, :] = rng.normal(0, s_t, (n, N))
        for t in range(1, T):
            noise[:, t, :] = (AR1 * noise[:, t-1, :]
                              + scale * rng.normal(0, s_t, (n, N)))
        arr += noise


def gen_uniform(n: int, rng) -> tuple:
    U  = rng.uniform(U_MIN, U_MAX, n)
    th = rng.uniform(-math.pi, math.pi, n)
    ux = (U[:, None, None] * np.cos(th[:, None, None])
          * np.ones((n, T_STEPS, N_PIL)))
    uy = (U[:, None, None] * np.sin(th[:, None, None])
          * np.ones((n, T_STEPS, N_PIL)))
    tgt = np.stack([U * np.cos(th), U * np.sin(th)], axis=-1)
    return ux, uy, np.zeros(n, dtype=int), tgt


def gen_wake(n: int, rng) -> tuple:
    U  = rng.uniform(U_MIN, U_MAX, n)
    th = rng.uniform(-math.pi, math.pi, n)
    tv = np.arange(T_STEPS) / FS
    ux = np.zeros((n, T_STEPS, N_PIL))
    uy = np.zeros((n, T_STEPS, N_PIL))
    for i in range(n):
        fs_s = ST * U[i] / D
        ph   = 2 * math.pi * fs_s * px / (CONV * U[i])
        ux[i] = U[i] * np.cos(th[i])
        uy[i] = (0.3 * U[i] * np.sin(2 * math.pi * fs_s * tv[:, None] - ph)
                 + U[i] * np.sin(th[i]))
    tgt = np.stack([U * np.cos(th), U * np.sin(th)], axis=-1)
    return ux, uy, np.ones(n, dtype=int), tgt


def gen_dipole(n: int, rng) -> tuple:
    U  = rng.uniform(U_MIN, U_MAX, n)
    th = rng.uniform(-math.pi, math.pi, n)
    tv = np.arange(T_STEPS) / FS
    ux = np.zeros((n, T_STEPS, N_PIL))
    uy = np.zeros((n, T_STEPS, N_PIL))
    for i in range(n):
        fd  = rng.uniform(DF_MIN, DF_MAX)
        xs  = rng.uniform(px[0], px[-1])
        ys  = rng.uniform(DS_MIN, DS_MAX)
        r2  = (px - xs)**2 + ys**2 + 1e-6
        amp = rng.uniform(0.05, 0.15) * U[i]
        uy[i] = (amp * np.sin(2 * math.pi * fd * tv[:, None]) / r2
                 + U[i] * np.sin(th[i]))
        ux[i] = U[i] * np.cos(th[i])
    tgt = np.stack([U * np.cos(th), U * np.sin(th)], axis=-1)
    return ux, uy, 2 * np.ones(n, dtype=int), tgt


def gen_dataset(npc: int, seed: int):
    rng   = np.random.default_rng(seed)
    parts = []
    for gen_fn in (gen_uniform, gen_wake, gen_dipole):
        ux, uy, lbl, tgt = gen_fn(npc, rng)
        _add_turbulence(ux, uy, rng)
        d = np.concatenate([_filt(ux), _filt(uy)], axis=-1).astype(np.float32)
        sigma = d.std(axis=(0, 1), keepdims=True) + 1e-8
        d += NOISE * sigma * rng.standard_normal(d.shape).astype(np.float32)
        parts.append((d, lbl, tgt.astype(np.float32)))
    D = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    R = np.concatenate([p[2] for p in parts])
    return D, Y, R


def standardize(D, stats=None):
    if stats is None:
        mu  = D.mean((0, 1), keepdims=True)
        std = D.std((0, 1), keepdims=True) + 1e-8
        return (D - mu) / std, (mu, std)
    return (D - stats[0]) / stats[1]


def split_data(D, Y, R, seed):
    rng  = np.random.default_rng(seed + 100)
    idx  = rng.permutation(len(Y))
    n_tr = int(0.7 * len(Y))
    n_va = int(0.15 * len(Y))
    i_tr = idx[:n_tr]; i_va = idx[n_tr:n_tr+n_va]; i_te = idx[n_tr+n_va:]
    return (D[i_tr], Y[i_tr], R[i_tr],
            D[i_va], Y[i_va], R[i_va],
            D[i_te], Y[i_te], R[i_te])


# ── spike encoder ─────────────────────────────────────────────────────────────

def encode_phasic(x: np.ndarray):
    """Phasic (change-detection) encoder.  x: (N, T, C) → on/off: (N, T, C)"""
    N, Ts, C = x.shape
    on  = np.zeros_like(x)
    off = np.zeros_like(x)
    ref = x[:, 0, :].copy()
    for t in range(1, Ts):
        d = x[:, t, :] - ref
        on[:, t, :]  = (d >=  PT).astype(np.float32)
        off[:, t, :] = (d <= -PT).astype(np.float32)
        ref = x[:, t, :].copy()
    return on, off


def encode_tonic(x: np.ndarray):
    """Tonic (integrate-and-fire) encoder.  x: (N, T, C) → plus/minus: (N, T, C)"""
    N, Ts, C = x.shape
    ft = 1.0 / (TG + 1e-9)
    lp = np.zeros_like(x)
    lp[:, 0, :] = x[:, 0, :]
    for t in range(1, Ts):
        lp[:, t, :] = LPA * lp[:, t-1, :] + (1 - LPA) * x[:, t, :]
    ap = np.zeros((N, C), np.float32)
    am = np.zeros((N, C), np.float32)
    pl = np.zeros((N, Ts, C), np.float32)
    mi = np.zeros((N, Ts, C), np.float32)
    for t in range(Ts):
        ap += np.maximum(lp[:, t, :], 0)
        am += np.maximum(-lp[:, t, :], 0)
        fp = ap >= ft;  fm = am >= ft
        pl[:, t, :] = fp.astype(np.float32)
        mi[:, t, :] = fm.astype(np.float32)
        ap -= fp * ft;  am -= fm * ft
    return pl, mi


def encode(D: np.ndarray, mode: str = "combined") -> np.ndarray:
    on, off = encode_phasic(D)
    pl, mi  = encode_tonic(D)
    z = np.zeros_like(on)
    if mode == "phasic":
        return np.concatenate([on, off, z, z], axis=-1).astype(np.float32)
    elif mode == "tonic":
        return np.concatenate([z, z, pl, mi], axis=-1).astype(np.float32)
    else:
        return np.concatenate([on, off, pl, mi], axis=-1).astype(np.float32)


# ── PyTorch models ─────────────────────────────────────────────────────────────

class SurrGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u >= THRESH).float()

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        sg = 1.0 / (1.0 + SLOPE * (u - THRESH).abs()) ** 2
        return grad_output * sg


spike_fn = SurrGrad.apply


class LIFLayer(nn.Module):
    """Single LIF layer with recurrent connections."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.fc   = nn.Linear(n_in, n_out)
        self.rec  = nn.Linear(n_out, n_out, bias=False)
        nn.init.normal_(self.fc.weight,   0, 1.0 / math.sqrt(n_in))
        nn.init.normal_(self.rec.weight,  0, 0.008)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x_seq):
        # x_seq: (T, B, n_in)
        T, B, _ = x_seq.shape
        u = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        s = torch.zeros_like(u)
        spikes = []
        for t in range(T):
            u  = BETA * u + self.fc(x_seq[t]) + self.rec(s) - THRESH * s
            s  = spike_fn(u)
            spikes.append(s)
        return torch.stack(spikes, dim=0), u  # (T, B, n_out), final u


class FlowSNN(nn.Module):
    def __init__(self, n_in: int = 64):
        super().__init__()
        self.layer1 = LIFLayer(n_in, H)
        self.layer2 = LIFLayer(H, H)
        self.cls_head = nn.Linear(H, NC)
        self.reg_head = nn.Linear(H, NR)

    def forward(self, x_seq):
        # x_seq: (T, B, n_in)
        s1, _   = self.layer1(x_seq)   # (T, B, H)
        s2, _   = self.layer2(s1)       # (T, B, H)
        readout = s2.mean(0)            # mean spike count over time (B, H)
        return self.cls_head(readout), self.reg_head(readout), s1, s2


class BaselineMLP(nn.Module):
    def __init__(self, n_in: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(),
            nn.Linear(256, 256),  nn.ReLU(),
            nn.Linear(256, 128),  nn.ReLU(),
        )
        self.cls_head = nn.Linear(128, NC)
        self.reg_head = nn.Linear(128, NR)

    def forward(self, x_seq):
        # x_seq: (T, B, n_in) → time-average
        feat = x_seq.mean(0)   # (B, n_in)
        h    = self.net(feat)
        return self.cls_head(h), self.reg_head(h)


class BaselineCNN(nn.Module):
    def __init__(self, n_in: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, 64,  kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(64,   128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128,  128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cls_head = nn.Linear(128, NC)
        self.reg_head = nn.Linear(128, NR)

    def forward(self, x_seq):
        # x_seq: (T, B, n_in) → (B, n_in, T)
        x   = x_seq.permute(1, 2, 0)
        h   = self.conv(x).squeeze(-1)   # (B, 128)
        return self.cls_head(h), self.reg_head(h)


class BaselineGRU(nn.Module):
    def __init__(self, n_in: int = 16):
        super().__init__()
        self.gru      = nn.GRU(n_in, H, batch_first=False)
        self.cls_head = nn.Linear(H, NC)
        self.reg_head = nn.Linear(H, NR)

    def forward(self, x_seq):
        # x_seq: (T, B, n_in)
        _, h_n = self.gru(x_seq)
        h      = h_n.squeeze(0)   # (B, H)
        return self.cls_head(h), self.reg_head(h)


# ── training helpers ───────────────────────────────────────────────────────────

def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=DEVICE)


def compute_energy(model):
    """Approximate 45-nm CMOS energy based on spike counts and MAC/AC ops."""
    if not isinstance(model, FlowSNN):
        return float("nan")
    with torch.no_grad():
        return float("nan")   # filled in during eval loop below


def train_epoch(model, X, Y, R, optimizer, is_snn=False):
    model.train()
    perm = torch.randperm(len(Y))
    total_loss = 0.0; n_batch = 0
    for s in range(0, len(Y), BATCH):
        bi = perm[s:s+BATCH]
        xb = to_tensor(X[bi]).permute(1, 0, 2)   # (T, B, C)
        yb = to_tensor(Y[bi], dtype=torch.long)
        rb = to_tensor(R[bi])
        optimizer.zero_grad()
        if is_snn:
            cls_out, reg_out, s1, s2 = model(xb)
            rate_loss = (s1.mean() + s2.mean()) * L_REG
        else:
            cls_out, reg_out = model(xb)
            rate_loss = 0.0
        ce_loss  = nn.functional.cross_entropy(cls_out, yb)
        reg_loss = nn.functional.mse_loss(reg_out, rb)
        loss     = ce_loss + 0.1 * reg_loss + rate_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n_batch += 1
    return total_loss / n_batch


@torch.no_grad()
def evaluate(model, X, Y, R, is_snn=False):
    model.eval()
    all_pred = []; all_reg = []; total_spikes_s1 = 0.0; total_spikes_s2 = 0.0
    for s in range(0, len(Y), BATCH):
        xb = to_tensor(X[s:s+BATCH]).permute(1, 0, 2)
        if is_snn:
            cls_out, reg_out, s1, s2 = model(xb)
            total_spikes_s1 += s1.mean().item()
            total_spikes_s2 += s2.mean().item()
        else:
            cls_out, reg_out = model(xb)
        all_pred.append(cls_out.argmax(-1).cpu().numpy())
        all_reg.append(reg_out.cpu().numpy())
    pred = np.concatenate(all_pred)
    reg  = np.concatenate(all_reg)
    acc  = float((pred == Y).mean())
    # heading MAE [degrees]
    true_heading = np.degrees(np.arctan2(R[:, 1], R[:, 0]))
    pred_heading = np.degrees(np.arctan2(reg[:, 1], reg[:, 0]))
    hmae = float(np.abs(true_heading - pred_heading).mean())
    # speed MAE [mm/s]
    true_speed = np.sqrt(R[:, 0]**2 + R[:, 1]**2)
    pred_speed = np.sqrt(reg[:, 0]**2 + reg[:, 1]**2)
    smae = float(np.abs(true_speed - pred_speed).mean() * 1000)
    return acc, hmae, smae


def energy_per_sample(model, X):
    """Approximate inference energy [nJ] using 45-nm CMOS model."""
    if not isinstance(model, FlowSNN):
        n_mac = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return n_mac * E_MAC * 1e9   # all MACs (dense activations)
    # SNN: AC per spike, MAC per weight (once per forward pass — amortised)
    with torch.no_grad():
        xb = to_tensor(X[:min(64, len(X))]).permute(1, 0, 2)
        _, _, s1, s2 = model(xb)
        sr1 = s1.mean().item(); sr2 = s2.mean().item()
    T = X.shape[1]
    n1 = model.layer1.fc.weight.numel() + model.layer1.rec.weight.numel()
    n2 = model.layer2.fc.weight.numel() + model.layer2.rec.weight.numel()
    # AC per timestep × spike rate × T
    e_ac = (sr1 * n1 + sr2 * n2) * T * E_AC
    # weight load (MAC, done once per batch element)
    e_mac = (n1 + n2) * E_MAC
    return (e_ac + e_mac) * 1e9   # nJ


def run_experiment(name, model, X_tr, Y_tr, R_tr, X_va, Y_va, R_va,
                   X_te, Y_te, R_te, n_epochs, is_snn=False):
    print(f"\n[{name}]  device={DEVICE}  epochs={n_epochs}", flush=True)
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_va_acc = -1; best_state = None
    t0 = time.time()
    for ep in range(n_epochs):
        tr_loss = train_epoch(model, X_tr, Y_tr, R_tr, opt, is_snn)
        va_acc, _, _ = evaluate(model, X_va, Y_va, R_va, is_snn)
        sch.step()
        if va_acc > best_va_acc:
            best_va_acc = va_acc
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"  ep{ep+1:3d}  tr_loss={tr_loss:.4f}  va_acc={va_acc:.4f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
    model.load_state_dict(best_state)
    te_acc, te_hmae, te_smae = evaluate(model, X_te, Y_te, R_te, is_snn)
    te_energy = energy_per_sample(model, X_te)
    print(f"  → test acc={te_acc:.4f}  hmae={te_hmae:.2f}°  smae={te_smae:.1f} mm/s"
          f"  energy={te_energy:.2f} nJ", flush=True)
    return {"acc": te_acc, "hmae": te_hmae, "smae": te_smae, "energy_nJ": te_energy}


# ── checkpoint helpers ─────────────────────────────────────────────────────────

def load_ckpt():
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            return json.load(f)
    return {}


def save_ckpt(data: dict):
    with open(CKPT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved checkpoint → {CKPT_FILE}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments already in the checkpoint file.")
    parser.add_argument("--epochs-snn",   type=int, default=30)
    parser.add_argument("--epochs-base",  type=int, default=25)
    parser.add_argument("--npc",          type=int, default=400,
                        help="Samples per class (paper uses 400).")
    args = parser.parse_args()

    results = load_ckpt() if args.resume else {}
    print(f"Running on device: {DEVICE}")

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f" SEED {seed}")
        print(f"{'='*60}")

        # ── generate & prepare data ──────────────────────────────────────────
        print("Generating dataset …", flush=True)
        D, Y, Reg = gen_dataset(args.npc, seed)
        D_std, stats = standardize(D)
        X_tr, Y_tr, R_tr, X_va, Y_va, R_va, X_te, Y_te, R_te = \
            split_data(D_std, Y, Reg, seed)

        # Encode for SNN
        print("Encoding spikes …", flush=True)
        E_tr  = encode(X_tr, "combined");  E_va = encode(X_va, "combined")
        E_te  = encode(X_te, "combined")
        Ep_tr = encode(X_tr, "phasic");    Ep_va = encode(X_va, "phasic")
        Ep_te = encode(X_te, "phasic")
        Et_tr = encode(X_tr, "tonic");     Et_va = encode(X_va, "tonic")
        Et_te = encode(X_te, "tonic")

        def key(name): return f"seed{seed}/{name}"

        torch.manual_seed(seed)

        # ── 1. main SNN ───────────────────────────────────────────────────────
        if key("main_snn") not in results:
            model = FlowSNN(n_in=64).to(DEVICE)
            res   = run_experiment(
                f"SNN seed={seed}", model,
                E_tr, Y_tr, R_tr, E_va, Y_va, R_va, E_te, Y_te, R_te,
                n_epochs=args.epochs_snn, is_snn=True)
            results[key("main_snn")] = res
            save_ckpt(results)

        # ── 2. ablation: phasic-only ──────────────────────────────────────────
        if key("abl_phasic") not in results:
            model = FlowSNN(n_in=64).to(DEVICE)
            res   = run_experiment(
                f"SNN-phasic seed={seed}", model,
                Ep_tr, Y_tr, R_tr, Ep_va, Y_va, R_va, Ep_te, Y_te, R_te,
                n_epochs=args.epochs_snn, is_snn=True)
            results[key("abl_phasic")] = res
            save_ckpt(results)

        # ── 3. ablation: tonic-only ───────────────────────────────────────────
        if key("abl_tonic") not in results:
            model = FlowSNN(n_in=64).to(DEVICE)
            res   = run_experiment(
                f"SNN-tonic seed={seed}", model,
                Et_tr, Y_tr, R_tr, Et_va, Y_va, R_va, Et_te, Y_te, R_te,
                n_epochs=args.epochs_snn, is_snn=True)
            results[key("abl_tonic")] = res
            save_ckpt(results)

        # ── 4. baseline MLP ───────────────────────────────────────────────────
        if key("baseline_mlp") not in results:
            model = BaselineMLP(n_in=16).to(DEVICE)
            res   = run_experiment(
                f"MLP seed={seed}", model,
                X_tr, Y_tr, R_tr, X_va, Y_va, R_va, X_te, Y_te, R_te,
                n_epochs=args.epochs_base, is_snn=False)
            results[key("baseline_mlp")] = res
            save_ckpt(results)

        # ── 5. baseline CNN ───────────────────────────────────────────────────
        if key("baseline_cnn") not in results:
            model = BaselineCNN(n_in=16).to(DEVICE)
            res   = run_experiment(
                f"CNN seed={seed}", model,
                X_tr, Y_tr, R_tr, X_va, Y_va, R_va, X_te, Y_te, R_te,
                n_epochs=args.epochs_base, is_snn=False)
            results[key("baseline_cnn")] = res
            save_ckpt(results)

        # ── 6. baseline GRU ───────────────────────────────────────────────────
        if key("baseline_gru") not in results:
            model = BaselineGRU(n_in=16).to(DEVICE)
            res   = run_experiment(
                f"GRU seed={seed}", model,
                X_tr, Y_tr, R_tr, X_va, Y_va, R_va, X_te, Y_te, R_te,
                n_epochs=args.epochs_base, is_snn=False)
            results[key("baseline_gru")] = res
            save_ckpt(results)

    # ── aggregate statistics ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (mean ± std over 3 seeds)")
    print(f"{'='*60}")

    experiment_names = [
        "main_snn", "abl_phasic", "abl_tonic",
        "baseline_mlp", "baseline_cnn", "baseline_gru"
    ]
    summary = {}
    for name in experiment_names:
        vals = {m: [] for m in ("acc", "hmae", "smae", "energy_nJ")}
        for seed in SEEDS:
            k = f"seed{seed}/{name}"
            if k in results:
                for m in vals:
                    vals[m].append(results[k][m])
        if vals["acc"]:
            acc_mean   = np.mean(vals["acc"])   * 100
            acc_std    = np.std(vals["acc"])    * 100
            hmae_mean  = np.mean(vals["hmae"])
            hmae_std   = np.std(vals["hmae"])
            smae_mean  = np.mean(vals["smae"])
            smae_std   = np.std(vals["smae"])
            enrg_mean  = np.mean(vals["energy_nJ"])
            enrg_std   = np.std(vals["energy_nJ"])
            print(f"  {name:20s}  acc={acc_mean:.1f}±{acc_std:.1f}%"
                  f"  hmae={hmae_mean:.1f}±{hmae_std:.1f}°"
                  f"  smae={smae_mean:.0f}±{smae_std:.0f} mm/s"
                  f"  energy={enrg_mean:.2f}±{enrg_std:.2f} nJ")
            summary[name] = {
                "acc_mean": round(acc_mean, 1), "acc_std": round(acc_std, 1),
                "hmae_mean": round(hmae_mean, 1), "hmae_std": round(hmae_std, 1),
                "smae_mean": round(smae_mean, 0), "smae_std": round(smae_std, 0),
                "energy_nJ_mean": round(enrg_mean, 2), "energy_nJ_std": round(enrg_std, 2),
            }

    results["summary"] = summary
    save_ckpt(results)
    print(f"\nAll done!  Full results in: {CKPT_FILE}")
    print("Share flowsnn_results.json with Claude to fill in the manuscript tables.")


if __name__ == "__main__":
    main()
