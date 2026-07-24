"""Training loop, metrics, and evaluation. Shared by every decoder."""

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import models as M


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def _to_seq(x, idx, device):
    """(B, T, C) numpy -> (T, B, C) tensor."""
    return torch.as_tensor(x[idx], dtype=torch.float32,
                           device=device).permute(1, 0, 2)


def _loss(model, cls_out, reg_out, aux, y, r, is_snn):
    total = (nn.functional.cross_entropy(cls_out, y)
             + C.LAMBDA_REG * nn.functional.mse_loss(reg_out, r))
    if is_snn:
        _, s1, s2 = aux
        total = total + C.LAMBDA_RATE * (s1.mean() + s2.mean())
    return total


def heading_and_speed_error(pred, true):
    """Wrapped angular error [deg] and speed error [mm/s]."""
    true_h = np.arctan2(true[:, 1], true[:, 0])
    pred_h = np.arctan2(pred[:, 1], pred[:, 0])
    delta = np.arctan2(np.sin(pred_h - true_h), np.cos(pred_h - true_h))
    hmae = float(np.degrees(np.abs(delta)).mean())
    smae = float(np.abs(np.linalg.norm(pred, axis=-1)
                        - np.linalg.norm(true, axis=-1)).mean() * 1000.0)
    return hmae, smae


@torch.no_grad()
def evaluate(model, x, y, r, norm, device, is_snn):
    """Returns metrics plus measured spike rates (SNN only)."""
    model.eval()
    mu, sd = norm
    preds, regs = [], []
    rate_in = rate_s1 = rate_s2 = 0.0
    n_batch = 0
    for idx in _batches(len(y), C.BATCH):
        xb = _to_seq(x, idx, device)
        cls_out, reg_out, aux = model(xb)
        if is_snn:
            xs, s1, s2 = aux
            rate_in += xs.mean().item()
            rate_s1 += s1.mean().item()
            rate_s2 += s2.mean().item()
        n_batch += 1
        preds.append(cls_out.argmax(-1).cpu().numpy())
        regs.append(reg_out.cpu().numpy())
    pred_cls = np.concatenate(preds)
    pred_reg = np.concatenate(regs) * sd + mu
    acc = float((pred_cls == y).mean())
    hmae, smae = heading_and_speed_error(pred_reg, r)
    rates = tuple(v / max(n_batch, 1) for v in (rate_in, rate_s1, rate_s2))
    return dict(acc=acc, hmae=hmae, smae=smae), pred_cls, rates


def fit(model, train_set, val_set, test_set, epochs, seed,
        device=None, verbose=True, tag=""):
    """Train with early stopping on validation loss; evaluate the best weights."""
    device = device or get_device()
    model = model.to(device)
    is_snn = isinstance(model, M.FlowSNN)

    x_tr, y_tr, r_tr = train_set
    x_va, y_va, r_va = val_set
    x_te, y_te, r_te = test_set

    mu = r_tr.mean(axis=0)
    sd = r_tr.std(axis=0) + 1e-8
    rn_tr = (r_tr - mu) / sd
    rn_va = (r_va - mu) / sd

    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed + 500)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for idx in _batches(len(y_tr), C.BATCH, rng):
            xb = _to_seq(x_tr, idx, device)
            yb = torch.as_tensor(y_tr[idx], dtype=torch.long, device=device)
            rb = torch.as_tensor(rn_tr[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            cls_out, reg_out, aux = model(xb)
            loss = _loss(model, cls_out, reg_out, aux, yb, rb, is_snn)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss, nb = 0.0, 0
            for idx in _batches(len(y_va), C.BATCH):
                xb = _to_seq(x_va, idx, device)
                yb = torch.as_tensor(y_va[idx], dtype=torch.long, device=device)
                rb = torch.as_tensor(rn_va[idx], dtype=torch.float32,
                                     device=device)
                cls_out, reg_out, aux = model(xb)
                val_loss += _loss(model, cls_out, reg_out, aux, yb, rb,
                                  is_snn).item()
                nb += 1
            val_loss /= max(nb, 1)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= C.PATIENCE:
                if verbose:
                    print(f"    early stop at epoch {epoch + 1}", flush=True)
                break
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    ep{epoch + 1:3d}  val_loss={val_loss:.4f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)

    model.load_state_dict(best_state)
    metrics, pred_cls, rates = evaluate(model, x_te, y_te, r_te, (mu, sd),
                                        device, is_snn)

    if is_snn:
        energy = M.snn_energy_joules(model, rates[0], rates[1])
        metrics["spike_rate_in"] = rates[0]
        metrics["spike_rate_h1"] = rates[1]
        metrics["spike_rate_h2"] = rates[2]
    else:
        energy = M.dense_energy_joules(model)
    metrics["energy_uJ"] = energy * 1e6
    metrics["power_mW"] = energy * C.INFERENCE_RATE * 1e3
    metrics["params"] = M.parameter_count(model)

    if verbose:
        print(f"  [{tag}] acc={metrics['acc']:.4f}  heading={metrics['hmae']:.2f}"
              f"deg  speed={metrics['smae']:.1f}mm/s"
              f"  energy={metrics['energy_uJ']:.3f}uJ", flush=True)
    return model, metrics, pred_cls, (mu, sd)


def aggregate(runs):
    """mean +/- std over seeds for every scalar metric."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out
