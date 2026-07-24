"""
Decoders and the operation-count energy model.

All four decoders expose the same interface: they consume a (T, B, C) sequence
and return (class logits, regression output). The SNN additionally returns its
per-layer spike trains, which the energy model consumes.
"""

import math
import torch
import torch.nn as nn

from . import config as C


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative backward."""

    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u >= C.THRESHOLD).float()

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        scale = 1.0 / (1.0 + C.SURR_SLOPE * (u - C.THRESHOLD).abs()) ** 2
        return grad_out * scale


spike = _SurrogateSpike.apply


class LIFLayer(nn.Module):
    """Feedforward leaky-integrate-and-fire layer with hard reset.

    U[t] = beta U[t-1] + W x[t] - theta S[t-1],  S[t] = 1 if U[t] >= theta

    Deliberately has no recurrent weight matrix: the manuscript's stated update
    rule is feedforward, and the energy model counts synaptic operations against
    exactly the weights present here.
    """

    def __init__(self, n_in, n_out, recurrent=False):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        nn.init.normal_(self.fc.weight, 0.0, 1.0 / math.sqrt(n_in))
        nn.init.zeros_(self.fc.bias)
        self.rec = None
        if recurrent:
            self.rec = nn.Linear(n_out, n_out, bias=False)
            nn.init.normal_(self.rec.weight, 0.0, 0.008)

    def forward(self, x_seq):
        t_steps, batch, _ = x_seq.shape
        n_out = self.fc.out_features
        drive = self.fc(x_seq)                     # (T, B, n_out), vectorised
        u = torch.zeros(batch, n_out, device=x_seq.device)
        s = torch.zeros_like(u)
        out = []
        for k in range(t_steps):
            u = C.BETA * u + drive[k] - C.THRESHOLD * s
            if self.rec is not None:
                u = u + self.rec(s)
            s = spike(u)
            out.append(s)
        return torch.stack(out, dim=0)


class FlowSNN(nn.Module):
    """Two LIF layers with a binned spike-count readout.

    Averaging spike counts over the whole window collapses the time axis, which
    costs exactly the information that separates a steady flow from an
    oscillating one. Splitting the window into READOUT_BINS equal parts and
    reading the count in each keeps coarse timing at negligible energy cost:
    the readout is a fixed multiply-accumulate term that the spike-driven
    synaptic operations dominate by three orders of magnitude.
    """

    def __init__(self, n_in=C.N_CHAN_ENC, recurrent=None, bins=None):
        super().__init__()
        recurrent = C.RECURRENT if recurrent is None else recurrent
        self.bins = C.READOUT_BINS if bins is None else bins
        self.recurrent = recurrent
        self.layer1 = LIFLayer(n_in, C.HIDDEN, recurrent)
        self.layer2 = LIFLayer(C.HIDDEN, C.HIDDEN, recurrent)
        feat = C.HIDDEN * self.bins
        self.cls_head = nn.Linear(feat, C.N_CLASSES)
        self.reg_head = nn.Linear(feat, C.N_REG)
        self.n_in = n_in

    def forward(self, x_seq):
        s1 = self.layer1(x_seq)
        s2 = self.layer2(s1)
        t_steps, batch, hidden = s2.shape
        if self.bins == 1:
            readout = s2.mean(dim=0)
        else:
            per_bin = t_steps // self.bins
            trimmed = s2[:per_bin * self.bins]
            readout = (trimmed.reshape(self.bins, per_bin, batch, hidden)
                       .mean(dim=1)              # (bins, B, H)
                       .permute(1, 0, 2)         # (B, bins, H)
                       .reshape(batch, -1))
        return self.cls_head(readout), self.reg_head(readout), (x_seq, s1, s2)


class BaselineMLP(nn.Module):
    """Two hidden layers, applied per frame, features averaged over time."""

    def __init__(self, n_in=C.N_CHAN_RAW):
        super().__init__()
        h = C.MLP_HIDDEN
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU())
        self.cls_head = nn.Linear(h, C.N_CLASSES)
        self.reg_head = nn.Linear(h, C.N_REG)
        self.n_in = n_in

    def forward(self, x_seq):
        feat = self.net(x_seq).mean(dim=0)
        return self.cls_head(feat), self.reg_head(feat), None


class BaselineCNN(nn.Module):
    """Two temporal convolutions and global average pooling."""

    def __init__(self, n_in=C.N_CHAN_RAW):
        super().__init__()
        ch = C.CNN_CHANNELS
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, ch, kernel_size=C.CNN_K1), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=C.CNN_K2), nn.ReLU())
        self.cls_head = nn.Linear(ch, C.N_CLASSES)
        self.reg_head = nn.Linear(ch, C.N_REG)
        self.n_in = n_in

    def forward(self, x_seq):
        x = x_seq.permute(1, 2, 0)          # (B, C, T)
        feat = self.conv(x).mean(dim=-1)    # (B, ch)
        return self.cls_head(feat), self.reg_head(feat), None


class BaselineGRU(nn.Module):
    def __init__(self, n_in=C.N_CHAN_RAW):
        super().__init__()
        self.gru = nn.GRU(n_in, C.GRU_HIDDEN)
        self.cls_head = nn.Linear(C.GRU_HIDDEN, C.N_CLASSES)
        self.reg_head = nn.Linear(C.GRU_HIDDEN, C.N_REG)
        self.n_in = n_in

    def forward(self, x_seq):
        _, h_n = self.gru(x_seq)
        feat = h_n.squeeze(0)
        return self.cls_head(feat), self.reg_head(feat), None


# ── energy model ──────────────────────────────────────────────────────────────
#
# Per-inference compute energy, where one inference decodes one T-step window.
#
# Dense decoders perform a multiply-accumulate for every weight at every
# timestep they process. The spiking decoder performs an accumulate only when a
# presynaptic spike arrives, so its synaptic-operation count is measured from
# actual spike activity on the held-out set; the dense readout over accumulated
# spike counts is charged as MACs.
#
# The earlier version of this code omitted the timestep factor for the dense
# decoders, which understated their energy by more than two orders of magnitude.

def dense_energy_joules(model):
    t = C.T_STEPS
    if isinstance(model, BaselineMLP):
        h = C.MLP_HIDDEN
        per_frame = model.n_in * h + h * h
        macs = t * per_frame + h * (C.N_CLASSES + C.N_REG)
    elif isinstance(model, BaselineCNN):
        ch = C.CNN_CHANNELS
        t1 = t - C.CNN_K1 + 1
        t2 = t1 - C.CNN_K2 + 1
        macs = (t1 * model.n_in * C.CNN_K1 * ch
                + t2 * ch * C.CNN_K2 * ch
                + ch * (C.N_CLASSES + C.N_REG))
    elif isinstance(model, BaselineGRU):
        h = C.GRU_HIDDEN
        per_frame = 3 * (model.n_in * h + h * h)
        macs = t * per_frame + h * (C.N_CLASSES + C.N_REG)
    else:
        raise TypeError(f"not a dense baseline: {type(model).__name__}")
    return macs * C.E_MAC


def snn_energy_joules(model, input_rate, s1_rate):
    """Spike-driven accumulates plus the dense readout.

    input_rate and s1_rate are mean spike probabilities per channel per timestep,
    measured on the evaluation set.
    """
    t = C.T_STEPS
    fan1 = C.HIDDEN
    fan2 = C.HIDDEN
    acs = (input_rate * model.n_in * t * fan1
           + s1_rate * C.HIDDEN * t * fan2)
    macs = C.HIDDEN * C.READOUT_BINS * (C.N_CLASSES + C.N_REG)
    return acs * C.E_AC + macs * C.E_MAC


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


BUILDERS = {
    "snn": lambda: FlowSNN(C.N_CHAN_ENC),
    "mlp": lambda: BaselineMLP(C.N_CHAN_RAW),
    "cnn": lambda: BaselineCNN(C.N_CHAN_RAW),
    "gru": lambda: BaselineGRU(C.N_CHAN_RAW),
}
