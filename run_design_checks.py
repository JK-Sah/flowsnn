"""
Two architecture choices that are easy to get wrong, measured rather than assumed.

    python run_design_checks.py   ->  results/design_checks.json

1. Recurrent connections in the LIF layers. Standard in the SNN literature and
   present in an earlier version of this code, but never justified against a
   feedforward trunk on this task.
2. Temporal resolution of the spike-count readout. Averaging counts over the
   whole window is the usual choice and it discards spike timing, which is where
   the uniform/wake distinction lives.

Both are run on seed 0 with the headline data size, which is enough to separate
the conditions; the chosen setting is then used for all three seeds in
run_experiments.py.
"""

import json
import time
from pathlib import Path

import torch

from flowsnn import config as C
from flowsnn import models as M
from flowsnn import train as T
import run_experiments as R

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def one(bins, recurrent, seed=0):
    torch.manual_seed(seed)
    tr, va, te, _ = R.prepare(C.NPC_MAIN, seed, "combined")
    model = M.FlowSNN(C.N_CHAN_ENC, recurrent=recurrent, bins=bins)
    t0 = time.time()
    model, metrics, pred, _ = T.fit(model, tr, va, te, C.EPOCHS_MAIN, seed,
                                    verbose=False, tag="")
    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["per_class_acc"] = R.per_class_accuracy(pred, te[1])
    return metrics


def main():
    out = {"recurrence": [], "readout_bins": []}

    print("recurrent connections")
    for rec in (False, True):
        m = one(C.READOUT_BINS, rec)
        m["recurrent"] = rec
        out["recurrence"].append(m)
        print(f"  recurrent={rec!s:5s} acc={m['acc']:.3f}"
              f" heading={m['hmae']:.2f}deg params={m['params']}"
              f" energy={m['energy_uJ']:.3f}uJ", flush=True)

    print("readout temporal bins")
    for bins in (1, 2, 4, 8, 16, 32):
        m = one(bins, C.RECURRENT)
        m["bins"] = bins
        out["readout_bins"].append(m)
        print(f"  bins={bins:<3d} acc={m['acc']:.3f}"
              f" heading={m['hmae']:.2f}deg speed={m['smae']:.1f}mm/s"
              f" energy={m['energy_uJ']:.3f}uJ", flush=True)

    (OUT / "design_checks.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'design_checks.json'}")


if __name__ == "__main__":
    main()
