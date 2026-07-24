"""
Evaluate the trained decoder on both CFD test domains.

Trains the headline spiking decoder per seed on the reduced-order model, then
scores it, without retraining, on three test sets:

    in-domain   held-out reduced-order samples
    cfd-2d      wake fluctuation from the 2-D Karman street
    cfd-3d      wake fluctuation from the finite-span 3-D wake at Re = 220

The 3-D column is the one that matters. The 2-D transfer test replaces an
analytic travelling wave with a simulated one that is still spanwise-uniform, so
it cannot say whether the decoder depends on that uniformity. The 3-D wake can.

    python run_transfer3d.py   ->  results/transfer3d.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from flowsnn import config as C
from flowsnn import data as D
from flowsnn import models as M
from flowsnn import train as T
from flowsnn import transfer as X
import run_experiments as R

OUT = Path(__file__).parent / "results"


def score(model, stats, reg_norm, device, per_class, seed, wake_source):
    sig, lab, reg = X.make_cfd_dataset(per_class, seed, wake_source=wake_source)
    x = D.encode(D.standardize(sig, stats), "combined")
    metrics, pred, _ = T.evaluate(model, x, lab, reg, reg_norm, device, True)
    metrics["per_class_acc"] = R.per_class_accuracy(pred, lab)
    return metrics


def main():
    device = T.get_device()
    rows = {"in_domain": [], "cfd_2d": [], "cfd_3d": []}

    for seed in C.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr, va, te, stats = R.prepare(C.NPC_MAIN, seed, "combined")
        model = M.FlowSNN(C.N_CHAN_ENC)
        model, metrics, pred, reg_norm = T.fit(
            model, tr, va, te, C.EPOCHS_MAIN, seed, device=device,
            verbose=False, tag=f"seed={seed}")
        metrics["per_class_acc"] = R.per_class_accuracy(pred, te[1])
        rows["in_domain"].append(metrics)

        for tag, src in (("cfd_2d", "2d"), ("cfd_3d", "3d")):
            rows[tag].append(score(model, stats, reg_norm, device,
                                   C.NPC_MAIN // 2, seed, src))

        print(f"seed {seed}: in={metrics['acc']:.3f} "
              f"2d={rows['cfd_2d'][-1]['acc']:.3f} "
              f"3d={rows['cfd_3d'][-1]['acc']:.3f}", flush=True)

    out = {}
    for tag, runs in rows.items():
        num = [{k: v for k, v in r.items() if isinstance(v, (int, float))}
               for r in runs]
        out[tag] = T.aggregate(num)
        out[tag]["per_class_acc"] = {
            n: float(np.mean([r["per_class_acc"][n] for r in runs]))
            for n in D.CLASS_NAMES}

    (OUT / "transfer3d.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 66)
    print(f"{'domain':12s} {'accuracy':>14s} {'heading':>10s}"
          f" {'uniform':>9s} {'wake':>7s} {'dipole':>8s}")
    print("=" * 66)
    for tag in ("in_domain", "cfd_2d", "cfd_3d"):
        a = out[tag]
        pc = a["per_class_acc"]
        print(f"{tag:12s} {a['acc_mean']*100:8.1f}+-{a['acc_std']*100:<4.1f}"
              f" {a['hmae_mean']:9.1f}"
              f" {pc['uniform']*100:8.0f}% {pc['wake']*100:6.0f}%"
              f" {pc['dipole']*100:7.0f}%")
    print(f"\nwrote {OUT / 'transfer3d.json'}")


if __name__ == "__main__":
    main()
