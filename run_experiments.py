"""
Run every experiment reported in the paper and write results/results.json.

    python run_experiments.py             # everything, seeds 0/1/2
    python run_experiments.py --quick     # small/fast smoke run
    python run_experiments.py --only main

Requires `python run_cfd.py` to have been run first for the transfer experiment.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flowsnn import config as C
from flowsnn import data as D
from flowsnn import models as M
from flowsnn import train as T
from flowsnn import transfer as X

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RESULTS = OUT / "results.json"


def load_results():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save_results(res):
    RESULTS.write_text(json.dumps(res, indent=2))


def prepare(per_class, seed, mode):
    """Reduced-order dataset, standardised, split, and encoded if needed."""
    sig, lab, reg = D.make_dataset(per_class, seed)
    sig_std, stats = D.standardize(sig)
    tr, va, te = D.split(sig_std, lab, reg, seed)
    if mode is None:
        return tr, va, te, stats
    enc = [(D.encode(s, mode), y, r) for (s, y, r) in (tr, va, te)]
    return enc[0], enc[1], enc[2], stats


def per_class_accuracy(pred, true):
    return {name: float((pred[true == i] == i).mean())
            for i, name in enumerate(D.CLASS_NAMES)}


def run_decoder(name, kind, mode, per_class, epochs, seeds, device,
                do_transfer, results):
    """Train one decoder across seeds; optionally evaluate CFD transfer.

    For the transfer evaluation nothing about the model is refitted, and the
    standardisation statistics come from the training domain, as they would on
    a deployed sensor.
    """
    runs, transfers, confusions, class_accs = [], [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr, va, te, stats = prepare(per_class, seed, mode)
        model = M.BUILDERS[kind]()
        model, metrics, pred, reg_norm = T.fit(
            model, tr, va, te, epochs, seed, device=device,
            tag=f"{name} seed={seed}")
        class_accs.append(per_class_accuracy(pred, te[1]))
        runs.append({k: v for k, v in metrics.items()
                     if isinstance(v, (int, float))})
        confusions.append(_confusion(pred, te[1]).tolist())

        if do_transfer:
            sig, lab, reg = X.make_cfd_dataset(max(per_class // 2, 50), seed)
            sig_std = D.standardize(sig, stats)
            x = D.encode(sig_std, mode) if mode is not None else sig_std
            tmetrics, tpred, _ = T.evaluate(
                model, x, lab, reg, reg_norm, device,
                isinstance(model, M.FlowSNN))
            tmetrics["per_class_acc"] = per_class_accuracy(tpred, lab)
            print(f"    -> CFD transfer: acc={tmetrics['acc']:.4f}"
                  f"  heading={tmetrics['hmae']:.2f}deg"
                  f"  per-class={ {k: round(v,3) for k,v in tmetrics['per_class_acc'].items()} }",
                  flush=True)
            transfers.append(tmetrics)

    entry = dict(agg=T.aggregate(runs), runs=runs,
                 confusion_mean=np.mean(confusions, axis=0).tolist(),
                 per_class_acc={n: float(np.mean([c[n] for c in class_accs]))
                                for n in D.CLASS_NAMES})
    if transfers:
        entry["transfer"] = T.aggregate(
            [{k: v for k, v in t.items() if isinstance(v, (int, float))}
             for t in transfers])
        entry["transfer_per_class"] = {
            n: float(np.mean([t["per_class_acc"][n] for t in transfers]))
            for n in D.CLASS_NAMES}
    results[name] = entry
    save_results(results)
    return entry


def _confusion(pred, true, n=C.N_CLASSES):
    m = np.zeros((n, n), dtype=float)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m / m.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    device = T.get_device()
    seeds = args.seeds if args.seeds is not None else C.SEEDS
    npc_main = 60 if args.quick else C.NPC_MAIN
    npc_abl = 60 if args.quick else C.NPC_ABLATION
    ep_main = 3 if args.quick else C.EPOCHS_MAIN
    ep_abl = 3 if args.quick else C.EPOCHS_ABLATION
    if args.quick:
        seeds = seeds[:1]

    print(f"device={device}  seeds={seeds}  npc={npc_main}/{npc_abl}",
          flush=True)
    results = load_results()
    t0 = time.time()

    jobs = [
        ("main_snn",     "snn", "combined", npc_main, ep_main, True),
        ("abl_phasic",   "snn", "phasic",   npc_abl,  ep_abl,  False),
        ("abl_tonic",    "snn", "tonic",    npc_abl,  ep_abl,  False),
        ("abl_combined", "snn", "combined", npc_abl,  ep_abl,  False),
        ("baseline_mlp", "mlp", None,       npc_abl,  ep_abl,  True),
        ("baseline_cnn", "cnn", None,       npc_abl,  ep_abl,  True),
        ("baseline_gru", "gru", None,       npc_abl,  ep_abl,  True),
    ]
    for name, kind, mode, npc, epochs, transfer in jobs:
        if args.only and args.only not in name:
            continue
        if name in results and not args.quick:
            print(f"[skip] {name} (already in results.json)", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        run_decoder(name, kind, mode, npc, epochs, seeds, device, transfer,
                    results)

    print(f"\ntotal {time.time() - t0:.0f}s -> {RESULTS}", flush=True)
    _summary(results)


def _summary(results):
    print("\n" + "=" * 78)
    print(f"{'experiment':16s} {'acc':>14s} {'heading':>12s} {'speed':>12s}"
          f" {'energy uJ':>11s}")
    print("=" * 78)
    for name, entry in results.items():
        a = entry["agg"]
        print(f"{name:16s} {a['acc_mean']*100:8.1f}+-{a['acc_std']*100:<4.1f}"
              f" {a['hmae_mean']:7.1f}+-{a['hmae_std']:<3.1f}"
              f" {a['smae_mean']:7.0f}+-{a['smae_std']:<3.0f}"
              f" {a['energy_uJ_mean']:11.3f}")
        if entry.get("transfer"):
            t = entry["transfer"]
            print(f"{'  -> CFD':16s} {t['acc_mean']*100:8.1f}+-"
                  f"{t['acc_std']*100:<4.1f} {t['hmae_mean']:7.1f}"
                  f"+-{t['hmae_std']:<3.1f}")


if __name__ == "__main__":
    main()
