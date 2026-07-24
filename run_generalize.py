"""
Does training on resolved wake data fix the transfer failure, and does it
generalise across shedding regimes?

The transfer test (Section 3.3) showed a decoder trained on the reduced-order
model's idealised travelling wave reads a real vortex wake as uniform flow. The
obvious remedy is to train on CFD wakes instead. The obvious objection to that
remedy is that training and testing on the same wake distribution proves
nothing. This script runs the version that answers the objection.

For each training source it trains one decoder per seed and evaluates it,
without retraining, on every wake domain:

    ROM          the reduced-order travelling wave (the original baseline)
    2d           the 2-D Karman street
    re220,300,380  the 3-D finite-span wake swept across Reynolds number

The comparison that matters is the off-diagonal: a decoder trained on the
Re = 220 wake and tested on Re = 380, a shedding regime it never saw. If wake
accuracy there is high, domain randomisation over resolved flow recovers the
capability; if it stays low, wake recognition is regime-specific and a real
sensor will need in-situ calibration. Either way the answer is in the table.

    python run_generalize.py            # all sources, held-out included
    python run_generalize.py --quick    # one seed, fewer samples

Needs the swept wake caches from run_cfd3d.py (wake3d_re*_cache.npz) and the
2-D caches from run_cfd.py.
"""

import argparse
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
ALL_DOMAINS = ["ROM", "2d", "re220", "re300", "re380"]


def cache_name(domain):
    return {"2d": "wake_cache.npz",
            "re220": "wake3d_re220_cache.npz",
            "re300": "wake3d_re300_cache.npz",
            "re380": "wake3d_re380_cache.npz"}.get(domain)


def available_domains():
    """Domains whose caches are present (ROM needs none)."""
    return [d for d in ALL_DOMAINS
            if d == "ROM" or (OUT / cache_name(d)).exists()]


def build_cfd_split(per_class, seed, wake_source):
    """A train/val/test split whose wake class comes from one CFD source.

    Encoded to the 64-channel spike tensor, matching R.prepare, so a decoder can
    be trained on it directly.
    """
    sig, lab, reg = X.make_cfd_dataset(per_class, seed, wake_source=wake_source)
    sig_std, stats = D.standardize(sig)
    enc = [(D.encode(s, "combined"), y, r)
           for (s, y, r) in D.split(sig_std, lab, reg, seed)]
    return enc[0], enc[1], enc[2], stats


def build_rom_split(per_class, seed):
    return R.prepare(per_class, seed, "combined")


def wake_source_of(domain):
    return None if domain == "ROM" else domain


def encode_test(wake_source, per_class, seed, stats):
    """A pure test set (no split) from one wake source, standardised by stats."""
    if wake_source is None:
        sig, lab, reg = D.make_dataset(per_class, seed + 777)
    else:
        sig, lab, reg = X.make_cfd_dataset(per_class, seed + 777,
                                           wake_source=wake_source)
    x = D.encode(D.standardize(sig, stats), "combined")
    return x, lab, reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train-on", nargs="*", default=None,
                    help="which sources to train a decoder on")
    args = ap.parse_args()
    domains = available_domains()
    train_domains = args.train_on if args.train_on is not None else domains
    print(f"domains available: {domains}", flush=True)

    device = T.get_device()
    seeds = [0] if args.quick else C.SEEDS
    npc = 80 if args.quick else C.NPC_MAIN
    epochs = 4 if args.quick else C.EPOCHS_MAIN

    results = {}
    for train_dom in train_domains:
        per_seed = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            ws = wake_source_of(train_dom)
            if ws is None:
                tr, va, te, stats = build_rom_split(npc, seed)
            else:
                tr, va, te, stats = build_cfd_split(npc, seed, ws)
            model = M.FlowSNN(C.N_CHAN_ENC)
            model, _, _, reg_norm = T.fit(model, tr, va, te, epochs, seed,
                                          device=device, verbose=False,
                                          tag=f"train={train_dom} s{seed}")
            # evaluate on every wake domain, held out
            row = {}
            for test_dom in domains:
                x, lab, reg = encode_test(wake_source_of(test_dom),
                                          max(npc // 2, 40), seed, stats)
                m, pred, _ = T.evaluate(model, x, lab, reg, reg_norm, device,
                                        True)
                pc = R.per_class_accuracy(pred, lab)
                row[test_dom] = dict(acc=m["acc"], hmae=m["hmae"],
                                     wake=pc["wake"], uniform=pc["uniform"],
                                     dipole=pc["dipole"])
            per_seed.append(row)

        agg = {}
        for test_dom in domains:
            cell = {k: float(np.mean([s[test_dom][k] for s in per_seed]))
                    for k in ("acc", "hmae", "wake", "uniform", "dipole")}
            # per-seed spread, so the transfer section (which reads the ROM row
            # of this matrix) can report mean +/- std from the same run
            cell.update({f"{k}_std": float(np.std([s[test_dom][k]
                                                   for s in per_seed]))
                         for k in ("acc", "hmae", "wake", "uniform", "dipole")})
            agg[test_dom] = cell
        results[train_dom] = agg
        print(f"[train on {train_dom}] wake acc by test domain: "
              + "  ".join(f"{td}={agg[td]['wake']*100:.0f}%"
                          for td in domains), flush=True)

    (OUT / "generalize.json").write_text(json.dumps(results, indent=2))
    _print_matrix(results)
    print(f"\nwrote {OUT / 'generalize.json'}")


def _print_matrix(results):
    print("\nwake-class accuracy (%), rows = train domain, cols = test domain")
    doms = list(next(iter(results.values())).keys())
    header = "train \\ test  " + "".join(f"{d:>8s}" for d in doms)
    print(header)
    print("-" * len(header))
    for train_dom in results:
        cells = "".join(f"{results[train_dom][td]['wake']*100:7.0f} "
                        for td in doms)
        print(f"{train_dom:12s}  {cells}")


if __name__ == "__main__":
    main()
