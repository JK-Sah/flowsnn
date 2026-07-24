# FlowSNN

A spiking decoder for a simulated artificial lateral line, and the experiment
that asks whether the fish lateral line's split into change-sensitive and
level-sensitive receptors does any computational work.

Short answer: it does. Give the network only the phasic (change) pathway and it
classifies flow structure but cannot estimate heading. Give it only the tonic
(level) pathway and heading is accurate while classification drops to chance.
Neither pathway is redundant, and the split follows the same line as the
canal/superficial neuromast distinction in the animal.

Every number below is produced by the scripts in this repository and can be
regenerated from scratch with the commands under "Install and run".

## Results

Three flow classes (uniform, vortex wake, oscillating dipole), eight compliant
pillars, 256 samples at 500 Hz, three seeds.

| Decoder | Structure accuracy | Heading MAE | Energy / inference |
|---|---|---|---|
| Spiking (events) | 87.1 ± 0.3% | 4.2° | 0.37 µJ |
| MLP | 70.9 ± 4.3% | 1.3° | 21.7 µJ |
| CNN | 94.4 ± 1.2% | 3.5° | 9.9 µJ |
| GRU | 68.1 ± 8.1% | 5.1° | 65.1 µJ |

The spiking decoder is about 59× cheaper than the MLP and
27× cheaper than the CNN, and 7.4 percentage points less
accurate than the best dense baseline. Energy is an operation count on a 45 nm
reference process, not a hardware measurement.

### Encoding ablation

| Encoding | Structure accuracy | Speed MAE | Heading MAE |
|---|---|---|---|
| Phasic only | 73.1 ± 1.9% | 65 mm/s | 16.5° |
| Tonic only | 45.4 ± 10.7% | 15 mm/s | 3.5° |
| Combined | 78.5 ± 4.9% | 18 mm/s | 4.6° |

Chance accuracy is 33.3%. A heading estimator that ignores its input scores
about 90°.

### Transfer to CFD-derived signals

Trained on the reduced-order sensor model, tested without retraining on signals
whose wake fluctuations come from the lattice-Boltzmann solver:
69.6 ± 1.5% accuracy (17.5 points below in-domain)
and 4.6° heading error.

## Install and run

```bash
pip install -r requirements.txt

python run_cfd.py            # lattice-Boltzmann calibration + transfer domain (~25 min)
python run_experiments.py    # all decoders, three seeds (~10 min)
python run_design_checks.py  # recurrence and readout-resolution sweeps
python make_figures.py       # regenerate every figure

python run_cfd3d.py dipole   # 3-D decay exponent   (needs a GPU)
python run_cfd3d.py wake     # 3-D spanwise wake    (needs a GPU)
```

`run_cfd.py` must run first — `run_experiments.py` reads its probe time series
to build the transfer test set. Both write to `results/`. CUDA and Apple MPS are
detected automatically; CPU-only works and is not much slower, because the
per-timestep LIF loop dominates. The 3-D runs in `run_cfd3d.py` are the
exception: they need CuPy and a GPU, and are impractical on CPU.

## Layout

| Path | What it is |
|---|---|
| `flowsnn/config.py` | every reported parameter, defined once |
| `flowsnn/data.py` | reduced-order source domain and the phasic/tonic encoder |
| `flowsnn/models.py` | the spiking decoder, three dense baselines, energy model |
| `flowsnn/lbm.py` | D2Q9 lattice-Boltzmann solver (wake, dipole, VIV) |
| `flowsnn/lbm3d.py` | D3Q19 solver, GPU-resident; 3-D dipole and finite-span wake |
| `flowsnn/transfer.py` | builds the CFD-derived test domain |
| `flowsnn/train.py` | training loop, early stopping, metrics |

## Notes on what changed from the first version

Four corrections, listed because they change published numbers.

The dipole generator divided by r² in SI units without normalising, which
produced fluctuation velocities around 20 m/s against a bulk flow of 0.05–0.40
m/s. The dipole class was separable by amplitude alone, and classification
accuracy was inflated as a result.

Turbulence intensity was applied against the top of the speed range rather than
each sample's own speed, so the slowest samples carried 48% intensity instead of
the intended 6%.

The energy model charged dense baselines one multiply-accumulate per parameter
per inference, with no factor for the 256 timesteps they actually process. That
understated dense energy by more than two orders of magnitude, in the direction
that flattered the spiking decoder — the previously reported ratios were
computed by a different code path than the one shipped.

The wake convection ratio was previously reported as recalibrated from 0.85 to
0.69 by CFD. That came from a spectral phase-slope fit over a record spanning
only a few shedding cycles, which is too coarse to trust. A cross-correlation
estimate gives 0.85 at two diameters downstream rising to
0.99 by eight; the original 0.85 was right for the near wake. The
dipole decay correction (1/r³ → 1/r²) does hold up: the measured exponent is
2.05.

## Author

J. K. Sah, Department of Mechanical and Industrial Engineering, Rochester
Institute of Technology — js8472@rit.edu —
[ORCID 0000-0002-7643-5137](https://orcid.org/0000-0002-7643-5137)
