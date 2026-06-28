# FlowSNN — Bio-Inspired Lateral-Line Flow Classification with Spiking Neural Networks

---

## Contents

| File | Description |
|---|---|
| `flowsnn_run_experiments.py` | Main PyTorch training script — runs all 3-seed experiments, produces `flowsnn_results.json` |
| `flowsnn_experiments.py` | NumPy prototype for rapid experimentation (no GPU required) |

---

## Requirements

Python 3.9 or later. Install dependencies with:

```bash
pip install torch scipy numpy tqdm
```

> **GPU support:** CUDA and Apple MPS are detected automatically. CPU-only runs work fine but take longer (~10–20 min on a modern laptop).

> **snnTorch** is optional. The script implements LIF neurons manually via PyTorch autograd — no snnTorch required to reproduce the results.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/JK-Sah/flowsnn.git
cd flowsnn
```

### 2. Install dependencies

```bash
pip install torch scipy numpy tqdm
```

### 3. Run all experiments (seeds 0, 1, 2)

```bash
python flowsnn_run_experiments.py
```

This runs the SNN, MLP, CNN, and GRU baselines across seeds 0–2 and writes results to `flowsnn_results.json`.

### 4. Resume an interrupted run

```bash
python flowsnn_run_experiments.py --resume
```

Checkpoints are saved after each seed so you can safely interrupt and continue.

---

## Output

`flowsnn_results.json` contains per-seed and aggregate (mean ± std) metrics for:

- Structure classification accuracy
- Bulk-speed MAE (mm/s)
- Heading MAE (degrees)
- Energy per inference (µJ)

These numbers correspond directly to **Tables 2–3** and **Section 4.1** of the paper.

---

## Expected Results (from paper)

| Model | Structure acc. | Heading MAE (deg) | Energy / inference |
|---|---|---|---|
| SNN (events) | 0.933 ± 0.012 | 7.4 ± 0.3 | ~0.14 µJ |
| MLP | 0.967 ± 0.008 | 1.7 | 21.7 µJ |
| CNN | 0.961 ± 0.005 | 2.6 | 10.3 µJ |
| GRU | 0.633 ± 0.017 | 4.7 | 18.1 µJ |

---

## Encoding Ablation (Table 2)

| Encoding | Structure acc. |
|---|---|
| Phasic only | 0.917 ± 0.025 |
| Tonic only | 0.611 ± 0.012 |
| Combined | 0.933 ± 0.012 |

---

## Runtime Estimate

| Hardware | Estimated time |
|---|---|
| CPU (modern laptop) | ~10–20 minutes |
| NVIDIA GPU (CUDA) | ~1–2 minutes |
| Apple Silicon (MPS) | ~2–4 minutes |

---

## Author

**J. K. Sah**  
Department of Mechanical and Industrial Engineering  
Kate Gleason College of Engineering  
Rochester Institute of Technology, Rochester, NY 14623, USA  
js8472@rit.edu  
ORCID: https://orcid.org/0000-0002-7643-5137

---