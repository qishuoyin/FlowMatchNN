# FlowMatchNN

Code accompanying **"A Theory on Flow Matching with Neural Networks"**
(NeurIPS 2026 submission). The paper establishes convergence, generalization,
and Wasserstein-1 sampling guarantees for flow matching trained with a
two-layer ReLU network. This repository provides implementations of the
two algorithms in the paper and empirical illustrations of those guarantees.

---

## Repository layout

```
FlowMatchNN/
├── neurips_2026.pdf                   # paper (gitignored)
├── flowmatchnn.yml                    # conda environment
│
├── flowmatchnn/                       # installable package
│   ├── data_generation.py             # Algorithm 1: data generating procedure
│   ├── learning_and_sampling.py       # Algorithm 2: training + sampling + evaluation
│   ├── model.py                       # two-layer ReLU velocity network
│   └── utils/
│       ├── data.py                    # source/target samplers and interpolant schedules
│       ├── velocity.py                # conditional velocity field
│       ├── losses.py                  # empirical flow-matching loss
│       ├── metrics.py                 # sliced Wasserstein-1 estimator
│       └── pca.py                     # PCA compression for real-data experiments
│
├── exp/                               # experiment entry points
│   ├── run_data.py                    # generate training dataset (Algorithm 1)
│   ├── run_exp1.py                    # Exp 1 (simulation): training loss vs iteration
│   ├── run_exp2.py                    # Exp 2 (simulation): W₁ distance vs iteration
│   ├── run_exp1_vis.py                # plot Exp 1 results
│   ├── run_exp2_vis.py                # plot Exp 2 results
│   ├── run_real_exp1.py               # Exp 1 (real data): training loss vs iteration
│   ├── run_real_exp2.py               # Exp 2 (real data): W₁ distance vs iteration
│   ├── run_real_exp1_vis.py           # plot real Exp 1 results
│   ├── run_real_exp2_vis.py           # plot real Exp 2 results
│   └── run_real_vis_combined.py       # combined real-data figure
│
├── scripts/                           # SLURM array scripts
│   ├── _grid.sh                       # simulation hyperparameter grid
│   ├── _real_grid.sh                  # real-data hyperparameter grid
│   ├── run_all_exp.sh                 # simulation: full pipeline (recommended)
│   ├── run_all_real_exp_mnist.sh      # MNIST: full pipeline
│   ├── run_all_real_exp_fashion_mnist.sh  # Fashion-MNIST: full pipeline
│   ├── download_mnist.sh              # fetch MNIST parquets
│   ├── download_fashion_mnist.sh      # fetch Fashion-MNIST parquets
│   └── ...                            # per-step scripts (data/exp1/exp2/vis)
│
├── figs/                              # output figures (committed)
├── data/                              # gitignored — generated datasets
├── results/                           # gitignored — experiment traces and configs
└── logs/                              # gitignored — SLURM stdout/stderr
```

Each completed run writes:

```
results/<run_name>/
├── config.json        # all resolved hyperparameters
├── exp1_trace.csv     # columns: t, loss
└── exp2_trace.csv     # columns: t, w1
```

---

## Experiments

### Track A — Simulation

Trains on a synthetic two-component Gaussian mixture target in `R^d`.
At every `eval_every` gradient-descent steps, records:

- **Exp 1:** training loss
- **Exp 2:** sliced Wasserstein-1 distance between model samples and held-out target samples

Hyperparameter grid (150 cells):

| Axis      | Values                                     |
|-----------|--------------------------------------------|
| `m` (width)   | 16, 32, 64, 128, 256                   |
| `η` (step size) | 1e-4, 1e-3, 1e-2                      |
| `d` (dimension) | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 |
| `n_train` | 500                                        |
| `n_test`  | 5 000                                      |

Fixed: `T = 500` iterations, `κ = 1.0`, `T_euler = 200`, `B_1 = 10`, `eval_every = 10`.

### Track B — Real image data

Applies the same flow-matching pipeline to **MNIST** and **Fashion-MNIST**.
Images are compressed with PCA (retaining `d_flat // 100` components) before
training. Records the same loss and W₁ metrics.

Hyperparameter grid (12 cells per dataset):

| Axis      | Values              |
|-----------|---------------------|
| `m`       | 128, 256, 512, 1024 |
| `η`       | 1e-4, 1e-3, 1e-2    |
| `n_train` | 500                 |
| `n_test`  | 2 000               |

---

## Installation

```bash
module load anaconda3/2024.6
conda env create -f flowmatchnn.yml
conda activate flowmatchnn
pip install -e .
```

Requires a CUDA-capable GPU (`torch.cuda.is_available()` must be `True`).

---

## Running the experiments

### Simulation (Track A)

Full grid on SLURM (recommended):

```bash
sbatch scripts/run_all_exp.sh        # 150 cells across 10 array tasks
```

Single cell locally:

```bash
IDX=0 bash scripts/run_all_exp.sh
```

### Real image data (Track B)

Download the datasets first:

```bash
bash scripts/download_mnist.sh
bash scripts/download_fashion_mnist.sh
```

Then submit:

```bash
sbatch scripts/run_all_real_exp_mnist.sh
sbatch scripts/run_all_real_exp_fashion_mnist.sh
```

Single cell locally:

```bash
RIDX=0 bash scripts/run_all_real_exp.sh --dataset mnist
```

### Customization

Edit `scripts/_grid.sh` (simulation) or `scripts/_real_grid.sh` (real data)
to change the hyperparameter grid — all downstream scripts pick up the change
automatically.

Pass extra CLI flags directly to the underlying Python script:

```bash
sbatch scripts/run_exp1.sh --T 1000 --kappa 0.5
```

Adjust SLURM concurrency at submission time:

```bash
sbatch --array=0-19     scripts/run_all_exp.sh   # 20 shards
sbatch --array=0-9%5    scripts/run_all_exp.sh   # cap at 5 concurrent
sbatch --array=42       scripts/run_all_exp.sh   # single cell 42
```
