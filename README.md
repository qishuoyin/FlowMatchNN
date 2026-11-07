# FlowMatchNN

Reference implementation for **"A Theory on Flow Matching with Neural
Networks"**. The paper studies convergence, generalization, and
Wasserstein-1 sampling guarantees for flow matching with a two-layer
ReLU velocity network.

The repository contains the implementations of Algorithms 1 and 2 and
experiments covering optimization, coupling, sampling, and representation
dimension. Training uses full-batch gradient descent, and sampling
integrates the learned velocity field with forward Euler steps.

---

## Repository layout

Python entry points live in `exp/`; matching grid definitions and run
scripts live in `scripts/`.

| Folder | Study | Default grid |
| --- | --- | --- |
| `exp1_optimization` | NTK spectrum and gradient-descent convergence | 3 dimensions × 2 sample sizes × 7 widths × 3 step-size multipliers × 5 seeds |
| `exp2_coupling` | Coupling × network-conditioning comparison | 3 couplings × 2 architectures × 3 dimensions × 5 seeds |
| `exp3_real_data` | PCA-dimension sweep on MNIST and Fashion-MNIST | 2 datasets × 16 `(d_pca, n)` pairs × 3 seeds |

```
FlowMatchNN/
├── pyproject.toml                     # package metadata
├── flowmatchnn.yml                    # conda environment
│
├── flowmatchnn/                       # reusable model, training, and evaluation code
│   ├── data_generation.py             # Algorithm 1: data generating procedure
│   ├── learning_and_sampling.py       # Algorithm 2: training, sampling, evaluation
│   ├── model.py                       # conditional and unconditional ReLU networks
│   └── utils/
│       ├── data.py                    # distributions, interpolant, and couplings
│       ├── velocity.py                # conditional velocity field
│       ├── losses.py                  # empirical flow-matching loss
│       ├── metrics.py                 # sliced and exact W₁ metrics
│       ├── ntk.py                     # closed-form NTK and spectral utilities
│       ├── pca.py                     # PCA and pixel-space evaluation
│       └── realdata.py                # parquet image loading
│
├── exp/                               # Python experiment entry points
│   ├── _common.py                     # shared CLI, paths, and CUDA checks
│   ├── exp1_optimization/            # NTK and simulation experiments
│   ├── exp2_coupling/                # coupling/architecture comparison
│   └── exp3_real_data/               # PCA and image experiments
│
├── scripts/                           # grids and run scripts; mirrors exp/
│   ├── _env.sh                        # shared conda-env activation
│   └── exp<k>_<name>/                 # grid definitions and run scripts
│
├── test/                              # NTK unit tests
├── figs/                              # aggregate figures and CSV tables
├── data/                              # generated/downloaded data (gitignored)
├── results/                           # per-run artifacts (gitignored)
└── logs/                              # run logs (gitignored)
```

## Experimental invariants

The experiment drivers enforce the following conventions:

1. **Training and evaluation data remain separate.** Real-data samplers
   draw their source pools from the training split. Held-out samples are
   used only for evaluation.
2. **Metrics have fixed meanings.** `velocity_risk_l2` is the first
   moment `E[||f-v||_2]` bounded in Theorem 3.3.
   `velocity_risk_sq` is retained as a separate diagnostic and is not
   compared with the same reference exponent.
3. **Distributional scores include context.** Coupling plots report the
   do-nothing reference and a matched finite-sample comparison. PCA plots
   report `pca_reconstruction_w1` and `holdout_floor_w1`. These are
   diagnostic reference levels, not universal mathematical lower bounds.
   In particular, a do-nothing pixel-space baseline is not used after PCA
   truncation because sliced W₁ is not monotone under added variance.
4. **Step sizes are calibrated per cell.** Where applicable,
   `ntk.eta_spectral` sets `eta = safety / lambda_max(H_tilde)`. Coupling
   arms are therefore not compared at silently different effective step
   sizes.
5. **Stopping status is explicit.** Plateau-enabled trainers record
   `stopped_iter` and `converged_ratio`, allowing readers to distinguish
   early stopping from reaching the iteration cap. A completed run is not
   automatically called converged.
6. **Reported cells aggregate seeds.** Visualization scripts write mean
   and standard deviation rather than selecting a single replication.
7. **Paths are stable.** Entry points change to the repository root, so
   relative `data/`, `results/`, and `figs/` paths do not depend on the
   caller's working directory.
8. **Every run is self-describing.** Resolved arguments and summary
   statistics are stored beside traces in `results/<run_name>/`.

---

## Installation

The provided environment uses Python 3.11, PyTorch with CUDA 12.1,
NumPy, SciPy, Matplotlib, POT, and pytest.

```bash
conda env create -f flowmatchnn.yml
conda activate flowmatchnn
python -m pip install -e .
```

The training entry points require a CUDA-capable GPU and fail explicitly
when CUDA is unavailable. The NTK unit tests can be run separately:

```bash
python -m pytest -q test/test_ntk.py
```

---

## Running the experiments

Run all commands from the repository root. The three main sweeps are
independent and may be submitted concurrently.

### 1. Download the image datasets

This step is required for the PCA and image experiments. The downloaders
place parquet files under `data/application/`.

```bash
bash scripts/exp3_real_data/download_mnist.sh
bash scripts/exp3_real_data/download_fashion_mnist.sh
```

### 2. Submit the experiment grids

```bash
sbatch scripts/exp1_optimization/run_ntk.sh
sbatch scripts/exp2_coupling/run_coupling.sh
sbatch scripts/exp3_real_data/run_pca_sweep.sh
```

### 3. Build aggregate figures and tables

These visualization steps are CPU-only and read the completed per-run
artifacts from `results/`:

```bash
bash scripts/exp1_optimization/run_ntk_vis.sh
bash scripts/exp2_coupling/run_coupling_vis.sh
bash scripts/exp3_real_data/run_pca_sweep_vis.sh
```

Together, these components provide a unified implementation for studying
optimization dynamics, coupling choices, and representation dimension in
neural flow matching. The shared training, evaluation, and visualization
code keeps the paper's theoretical and empirical analyses reproducible
within one consistent framework.
