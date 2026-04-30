# FlowMatchNN

Implementation of **Flow Matching with Neural Networks** (NeurIPS 2026).

This repo provides:
1. A reusable package (`flowmatch_nn/`) implementing Algorithm 1 (training the
   neural-network velocity field) and Algorithm 2 (the flow / sampling
   procedure).
2. Reproducible experiments that produce two figures:
   - **Loss vs. time step `t`** (loss values per time step `t` of Alg 2)
   - **TV-distance vs. time step `t`** (TV between samples and target per `t`)
3. `argparse`-based CLI for every experiment so all hyperparameters are
   overridable from the command line.
4. SLURM shell scripts for submitting jobs to the Princeton cluster
   (`/scratch/gpfs/...`).

---

## 1. Repository layout

```
FlowMatchNN/
├── README.md
├── pyproject.toml                   # package metadata + deps
├── requirements.txt                 # pinned deps for cluster envs
│
├── flowmatch_nn/                    # the installable package
│   ├── __init__.py
│   ├── interpolant.py               # mu(tau), sigma(tau), x_tau = mu*x0 + sigma*x1
│   ├── models.py                    # MLP velocity field v_theta(x, tau)
│   ├── data.py                      # samplers for x0 (target) and x1 = N(0, I_d)
│   ├── algorithm1.py                # Alg 1: train v_theta on the FM loss
│   ├── algorithm2.py                # Alg 2: flow / sampling with kappa, T
│   ├── losses.py                    # FM loss; per-t loss bookkeeping
│   ├── metrics.py                   # TV distance estimators (histogram / k-NN)
│   ├── utils.py                     # seeding, logging, IO helpers
│   └── config.py                    # dataclass holding all hyperparameters
│
├── scripts/                         # entry points (use argparse)
│   ├── train.py                     # runs Alg 1, saves checkpoint
│   ├── run_alg2.py                  # runs Alg 2, logs per-t loss + samples
│   ├── eval_tv.py                   # computes TV-distance per t from samples
│   └── plot_results.py              # generates the two figures
│
├── slurm/                           # cluster submission scripts
│   ├── env.sh                       # module loads + venv activate
│   ├── train.slurm
│   ├── run_alg2.slurm
│   ├── eval_tv.slurm
│   └── full_pipeline.slurm          # train -> alg2 -> tv -> plot
│
├── configs/                         # default YAML/JSON configs (optional)
│   └── default.yaml
│
├── results/                         # auto-created; per-run outputs
│   └── <run_name>/
│       ├── config.json
│       ├── checkpoint.pt
│       ├── alg2_trace.npz           # loss[t], samples[t], times[t]
│       ├── tv.npz                   # tv_distance[t]
│       └── figures/
│           ├── loss_vs_t.pdf
│           └── tv_vs_t.pdf
│
└── tests/                           # minimal pytest sanity checks
    ├── test_interpolant.py
    ├── test_algorithm1.py
    └── test_algorithm2.py
```

---

## 2. Mathematical setup (from the paper / README)

Let `x_0` denote the target distribution we want to learn and
`x_1 ~ N(0, I_d)` denote the easy reference distribution.

Stochastic interpolant on `tau ∈ [0, 1]`:
```
x_tau = mu(tau) * x_0 + sigma(tau) * x_1
mu(tau)    = 1 - tau
sigma(tau) = tau
```

So `x_0` is recovered at `tau = 0` and pure noise at `tau = 1`.

> **Confirm against paper:** the README gives `mu(tau) = 1 - tau`,
> `sigma(tau) = tau`, and `x_1 ~ N(0, 1)`. This is the standard linear
> interpolant. If the paper uses a different convention for which endpoint
> is data vs. noise, we'll just flip the roles in `interpolant.py`.

### Algorithm 1 — training (placeholder, to confirm vs. paper)

Inputs (from README):
- `n = 100` (sample size, batch of `(x_0, x_1)` pairs)
- `d = 5` (dimension)
- `x_1 ~ N(0, I_d)`
- target `x_0` distribution: **needs confirmation from paper.**
  Reasonable defaults: a 2-mode Gaussian mixture in `R^d`, or whatever the
  paper's toy example uses.

Per training step:
1. Sample `tau ~ Uniform[0, 1]`.
2. Sample `x_0 ~ p_data`, `x_1 ~ N(0, I_d)`.
3. Form `x_tau = mu(tau) x_0 + sigma(tau) x_1`.
4. Target velocity `u(tau, x_tau) = mu'(tau) x_0 + sigma'(tau) x_1
                                 = -x_0 + x_1`.
5. Loss `L(theta) = E || v_theta(tau, x_tau) - u(tau, x_tau) ||^2`.
6. Gradient step on `theta`.

### Algorithm 2 — flow / sampling (placeholder, to confirm vs. paper)

Inputs (from README): `kappa = 1`, `T = 200`.

Discretize `tau ∈ [0, 1]` into `T` steps `t = 0, 1, ..., T-1` with step
size `dt = 1/T`. Starting from `x_1 ~ N(0, I_d)`, integrate the learned
velocity field backwards to `tau = 0`. With a stochastic component
controlled by `kappa`, this looks like an SDE step:
```
x_{tau - dt} = x_tau - v_theta(tau, x_tau) * dt + kappa * sqrt(dt) * eta,
                                                            eta ~ N(0, I_d)
```

> **Confirm against paper:** I'm guessing the SDE form because `kappa`
> typically scales a noise term in flow-matching-with-stochasticity
> variants. Please paste Alg 2 (or the relevant equation) so I match
> exactly.

### What the figures plot

For each `t ∈ {0, ..., T-1}` of Alg 2:
- **`loss_vs_t.pdf`**: the FM loss
  `|| v_theta(tau_t, x_tau_t) - u(tau_t, x_tau_t) ||^2`
  evaluated at the current sample `x_tau_t` produced by Alg 2 (averaged
  over the batch of `n` samples).
- **`tv_vs_t.pdf`**: TV distance between the empirical distribution of the
  `n` samples at step `t` and the target distribution.
  Estimator: histogram-based per-dimension TV (cheap, works for `d = 5`),
  with an optional k-NN-based multivariate TV as a sanity check.

---

## 3. Hyperparameters (defaults; all overridable via argparse)

| Name              | Default       | Where         | Notes                                    |
|-------------------|---------------|---------------|------------------------------------------|
| `n`               | 100           | Alg 1 / Alg 2 | sample size / batch size                 |
| `d`               | 5             | Alg 1 / Alg 2 | dimension                                |
| `kappa`           | 1.0           | Alg 2         | noise scale                              |
| `T`               | 200           | Alg 2         | number of time steps                     |
| `lr`              | 1e-3          | Alg 1         | Adam learning rate                       |
| `n_train_steps`   | 20000         | Alg 1         | training iterations                      |
| `hidden_dim`      | 128           | model         | MLP width                                |
| `n_layers`        | 4             | model         | MLP depth                                |
| `activation`      | `silu`        | model         |                                          |
| `seed`            | 0             | global        |                                          |
| `device`          | `cuda`        | global        | falls back to `cpu` automatically        |
| `target`          | `gmm`         | data          | `gmm` / `gaussian` / `checkerboard` / ...|
| `n_eval`          | 5000          | Alg 2 / TV    | samples used for TV estimate             |
| `out_dir`         | `results/run` | IO            |                                          |

---

## 4. Usage

### Local quickstart

```bash
pip install -e .

# 1. Train v_theta (Algorithm 1)
python scripts/train.py --out_dir results/run0 --n 100 --d 5 \
                       --n_train_steps 20000 --seed 0

# 2. Run Algorithm 2, log per-t loss and samples
python scripts/run_alg2.py --out_dir results/run0 --kappa 1.0 --T 200

# 3. Compute TV distance per t
python scripts/eval_tv.py --out_dir results/run0 --n_eval 5000

# 4. Make the two figures
python scripts/plot_results.py --out_dir results/run0
```

### Cluster (SLURM)

```bash
sbatch slurm/full_pipeline.slurm           # uses defaults
sbatch slurm/full_pipeline.slurm --kappa 0.5 --T 400   # override
```

Each `*.slurm` script:
- requests 1 GPU + a small CPU/RAM allocation,
- sources `slurm/env.sh` (module load python, activate venv),
- forwards extra CLI flags through to the underlying Python script.

---

## 5. Reproducibility

- All randomness routed through a single `torch.Generator` keyed by `--seed`.
- Each run dumps `config.json` containing the resolved argparse namespace.
- `results/<run_name>/` is self-contained: checkpoint + traces + figures.

---

## 6. Open questions for you (please confirm before I implement)

1. **The PDF isn't accessible to me on the cluster.** Could you paste
   Algorithm 1 and Algorithm 2 (or attach the PDF inside this workspace,
   e.g. `/scratch/gpfs/JQFAN/qy1448/FlowMatchNN/paper.pdf`)? I want to
   match the loss form and the SDE/ODE step exactly.
2. **Target distribution `p_data` for `x_0`**: the README only specifies
   `x_1 ~ N(0, 1)`. What does the paper use as the *target*? (Gaussian
   mixture? A specific mean/covariance? A non-Gaussian toy?)
3. **TV-distance estimator**: is the paper's TV computed per-dimension and
   averaged, or via a multivariate estimator? I'll default to per-dim
   histogram TV until told otherwise.
4. **Cluster account / partition**: which SLURM `--account` and
   `--partition` should the `*.slurm` headers use (e.g., `pli`, `gpu`)?
   Default I'll put in is `gpu` with a `# TODO` comment.

Once you confirm the four points above (especially #1 and #2), I'll
implement the package, scripts, and SLURM files in that order.
