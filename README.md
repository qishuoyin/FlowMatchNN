# FlowMatchNN

Implementation and experiments for **"A Theory on Flow Matching with Neural
Networks"** (NeurIPS 2026 submission, paper at `neurips_2026.pdf`).

The paper is purely theoretical (Theorems 3.2, 3.3, 3.4 — convergence,
generalization, Wasserstein-1 sampling guarantees) and contains **no
experiments of its own**. This repo provides:

1. A package `flowmatchnn/` that implements Algorithm 1 (data generating
   procedure) and Algorithm 2 (learning + sampling) literally as written.
2. Two empirical plots probed at every gradient-descent iteration `t` of
   Algorithm 2:
   - **Exp 1:** loss `L(θ^(t))` from Eq. (2) vs `t`
   - **Exp 2:** Wasserstein-1 distance between Algorithm-2 samples (under
     `θ^(t)`) and the data distribution `P_1`, vs `t`. This is the metric
     bounded by Theorem 3.4 in the paper.
3. `argparse`-driven scripts so all hyperparameters are CLI-overridable.
4. Shell scripts under `exp/scripts/` that run/submit each step.
5. A conda env file `flowmatchnn.yml` for installing on the cluster.

---

## 1. Math, copied directly from the paper

### Setup (paper §3.1.1)

- Data `{x_{1,i}}_{i ∈ [n]}` are i.i.d. from a target distribution `P_1` on `R^d`.
- Source samples `x_{0,i}` are drawn from a conditional `P(x_0 | x_{1,i})` (the
  paper assumes an oracle mapping; in practice we draw `x_0 ~ N(0, I_d)`
  independently of `x_1`, the standard FM realization consistent with
  `P_0 = N(0, I_d)`).
- Scalar schedules `σ_τ, μ_τ` satisfy `σ_0 = μ_1 = 1` and `σ_1 = μ_0 = 0`.
- Interpolant: `x_τ = σ_τ · x_0 + μ_τ · x_1`, so `x_τ|_{τ=0} = x_0` (noise) and
  `x_τ|_{τ=1} = x_1` (data).
- Conditional velocity field (Eq. 1):
  `v_τ(x_τ | x_0) = σ'_τ · x_0 + (μ'_τ / μ_τ) · (x_τ − σ_τ · x_0)`.
  Evaluated on a paired `(x_0, x_1)` (i.e. on the path) this collapses to
  `v_τ = σ'_τ · x_0 + μ'_τ · x_1`.

### Network (paper §3.1.1)

Two-layer ReLU, **both layers trainable**:

```
f_θ(x_τ, τ, x_0) = (1/√m) Σ_{r=1}^{m} a_r · ReLU(w_r^⊤ [x_τ; τ; x_0])
```

with `a_r ∈ R^d`, `w_r ∈ R^{2d+1}`, `θ = [a_1, …, a_m, w_1, …, w_m]`.

Initialization: `w_r ~ N(0, κ² I_{2d+1})`, `a_r ~ N(0, I_d)`.

### Empirical loss (Eq. 2)

```
L(θ) = (1/n) Σ_{i=1}^n  || f_θ(x_{τ_i, i}, τ_i, x_{0,i}) − v_{τ_i}(x_{τ_i, i} | x_{0,i}) ||_2^2
```

### Algorithm 1 — Data generating procedure (paper p. 4)

```
input : n, dataset {x_{1,i}}_{i ∈ [n]}
for i = 1, …, n:
    sample τ_i ~ Uniform([0, 1])
    sample x_{0,i} ~ P(x_0 | x_{1,i})
    x_{τ,i} ← σ_{τ_i} · x_{0,i} + μ_{τ_i} · x_{1,i}
return {(τ_i, x_{τ_i, i}, x_{0,i})}_{i ∈ [n]}
```

### Algorithm 2 — Learning and Sampling (paper p. 5)

```
input : augmented dataset {x_{0,i}, τ_i, x_{1,i}}_{i ∈ [n]}, step size η, constant B_1
init  : w_r ~ N(0, κ² I_d), a_r ~ N(0, I_d)
for t = 1, …, T:
    θ^(t) ← θ^(t−1) − η · ∇_θ L(θ^(t−1))           # full-batch GD on Eq. (2)
sample x̂_0
solve  d x̂_τ / dτ = f_{θ̂}(x̂_τ, τ, x̂_0)             # Euler from τ=0 to τ=1
B̃_1 ← √d · B_1
truncate x̂_1 ←  x̂_1 · 1{||x̂_1|| ≤ 2 B̃_1} + (projection onto sphere of radius 2 B̃_1) · 1{||x̂_1|| > 2 B̃_1}
return x̂_1
```

`T` is the number of **gradient-descent iterations** (not Euler steps).

---

## 2. Note on the original README's convention

The original README specified `x_1 ~ N(0,1)` and `σ(τ) = τ, μ(τ) = 1 − τ`.
Under the paper's boundary conditions `σ_0 = μ_1 = 1`, `σ_1 = μ_0 = 0`, the
unique linear choice is the **opposite**: `σ(τ) = 1 − τ`, `μ(τ) = τ`. The
original spec corresponds to swapping the labels `x_0 ↔ x_1`, which would
also break Algorithm 2's sampling step (line 6 needs to draw a *fresh*
`x̂_0` from the easy distribution `P_0`).

This implementation follows the paper:

| Symbol      | Meaning           | Distribution      | Schedule choice           |
|-------------|-------------------|-------------------|---------------------------|
| `x_0`       | source / noise    | `N(0, I_d)`       | weighted by `σ_τ = 1 − τ` |
| `x_1`       | target / data     | `P_1` (TBD)       | weighted by `μ_τ = τ`     |

The schedule `σ, μ` is plugged in via two functions in
`flowmatchnn/utils/data.py` and can be swapped if you really want the
original ordering.

---

## 3. Experiments

For each gradient-descent iteration `t = 0, 1, …, T`:

- **Exp 1 (`run_exp1.py`):** record `loss[t] = L(θ^(t))` evaluated on the
  same `n_train` training pairs (default `n_train = 500`).
- **Exp 2 (`run_exp2.py`):** run Algorithm 2's sampling step using `θ^(t)`
  to draw `n_train` samples `{x̂_1^(j)}` (matched to the training-set size,
  so there is no separate `n_eval` knob), then estimate
  `W_1(P_1, P_{x̂_1})`.

Visualization (`run_exp1_vis.py`, `run_exp2_vis.py`) reads the trace files
in `results/` and writes each figure as a PNG into `figs/`. There are two
plot modes selected by `--mode`:

- `--mode setting`: one PNG per `(m, η, n_train, n_test)` setting, with
  every `d` in `D_LIST` overlaid as a separate curve (color = `d`).
  - `figs/loss_vs_t__<setting>.png`
  - `figs/w1_vs_t__<setting>.png`
- `--mode global`: one PNG per `(n_train, n_test)`, with every `(m, η)`
  pair overlaid as a separate curve. The y-value is the tail mean over
  the last `--tail` evaluations. The window size is set by `TAIL` in
  `scripts/_grid.sh` (override at submit time via
  `TAIL=10 sbatch scripts/run_exp2_vis.sh`).
  - `figs/loss_vs_d__ntr<ntr>_nte<nte>.png`
  - `figs/w1_vs_d__ntr<ntr>_nte<nte>.png`

In the global plot, color encodes `η` (light → dark for large → small),
and marker / linestyle encodes `m` cycling through 3 distinct pairs
(`o`/`-`, `s`/`--`, `^`/`:`); with 5 `m` values the encoding wraps
around, so e.g. `m=16` and `m=128` share marker/linestyle and are
disambiguated only by `η`-color. See `_M_LINESTYLES` and `_M_MARKERS`
in `exp/_vis_common.py` to extend.

### W_1-distance estimator

The paper bounds `W_1(P_1, hat P_1)` (Theorem 3.4), so we evaluate
convergence in the same metric. Computing the multivariate `W_1` in
`R^d` exactly costs `O(n^3)` (linear program), which is impractical
inside a training loop swept across `d ∈ {5, …, 50}` and
`n_test = 5000`. The standard practical surrogate — and the
one the optimal-transport literature uses for high-dimensional `W_1`
estimation — is the **sliced Wasserstein-1**:

```
SW_1(P, Q) = (1 / K) Σ_{k=1}^K  W_1(u_k · P,  u_k · Q),
                                   u_k iid uniform on S^{d-1}.
```

`SW_1` is a metric, equals 0 iff `P = Q`, and lower-bounds the true
multivariate `W_1`. Each 1-D `W_1` has a closed form via the CDF
formulation `W_1 = ∫ |F_p − F_q|`, evaluated on the sorted union of
the two projected sample sets — so unequal `n_train` / `n_test` are
handled exactly, no LP solver required. Default `K = 128`
projections; override at submit time with `--n_proj 256`. Directions
are drawn once per run (seeded off `--seed`) and reused for every
evaluation, so the trace is comparable across `t` (no projection-noise
jitter). Implementation: `flowmatchnn/utils/metrics.py`.

### Hyperparameters (final, locked in)

| Name        | Value   | Where         | Notes |
|-------------|---------|---------------|-------|
| `n_train`   | swept   | Algs 1 & 2    | Training-set size for Eq. (2); also reused as the # of Algorithm-2 samples drawn under `θ^(t)` for `W_1`. See grid below. |
| `n_test`    | swept   | Exp 2         | Fresh `P_1` samples used as the `W_1` reference set. See grid below. |
| `d`         | swept   | Algs 1 & 2    | Ambient dimension. See grid below. |
| `κ`         | 1.0     | Alg 2 (init)  |  |
| `T`         | 500     | Alg 2 (GD)    | Full-batch GD iterations. |
| `T_euler`   | 200     | Alg 2 (sample)| Euler steps for the ODE solve at sampling time. |
| `B_1`       | 10.0    | Alg 2 (sample)| Truncation constant; `B̃_1 = √d · B_1`, threshold `2 B̃_1`. |
| `eval_every`| 10      | Both exps     | Evaluate every 10 GD steps → 51 points: `t ∈ {0, 10, …, 500}`. |
| `n_proj`    | 128     | Exp 2         | # random unit directions used by the sliced-W_1 estimator. |
| `seed`      | 0       |               |  |
| `device`    | `cuda`  |               | **Hard fail** if `torch.cuda.is_available() == False`. No CPU fallback. |

### `(m, η, d, n_train, n_test)` grid search

Paper requires `m` polynomial in `n, d, …` and small `η`. We sweep five
hyperparameters jointly (full cross-product):

| Hyperparameter | Grid                                                  |
|----------------|-------------------------------------------------------|
| `m` (width)    | `{16, 32, 64, 128, 256}` (geometric, 5 values)        |
| `η` (step size)| `{1e-4, 1e-3, 1e-2}`                                  |
| `d` (ambient)  | `{5, 10, 15, …, 50}` (10 values, step 5)              |
| `n_train`      | `{500}`                                               |
| `n_test`       | `{5000}`                                              |

→ `5 · 3 · 10 · 1 · 1 = 150` cells. Index mapping (`m` outermost,
`n_test` innermost) and the `cell_indices` dispatcher live in
`scripts/_grid.sh`. Each cell produces its own
`results/m{m}_eta{eta}_d{d}_ntr{n_train}_nte{n_test}/` directory.

`n_train` and `n_test` are pinned to a single value: an empirical
audit showed `n_test ∈ {5000, 10000}` produced visually identical
plots (median |Δw₁|/w₁ ≈ 1.2e-4), and `n_train` was collapsed to
the value where the W₁-vs-d trend is cleanest. Sweeping `m`
geometrically (rather than two adjacent points) is what makes the
NTK saturation visible in the global plot.

Plots are produced at two different aggregation levels (see §3 above):
one *per-setting* `(m, η, n_train, n_test)` figure (10 d-curves
overlaid) and one *global* `(n_train, n_test)` figure (15 `(m, η)`
curves overlaid). To run an even smaller grid, edit the `*_LIST`
arrays in `_grid.sh`; the plot scripts pick up the new lists
automatically.

#### Compute budget and SLURM sharding

The previous 9-cell pilot job (`d = 5, n_train = 100, n_test = 5000`,
full pipeline data → exp1+exp1_vis → exp2+exp2_vis) ran in **1 min 27 s
total ≈ 10 s/cell**. Per-cell compute scales (very roughly) as
`m · (2d + 1) · n_train · T`, so against the current 150-cell grid
(`n_train = 500`, `d ∈ {5, …, 50}`, `T = 500`):

| Cell                                          | Scale vs pilot | Estimated wall time |
|-----------------------------------------------|---------------:|--------------------:|
| `m=16,  d=5,   n_train=500, T=500`  (lightest)|    ~5×         |     ~50 s           |
| Geometric-mean cell (m≈64, d≈25, n=500, T=500)|    ~95×        |     ~16 min         |
| `m=256, d=50,  n_train=500, T=500`  (heaviest)|    ~735×       |     ~2 h            |

The scripts default to **`#SBATCH --array=0-9` — 10 SLURM array tasks**,
each processing a stride-10 shard of 15 cells:

- shard `k ∈ {0..9}` runs cells `{k, k+10, k+20, …}` (round robin), so
  heavy and light cells are interleaved across shards.
- `cell_indices` / `setting_indices` in `scripts/_grid.sh` read
  `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` to compute the
  stride, so changing the shard count is just
  `sbatch --array=0-N scripts/run_all_exp.sh` (use contiguous ranges).

### Source coupling `P(x_0 | x_1)`

Locked in: `P(x_0 | x_1) = N(x_1, I_d)`. The marginal `P_0` is therefore
`P_1 * N(0, I_d)`; at sampling time we draw fresh `x̂_0` from this
marginal by drawing `x_1' ~ P_1` then `x̂_0 ~ N(x_1', I_d)`. See
`flowmatchnn/utils/data.py` (`sample_p_x0_given_x1`, `sample_p0_marginal`).

### Target `P_1`

Two-component Gaussian mixture in `R^d` (with `d` the swept ambient
dimension), centers at `±2·e_1`, identity covariance, equal weights.
See `sample_p1_gmm` in `flowmatchnn/utils/data.py`.

---

## 4. Repository layout

```
FlowMatchNN/
├── README.md
├── .gitignore
├── neurips_2026.pdf                # gitignored
├── flowmatchnn.yml                 # conda env (install on cluster)
│
├── flowmatchnn/                    # the installable package
│   ├── __init__.py
│   ├── data_generation.py          # Algorithm 1
│   ├── learning_and_sampling.py    # Algorithm 2
│   ├── model.py                    # f_θ two-layer ReLU per §3.1.1
│   └── utils/
│       ├── __init__.py
│       ├── data.py                 # P_0/P_1 samplers + interpolant (σ_τ, μ_τ, σ'_τ, μ'_τ)
│       ├── velocity.py             # v_τ(x_τ | x_0) per Eq. (1)
│       ├── losses.py               # Eq. (2)
│       └── metrics.py              # sliced Wasserstein-1 distance
│
├── exp/                            # python entry points
│   ├── run_exp1.py                 # loss vs t
│   ├── run_exp2.py                 # W_1  vs t
│   ├── run_exp1_vis.py             # plot loss vs t
│   └── run_exp2_vis.py             # plot W_1  vs t
│
├── scripts/                        # shell scripts (root-level)
│   ├── run_data.sh                 # generate the augmented dataset (Alg 1)
│   ├── run_exp1.sh                 # run Exp 1
│   ├── run_exp2.sh                 # run Exp 2
│   ├── run_exp1_vis.sh             # plot Exp 1
│   ├── run_exp2_vis.sh             # plot Exp 2
│   ├── run_exp1_all.sh             # data → exp1 → vis
│   └── run_exp2_all.sh             # data → exp2 → vis
│
├── figs/                           # final figures (committed)
│
├── data/                           # gitignored — Algorithm 1 outputs (.pkl)
│   └── <run_name>/
│       └── dataset.pkl             # {τ_i, x_τ,i, x_0,i, x_1,i}
│
├── logs/                           # gitignored — stdout/stderr from cluster jobs
└── results/                        # gitignored — Exp 1 / Exp 2 outputs
    └── <run_name>/
        ├── config.json             # snapshot of CLI args (see §6)
        ├── exp1_trace.csv          # columns: t, loss
        ├── exp2_trace.csv          # columns: t, w1
        └── exp2_samples.pkl        # generated x̂_1 per evaluated t (optional)
```

### On `slurm/`

You're right — a separate `slurm/` directory isn't needed. Each shell script
under `scripts/` will carry its own `#SBATCH` headers at the top, so
`bash scripts/run_exp1.sh` works locally and
`sbatch scripts/run_exp1.sh` submits to the cluster. One file per task,
no duplication.

### File formats in `data/` and `results/`

- `data/<run_name>/dataset.pkl` — Python pickle of a dict
  `{"tau": (n,), "x_tau": (n, d), "x_0": (n, d), "x_1": (n, d)}`.
  Pickle (rather than CSV) because these are 2-D float arrays; CSV would
  flatten and lose the shape semantics.
- `results/<run_name>/exp1_trace.csv` — two columns, `t,loss`.
- `results/<run_name>/exp2_trace.csv` — two columns, `t,w1`.
- `results/<run_name>/exp2_samples.pkl` — optional pickle of
  `{t_value: ndarray (n_train, d)}` so you can reconstruct the samples
  behind any `W_1` value if you need to.

---

## 5. Installation and usage

### Cluster install (one time)

```bash
module load anaconda3/2024.6                  # or whichever module is current
conda env create -f flowmatchnn.yml
conda activate flowmatchnn
pip install -e .
```

### Run locally

```bash
conda activate flowmatchnn
IDX=0 bash scripts/run_exp1_all.sh            # one cell only (data → exp1 → figure)
IDX=0 bash scripts/run_exp2_all.sh            # one cell only (data → exp2 → figure)
```

Without `IDX`, the local mode iterates the full 150-cell grid sequentially
in a single GPU allocation — fine for a tiny edited grid, way too slow for
the default sweep.

### Submit to SLURM (full grid)

The recommended one-shot driver is `run_all_exp.sh`: a single `sbatch`
fans the 150-cell grid out across 10 SLURM array tasks (each task = one
stride-10 shard) and runs **dataset → train → both vis passes** in
order:

```bash
sbatch scripts/run_all_exp.sh     # ~15 cells/shard, all metrics + figures
```

Internally `run_all_exp.sh` runs, in order:

1. `run_data.sh` — build `data/<cell>/dataset.pkl` per cell (cheap).
   Inline in the training array.
2. `run_exp1.sh` — train each cell once. `run_exp1.py` writes BOTH
   `results/<cell>/exp1_trace.csv` (loss) AND `exp2_trace.csv` (W₁)
   from the same trajectory, so we do not also call `run_exp2.sh`
   (that would re-train every cell from a different seed if you want
   a second trajectory). Inline in the training array.
3. `run_exp1_vis.sh` — per-setting `loss-vs-t` (10 d-curves) + global
   `loss-vs-d` (15 `(m, η)` curves). Submitted as a **separate** sbatch
   job with `--dependency=afterany` on the training array.
4. `run_exp2_vis.sh` — per-setting `W₁-vs-t` + global `W₁-vs-d`.
   Same dependency strategy as step 3.

Vis is split into a dependent job because each shard only computes its
own slice of `D_LIST`, but vis needs CSVs from *every* shard. Running
vis inline-per-shard (the previous design) raced: a shard that finished
training quickly rendered figures from CSVs that were still being
written by peer shards, producing per-setting plots with `ok=4/10`
d-curves and a global plot keyed off the fastest, most-incomplete
shard. With the dependency split, vis starts only after every training
shard has reported in.

For finer-grained workflows the individual scripts can still be
submitted on their own:

```bash
sbatch scripts/run_data.sh        # build dataset.pkl per cell (cheap)
sbatch scripts/run_exp1.sh        # train + write exp1/exp2 CSVs, 10 shards
sbatch scripts/run_exp2.sh        # (only needed if you want a separate trajectory)
sbatch scripts/run_exp1_vis.sh    # render figs/loss_vs_{t,d}__... .png
sbatch scripts/run_exp2_vis.sh    # render figs/w1_vs_{t,d}__... .png
```

All shell scripts forward extra CLI flags through to the underlying Python
script, e.g.

```bash
sbatch scripts/run_exp1.sh --T 500 --kappa 0.5
```

To change the shard count or concurrency, override `--array` at submit
time (always use contiguous `0-(N-1)` ranges so the stride math works):

```bash
sbatch --array=0-19      scripts/run_exp1.sh    # 20 shards (~7-8 cells each)
sbatch --array=0-9%5     scripts/run_exp1.sh    # 10 shards, cap concurrency at 5
sbatch --array=42        scripts/run_exp1.sh    # single cell (cell 42 only)
```

Job stdout/stderr go to `logs/`; the augmented dataset goes to
`data/<run_name>/dataset.pkl`; experiment traces go to
`results/<run_name>/{exp1_trace.csv, exp2_trace.csv}`; the figures
go to `figs/{loss,w1}_vs_t__<setting>.png` (per-setting, all-d
overlay) and `figs/{loss,w1}_vs_d__ntr<ntr>_nte<nte>.png` (global,
all-(m,η) overlay). The `(m, η, d, n_train, n_test)` grid itself is
defined in `scripts/_grid.sh` — edit it in one place.

---

## 6. Confirmed configuration (resolved)

| # | Question                                  | Resolution |
|---|-------------------------------------------|------------|
| 1 | Schedule convention                       | Paper's `σ(τ) = 1 − τ`, `μ(τ) = τ`. |
| 2 | Source `P(x_0 \| x_1)`                    | `N(x_1, I_d)` (paired with `x_1`). |
| 3 | Target `P_1`                              | Two-component GMM in `R^5`, centers `±2·e_1`, identity cov. |
| 4 | `(m, η, d, n_train, n_test)` and other defaults | 5-D grid search over `m ∈ {16, 32, 64, 128, 256}`, `η ∈ {1e-4, 1e-3, 1e-2}`, `d ∈ {5, 10, 15, …, 50}` (10 values), `n_train ∈ {500}`, `n_test ∈ {5000}` (= 150 cells). `n_train` is also reused as the model-side count for the `W_1` estimate. Fixed: `T = 500`, `B_1 = 10`, `T_euler = 200`, `eval_every = 10`, `n_proj = 128`. |
| 5 | SLURM partition / account / time          | `--partition=ailab`, `--account=jqfan`, `--time=24:00:00`, `--gres=gpu:1`, `--mem=16G`. Headers embedded in every script under `scripts/`. |
| 6 | CUDA version                              | `pytorch-cuda=12.1`, pinned in `flowmatchnn.yml`. Scripts hard-fail when CUDA is unavailable (no CPU fallback). |
