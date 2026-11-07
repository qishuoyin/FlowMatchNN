"""
Exp 1 (real data): training-loss trajectory L(theta^(t)) vs GD iteration t.

Pipeline:
    1. Load train and test splits from data/application/<dataset>/*.parquet.
       Each parquet must have an 'image' column whose rows are either
         - a dict  {'bytes': <PNG/JPEG bytes>}   (HuggingFace format), or
         - a numpy array of shape (H, W) or (H, W, C).
    2. Flatten images, normalise to [0, 1].
    3. Fit PCA on the training images; retain n_pca components
       (default: d_flat // 100 — reduces each spatial side by ~10x).
    4. Project train + test images to PCA space.
    5. Build the flow-matching regression dataset from train PCA codes:
         P(x_0 | x_1) = N(x_1, I_{d_pca})    (same coupling as simulation)
         x_tau = (1-tau)*x_0 + tau*x_1
    6. Run train_and_trace_real, which records both
         - the training loss L(theta^(t))            → exp1_trace.csv
         - the sliced W_1 distance to the test set   → exp2_trace.csv
       from the same training trajectory, so a single invocation of
       run_real_train fully populates both CSVs.
    7. Write:
         results/<run_name>/exp1_trace.csv   (t, loss)
         results/<run_name>/exp2_trace.csv   (t, w1)
         results/<run_name>/config.json      (all resolved args + img_h/w)
         data/<run_name>/dataset.pkl         (tau, x_tau, x_0, x_1 in PCA)
         data/<run_name>/pca_model.pkl       (PCAModel: components + mean)
         data/<run_name>/model.pt            (trained TwoLayerReLU weights)

Run-name convention (stable across PCA choices, suitable for grid sweeps):
    real_{dataset}_m{m}_eta{eta}_ntr{n_train}_nte{n_test}

Both exp1_trace.csv (loss) and exp2_trace.csv (W_1) come from the same
training run, so a single run_real_train.sh submission populates both.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from flowmatchnn import learning_and_sampling
from flowmatchnn.utils import data as data_utils
from flowmatchnn.utils import pca as pca_utils

# The shared helpers (_common.py, _vis_common.py, _real_vis_common.py) live
# one level up in exp/. Entry points are executed as plain files, so only
# this subfolder lands on sys.path; add exp/ explicitly before importing.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo, require_cuda, write_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Flow-matching Exp 1 (loss vs t) on a real image dataset."
    )
    # ---- dataset ----
    p.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing <train_split>.parquet and <test_split>.parquet. "
             "Defaults to data/application/<dataset>.",
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name used in run_name and data_dir resolution. "
             "Inferred from the basename of --data_dir if omitted.",
    )
    p.add_argument(
        "--train_split", type=str, default="train",
        help="Name of the training parquet split (default: 'train').",
    )
    p.add_argument(
        "--test_split", type=str, default="test",
        help="Name of the held-out parquet split used for W_1 reference "
             "(default: 'test').",
    )
    p.add_argument(
        "--image_col", type=str, default="image",
        help="Parquet column that holds the image data (default: 'image').",
    )
    # ---- PCA ----
    p.add_argument(
        "--n_pca", type=int, default=None,
        help="PCA components to keep. Default: d_flat // 100.",
    )
    # ---- dataset size ----
    p.add_argument("--n_train", type=int, default=500)
    p.add_argument("--n_test",  type=int, default=2000)
    # ---- model / training ----
    p.add_argument("--m",          type=int,   default=128)
    p.add_argument("--kappa",      type=float, default=1.0)
    p.add_argument("--T",          type=int,   default=500)
    p.add_argument("--eta",        type=float, default=1e-3)
    p.add_argument("--T_euler",    type=int,   default=200)
    p.add_argument("--B1",         type=float, default=10.0)
    p.add_argument("--eval_every", type=int,   default=10)
    p.add_argument("--n_proj",     type=int,   default=128)
    p.add_argument("--seed",       type=int,   default=0)
    # ---- IO ----
    p.add_argument(
        "--run_name", type=str, default=None,
        help="Subdir under data/ and results/. "
             "Default: real_{dataset}_m{m}_eta{eta}_ntr{n_train}_nte{n_test}.",
    )
    p.add_argument("--data_root",    type=str, default="data")
    p.add_argument("--results_root", type=str, default="results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _decode_row(row) -> np.ndarray:
    """
    Decode one value from the image column.

    Handles:
      - dict {'bytes': <bytes>}       — HuggingFace PNG/JPEG
      - numpy array (H, W) or (H,W,C)
      - raw bytes
    """
    if isinstance(row, dict) and "bytes" in row:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(row["bytes"])).convert("L"),
                        dtype=np.float32)
    if isinstance(row, np.ndarray):
        return row.astype(np.float32)
    if isinstance(row, (bytes, bytearray)):
        from PIL import Image
        return np.array(Image.open(io.BytesIO(row)).convert("L"),
                        dtype=np.float32)
    raise TypeError(f"Unrecognised image format: {type(row)}")


def load_split(
    parquet_path: Path, image_col: str
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load a parquet split → (X_flat, (img_h, img_w)).

    X_flat has shape (N, H*W) with pixel values in [0, 1].
    img_h, img_w are auto-detected from the first row.
    """
    import pandas as pd
    df = pd.read_parquet(str(parquet_path))
    first = _decode_row(df[image_col].iloc[0])
    if first.ndim == 1:
        raise ValueError(
            f"Image in {parquet_path} is already 1-D; "
            "cannot auto-detect (img_h, img_w). Check the dataset format."
        )
    if first.ndim == 3:
        first = first.mean(axis=-1)
    img_h, img_w = first.shape

    imgs = []
    for row in df[image_col]:
        arr = _decode_row(row)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        imgs.append(arr.ravel())

    return (np.stack(imgs, axis=0) / 255.0).astype(np.float32), (img_h, img_w)


# ---------------------------------------------------------------------------
# Dataset construction in PCA space
# ---------------------------------------------------------------------------


def _build_pca_dataset(x_1_np: np.ndarray, *, seed: int,
                        device: torch.device) -> dict:
    """(tau, x_tau, x_0, x_1) from PCA-embedded target images."""
    x_1 = torch.from_numpy(x_1_np).to(device=device, dtype=torch.float32)
    g = torch.Generator(device=device).manual_seed(seed)
    tau, x_tau, x_0 = data_utils.make_dataset(x_1, generator=g,
                                               dtype=torch.float32)
    return {"tau": tau, "x_tau": x_tau, "x_0": x_0, "x_1": x_1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    device = require_cuda()

    if args.data_dir is None and args.dataset is None:
        raise SystemExit(
            "[run_real_train] supply --dataset <name> or --data_dir <path>."
        )
    if args.data_dir is None:
        args.data_dir = f"data/application/{args.dataset}"
    data_dir = Path(args.data_dir)
    dataset_name = args.dataset or data_dir.name

    # ---- load images ----
    train_path = data_dir / f"{args.train_split}.parquet"
    test_path  = data_dir / f"{args.test_split}.parquet"
    print(f"[run_real_train] loading {train_path} …")
    X_train_full, (img_h, img_w) = load_split(train_path, args.image_col)
    print(f"[run_real_train] loading {test_path} …")
    X_test_full, _ = load_split(test_path, args.image_col)

    n_train_avail, d_flat = X_train_full.shape
    n_test_avail = X_test_full.shape[0]
    print(f"[run_real_train] dataset={dataset_name}  img={img_h}x{img_w}  "
          f"d_flat={d_flat}  train={n_train_avail}  test={n_test_avail}")

    # ---- PCA ----
    n_pca = args.n_pca if args.n_pca is not None else max(1, d_flat // 100)
    print(f"[run_real_train] n_pca={n_pca} ({100*n_pca/d_flat:.1f}% of dims)")
    pca_model = pca_utils.fit(X_train_full, n_pca)
    total_var = float((X_train_full - X_train_full.mean(0)).var(0).sum()) + 1e-12
    print(f"[run_real_train] PCA explains "
          f"{100*float(pca_model.explained_var.sum())/total_var:.1f}% of variance")

    Z_train_full = pca_utils.transform(X_train_full, pca_model)
    Z_test_full  = pca_utils.transform(X_test_full,  pca_model)

    rng = np.random.default_rng(args.seed)
    n_train = min(args.n_train, n_train_avail)
    n_test  = min(args.n_test,  n_test_avail)
    Z_train = Z_train_full[rng.choice(n_train_avail, size=n_train, replace=False)]
    Z_test  = Z_test_full[ rng.choice(n_test_avail,  size=n_test,  replace=False)]

    dataset  = _build_pca_dataset(Z_train, seed=args.seed, device=device)
    x_1_ref  = torch.from_numpy(Z_test).to(device=device, dtype=torch.float32)

    # ---- output dirs ----
    run_name = args.run_name or (
        f"real_{dataset_name}_m{args.m}_eta{args.eta:g}"
        f"_ntr{n_train}_nte{n_test}"
    )
    out_data    = Path(args.data_root)    / run_name
    out_results = Path(args.results_root) / run_name
    out_data.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    np_payload = {k: v.detach().cpu().numpy() for k, v in dataset.items()}
    with open(out_data / "dataset.pkl",   "wb") as f: pickle.dump(np_payload, f)
    with open(out_data / "pca_model.pkl", "wb") as f: pickle.dump(pca_model, f)

    # ---- train ----
    cfg = learning_and_sampling.TrainConfig(
        n_train=n_train, d=n_pca, m=args.m, kappa=args.kappa,
        T=args.T, eta=args.eta, B1=args.B1, T_euler=args.T_euler,
        n_test=n_test, eval_every=args.eval_every, seed=args.seed,
        target_center_norm=2.0, n_proj=args.n_proj,
    )
    print(f"[run_real_train] training  run={run_name}  T={args.T} …")
    trace = learning_and_sampling.train_and_trace_real(
        cfg, dataset=dataset, x_1_ref=x_1_ref, device=device,
    )

    # ---- write outputs ----
    write_csv(out_results / "exp1_trace.csv", "t,loss",
              list(zip(trace["t"], trace["loss"])))
    write_csv(out_results / "exp2_trace.csv", "t,w1",
              list(zip(trace["t"], trace["w1"])))
    torch.save(trace["model"].state_dict(), out_data / "model.pt")

    config = {
        **vars(args),
        "dataset": dataset_name, "n_pca": n_pca,
        "d_flat": int(d_flat), "img_h": int(img_h), "img_w": int(img_w),
        "n_train_actual": int(n_train), "n_test_actual": int(n_test),
        "run_name": run_name,
    }
    with open(out_results / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"[run_real_train] wrote results/{run_name}/  "
          f"evals={len(trace['t'])}  "
          f"final_loss={trace['loss'][-1]:.4g}  "
          f"final_w1={trace['w1'][-1]:.4g}")


if __name__ == "__main__":
    main()
