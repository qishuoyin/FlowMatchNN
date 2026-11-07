"""
Build the regression dataset and persist it to disk.

For each i in 1..n_train: draw an interpolation time tau_i, a target
sample x_{1,i}, a source sample x_{0,i} ~ P(x_0 | x_{1,i}), and form
x_{tau,i} = sigma(tau_i) * x_{0,i} + mu(tau_i) * x_{1,i}. The result
is pickled to data/<run_name>/dataset.pkl.

The dataset depends only on (n_train, d, seed, target_center_norm); it
is independent of (m, eta). When sweeping the (m, eta) grid you can
either regenerate per cell (cheap at n=100) or pass a shared --run_name
to share one pickle across the whole sweep.
"""

from __future__ import annotations

import pickle

from flowmatchnn import data_generation

# The shared helpers (_common.py, _vis_common.py, _real_vis_common.py) live
# one level up in exp/. Entry points are executed as plain files, so only
# this subfolder lands on sys.path; add exp/ explicitly before importing.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (
    chdir_to_repo,
    parse_args,
    require_cuda,
    resolve_paths,
    write_config,
)


def main() -> None:
    """
    Generate the dataset and write it to data/<run_name>/dataset.pkl.
    """
    args = parse_args()

    chdir_to_repo()
    device = require_cuda()
    paths = resolve_paths(args)

    ds = data_generation.generate_dataset(
        n=args.n_train,
        d=args.d,
        seed=args.seed,
        device=device,
        target_center_norm=args.target_center_norm,
    )
    payload = {k: v.detach().cpu().numpy() for k, v in ds.items()}

    with open(paths["dataset_pkl"], "wb") as f:
        pickle.dump(payload, f)

    write_config(args, paths)
    print(f"[run_data] wrote {paths['dataset_pkl']}  "
          f"(n={args.n_train}, d={args.d}, seed={args.seed})")


if __name__ == "__main__":
    main()
