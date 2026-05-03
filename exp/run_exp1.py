"""
Exp 1: training-loss trajectory L(theta^(t)) vs GD iteration t.

Pipeline:
    1. Load (or build on-the-fly) the regression dataset under
       data/<run_name>/dataset.pkl.
    2. Initialise the two-layer ReLU network and run T full-batch GD
       steps on the velocity-matching MSE.
    3. Every eval_every iterations, record both
         - the training loss L(theta^(t))
         - the sliced Wasserstein-1 distance between fresh model
           samples and a fixed reference set drawn from P_1.
    4. Write
         results/<run_name>/exp1_trace.csv   (columns: t, loss)
         results/<run_name>/exp2_trace.csv   (columns: t, w1).

Both CSVs come out of the same training trajectory, so a single
invocation of either run_exp1 or run_exp2 fully populates the cell.
"""

from __future__ import annotations

import pickle

import torch

from flowmatchnn import data_generation, learning_and_sampling

from _common import (
    chdir_to_repo,
    parse_args,
    require_cuda,
    resolve_paths,
    write_config,
    write_csv,
)


def _load_or_make_dataset(args, paths, device):
    """
    Load dataset.pkl if it exists, otherwise build it and persist
    under data/<run_name>/.
    """
    if paths["dataset_pkl"].exists():
        with open(paths["dataset_pkl"], "rb") as f:
            np_payload = pickle.load(f)
        return {
            k: torch.from_numpy(v).to(device=device, dtype=torch.float32)
            for k, v in np_payload.items()
        }
    ds = data_generation.generate_dataset(
        n=args.n_train,
        d=args.d,
        seed=args.seed,
        device=device,
        target_center_norm=args.target_center_norm,
    )
    np_payload = {k: v.detach().cpu().numpy() for k, v in ds.items()}
    with open(paths["dataset_pkl"], "wb") as f:
        pickle.dump(np_payload, f)
    return ds


def main() -> None:
    """
    Train the network with periodic evals and write both trace CSVs.
    """
    args = parse_args()

    chdir_to_repo()
    device = require_cuda()
    paths = resolve_paths(args)

    dataset = _load_or_make_dataset(args, paths, device)

    cfg = learning_and_sampling.TrainConfig(
        n_train=args.n_train,
        d=args.d,
        m=args.m,
        kappa=args.kappa,
        T=args.T,
        eta=args.eta,
        B1=args.B1,
        T_euler=args.T_euler,
        n_test=args.n_test,
        eval_every=args.eval_every,
        seed=args.seed,
        target_center_norm=args.target_center_norm,
        n_proj=args.n_proj,
    )
    trace = learning_and_sampling.train_and_trace(
        cfg, dataset=dataset, device=device,
    )

    write_csv(
        paths["exp1_csv"], "t,loss",
        list(zip(trace["t"], trace["loss"])),
    )
    write_csv(
        paths["exp2_csv"], "t,w1",
        list(zip(trace["t"], trace["w1"])),
    )
    write_config(args, paths)
    print(f"[run_exp1] wrote {paths['exp1_csv']} and {paths['exp2_csv']}  "
          f"(m={args.m}, eta={args.eta}, T={args.T}, evals={len(trace['t'])})")


if __name__ == "__main__":
    main()
