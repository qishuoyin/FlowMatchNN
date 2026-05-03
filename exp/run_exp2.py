"""
Exp 2: W_1(model samples, P_1) trajectory vs GD iteration t.

Runs the exact same training-and-eval loop as run_exp1 (same seed,
same hyperparameters) and writes both CSVs. Provided as its own entry
point only so the per-experiment naming convention `run_exp{1,2}.py`
is preserved; either script alone fully populates results/<run_name>/.

For details of the training/eval pipeline see the docstring at the
top of run_exp1.py.
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
    print(f"[run_exp2] wrote {paths['exp1_csv']} and {paths['exp2_csv']}  "
          f"(m={args.m}, eta={args.eta}, T={args.T}, evals={len(trace['t'])})")


if __name__ == "__main__":
    main()
