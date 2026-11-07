"""
Exp 2 visualisation (real data): sliced W_1 vs t.

Reads results/<run_name>/exp2_trace.csv; writes to
figs/application/<dataset>/w1/<param_suffix>.png.

No model or GPU required.

Usage:
    python exp/exp3_real_data/run_real_w1_vis.py --run_name real_mnist_m128_eta1e-3_ntr500_nte2000
    sbatch scripts/exp3_real_data/run_real_w1_vis.sh   (iterates all grid cells)
"""

from __future__ import annotations

from pathlib import Path

# The shared helpers (_common.py, _vis_common.py, _real_vis_common.py) live
# one level up in exp/. Entry points are executed as plain files, so only
# this subfolder lands on sys.path; add exp/ explicitly before importing.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo
from _real_vis_common import (
    app_figs_subdir,
    load_run_config,
    param_suffix,
    parse_vis_args,
    plot_trace,
)


def main() -> None:
    args = parse_vis_args()
    chdir_to_repo()

    run_results_dir = Path(args.results_root) / args.run_name
    figs_dir        = Path(args.figs_dir)

    cfg     = load_run_config(run_results_dir)
    dataset = cfg.get("dataset", args.run_name)
    sfx     = param_suffix(args.run_name, dataset)

    w1_dir  = app_figs_subdir(figs_dir, dataset, "w1")
    w1_path = w1_dir / f"{sfx}.png"
    plot_trace(
        run_results_dir / "exp2_trace.csv", w1_path,
        ylabel=r"sliced $W_1$",
        title=f"$W_1$ vs $t$ — {dataset}  ({sfx})",
    )
    print(f"[run_real_w1_vis] wrote {w1_path}")


if __name__ == "__main__":
    main()
