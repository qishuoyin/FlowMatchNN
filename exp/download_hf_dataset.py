"""
Download a Hugging Face dataset and save it to data/application/<output_name>/.

Generic entry point. Dataset-specific shell scripts in scripts/ pass the
correct parameters for each dataset:

    scripts/download_mnist.sh      ylecun/mnist        image col: image

Key flag for cross-dataset consistency:
    --normalize_col <old>   Rename column <old> to 'image' in the saved
                            parquet. Use this whenever the source dataset
                            stores images under a different column name.
                            After normalising, every downstream script can
                            use --image_col image.

Usage:
    bash scripts/download_mnist.sh
    python exp/download_hf_dataset.py --dataset ylecun/mnist
    python exp/download_hf_dataset.py \\
        --dataset owner/name --normalize_col img --output_name mydata
"""

import argparse
import os
from pathlib import Path

# Redirect HF cache to scratch so it doesn't fill the home-dir quota.
# The shell script sets HF_HOME before invoking this; this fallback covers
# direct invocations (e.g. interactive debugging).
if "HF_HOME" not in os.environ:
    _repo_root = Path(__file__).resolve().parent.parent
    os.environ["HF_HOME"] = str(_repo_root / ".cache" / "huggingface")

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset to data/application/."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="HuggingFace dataset path, e.g. 'ylecun/mnist'.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Dataset config/subset name (passed as 'name' to load_dataset).",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Single split to download, e.g. 'train'. "
             "Downloads all splits when omitted.",
    )
    parser.add_argument(
        "--save_format", type=str, default="parquet",
        choices=["parquet", "json", "csv"],
        help="File format to save each split (default: parquet).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Full override of the output directory. "
             "Default: data/application/<output_name>.",
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Short name used for the output subdirectory under "
             "data/application/. Defaults to the last segment of --dataset "
             "(e.g. 'mnist' for 'ylecun/mnist').",
    )
    parser.add_argument(
        "--normalize_col", type=str, default=None,
        metavar="OLD_COL",
        help="Rename column OLD_COL to 'image' in every saved parquet. "
             "Use when the source dataset stores images under a non-standard "
             "name (e.g. CIFAR-10 uses 'img'). "
             "No-op when OLD_COL is not present in a split.",
    )
    return parser.parse_args()


def _dataset_short_name(dataset_id: str) -> str:
    """'owner/name' → 'name'; 'name' → 'name'."""
    return dataset_id.split("/")[-1]


def _save_split(split_ds, out_path: Path, save_format: str,
                normalize_col: str | None) -> None:
    """
    Optionally rename normalize_col → 'image', then write to disk.

    Renaming is done in-memory on the HuggingFace Dataset object before
    saving, so the parquet on disk always has a consistent 'image' column.
    """
    if normalize_col and normalize_col in split_ds.column_names:
        split_ds = split_ds.rename_column(normalize_col, "image")
        print(f"    renamed column '{normalize_col}' → 'image'")

    if save_format == "parquet":
        split_ds.to_parquet(str(out_path))
    elif save_format == "json":
        split_ds.to_json(str(out_path))
    elif save_format == "csv":
        split_ds.to_csv(str(out_path))


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_name = args.output_name or _dataset_short_name(args.dataset)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "data" / "application" / output_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.dataset}' → {out_dir}")
    if args.normalize_col:
        print(f"  column normalisation: '{args.normalize_col}' → 'image'")

    load_kwargs = dict(path=args.dataset)
    if args.name:
        load_kwargs["name"] = args.name
    if args.split:
        load_kwargs["split"] = args.split

    ds = load_dataset(**load_kwargs)

    # load_dataset returns a DatasetDict (no split arg) or Dataset (split arg).
    if hasattr(ds, "items"):
        splits = dict(ds)
    else:
        splits = {args.split or "data": ds}

    for split_name, split_ds in splits.items():
        out_path = out_dir / f"{split_name}.{args.save_format}"
        print(f"  Saving split '{split_name}' ({len(split_ds)} rows) → {out_path}")
        _save_split(split_ds, out_path, args.save_format, args.normalize_col)

    print("Done.")


if __name__ == "__main__":
    main()
