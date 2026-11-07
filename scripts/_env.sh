#!/usr/bin/env bash
# Activate the `flowmatchnn` conda env so that `python` resolves to the
# interpreter that has the editable `flowmatchnn` package installed.
#
# Why this file exists: under `sbatch`, the shell is non-interactive and
# does NOT source ~/.bashrc, so the conda init that normally puts the
# `conda activate flowmatchnn` env on PATH is missing. Bare `python`
# then resolves to the system anaconda3/2024.6 base, which does not
# have `flowmatchnn` -> ModuleNotFoundError on `from flowmatchnn import ...`.
#
# Sourced (not executed) from every leaf script under scripts/.

# Idempotent fast path: nothing to do if `flowmatchnn` is already importable.
# Covers interactive shells where the user already ran `conda activate
# flowmatchnn` before invoking the script directly.
# Run the probe from / — `python -c` puts the cwd on sys.path, so probing
# from the repo root would find the flowmatchnn/ source dir and pass even
# under a python that does not have the package installed.
if (cd / && python -c "import flowmatchnn") >/dev/null 2>&1; then
  return 0
fi

# `conda activate` reads PS1 and other unset vars; the leaf scripts run
# with `set -u`, so disable it for the duration of activation.
set +u

CONDA_SH=/usr/licensed/anaconda3/2024.6/etc/profile.d/conda.sh
ENV_PREFIX=/scratch/gpfs/JQFAN/qy1448/conda/envs/flowmatchnn

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_PREFIX"
elif [[ -x "$ENV_PREFIX/bin/python" ]]; then
  # Fallback when the system conda init script is unavailable: prepend
  # the env's bin/ to PATH. Sufficient for the editable install (.pth
  # files in site-packages get picked up automatically), but misses any
  # conda-managed env vars (e.g. LD_LIBRARY_PATH for cuDNN bundled in
  # the env). Acceptable because pytorch wheels ship their CUDA libs.
  export PATH="$ENV_PREFIX/bin:$PATH"
else
  echo "[_env.sh] ERROR: cannot find $CONDA_SH or $ENV_PREFIX/bin/python" >&2
  echo "[_env.sh] create the env first: conda env create -f flowmatchnn.yml && pip install -e ." >&2
  exit 1
fi

set -u
