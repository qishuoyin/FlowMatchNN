"""
Exp 2 (real data): W_1(model samples, P_1) trajectory vs GD iteration t.

Runs the exact same training-and-eval loop as run_real_exp1 (same seed,
same hyperparameters) and writes both CSVs. Provided as its own entry
point only so the per-experiment naming convention run_real_exp{1,2}.py
is preserved, mirroring the simulation scripts run_exp{1,2}.py; either
script alone fully populates results/<run_name>/.

For details of the training/eval pipeline see run_real_exp1.py.
"""

from run_real_exp1 import main

if __name__ == "__main__":
    main()
