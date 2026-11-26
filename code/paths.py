"""Where the scripts read predictions and write figures.

Defaults point at the output of train_gru.py. Environment variables switch to another set,
for example the STID forecaster:

  PRED_DIR=predictions_stid MODEL=stid FIG_DIR=figures_stid python code/baselines.py
"""
import os

PRED_DIR = os.environ.get("PRED_DIR", "predictions_v2")
FIG_DIR = os.environ.get("FIG_DIR", "figures_v2")
MODEL = os.environ.get("MODEL", "gru")
RES_DIR = os.environ.get("RES_DIR", "results_v2")


def pred_file(net):
    return f"{PRED_DIR}/predictions_{MODEL}_{net}_h12.npz"


def quantile_file(net):
    """Quantile forecasts for a network, or None if there are none."""
    for f in (f"{PRED_DIR}/quantile_{MODEL}_{net}_h12.npz",
              f"{PRED_DIR}/quantile_gru_h12.npz" if net == "pems04" and MODEL == "gru" else None):
        if f and os.path.exists(f):
            return f
    return None
