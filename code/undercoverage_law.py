"""How large is the minority-state undercoverage, and can it be predicted in advance?

If absolute residuals in state g are roughly exponential with mean b_g (Laplace errors) and
the state has share pi_g, the single split-conformal pad q solves
    sum_g pi_g * (1 - exp(-q / b_g)) = 1 - alpha,
and coverage in state g is 1 - exp(-q / b_g). When free flow dominates, q is about
b_free * ln(1 / alpha), which gives the rule of thumb
    coverage_g = 1 - alpha ** (b_free / b_g).
Everything on the right-hand side is known from calibration data before any test point is
seen. This script compares the prediction with measured test coverage by state, by horizon,
and sensor by sensor.

Run: python code/undercoverage_law.py > results_v2/undercoverage_law.txt
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems

ALPHA = 0.10
FILES = [("STID      D4 h=12", "predictions_stid_gpu/predictions_stid_pems04_h12.npz"),
         ("STID      D8 h=12", "predictions_stid_gpu/predictions_stid_pems08_h12.npz"),
         ("retrained D4 h=12", "predictions_v2/predictions_gru_pems04_h12.npz"),
         ("retrained D8 h=12", "predictions_v2/predictions_gru_pems08_h12.npz"),
         ("original  D4 h=12", "predictions/predictions_gru_pems04_h12.npz"),
         ("original  D8 h=12", "predictions/predictions_gru_pems08_h12.npz"),
         ("original  D4 h=6 ", "predictions/predictions_gru_h6.npz"),
         ("original  D4 h=3 ", "predictions/predictions_gru_h3.npz")]

rows = []
print(f"{'forecaster':<20}{'state':<7}{'share':>7}{'MAE':>7}{'ratio':>7} | {'measured':>9}{'mixture':>9}{'rule':>7}")
for label, f in FILES:
    if not os.path.exists(f):
        continue
    d = np.load(f); oc = d["o_c"]
    gc = fd_regime_from_pems(d["occ_cal"], oc[d["sid_cal"]]); gt = fd_regime_from_pems(d["occ_test"], oc[d["sid_test"]])
    ac, at = np.abs(d["y_cal"] - d["yhat_cal"]), np.abs(d["y_test"] - d["yhat_test"])
    pi = np.array([(gc == g).mean() for g in range(3)]); b = np.array([ac[gc == g].mean() for g in range(3)])
    q = split_conformal_quantile(ac, ALPHA)
    ql = brentq(lambda x: (pi * (1 - np.exp(-x / b))).sum() - (1 - ALPHA), 0.01, 500)
    for g, name in enumerate(["free", "trans", "cong"]):
        meas, mix, rule = (at[gt == g] <= q).mean(), 1 - np.exp(-ql / b[g]), 1 - ALPHA ** (b[0] / b[g])
        rows.append((meas, mix, rule, g))
        print(f"{label:<20}{name:<7}{pi[g]:7.3f}{b[g]:7.2f}{b[g]/b[0]:7.2f} | {meas:9.3f}{mix:9.3f}{rule:7.3f}")
    # sensor by sensor, congested state: predict from the sensor's own calibration MAE
    N = len(oc); pred, obs = [], []
    for s in range(N):
        mc, mt = (d["sid_cal"] == s) & (gc == 2), (d["sid_test"] == s) & (gt == 2)
        if mc.sum() >= 200 and mt.sum() >= 200:
            pred.append(1 - np.exp(-q / ac[mc].mean())); obs.append((at[mt] <= q).mean())
    pred, obs = np.array(pred), np.array(obs)
    print(f"{'':<20}per-sensor congested coverage, {len(obs)} sensors: corr(predicted, measured) = "
          f"{np.corrcoef(pred, obs)[0, 1]:.2f}, mean abs gap = {np.abs(pred - obs).mean():.3f}\n")
r = np.array(rows); nf = r[:, 3] > 0
print(f"transition and congested rows ({nf.sum()}): mean abs gap  mixture {np.abs(r[nf,0]-r[nf,1]).mean():.3f}   rule {np.abs(r[nf,0]-r[nf,2]).mean():.3f}")
