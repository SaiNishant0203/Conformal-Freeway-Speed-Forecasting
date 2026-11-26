"""Where does the error of the undercoverage formula come from?

Split conformal with one pad q covers regime g with probability F_g(q), where F_g is the
distribution of the absolute residual in that regime. Dewolf et al. (arXiv 2309.08313) show
that a score is conditionally valid exactly when it is pivotal, that is, when the residuals of
all groups share one shape up to scale. Under that assumption F_g(q) = F(q / b_g) for a common
shape F and regime scales b_g, and the closed form 1 - exp(-q / b_g) is the special case of
Laplace errors.

Four predictions of test coverage in each regime, each using calibration data only:
  laplace     1 - exp(-q / b_g), b_g the regime's mean absolute residual
  normal      half-normal with the regime's root mean squared residual
  common      F estimated from all calibration residuals after dividing by b_g (the pivotal
              assumption, with no parametric form)
  own         the regime's own calibration distribution F_g (no shape assumption at all)

The gap between measurement and prediction then splits into
  drift       measured - own          calibration and test periods differ
  shape       own - common            regimes do not share one residual shape
  parametric  common - laplace        the common shape is not Laplace
Also reported: the Kolmogorov-Smirnov distance between each regime's scaled residual
distribution and the pooled one, as a direct measure of how far the pivotal assumption is off.

Run: python code/formula_decomposition.py > results_audit/formula_decomposition.txt
"""
import json
import os
import numpy as np
from scipy.special import erf
from pairs import PAIRS, load, label, cq, ALPHA

os.makedirs("results_audit", exist_ok=True)
REG = ["free", "transition", "congested"]
out = []
for net, model, folder in PAIRS:
    d = load(net, model, folder)
    sc, st = np.abs(d["y_cal"] - d["yhat_cal"]), np.abs(d["y_test"] - d["yhat_test"])
    okc, okt, gc, gt = d["ok_cal"], d["ok_test"], d["g_cal"], d["g_test"]
    q = cq(sc[okc], 1 - ALPHA)
    b = np.array([sc[okc & (gc == g)].mean() for g in range(3)]); rms = np.array([np.sqrt((sc[okc & (gc == g)] ** 2).mean()) for g in range(3)])
    z_all = np.sort(np.concatenate([sc[okc & (gc == g)] / b[g] for g in range(3)]))
    print(f"\n==================== {label(net, model)}: pad {q:.2f} mph, regime mean abs residuals " + " / ".join(f"{v:.2f}" for v in b) + " ====================")
    print(f"{'regime':<11}{'measured':>9}{'laplace':>9}{'normal':>8}{'common':>8}{'own':>7} | {'drift':>7}{'shape':>7}{'param.':>7} | {'KS':>6}")
    for g in range(3):
        meas = float((st[okt & (gt == g)] <= q).mean())
        lap = float(1 - np.exp(-q / b[g])); nor = float(erf(q / (rms[g] * np.sqrt(2))))
        com = float(np.searchsorted(z_all, q / b[g], side="right") / len(z_all))
        zg = np.sort(sc[okc & (gc == g)] / b[g]); own = float(np.searchsorted(zg, q / b[g], side="right") / len(zg))
        grid = np.linspace(0, 6, 601)
        ks = float(np.max(np.abs(np.searchsorted(zg, grid, side="right") / len(zg) - np.searchsorted(z_all, grid, side="right") / len(z_all))))
        row = dict(net=net, model=model, regime=REG[g], measured=meas, laplace=lap, normal=nor, common=com, own=own,
                   drift=meas - own, shape=own - com, parametric=com - lap, ks=ks)
        out.append(row)
        print(f"{REG[g]:<11}{meas:9.3f}{lap:9.3f}{nor:8.3f}{com:8.3f}{own:7.3f} | {row['drift']:7.3f}{row['shape']:7.3f}{row['parametric']:7.3f} | {ks:6.3f}")
json.dump(out, open("results_audit/formula_decomposition.json", "w"), indent=1)
print("\n==================== mean absolute gap to measured coverage, by regime ====================")
print(f"{'regime':<11}{'laplace':>9}{'normal':>8}{'common':>8}{'own':>7} | mean |drift| |shape| |parametric|")
for r in REG:
    rows = [x for x in out if x["regime"] == r]
    m = lambda k: np.mean([abs(x["measured"] - x[k]) for x in rows])
    print(f"{r:<11}{m('laplace'):9.3f}{m('normal'):8.3f}{m('common'):8.3f}{m('own'):7.3f} | "
          f"{np.mean([abs(x['drift']) for x in rows]):11.3f}{np.mean([abs(x['shape']) for x in rows]):8.3f}{np.mean([abs(x['parametric']) for x in rows]):13.3f}")
