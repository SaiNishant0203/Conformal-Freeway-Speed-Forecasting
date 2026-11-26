"""Per-sensor view of the residual-quantile conformal method (default configuration) against
the paper's FD Mondrian intervals and CQR on the GRU quantile model: how coverage varies
across sensors, and which sensors lose the most relative to CQR.

Run: python code/fd_informed_sensors.py pems08 >> results_v2/fd_informed_sensors.txt
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import quantile_file
from fd_informed import build, winkler, ALPHA

net = sys.argv[1]
d, Xc, Xt, N = build(net)
yc, pc, yt, pt, sid = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"], d["sid_test"]
rc = yc - pc
gfd = fd_regime_from_pems(d["occ_test"], d["o_c"][sid]); gfc = fd_regime_from_pems(d["occ_cal"], d["o_c"][d["sid_cal"]])
half = ((len(yc) // N) // 2) * N; A, B = slice(0, half), slice(half, None)
kw = dict(max_iter=300, learning_rate=0.1, random_state=0)
lo_m = HGB(loss="quantile", quantile=ALPHA / 2, **kw).fit(Xc[A], rc[A]); hi_m = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kw).fit(Xc[A], rc[A])
s = np.maximum(lo_m.predict(Xc[B]) - rc[B], rc[B] - hi_m.predict(Xc[B]))
eg = np.array([split_conformal_quantile(s[gfc[B] == g], ALPHA) for g in range(3)])
iv = {"new": (pt + lo_m.predict(Xt) - eg[gfd], pt + hi_m.predict(Xt) + eg[gfd])}
qg = np.array([split_conformal_quantile(np.abs(rc)[gfc == g], ALPHA) for g in range(3)])
iv["FD Mondrian"] = (pt - qg[gfd], pt + qg[gfd])
qd = np.load(quantile_file(net)); e = split_conformal_quantile(np.maximum(qd["qlo_cal"] - yc, yc - qd["qhi_cal"]), ALPHA)
iv["CQR"] = (qd["qlo_test"] - e, qd["qhi_test"] + e)
q = split_conformal_quantile(np.abs(rc), ALPHA); iv["split CP"] = (pt - q, pt + q)

print(f"\n==================== {net}: coverage across the {N} sensors (target 0.90) ====================")
print(f"{'method':<13}{'min':>7}{'p10':>7}{'median':>8}{'p90':>7}{'max':>7}{'share below 0.85':>18}")
cov, wk = {}, {}
for name, (lo, hi) in iv.items():
    ins = (yt >= lo) & (yt <= hi); w = winkler(yt, lo, hi)
    cov[name] = np.array([ins[sid == k].mean() for k in range(N)]); wk[name] = np.array([w[sid == k].mean() for k in range(N)])
    c = cov[name]
    print(f"{name:<13}{c.min():7.3f}{np.percentile(c, 10):7.3f}{np.median(c):8.3f}{np.percentile(c, 90):7.3f}{c.max():7.3f}{(c < 0.85).mean():18.3f}")
chg = 100 * (wk["CQR"] - wk["new"]) / wk["CQR"]
print("\nsensors where the new method loses most to CQR (negative = worse than CQR):")
print(f"{'sensor':>7}{'change %':>10}{'Wink new':>10}{'Wink CQR':>10}{'Wink FD':>9}{'cov new':>9}{'cov CQR':>9}{'MAE':>7}{'cong share':>12}{'o_c':>8}")
for k in np.argsort(chg)[:5]:
    m = sid == k
    print(f"{k:>7}{chg[k]:10.1f}{wk['new'][k]:10.2f}{wk['CQR'][k]:10.2f}{wk['FD Mondrian'][k]:9.2f}{cov['new'][k]:9.3f}{cov['CQR'][k]:9.3f}"
          f"{np.abs(yt - pt)[m].mean():7.2f}{(gfd[m] == 2).mean():12.3f}{d['o_c'][k]:8.4f}")
print(f"sensors worse than CQR: {(chg < 0).sum()} of {N}; worse by more than 10%: {(chg < -10).sum()}")
