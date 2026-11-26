"""How much conditional miscoverage does each method leave, measured without choosing groups?

Follows the idea of Braun, Holzmuller, Jordan and Bach ("Conditional coverage diagnostics for
conformal prediction", arXiv 2512.11779): if intervals had exactly 90% coverage for every
input, no classifier could predict from the input whether a point will be covered better than
the constant 0.90 does. The amount by which a classifier beats the constant, in squared error,
estimates the mean squared gap between the true conditional coverage and 0.90 (it can only
underestimate it). The square root is reported, in coverage units: 0.05 means the conditional
coverage is typically about 5 points away from the target.

The classifier is gradient boosting on what is known at forecast time: origin speed, forecast,
interval width, speed change over 15 minutes, time of day, weekend, and two sensor descriptors
(mean training speed and mean calibration error). It is trained on even test days and scored on
odd test days, on at most 400,000 points each.

Run: python code/ert_diagnostic.py > results_audit/ert_diagnostic.txt
"""
import json
import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from pairs import PAIRS, load, label, cq, ALPHA, DAY

os.makedirs("results_audit", exist_ok=True)
out = []; rng = np.random.default_rng(0)
for net, model, folder in PAIRS:
    d = load(net, model, folder); N = d["N"]
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    okc, okt, gc, gt = d["ok_cal"], d["ok_test"], d["g_cal"], d["g_test"]
    sa = np.abs(yc - pc); s = np.maximum(d["qlo_cal"] - yc, yc - d["qhi_cal"])
    q = cq(sa[okc], 1 - ALPHA); qg = np.array([cq(sa[okc & (gc == g)], 1 - ALPHA) for g in range(3)])
    e = cq(s[okc], 1 - ALPHA); eg = np.array([cq(s[okc & (gc == g)], 1 - ALPHA) for g in range(3)])
    es = eg[gt].astype(float).copy()
    for g in range(3):
        m = okc & (gc == g)
        for k in range(N):
            v = s[m[:, k], k]
            if len(v) >= 300:
                es[gt[:, k] == g, k] = cq(v, 1 - ALPHA)
    iv = {"split conformal": (pt - q, pt + q), "Mondrian on regime": (pt - qg[gt], pt + qg[gt]),
          "CQR": (d["qlo_test"] - e, d["qhi_test"] + e), "CQR per regime": (d["qlo_test"] - eg[gt], d["qhi_test"] + eg[gt]),
          "CQR per sensor x regime": (d["qlo_test"] - es, d["qhi_test"] + es)}
    with np.errstate(all="ignore"):
        ms = np.nan_to_num(np.nanmean(np.where(d["speed"][:d["tr"]] > 0, d["speed"][:d["tr"]], np.nan), 0))
        me = np.nan_to_num(np.nanmean(np.where(okc, sa, np.nan), 0))
    t = d["t_test"]; ang = (d["tod"][t] * 2 * np.pi / DAY)[:, None] * np.ones((1, N))
    base = [d["spd_test"], pt, d["spd_test"] - d["speed"][t - 3], np.sin(ang), np.cos(ang),
            (d["dow"][t] >= 5)[:, None] * np.ones((1, N)), np.tile(ms, (len(t), 1)), np.tile(me, (len(t), 1))]
    day = ((t // DAY) % 2)[:, None] * np.ones((1, N), int)
    tr_idx = np.flatnonzero((okt & (day == 0)).ravel()); ev_idx = np.flatnonzero((okt & (day == 1)).ravel())
    tr_idx = rng.choice(tr_idx, min(400000, len(tr_idx)), replace=False); ev_idx = rng.choice(ev_idx, min(400000, len(ev_idx)), replace=False)
    print(f"\n==================== {label(net, model)} ====================")
    print(f"{'method':<26}{'coverage':>9}{'typical conditional gap':>25}")
    for name, (lo, hi) in iv.items():
        X = np.stack([a.ravel() for a in base + [hi - lo]], 1).astype(np.float32)
        c = ((yt >= lo) & (yt <= hi)).ravel().astype(int)
        clf = HGBC(max_iter=150, learning_rate=0.1, random_state=0).fit(X[tr_idx], c[tr_idx])
        p = clf.predict_proba(X[ev_idx])[:, 1]; ce = c[ev_idx]
        gain = float(np.mean((1 - ALPHA - ce) ** 2) - np.mean((p - ce) ** 2))
        gap = float(np.sqrt(max(gain, 0.0)))
        out.append(dict(net=net, model=model, method=name, coverage=float(ce.mean()), gap=gap))
        print(f"{name:<26}{ce.mean():9.3f}{gap:25.3f}", flush=True)
json.dump(out, open("results_audit/ert_diagnostic.json", "w"), indent=1)
