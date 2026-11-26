"""Confidence intervals for the headline coverage numbers that respect dependence in time and
across sensors.

Test points are far from independent: errors at one sensor persist for many steps and sensors
move together. Whole test days are therefore resampled with replacement (2,000 draws), which
keeps everything that happens within a day together. This is a day-block bootstrap, in the
spirit of the block-based standard errors of Zhai, Cheng and Wu (arXiv 2609.33868). Intervals
are conditional on the calibration split, which is held fixed.

With only 12 to 13 test days on PeMS04 and PeMS08 the intervals there are rough; METR-LA has
24 test days and PEMS-BAY 36.

Run: python code/block_bootstrap.py > results_audit/block_bootstrap.txt
"""
import json
import os
import numpy as np
from pairs import PAIRS, load, label, winkler, cq, window_regime, ALPHA, H, DAY

os.makedirs("results_audit", exist_ok=True)
rng = np.random.default_rng(0); B = 2000
out = []
for net, model, folder in PAIRS:
    d = load(net, model, folder)
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    okc, okt, gc, gt = d["ok_cal"], d["ok_test"], d["g_cal"], d["g_test"]
    wt = window_regime(d["speed"], d["ca"], d["T"] - H)
    day = (d["t_test"] // DAY); day = day - day.min(); nd = int(day.max()) + 1
    sa = np.abs(yc - pc); s = np.maximum(d["qlo_cal"] - yc, yc - d["qhi_cal"])
    q = cq(sa[okc], 1 - ALPHA); qg = np.array([cq(sa[okc & (gc == g)], 1 - ALPHA) for g in range(3)])
    e = cq(s[okc], 1 - ALPHA); eg = np.array([cq(s[okc & (gc == g)], 1 - ALPHA) for g in range(3)])
    iv = {"split": (pt - q, pt + q), "mondrian": (pt - qg[gt], pt + qg[gt]),
          "cqr": (d["qlo_test"] - e, d["qhi_test"] + e), "cqr_regime": (d["qlo_test"] - eg[gt], d["qhi_test"] + eg[gt])}
    # per-day sufficient statistics: covered and total counts in the congested regime, in the
    # window-transition regime and overall, and Winkler sums
    stats = {}
    for name, (lo, hi) in iv.items():
        ins = (yt >= lo) & (yt <= hi); wk = winkler(yt, lo, hi)
        for key, m in (("cong", okt & (gt == 2)), ("trans", okt & (gt == 1)), ("wtrans", okt & (wt == 1)), ("all", okt)):
            stats[(name, key, "c")] = np.array([ins[day == k][m[day == k]].sum() for k in range(nd)], float)
            stats[(name, key, "n")] = np.array([m[day == k].sum() for k in range(nd)], float)
        stats[(name, "all", "w")] = np.array([wk[day == k][okt[day == k]].sum() for k in range(nd)], float)
    draws = rng.integers(0, nd, size=(B, nd))

    def ci(fn):
        v = np.array([fn(dr) for dr in draws]); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    cov = lambda name, key: (lambda dr: stats[(name, key, "c")][dr].sum() / max(stats[(name, key, "n")][dr].sum(), 1))
    wkf = lambda name: (lambda dr: stats[(name, "all", "w")][dr].sum() / stats[(name, "all", "n")][dr].sum())
    full = np.arange(nd)
    print(f"\n==================== {label(net, model)}: {nd} test days ====================")
    print(f"{'quantity':<52}{'estimate':>9}   95% interval")
    items = [("split conformal, congested coverage", cov("split", "cong")),
             ("Mondrian, congested coverage", cov("mondrian", "cong")),
             ("CQR, congested coverage", cov("cqr", "cong")),
             ("CQR per-regime, congested coverage", cov("cqr_regime", "cong")),
             ("CQR per-regime minus CQR, congested coverage", lambda dr: cov("cqr_regime", "cong")(dr) - cov("cqr", "cong")(dr)),
             ("CQR, transition coverage", cov("cqr", "trans")),
             ("CQR per-regime, transition coverage", cov("cqr_regime", "trans")),
             ("CQR per-regime, overall coverage", cov("cqr_regime", "all")),
             ("CQR per-regime, window-transition coverage (Wang)", cov("cqr_regime", "wtrans")),
             ("Winkler: Mondrian minus CQR per-regime", lambda dr: wkf("mondrian")(dr) - wkf("cqr_regime")(dr)),
             ("Winkler: split conformal minus CQR per-regime", lambda dr: wkf("split")(dr) - wkf("cqr_regime")(dr))]
    for name, fn in items:
        est = fn(full); lo, hi = ci(fn)
        out.append(dict(net=net, model=model, quantity=name, estimate=float(est), lo=lo, hi=hi, days=nd))
        print(f"{name:<52}{est:9.3f}   [{lo:.3f}, {hi:.3f}]", flush=True)
json.dump(out, open("results_audit/block_bootstrap.json", "w"), indent=1)
