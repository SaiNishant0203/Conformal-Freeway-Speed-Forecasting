"""Sensitivity sweep for the per-sensor alignment test in alignment.py.
Varies breakdown threshold, bin count, min obs per bin, and restricts to sensors that
actually reach congestion (>= 200 test obs above 1.15 o_c) so o_c is well identified.
Run: python code/alignment_sensitivity.py > results_v2/alignment_sensitivity.txt
"""
import sys
import numpy as np
sys.path.insert(0, "code")
import alignment as A
from conformal import split_conformal_quantile
from paths import pred_file

for net in ["pems04", "pems08"]:
    d = dict(np.load(pred_file(net)))
    oc = d["o_c"]; N = len(oc)
    q = split_conformal_quantile(np.abs(d["y_cal"] - d["yhat_cal"]), 0.10)
    yt, pt, ot, st = d["y_test"], d["yhat_test"], d["occ_test"], d["sid_test"]
    ins = (np.abs(yt - pt) <= q).astype(float)
    congested_enough = np.bincount(st[ot > 1.15 * oc[st]], minlength=N) >= 200
    print(f"\n==== {net}: {congested_enough.sum()}/{N} sensors reach congestion (>=200 obs above 1.15 o_c)")
    print(f"{'thr':>5}{'bins':>5}{'minN':>5}{'subset':>8} | {'n':>4}{'CV raw':>8}{'CV o/oc':>8}{'ratio':>7}{'corr(raw,oc)':>13}")
    for thr in (0.6, 0.75, 0.8):
        for nb in (20, 40):
            for mn in (30, 100):
                A.THR, A.NB, A.MIN_N = thr, nb, mn
                res = {}
                for name, scale in [("raw", np.ones(N)), ("norm", oc)]:
                    x = ot / scale[st]; hi = np.percentile(x, 99.5)
                    res[name] = np.array([A.breakdown(x[st == s], ins[st == s], 0, hi) for s in range(N)])
                for sub_name, sub in [("all", np.ones(N, bool)), ("congd", congested_enough)]:
                    ok = sub & np.isfinite(res["raw"]) & np.isfinite(res["norm"])
                    r, nn = res["raw"][ok], res["norm"][ok]
                    cr, cn = r.std() / r.mean(), nn.std() / nn.mean()
                    print(f"{thr:5.2f}{nb:5d}{mn:5d}{sub_name:>8} | {ok.sum():4d}{cr:8.3f}{cn:8.3f}{cr/cn:7.2f}{np.corrcoef(r, oc[ok])[0,1]:13.3f}")
