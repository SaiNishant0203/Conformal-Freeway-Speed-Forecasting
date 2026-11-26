"""Regime-wise coverage on a dataset that has speed only (METR-LA), and on PeMS04/08 with the
same speed regimes for comparison.

Regimes follow the fixed thresholds used by Wang et al. (arXiv 2606.18367), applied here to
the speed at the forecast origin: free flow at 55 mph or more, congested below 25 mph,
transition in between. No flow or occupancy is used anywhere in this script, so it says
nothing about the fundamental diagram; it tests the parts of the method that do not need it.

Methods (alpha = 0.10, 60-minute horizon):
  split CP        one pad
  Mondrian        one pad per speed regime
  normalized CP   gradient boosting predicts |residual| from origin speed and forecast
  CQR             conformalized quantile regression on the forecaster's own 5%/95% quantiles
  climatology     5% to 95% range of future speed in each regime (ignores the forecast)
  rq              residual-quantile conformal: gradient boosting predicts the 5% and 95%
                  quantiles of y - yhat from speed-only features, then a CQR correction
  rq + Mondrian   the same with the correction computed per speed regime
  refined CQR     second-stage shift of the CQR interval ends (speed-only features plus the
                  interval width), then conformalized; also with a per-regime correction
  CQR + Mondrian  plain CQR with the correction computed per speed regime

Second-stage models are fit on the first half of the calibration period and conformalized on
the second half. Rows with a missing target or a missing origin speed (recorded as 0) are
dropped from calibration and evaluation.

Run: MODEL=stid PRED_DIR=predictions_stid python code/speed_regime_study.py metr_la
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import brentq
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.path.insert(0, "code")
from conformal import split_conformal_quantile
from paths import pred_file, quantile_file, MODEL, RES_DIR
from train_stid import load_raw

ALPHA, H, L, DAY = 0.10, 12, 12, 288
NAMES = ["free", "trans", "cong"]


def winkler(y, lo, hi):
    return (hi - lo) + (2 / ALPHA) * np.clip(lo - y, 0, None) + (2 / ALPHA) * np.clip(y - hi, 0, None)


def regime(spd):
    g = np.ones(len(spd), dtype=int); g[spd >= 55] = 0; g[spd < 25] = 2
    return g


def features(speed, tod, dow, t0, t1, yhat, mean_spd):
    t = np.arange(t0, t1); N = speed.shape[1]
    hist = np.stack([speed[t - k] for k in range(L)], 0)
    f = lambda a: np.asarray(a, dtype=np.float32).reshape(-1)
    ang = (tod[t0:t1][:, None] * np.ones((1, N))) * 2 * np.pi / DAY
    wk = (dow[t0:t1] >= 5)[:, None] * np.ones((1, N))
    hmask = np.where(hist > 0, hist, np.nan)
    with np.errstate(all="ignore"):
        hmin, hstd = np.nanmin(hmask, 0), np.nanstd(hmask, 0)
    return np.stack([f(speed[t]), yhat, f(speed[t] - speed[t - 3]), f(speed[t] - speed[t - 11]), f(np.nan_to_num(hmin)),
                     f(np.nan_to_num(hstd)), f(np.sin(ang)), f(np.cos(ang)), f(wk), f(np.tile(mean_spd, (len(t), 1))),
                     f((hist == 0).sum(0))], 1)


def main(net):
    d = dict(np.load(pred_file(net)))
    speed, _, _, _, tod, dow = load_raw(net)
    T, N = speed.shape; tr, ca = int(0.6 * T), int(0.8 * T)
    with np.errstate(all="ignore"):
        mean_spd = np.nan_to_num(np.nanmean(np.where(speed[:tr] > 0, speed[:tr], np.nan), 0))
    Xc = features(speed, tod, dow, tr, ca - H, d["yhat_cal"], mean_spd)
    Xt = features(speed, tod, dow, ca, T - H, d["yhat_test"], mean_spd)
    assert np.array_equal(Xc[:, 0], d["spd_cal"])
    half = ((len(d["y_cal"]) // N) // 2) * N
    part = np.arange(len(d["y_cal"])) < half
    vc = (d["y_cal"] > 0) & (d["spd_cal"] > 0); vt = (d["y_test"] > 0) & (d["spd_test"] > 0)
    A, B = vc & part, vc & ~part
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"][vt], d["yhat_test"][vt]
    rc = yc - pc; gc = regime(d["spd_cal"]); gt = regime(d["spd_test"][vt]); Xt = Xt[vt]
    rows = []

    def add(name, lo, hi):
        ins, wk = (yt >= lo) & (yt <= hi), winkler(yt, lo, hi)
        rows.append((name, ins.mean(), (hi - lo).mean(), wk.mean(), [ins[gt == g].mean() for g in range(3)],
                     [wk[gt == g].mean() for g in range(3)]))

    sc = np.abs(rc)
    q = split_conformal_quantile(sc[vc], ALPHA); add("split CP", pt - q, pt + q)
    qg = np.array([split_conformal_quantile(sc[vc & (gc == g)], ALPHA) for g in range(3)]); add("Mondrian", pt - qg[gt], pt + qg[gt])
    m = HGB(max_iter=200, random_state=0).fit(Xc[A][:, :2], sc[A])
    sB, sT = np.maximum(m.predict(Xc[B][:, :2]), 0.5), np.maximum(m.predict(Xt[:, :2]), 0.5)
    qn = split_conformal_quantile(sc[B] / sB, ALPHA); add("normalized CP", pt - qn * sT, pt + qn * sT)
    qf = quantile_file(net)
    if qf:
        qd = np.load(qf)
        e = split_conformal_quantile(np.maximum(qd["qlo_cal"] - yc, yc - qd["qhi_cal"])[vc], ALPHA)
        add("CQR", qd["qlo_test"][vt] - e, qd["qhi_test"][vt] + e)
        # second-stage refinement of the CQR interval with speed-only features plus its width
        Fc = np.hstack([Xc, (qd["qhi_cal"] - qd["qlo_cal"])[:, None]]); Ft = np.hstack([Xt, (qd["qhi_test"] - qd["qlo_test"])[vt][:, None]])
        kq = dict(max_iter=300, learning_rate=0.1, random_state=0)
        gl = HGB(loss="quantile", quantile=ALPHA / 2, **kq).fit(Fc[A], (yc - qd["qlo_cal"])[A])
        gh = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kq).fit(Fc[A], (yc - qd["qhi_cal"])[A])
        l2c, h2c = qd["qlo_cal"][B] + gl.predict(Fc[B]), qd["qhi_cal"][B] + gh.predict(Fc[B])
        l2t, h2t = qd["qlo_test"][vt] + gl.predict(Ft), qd["qhi_test"][vt] + gh.predict(Ft)
        s3 = np.maximum(l2c - yc[B], yc[B] - h2c); e3 = split_conformal_quantile(s3, ALPHA)
        add("refined CQR", l2t - e3, h2t + e3)
        eg3 = np.array([split_conformal_quantile(s3[gc[B] == g], ALPHA) for g in range(3)])
        add("refined CQR+Mond.", l2t - eg3[gt], h2t + eg3[gt])
        eg0 = np.array([split_conformal_quantile(np.maximum(qd["qlo_cal"] - yc, yc - qd["qhi_cal"])[vc & (gc == g)], ALPHA) for g in range(3)])
        add("CQR + Mondrian", qd["qlo_test"][vt] - eg0[gt], qd["qhi_test"][vt] + eg0[gt])
    lo = np.array([np.quantile(yc[vc & (gc == g)], ALPHA / 2) for g in range(3)])
    hi = np.array([np.quantile(yc[vc & (gc == g)], 1 - ALPHA / 2) for g in range(3)]); add("climatology", lo[gt], hi[gt])
    kw = dict(max_iter=300, learning_rate=0.1, random_state=0)
    lo_m = HGB(loss="quantile", quantile=ALPHA / 2, **kw).fit(Xc[A], rc[A])
    hi_m = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kw).fit(Xc[A], rc[A])
    s = np.maximum(lo_m.predict(Xc[B]) - rc[B], rc[B] - hi_m.predict(Xc[B])); lt, ht = lo_m.predict(Xt), hi_m.predict(Xt)
    e = split_conformal_quantile(s, ALPHA); add("rq", pt + lt - e, pt + ht + e)
    eg = np.array([split_conformal_quantile(s[gc[B] == g], ALPHA) for g in range(3)]); add("rq + Mondrian", pt + lt - eg[gt], pt + ht + eg[gt])

    at = np.abs(yt - pt)
    print(f"\n==================== {net}, forecaster {MODEL}: {vt.sum():,} valid test points of {len(vt):,} ====================")
    print("regime shares (test) free/trans/cong:", np.round([(gt == g).mean() for g in range(3)], 3),
          "  MAE by regime:", np.round([at[gt == g].mean() for g in range(3)], 2), f"  overall MAE {at.mean():.2f}")
    hdr = f"{'method':<18}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | {'Wk F':>6}{'Wk T':>6}{'Wk C':>6}"
    print(hdr); print("-" * len(hdr))
    for name, c, w, k, cs, ks in rows:
        print(f"{name:<18}{c:6.3f}{w:7.2f}{k:7.2f} | " + "".join(f"{v:6.3f}" for v in cs) + " | " + "".join(f"{v:6.1f}" for v in ks))
    # undercoverage formula from calibration data only
    pi = np.array([(gc[vc] == g).mean() for g in range(3)]); b = np.array([sc[vc & (gc == g)].mean() for g in range(3)])
    ql = brentq(lambda x: (pi * (1 - np.exp(-x / b))).sum() - (1 - ALPHA), 0.01, 500)
    os.makedirs(RES_DIR, exist_ok=True)
    with open(f"{RES_DIR}/speed_regimes_{net}_{MODEL}.json", "w") as fh:
        json.dump(dict(net=net, model=MODEL, n_test=int(vt.sum()), mae=float(at.mean()),
                       shares=[float((gt == g).mean()) for g in range(3)], mae_regime=[float(at[gt == g].mean()) for g in range(3)],
                       formula_mixture=[float(v) for v in 1 - np.exp(-ql / b)], formula_rule=[float(v) for v in 1 - ALPHA ** (b[0] / b)],
                       rows=[dict(method=n_, cov=float(c_), width=float(w_), wink=float(k_), cov_regime=[float(v) for v in cs_],
                                  wink_regime=[float(v) for v in ks_]) for n_, c_, w_, k_, cs_, ks_ in rows]), fh, indent=1)
    print("single-pad coverage by regime: measured", np.round(rows[0][4], 3), " mixture formula", np.round(1 - np.exp(-ql / b), 3),
          " rule 1-a^(MAE_free/MAE_g)", np.round(1 - ALPHA ** (b[0] / b), 3))


if __name__ == "__main__":
    for net in sys.argv[1:]:
        main(net)
