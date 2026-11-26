"""On a forecaster that has its own quantile outputs, does anything based on the fundamental
diagram still improve the intervals?

Compared, all at alpha = 0.10 on the same test split:
  CQR                 standard conformalized quantile regression on the forecaster's quantiles
  CQR + FD-Mondrian   the same, with the conformal correction computed per FD state
  rq (point)          second-stage residual-quantile model on the point forecast (fd_informed.py)
  refine A / B / C    second-stage refinement of the CQR interval: gradient boosting predicts
                      the 5% quantile of y - q_lo and the 95% quantile of y - q_hi from origin
                      features, the interval ends are shifted by those amounts, and the result
                      is conformalized. Feature sets as in fd_informed.py, plus the forecaster's
                      own interval width: A speed only, B + raw occupancy and flow, C + FD features
  refine C + Mondrian refine C with the correction computed per FD state

Second-stage models are fit on the first half of the calibration period and conformalized on
the second half. Plain CQR uses the whole calibration period.

Run: PRED_DIR=predictions_stid_gpu MODEL=stid python code/strong_forecaster_study.py pems04
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import quantile_file, MODEL
from fd_informed import build, winkler, ALPHA


def main(net):
    d, Xc, Xt, N = build(net)
    qd = np.load(quantile_file(net))
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    gfd = fd_regime_from_pems(d["occ_test"], d["o_c"][d["sid_test"]])
    gfc = fd_regime_from_pems(d["occ_cal"], d["o_c"][d["sid_cal"]])
    half = ((len(yc) // N) // 2) * N; A, B = slice(0, half), slice(half, None)
    rows = []

    def add(name, lo, hi):
        ins, wk = (yt >= lo) & (yt <= hi), winkler(yt, lo, hi)
        rows.append((name, ins.mean(), (hi - lo).mean(), wk.mean(), [ins[gfd == g].mean() for g in range(3)],
                     [wk[gfd == g].mean() for g in range(3)]))

    lc, hc, lt, ht = qd["qlo_cal"], qd["qhi_cal"], qd["qlo_test"], qd["qhi_test"]
    s = np.maximum(lc - yc, yc - hc)
    e = split_conformal_quantile(s, ALPHA); add("CQR", lt - e, ht + e)
    eg = np.array([split_conformal_quantile(s[gfc == g], ALPHA) for g in range(3)]); add("CQR + FD-Mondrian", lt - eg[gfd], ht + eg[gfd])
    print(f"\n==================== {net}, forecaster {MODEL} ====================", flush=True)
    print("CQR corrections by FD state (mph):", np.round(eg, 2), " pooled:", round(float(e), 2), flush=True)

    kw = dict(max_iter=300, learning_rate=0.1, random_state=0)
    rc = yc - pc
    lo_m = HGB(loss="quantile", quantile=ALPHA / 2, **kw).fit(Xc[A], rc[A]); hi_m = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kw).fit(Xc[A], rc[A])
    s2 = np.maximum(lo_m.predict(Xc[B]) - rc[B], rc[B] - hi_m.predict(Xc[B])); e2 = split_conformal_quantile(s2, ALPHA)
    add("rq (point)", pt + lo_m.predict(Xt) - e2, pt + hi_m.predict(Xt) + e2)

    wc, wt = (hc - lc)[:, None], (ht - lt)[:, None]
    for label, k in [("refine A speed", 10), ("refine B +occ/flow", 14), ("refine C +FD", 21)]:
        Fc, Ft = np.hstack([Xc[:, :k], wc]), np.hstack([Xt[:, :k], wt])
        gl = HGB(loss="quantile", quantile=ALPHA / 2, **kw).fit(Fc[A], (yc - lc)[A])
        gh = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kw).fit(Fc[A], (yc - hc)[A])
        l2c, h2c = lc[B] + gl.predict(Fc[B]), hc[B] + gh.predict(Fc[B])
        l2t, h2t = lt + gl.predict(Ft), ht + gh.predict(Ft)
        s3 = np.maximum(l2c - yc[B], yc[B] - h2c); e3 = split_conformal_quantile(s3, ALPHA)
        add(label, l2t - e3, h2t + e3)
        if k == 21:
            eg3 = np.array([split_conformal_quantile(s3[gfc[B] == g], ALPHA) for g in range(3)])
            add("refine C + Mondrian", l2t - eg3[gfd], h2t + eg3[gfd])

    at = np.abs(yt - pt)
    print(f"point MAE {at.mean():.2f}  by FD state " + " ".join(f"{at[gfd == g].mean():.2f}" for g in range(3)))
    hdr = f"{'method':<21}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | {'Wk F':>6}{'Wk T':>6}{'Wk C':>6}"
    print(hdr); print("-" * len(hdr))
    for name, c, w, k, cs, ks in rows:
        print(f"{name:<21}{c:6.3f}{w:7.2f}{k:7.2f} | " + "".join(f"{v:6.3f}" for v in cs) + " | " + "".join(f"{v:6.1f}" for v in ks), flush=True)


if __name__ == "__main__":
    for net in sys.argv[1:]:
        main(net)
