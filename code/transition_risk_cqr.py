"""Can calibration be aimed at the windows where the regime is about to change?

Under the window-based regimes of Wang et al. the hard case is "transition": the hour ahead
contains a change between free flow and congestion. That label depends on the future, so no
method can calibrate on it directly. What can be done at forecast time is to predict it.

Transition-risk calibration, tested here:
  1. On the first half of the calibration period, fit a classifier for "the coming window is a
     transition" from what is known at the origin (speed, its recent change, the forecast and its
     quantiles, time of day, weekend, sensor descriptors).
  2. On the second half, sort points into ten equal-sized bins of predicted transition risk and
     compute the CQR correction separately in each bin.
  3. At test time, apply the correction of the bin the point falls in.
Coverage is then guaranteed within each risk bin, which is known at forecast time. Coverage
within the transition windows themselves is not guaranteed; the question is how close it gets.

Compared with CQR and CQR with a per-regime correction (regime at the origin), both calibrated
on the same second half so that the comparison is like for like.

Run: python code/transition_risk_cqr.py > results_audit/transition_risk_cqr.txt
"""
import json
import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.metrics import roc_auc_score
from pairs import PAIRS, load, label, winkler, cq, window_regime, ALPHA, H, DAY

os.makedirs("results_audit", exist_ok=True)
out = []; rng = np.random.default_rng(0); K = 10
for net, model, folder in PAIRS:
    d = load(net, model, folder); N = d["N"]; sp = d["speed"]
    with np.errstate(all="ignore"):
        tr_sp = sp[:d["tr"]]; ms = np.nan_to_num(np.nanmean(np.where(tr_sp > 0, tr_sp, np.nan), 0))
        slow = np.nan_to_num(((tr_sp > 0) & (tr_sp < 55)).sum(0) / np.maximum((tr_sp > 0).sum(0), 1))

    def feats(s):
        t = d[f"t_{s}"]; ang = (d["tod"][t] * 2 * np.pi / DAY)[:, None] * np.ones((1, N))
        cols = [d[f"spd_{s}"], d[f"yhat_{s}"], d[f"qlo_{s}"], d[f"qhi_{s}"], d[f"qhi_{s}"] - d[f"qlo_{s}"],
                d[f"spd_{s}"] - sp[t - 3], d[f"spd_{s}"] - sp[t - 11], np.sin(ang), np.cos(ang),
                (d["dow"][t] >= 5)[:, None] * np.ones((1, N)), np.tile(ms, (len(t), 1)), np.tile(slow, (len(t), 1))]
        return np.stack([c.ravel() for c in cols], 1).astype(np.float32)

    Xc, Xt = feats("cal"), feats("test")
    wc = window_regime(sp, d["tr"], d["ca"] - H).ravel(); wt = window_regime(sp, d["ca"], d["T"] - H).ravel()
    okc, okt = d["ok_cal"].ravel() & (wc >= 0), d["ok_test"].ravel() & (wt >= 0)
    yc, yt = d["y_cal"].ravel(), d["y_test"].ravel(); gc, gt = d["g_cal"].ravel(), d["g_test"].ravel()
    lc, hc, lt, ht = d["qlo_cal"].ravel(), d["qhi_cal"].ravel(), d["qlo_test"].ravel(), d["qhi_test"].ravel()
    half = (len(d["t_cal"]) // 2) * N; first = np.arange(len(yc)) < half
    A = np.flatnonzero(okc & first); B = okc & ~first
    A = rng.choice(A, min(500000, len(A)), replace=False)
    clf = HGBC(max_iter=200, learning_rate=0.1, random_state=0).fit(Xc[A], (wc[A] == 1).astype(int))
    pB, pT = clf.predict_proba(Xc[B])[:, 1], clf.predict_proba(Xt)[:, 1]
    auc = float(roc_auc_score((wt[okt] == 1).astype(int), pT[okt]))
    s = np.maximum(lc - yc, yc - hc)[B]; gB = gc[B]
    edges = np.quantile(pB, np.linspace(0, 1, K + 1)[1:-1]); bB, bT = np.searchsorted(edges, pB), np.searchsorted(edges, pT)
    corr = {"CQR": np.full(len(yt), cq(s, 1 - ALPHA)),
            "CQR per origin regime": np.array([cq(s[gB == g], 1 - ALPHA) for g in range(3)])[gt],
            "CQR per transition-risk bin": np.array([cq(s[bB == k], 1 - ALPHA) for k in range(K)])[bT]}
    print(f"\n==================== {label(net, model)}: transition classifier AUC {auc:.3f}; share of transition windows {float((wt[okt] == 1).mean()):.3f} ====================")
    print(f"{'correction':<29}{'cov':>6}{'width':>7}{'Wink':>7} | window regime: {'free':>6}{'trans':>6}{'cong':>6} | origin regime: {'free':>6}{'trans':>6}{'cong':>6} | {'Wk trans':>8}")
    st = np.maximum(lt - yt, yt - ht)
    for name, cr in corr.items():
        lo, hi = lt - cr, np.maximum(ht + cr, lt - cr); ins = st <= cr; wk = winkler(yt, lo, hi)
        cw = [float(ins[okt & (wt == g)].mean()) for g in range(3)]; co = [float(ins[okt & (gt == g)].mean()) for g in range(3)]
        row = dict(net=net, model=model, correction=name, cov=float(ins[okt].mean()), width=float((hi - lo)[okt].mean()), wink=float(wk[okt].mean()),
                   cov_window=cw, cov_origin=co, wink_window_transition=float(wk[okt & (wt == 1)].mean()), auc=auc)
        out.append(row)
        print(f"{name:<29}{row['cov']:6.3f}{row['width']:7.2f}{row['wink']:7.2f} | {'':15}" + "".join(f"{c:6.3f}" for c in cw) + f" | {'':15}"
              + "".join(f"{c:6.3f}" for c in co) + f" | {row['wink_window_transition']:8.1f}", flush=True)
    # Selection on the outcome: inside a risk bin coverage is on target, yet the windows that turn
    # out to be transitions are covered less often than the ones that do not.
    ins = st <= corr["CQR per transition-risk bin"]
    print(f"{'risk bin':<9}{'share trans.':>13}{'coverage':>10}{'if transition':>15}{'if not':>8}")
    sel = []
    for k in range(K):
        m = okt & (bT == k); tr_ = m & (wt == 1); nt = m & (wt != 1)
        r_ = dict(bin=k, share=float(tr_.sum() / max(m.sum(), 1)), cov=float(ins[m].mean()),
                  cov_trans=float(ins[tr_].mean()) if tr_.sum() >= 200 else None, cov_other=float(ins[nt].mean()) if nt.sum() >= 200 else None)
        sel.append(r_)
        f = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        print(f"{k + 1:<9}{r_['share']:13.3f}{r_['cov']:10.3f}{f(r_['cov_trans']):>15}{f(r_['cov_other']):>8}")
    out[-1]["by_bin"] = sel
    # A valid way to look at the hard cases: coverage by predicted risk decile, which is known at
    # forecast time, for the standard methods (all calibrated on the same second half).
    sa = np.abs(yc - d["yhat_cal"].ravel())[B]; sat = np.abs(yt - d["yhat_test"].ravel())
    pads = {"split conformal": np.full(len(yt), cq(sa, 1 - ALPHA)),
            "Mondrian on origin regime": np.array([cq(sa[gB == g], 1 - ALPHA) for g in range(3)])[gt]}
    by_risk = {name: [float((sat <= pd_)[okt & (bT == k)].mean()) for k in range(K)] for name, pd_ in pads.items()}
    for name in ("CQR", "CQR per origin regime", "CQR per transition-risk bin"):
        by_risk[name] = [float((st <= corr[name])[okt & (bT == k)].mean()) for k in range(K)]
    print("coverage by predicted transition-risk decile (low to high):")
    for name, v in by_risk.items():
        print(f"  {name:<29}" + " ".join(f"{x:.3f}" for x in v))
    out[-1]["coverage_by_risk_decile"] = by_risk
    print("corrections by risk bin (mph), lowest to highest risk: " + " ".join(f"{cq(s[bB == k], 1 - ALPHA):.2f}" for k in range(K)))
json.dump(out, open("results_audit/transition_risk_cqr.json", "w"), indent=1)
