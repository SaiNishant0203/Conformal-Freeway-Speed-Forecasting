"""Coverage of 60-minute intervals under the regime definition of Wang et al. (arXiv 2606.18367),
for a direct comparison with their Table 2.

Their regimes are defined on the forecast window, not at the origin: free flow if every reading
in the window is above 55 mph, congested if every reading is below 25 mph, transition otherwise.
A window regime is not known when the forecast is made, so no method can calibrate on it; it is
used here only to stratify the evaluation.

Methods: split conformal (one pad), CQR, CQR with a per-regime correction (regime taken at the
origin, which is known at forecast time), and a historical baseline (5% to 95% range of the
speed one hour later, per sensor and origin regime, from calibration data).

They evaluate 30 congestion-prone sensors per dataset; the second block repeats the evaluation
on the 30 sensors with the largest share of training readings below 55 mph.

Run: python code/wang_regime_eval.py > results_audit/wang_regime_eval.txt
"""
import json
import os
import numpy as np
from pairs import PAIRS, load, label, winkler, cq, ALPHA

os.makedirs("results_audit", exist_ok=True)
out = []
for net, model, folder in PAIRS:
    d = load(net, model, folder)
    from pairs import window_regime
    wt = window_regime(d["speed"], d["ca"], d["T"] - 12)
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    okc, okt, gc, gt = d["ok_cal"], d["ok_test"], d["g_cal"], d["g_test"]
    iv = {}
    q = cq(np.abs(yc - pc)[okc], 1 - ALPHA); iv["split conformal"] = (pt - q, pt + q)
    s = np.maximum(d["qlo_cal"] - yc, yc - d["qhi_cal"])
    e = cq(s[okc], 1 - ALPHA); iv["CQR"] = (d["qlo_test"] - e, d["qhi_test"] + e)
    eg = np.array([cq(s[okc & (gc == g)], 1 - ALPHA) for g in range(3)]); iv["CQR, per-regime correction"] = (d["qlo_test"] - eg[gt], d["qhi_test"] + eg[gt])
    lo = np.zeros(pt.shape, np.float32); hi = np.zeros(pt.shape, np.float32)
    for g in range(3):                                  # per sensor and origin regime, pooled fallback when thin
        m = okc & (gc == g); pooled = np.quantile(yc[m], [ALPHA / 2, 1 - ALPHA / 2])
        for sidx in range(d["N"]):
            v = yc[m[:, sidx], sidx]; ql, qh = np.quantile(v, [ALPHA / 2, 1 - ALPHA / 2]) if len(v) >= 100 else pooled
            sel = gt[:, sidx] == g; lo[sel, sidx] = ql; hi[sel, sidx] = qh
    iv["historical baseline"] = (lo, hi)
    train = d["speed"][:d["tr"]]
    with np.errstate(all="ignore"):
        slow = np.nan_to_num(((train > 0) & (train < 55)).sum(0) / np.maximum((train > 0).sum(0), 1))
    top30 = np.zeros(d["N"], bool); top30[np.argsort(-slow)[:30]] = True
    for subset, cols in (("all sensors", np.ones(d["N"], bool)), ("30 most congestion-prone sensors", top30)):
        v = okt & (wt >= 0) & cols[None, :]
        shares = [float((wt[v] == g).mean()) for g in range(3)]
        print(f"\n=== {label(net, model)}, {subset}: {int(v.sum()):,} test points; window-regime shares free/trans/cong "
              + " / ".join(f"{x:.3f}" for x in shares) + f"; MAE by window regime " + " / ".join(f"{np.abs(yt - pt)[v & (wt == g)].mean():.2f}" for g in range(3)))
        print(f"{'method':<28}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | {'wid T':>6}{'Wk T':>6}")
        for name, (l, h) in iv.items():
            ins = (yt >= l) & (yt <= h); wk = winkler(yt, l, h); wd = h - l
            cs = [float(ins[v & (wt == g)].mean()) for g in range(3)]
            print(f"{name:<28}{ins[v].mean():6.3f}{wd[v].mean():7.2f}{wk[v].mean():7.2f} | " + "".join(f"{c:6.3f}" for c in cs)
                  + f" | {wd[v & (wt == 1)].mean():6.1f}{wk[v & (wt == 1)].mean():6.1f}", flush=True)
            out.append(dict(net=net, model=model, subset=subset, method=name, cov=float(ins[v].mean()), width=float(wd[v].mean()),
                            wink=float(wk[v].mean()), cov_window_regime=cs, shares=shares))
json.dump(out, open("results_audit/wang_regime_eval.json", "w"), indent=1)
