"""Coverage across many overlapping groups at once: regime, time of day, weekend, sensor.

Per-regime (Mondrian) calibration guarantees coverage in each regime but says nothing about
other ways of slicing the data. Group-conditional calibration (Jung et al., "Batch multivalid
conformal prediction", ICLR 2023; Gibbs, Cherian and Candes, "Conformal prediction with
conditional guarantees") fits the correction as a linear function of group indicators by
pinball regression, which targets coverage in every listed group simultaneously, including
groups that overlap.

All methods start from the CQR score max(q_lo - y, y - q_hi) and differ only in the correction:
  pooled             one correction
  regime             one per origin regime
  regime x hour      one per regime and 4-hour block of the day (18 cells)
  sensor x regime    one per sensor and regime, falling back to the regime when a cell has
                     fewer than 300 calibration points
  multigroup         linear quantile regression of the score on indicators of regime (3),
                     4-hour block (6), weekend (1) and sensor speed class (quartiles of mean
                     training speed, 4), fit on all calibration points

Reported: overall coverage and Winkler score, the lowest coverage within each family of groups,
the lowest coverage over regime x hour cells with at least 2,000 test points, and the 10th
percentile of per-sensor coverage.

Run: python code/multigroup_study.py > results_audit/multigroup_study.txt
"""
import json
import os
import numpy as np
from pairs import PAIRS, load, label, winkler, cq, ALPHA, DAY

os.makedirs("results_audit", exist_ok=True)
out = []
for net, model, folder in PAIRS:
    d = load(net, model, folder); N = d["N"]
    yc, yt = d["y_cal"], d["y_test"]; okc, okt, gc, gt = d["ok_cal"], d["ok_test"], d["g_cal"], d["g_test"]
    sc = np.maximum(d["qlo_cal"] - yc, yc - d["qhi_cal"]); st = np.maximum(d["qlo_test"] - yt, yt - d["qhi_test"])
    hour = lambda t: ((d["tod"][t] // 48)[:, None] * np.ones((1, N), int)).astype(int)        # 4-hour blocks
    wknd = lambda t: ((d["dow"][t] >= 5)[:, None] * np.ones((1, N), int)).astype(int)
    hc, ht, wc, wt_ = hour(d["t_cal"]), hour(d["t_test"]), wknd(d["t_cal"]), wknd(d["t_test"])
    with np.errstate(all="ignore"):
        ms = np.nan_to_num(np.nanmean(np.where(d["speed"][:d["tr"]] > 0, d["speed"][:d["tr"]], np.nan), 0))
    cls = np.searchsorted(np.quantile(ms, [0.25, 0.5, 0.75]), ms)                               # sensor speed class 0..3
    kc, kt = np.tile(cls, (len(yc), 1)), np.tile(cls, (len(yt), 1))
    sidc, sidt = np.tile(np.arange(N), (len(yc), 1)), np.tile(np.arange(N), (len(yt), 1))
    corr = {}
    corr["pooled"] = np.full(st.shape, cq(sc[okc], 1 - ALPHA))
    eg = np.array([cq(sc[okc & (gc == g)], 1 - ALPHA) for g in range(3)]); corr["regime"] = eg[gt]
    c = np.zeros(st.shape)
    for g in range(3):
        for h in range(6):
            m = okc & (gc == g) & (hc == h); c[(gt == g) & (ht == h)] = cq(sc[m], 1 - ALPHA) if m.sum() >= 300 else eg[g]
    corr["regime x hour"] = c
    c = eg[gt].astype(float).copy(); n_own = 0
    for g in range(3):
        m = okc & (gc == g)
        for s in range(N):
            v = sc[m[:, s], s]
            if len(v) >= 300:
                c[gt[:, s] == g, s] = cq(v, 1 - ALPHA); n_own += 1
    corr["sensor x regime"] = c
    def design(g, h, w, k):
        Z = np.zeros((g.size, 14), np.float32); r = np.arange(g.size)
        Z[r, g.ravel()] = 1; Z[r, 3 + h.ravel()] = 1; Z[:, 9] = w.ravel(); Z[r, 10 + k.ravel()] = 1
        return Z[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13]]            # drop one level of hour and class (regime acts as intercept)
    # pinball regression of the score on the indicators, by full-batch subgradient descent
    # (convex, 12 coefficients); starts from the per-regime corrections
    Zc = design(gc, hc, wc, kc)[okc.ravel()]; sv = sc.ravel()[okc.ravel()]
    beta = np.zeros(12); beta[:3] = eg; step = float(np.std(sv))
    best, best_loss = beta.copy(), np.inf
    for it in range(400):
        r = sv - Zc @ beta
        loss = float(np.mean(np.maximum((1 - ALPHA) * r, -ALPHA * r)))
        if loss < best_loss:
            best, best_loss = beta.copy(), loss
        grad = Zc.T @ ((r < 0).astype(np.float32) - (1 - ALPHA)) / len(sv)
        beta = beta - step * 2.0 / np.sqrt(it + 1) * grad / (Zc.mean(0) + 1e-9)
    corr["multigroup"] = (design(gt, ht, wt_, kt) @ best).reshape(st.shape)
    fams = {"regime": (gt, 3), "hour": (ht, 6), "weekend": (wt_, 2), "class": (kt, 4)}
    print(f"\n==================== {label(net, model)} ({n_own} of {3 * N} sensor x regime cells have their own correction) ====================")
    print(f"{'correction':<17}{'cov':>6}{'Wink':>7} | lowest coverage in: {'regime':>7}{'hour':>7}{'wkend':>7}{'class':>7}{'reg x hour':>11} | {'sensor p10':>10}{'sensor min':>11}")
    for name, cr in corr.items():
        lo, hi = d["qlo_test"] - cr, d["qhi_test"] + cr; hi = np.maximum(hi, lo)
        ins = st <= cr; wk = winkler(yt, lo, hi)
        mins = {f: min(float(ins[okt & (a == v)].mean()) for v in range(nv)) for f, (a, nv) in fams.items()}
        cells = [float(ins[okt & (gt == g) & (ht == h)].mean()) for g in range(3) for h in range(6) if (okt & (gt == g) & (ht == h)).sum() >= 2000]
        ps = np.array([ins[okt[:, s], s].mean() for s in range(N) if okt[:, s].sum() >= 500])
        row = dict(net=net, model=model, correction=name, cov=float(ins[okt].mean()), wink=float(wk[okt].mean()), min_regime=mins["regime"],
                   min_hour=mins["hour"], min_weekend=mins["weekend"], min_class=mins["class"], min_regime_hour=min(cells), n_cells=len(cells),
                   sensor_p10=float(np.percentile(ps, 10)), sensor_min=float(ps.min()))
        out.append(row)
        print(f"{name:<17}{row['cov']:6.3f}{row['wink']:7.2f} | {'':20}{mins['regime']:7.3f}{mins['hour']:7.3f}{mins['weekend']:7.3f}{mins['class']:7.3f}{min(cells):11.3f} | "
              f"{row['sensor_p10']:10.3f}{row['sensor_min']:11.3f}", flush=True)
json.dump(out, open("results_audit/multigroup_study.json", "w"), indent=1)
