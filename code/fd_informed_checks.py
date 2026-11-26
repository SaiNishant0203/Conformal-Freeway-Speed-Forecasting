"""Robustness checks for the residual-quantile conformal method with FD-Mondrian correction
(fd_informed.py, "final" mode). Nothing here is tuned on the test split: every configuration
is run once and reported.

  seeds      five random seeds of the gradient boosting models
  settings   number of trees, learning rate, tree size
  splits     which part of the calibration period fits the second-stage model and which part
             sets the conformal correction: chronological 30/70, 50/50 (default), 70/30,
             the two halves swapped, and alternating days
  detail     default configuration only: Winkler score day by day and sensor by sensor against
             the paper's FD Mondrian intervals and CQR on the GRU quantile model

Each configuration appends one line to results_v2/checks_<net>.jsonl, and finished ones are
skipped, so the script can be stopped and rerun.

Run: python code/fd_informed_checks.py pems08 [--budget-min 8]
     python code/fd_informed_checks.py summary > results_v2/fd_informed_checks.txt
"""
import json
import os
import sys
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import pred_file, quantile_file, RES_DIR
from fd_informed import build, winkler, ALPHA, DAY

START = time.time()
DEFAULT = dict(seed=0, iters=300, lr=0.1, leaves=31, split="chrono50")
CONFIGS = ([("default", {})]
           + [(f"seed {s}", dict(seed=s)) for s in (1, 2, 3, 4)]
           + [("100 trees", dict(iters=100)), ("600 trees", dict(iters=600)),
              ("lr 0.05, 600 trees", dict(lr=0.05, iters=600)),
              ("small trees (15 leaves)", dict(leaves=15)), ("large trees (63 leaves)", dict(leaves=63))]
           + [("fit 30% / conformalize 70%", dict(split="chrono30")), ("fit 70% / conformalize 30%", dict(split="chrono70")),
              ("halves swapped", dict(split="swap")), ("alternating days", dict(split="days"))])


def split_masks(mode, n, N):
    steps = n // N; t = np.arange(n) // N
    if mode.startswith("chrono"):
        fit = t < int(steps * int(mode[6:]) / 100)
    elif mode == "swap":
        fit = t >= steps // 2
    elif mode == "days":
        fit = (t // DAY) % 2 == 0
    return fit, ~fit


def run_config(net, data, cfg, detail=False):
    d, Xc, Xt, N = data
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    rc = yc - pc
    gfd = fd_regime_from_pems(d["occ_test"], d["o_c"][d["sid_test"]])
    gfc = fd_regime_from_pems(d["occ_cal"], d["o_c"][d["sid_cal"]])
    A, B = split_masks(cfg["split"], len(yc), N)
    kw = dict(max_iter=cfg["iters"], learning_rate=cfg["lr"], max_leaf_nodes=cfg["leaves"], random_state=cfg["seed"])
    lo_m = HGB(loss="quantile", quantile=ALPHA / 2, **kw).fit(Xc[A], rc[A])
    hi_m = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kw).fit(Xc[A], rc[A])
    s = np.maximum(lo_m.predict(Xc[B]) - rc[B], rc[B] - hi_m.predict(Xc[B]))
    eg = np.array([split_conformal_quantile(s[gfc[B] == g], ALPHA) for g in range(3)])
    lo, hi = pt + lo_m.predict(Xt) - eg[gfd], pt + hi_m.predict(Xt) + eg[gfd]
    ins, wk = (yt >= lo) & (yt <= hi), winkler(yt, lo, hi)
    out = dict(cov=float(ins.mean()), width=float((hi - lo).mean()), wink=float(wk.mean()),
               cov_state=[float(ins[gfd == g].mean()) for g in range(3)],
               wink_state=[float(wk[gfd == g].mean()) for g in range(3)], corr=[float(v) for v in eg])
    if detail:
        sc = np.abs(rc)
        qg = np.array([split_conformal_quantile(sc[gfc == g], ALPHA) for g in range(3)])
        wk_fd = winkler(yt, pt - qg[gfd], pt + qg[gfd])
        qd = np.load(quantile_file(net))
        e = split_conformal_quantile(np.maximum(qd["qlo_cal"] - yc, yc - qd["qhi_cal"]), ALPHA)
        wk_cqr = winkler(yt, qd["qlo_test"] - e, qd["qhi_test"] + e)
        day = (np.arange(len(yt)) // N) // DAY; sid = d["sid_test"]
        per = lambda key, w: np.array([w[key == k].mean() for k in np.unique(key)])
        dn, df, dc = per(day, wk), per(day, wk_fd), per(day, wk_cqr)
        sn, sf, sq = per(sid, wk), per(sid, wk_fd), per(sid, wk_cqr)
        cong = gfd == 2
        cs = np.array([k for k in np.unique(sid[cong]) if (cong & (sid == k)).sum() >= 200])
        cn = np.array([wk[cong & (sid == k)].mean() for k in cs]); cq = np.array([wk_cqr[cong & (sid == k)].mean() for k in cs])
        cf = np.array([wk_fd[cong & (sid == k)].mean() for k in cs])
        out["detail"] = dict(
            days=len(dn), day_new=dn.round(2).tolist(), day_fd=df.round(2).tolist(), day_cqr=dc.round(2).tolist(),
            days_better_than_fd=int((dn < df).sum()), days_better_than_cqr=int((dn < dc).sum()),
            sensors=len(sn), sensors_better_than_fd=int((sn < sf).sum()), sensors_better_than_cqr=int((sn < sq).sum()),
            median_sensor_gain_vs_cqr_pct=float(np.median(100 * (sq - sn) / sq)),
            worst_sensor_change_vs_cqr_pct=float((100 * (sq - sn) / sq).min()),
            cong_sensors=len(cs), cong_better_than_fd=int((cn < cf).sum()), cong_better_than_cqr=int((cn < cq).sum()),
            base_fd=float(wk_fd.mean()), base_cqr=float(wk_cqr.mean()),
            sensor_cov_p10=float(np.percentile([ins[sid == k].mean() for k in np.unique(sid)], 10)))
    return out


def run(net, budget_min, only=None):
    path = f"{RES_DIR}/checks_{net}.jsonl"
    done = {json.loads(l)["name"] for l in open(path)} if os.path.exists(path) else set()
    todo = [(n, c) for n, c in CONFIGS if n not in done and (only is None or n in only)]
    if not todo:
        print("ALL DONE", net); return
    data = build(net); last = 0.0
    for name, change in todo:
        if budget_min and time.time() - START + 1.2 * last + 20 > budget_min * 60:
            print("PAUSED before", name); return
        t0 = time.time(); cfg = dict(DEFAULT, **change)
        res = run_config(net, data, cfg, detail=(name == "default"))
        last = time.time() - t0
        with open(path, "a") as f:
            f.write(json.dumps(dict(name=name, cfg=cfg, seconds=round(last, 1), **res)) + "\n")
        print(f"done {name} ({last:.0f}s)  Winkler {res['wink']:.2f}", flush=True)
    print("ALL DONE", net)


def summary():
    for net in ["pems04", "pems08"]:
        path = f"{RES_DIR}/checks_{net}.jsonl"
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        print(f"\n==================== {net}: {len(rows)}/{len(CONFIGS)} configurations ====================")
        print(f"{'configuration':<30}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | {'Wk F':>6}{'Wk T':>6}{'Wk C':>6}")
        for r in rows:
            print(f"{r['name']:<30}{r['cov']:6.3f}{r['width']:7.2f}{r['wink']:7.2f} | "
                  + "".join(f"{v:6.3f}" for v in r["cov_state"]) + " | " + "".join(f"{v:6.1f}" for v in r["wink_state"]))
        w = np.array([r["wink"] for r in rows]); cc = np.array([r["cov_state"][2] for r in rows]); ct = np.array([r["cov_state"][1] for r in rows])
        print(f"Winkler over all configurations: min {w.min():.2f}  max {w.max():.2f}   congested coverage: {cc.min():.3f} to {cc.max():.3f}"
              f"   transition coverage: {ct.min():.3f} to {ct.max():.3f}")
        dt = next((r["detail"] for r in rows if r["name"] == "default"), None)
        if dt:
            print(f"baselines on the same test split: FD Mondrian {dt['base_fd']:.2f}, CQR (forecaster quantiles) {dt['base_cqr']:.2f}")
            print(f"worst configuration vs CQR: {100 * (dt['base_cqr'] - w.max()) / dt['base_cqr']:.1f}% lower Winkler; best: {100 * (dt['base_cqr'] - w.min()) / dt['base_cqr']:.1f}%")
            print(f"day by day ({dt['days']} test days): better than FD Mondrian on {dt['days_better_than_fd']}, better than CQR on {dt['days_better_than_cqr']}")
            for lab, key in [("new", "day_new"), ("CQR", "day_cqr"), ("FD ", "day_fd")]:
                print(f"   daily Winkler {lab}:", " ".join(f"{v:5.1f}" for v in dt[key]))
            print(f"sensor by sensor ({dt['sensors']}): better than FD Mondrian on {dt['sensors_better_than_fd']}, better than CQR on {dt['sensors_better_than_cqr']}; "
                  f"median gain vs CQR {dt['median_sensor_gain_vs_cqr_pct']:.1f}%, worst sensor {dt['worst_sensor_change_vs_cqr_pct']:.1f}%")
            print(f"congested state only, sensors with >= 200 congested test points ({dt['cong_sensors']}): better than FD Mondrian on "
                  f"{dt['cong_better_than_fd']}, better than CQR on {dt['cong_better_than_cqr']}")
            print(f"per-sensor coverage, 10th percentile across sensors: {dt['sensor_cov_p10']:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        b = float(sys.argv[sys.argv.index("--budget-min") + 1]) if "--budget-min" in sys.argv else None
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
        run(sys.argv[1], b, only)
