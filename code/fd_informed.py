"""Does fundamental-diagram information buy interval efficiency beyond what speed carries?

Ablation over three feature sets available at the forecast origin:
  A  speed only   : origin speed, forecast, speed trend / min / std over the last hour,
                    time of day, weekend flag, sensor mean speed
  B  A + raw      : occupancy, flow, and their 15-minute trends (no per-sensor normalization)
  C  B + FD       : o/o_c, flow / capacity, trend of o/o_c, minutes since o/o_c was last below 1
                    (time spent beyond critical), minutes since it was last above 1, speed /
                    free-flow speed, and the sensor's o_c

Two calibrated methods are built on each feature set:
  norm   normalized conformal: gradient boosting predicts |y - yhat|, score |r| / sigma
  rq     residual-quantile conformal: gradient boosting predicts the 5% and 95% quantiles of
         the signed residual y - yhat, then the usual CQR correction (asymmetric intervals)

Models are fit on the first half of the calibration period and conformalized on the second
half; evaluation is on the full test split. Per-sensor FD parameters (o_c, capacity,
free-flow speed) come from the training split only.

Run: python code/fd_informed.py pems04 > results_v2/fd_informed_pems04.txt
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import pred_file

ALPHA, H, L = 0.10, 12, 12
DAY = 288


def fd_params(flow, occ, speed, o_c, n_bins=40):
    """Capacity (apex of the 90th-percentile flow envelope) and free-flow speed per sensor."""
    N = flow.shape[1]; cap = np.zeros(N); vf = np.zeros(N)
    for s in range(N):
        lo, hi = np.nanpercentile(occ[:, s], [1, 99]); ed = np.linspace(lo, hi, n_bins + 1)
        env = [np.quantile(flow[(occ[:, s] >= ed[b]) & (occ[:, s] < ed[b + 1]), s], 0.90)
               for b in range(n_bins) if ((occ[:, s] >= ed[b]) & (occ[:, s] < ed[b + 1])).sum() >= 20]
        cap[s] = max(env) if env else np.quantile(flow[:, s], 0.99)
        vf[s] = np.quantile(speed[:, s], 0.85)
    return cap, vf


def run_lengths(above):
    """For each time step: steps since the condition was last False (capped at 6 hours)."""
    T, N = above.shape; out = np.zeros((T, N), dtype=np.float32); c = np.zeros(N, dtype=np.float32)
    for t in range(T):
        c = np.where(above[t], c + 1, 0); out[t] = np.minimum(c, 72)
    return out


def build(net):
    raw = np.load(f"Dataset/{net}.npz")["data"]
    flow, occ, speed = (raw[..., k].astype(np.float32) for k in range(3))
    T, N = speed.shape; tr, ca = int(0.6 * T), int(0.8 * T)
    d = dict(np.load(pred_file(net))); o_c = d["o_c"].astype(np.float32)
    cap, vf = fd_params(flow[:tr], occ[:tr], speed[:tr], o_c)
    on = occ / o_c
    since_below = run_lengths(on >= 1.0)      # time spent at or beyond critical occupancy
    since_above = run_lengths(on < 1.0)       # time since last beyond critical
    mean_spd = speed[:tr].mean(0)

    def feats(t0, t1, yhat):
        t = np.arange(t0, t1)
        hist = np.stack([speed[t - k] for k in range(L)], 0)        # [L, n_t, N], k=0 is origin
        tod = (t % DAY)[:, None] * np.ones((1, N)) * 2 * np.pi / DAY
        wk = (((t // DAY) % 7) >= 5)[:, None] * np.ones((1, N))
        f = lambda a: np.asarray(a, dtype=np.float32).reshape(-1)
        A = [f(speed[t]), yhat, f(speed[t] - speed[t - 3]), f(speed[t] - speed[t - 11]), f(hist.min(0)),
             f(hist.std(0)), f(np.sin(tod)), f(np.cos(tod)), f(wk), f(np.tile(mean_spd, (len(t), 1)))]
        B = [f(occ[t]), f(flow[t]), f(occ[t] - occ[t - 3]), f(flow[t] - flow[t - 3])]
        C = [f(on[t]), f(flow[t] / cap), f(on[t] - on[t - 3]), f(since_below[t]), f(since_above[t]),
             f(speed[t] / vf), f(np.tile(o_c, (len(t), 1)))]
        return np.stack(A + B + C, 1)

    Xc, Xt = feats(tr, ca - H, d["yhat_cal"]), feats(ca, T - H, d["yhat_test"])
    assert np.array_equal(Xc[:, 0], d["spd_cal"]) and np.array_equal(Xt[:, 0], d["spd_test"])
    return d, Xc, Xt, N


def winkler(y, lo, hi):
    return (hi - lo) + (2 / ALPHA) * np.clip(lo - y, 0, None) + (2 / ALPHA) * np.clip(y - hi, 0, None)


def report(name, y, lo, hi, gfd, gsp):
    ins = (y >= lo) & (y <= hi); wk = winkler(y, lo, hi)
    c = [ins[gfd == g].mean() for g in range(3)]; w = [wk[gfd == g].mean() for g in range(3)]
    ws = min(ins[gsp == g].mean() for g in range(3))
    print(f"{name:<14}{ins.mean():6.3f}{(hi - lo).mean():7.2f}{wk.mean():7.2f} | "
          + "".join(f"{v:6.3f}" for v in c) + " | " + "".join(f"{v:6.1f}" for v in w)
          + f" | {min(c):7.3f}{ws:8.3f}", flush=True)


def main(net):
    d, Xc, Xt, N = build(net)
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    rc = yc - pc
    gfd = fd_regime_from_pems(d["occ_test"], d["o_c"][d["sid_test"]])
    gfc = fd_regime_from_pems(d["occ_cal"], d["o_c"][d["sid_cal"]])
    thr = np.quantile(d["spd_cal"], [(gfc == 2).mean(), (gfc >= 1).mean()])
    gsp = np.zeros(len(yt), int); gsp[d["spd_test"] < thr[1]] = 1; gsp[d["spd_test"] < thr[0]] = 2
    half = ((len(yc) // N) // 2) * N
    A, B = slice(0, half), slice(half, None)
    sets = {"A speed": slice(0, 10), "B +occ/flow": slice(0, 14), "C +FD": slice(0, 21)}
    print(f"\n==================== {net} (n_test={len(yt):,}; fit on {half:,}, conformalize on {len(yc) - half:,}) ====================")
    hdr = (f"{'method':<14}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | "
           f"{'Wk F':>6}{'Wk T':>6}{'Wk C':>6} | {'worstFD':>7}{'worstSp':>8}")
    print(hdr); print("-" * len(hdr), flush=True)
    kw = dict(max_iter=300, learning_rate=0.1, random_state=0)
    for label, cols in sets.items():
        m = HGB(**kw).fit(Xc[A][:, cols], np.abs(rc[A]))
        sc_, st_ = np.maximum(m.predict(Xc[B][:, cols]), 0.5), np.maximum(m.predict(Xt[:, cols]), 0.5)
        q = split_conformal_quantile(np.abs(rc[B]) / sc_, ALPHA)
        report(f"norm {label}", yt, pt - q * st_, pt + q * st_, gfd, gsp)
        lo_m = HGB(loss="quantile", quantile=ALPHA / 2, **kw).fit(Xc[A][:, cols], rc[A])
        hi_m = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kw).fit(Xc[A][:, cols], rc[A])
        lc, hc = lo_m.predict(Xc[B][:, cols]), hi_m.predict(Xc[B][:, cols])
        e = split_conformal_quantile(np.maximum(lc - rc[B], rc[B] - hc), ALPHA)
        report(f"rq   {label}", yt, pt + lo_m.predict(Xt[:, cols]) - e, pt + hi_m.predict(Xt[:, cols]) + e, gfd, gsp)


def final(net):
    """Residual-quantile conformal on the full feature set, with the conformal correction
    computed once (pooled) or separately per FD state (Mondrian), so that each FD state gets
    its own finite-sample guarantee."""
    d, Xc, Xt, N = build(net)
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    rc = yc - pc
    gfd = fd_regime_from_pems(d["occ_test"], d["o_c"][d["sid_test"]])
    gfc = fd_regime_from_pems(d["occ_cal"], d["o_c"][d["sid_cal"]])
    thr = np.quantile(d["spd_cal"], [(gfc == 2).mean(), (gfc >= 1).mean()])
    gsp = np.zeros(len(yt), int); gsp[d["spd_test"] < thr[1]] = 1; gsp[d["spd_test"] < thr[0]] = 2
    half = ((len(yc) // N) // 2) * N
    A, B = slice(0, half), slice(half, None)
    kw = dict(max_iter=300, learning_rate=0.1, random_state=0)
    lo_m = HGB(loss="quantile", quantile=ALPHA / 2, **kw).fit(Xc[A], rc[A])
    hi_m = HGB(loss="quantile", quantile=1 - ALPHA / 2, **kw).fit(Xc[A], rc[A])
    lc, hc, lt, ht = lo_m.predict(Xc[B]), hi_m.predict(Xc[B]), lo_m.predict(Xt), hi_m.predict(Xt)
    s = np.maximum(lc - rc[B], rc[B] - hc)
    print(f"\n==================== {net}: residual-quantile conformal, full feature set ====================")
    hdr = (f"{'method':<14}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | "
           f"{'Wk F':>6}{'Wk T':>6}{'Wk C':>6} | {'worstFD':>7}{'worstSp':>8}")
    print(hdr); print("-" * len(hdr), flush=True)
    e = split_conformal_quantile(s, ALPHA)
    report("rq pooled", yt, pt + lt - e, pt + ht + e, gfd, gsp)
    eg = np.array([split_conformal_quantile(s[gfc[B] == g], ALPHA) for g in range(3)])
    report("rq FD-Mondrian", yt, pt + lt - eg[gfd], pt + ht + eg[gfd], gfd, gsp)
    print("per-state corrections (mph):", np.round(eg, 2), " pooled:", round(e, 2))
    mid = pt + 0.5 * (lt + ht)
    print("MAE of GRU forecast by FD state:     ", np.round([np.abs(yt - pt)[gfd == g].mean() for g in range(3)], 2))
    print("MAE of interval midpoint by FD state:", np.round([np.abs(yt - mid)[gfd == g].mean() for g in range(3)], 2))


if __name__ == "__main__":
    (final if len(sys.argv) > 2 and sys.argv[2] == "final" else main)(sys.argv[1])
