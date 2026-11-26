"""Online conformal methods at a 60-minute horizon, where the outcome of a forecast is only
known 12 steps later, compared with batch calibration.

Each sensor runs its own online tracker over the test period. A tracker may only use feedback
from forecasts whose target time has passed (delay = 12 steps); a "no delay" variant (feedback
after one step) is included to show what the delay costs, although it cannot be used in practice.

Methods
  batch        fixed correction from the calibration split
  ACI          adaptive conformal inference (Gibbs and Candes 2021): the miscoverage level
               alpha_t moves by gamma * (alpha - miss) and the pad is the matching quantile of
               the calibration scores
  QT           quantile tracking (the proportional term of conformal PID control, Angelopoulos
               et al. 2023): the pad itself moves by eta * (miss - alpha)
  R-ACI, R-QT  one tracker per sensor and origin regime (the regime-conditional variant of
               Wang et al. 2026)
Scores: |y - yhat| ("abs") and the CQR score max(q_lo - y, y - q_hi) ("cqr").

Also reported: how long misses stay correlated in time (the lag at which the autocorrelation of
the batch miss indicator falls below 1/e), which gives the delay-to-memory ratio of El Halabi
and Brandt (arXiv 2609.07251).

Run: python code/online_delay_study.py > results_audit/online_delay_study.txt
"""
import json
import os
import sys
import numpy as np
from pairs import PAIRS, load, label, winkler, window_regime, ALPHA, H

os.makedirs("results_audit", exist_ok=True)


def run_online(s_test, ok, g, srt_by, q0_by, kind, rate, delay, per_regime, scale):
    """s_test [T, N] scores, ok mask, g origin regime. Returns the pad used at each step [T, N]."""
    T, N = s_test.shape; R = 3 if per_regime else 1
    gi = g.astype(int) if per_regime else np.zeros_like(g, dtype=int)
    cols = np.arange(N)
    if kind == "aci":
        state = np.full((N, R), ALPHA)
    else:
        state = np.tile(np.asarray(q0_by, dtype=float)[None, :R] if per_regime else np.array([[q0_by[0]]]), (N, 1))
    pads = np.empty((T, N))
    for t in range(T):
        cur = state[cols, gi[t]]
        if kind == "aci":
            a = cur; pad = np.empty(N)
            for r in range(R):
                m = gi[t] == r
                if m.any():
                    srt = srt_by[r]; n = len(srt)
                    k = np.clip(np.ceil((n + 1) * (1 - a[m])).astype(int), 1, n)
                    pad[m] = np.where(a[m] <= 0, np.inf, np.where(a[m] >= 1, -np.inf, srt[k - 1]))
        else:
            pad = cur
        pads[t] = pad
        u = t - delay + 1                                   # forecast whose outcome becomes known now
        if u >= 0:
            v = ok[u]; miss = (s_test[u] > pads[u]).astype(float); r_u = gi[u]
            if kind == "aci":
                np.add.at(state, (cols[v], r_u[v]), rate * (ALPHA - miss[v]))
            else:
                np.add.at(state, (cols[v], r_u[v]), rate * scale * (miss[v] - ALPHA))
    return pads


def memory_lag(miss, ok):
    """Lag (in steps) at which the autocorrelation of the miss indicator, averaged over sensors
    with enough misses, falls below 1/e."""
    lags = np.arange(1, 145); acf = []
    x = np.where(ok, miss, np.nan)
    keep = np.nansum(x, 0) >= 50
    xc = x[:, keep] - np.nanmean(x[:, keep], 0)
    var = np.nanmean(xc ** 2, 0)
    for L in lags:
        acf.append(np.nanmean(np.nanmean(xc[L:] * xc[:-L], 0) / var))
    acf = np.array(acf); below = np.where(acf < 1 / np.e)[0]
    return (int(lags[below[0]]) if len(below) else 145), acf


def main():
    out = []
    for net, model, folder in PAIRS:
        d = load(net, model, folder)
        yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
        okc, okt, gc, gt = d["ok_cal"], d["ok_test"], d["g_cal"], d["g_test"]
        wt = window_regime(d["speed"], d["ca"], d["T"] - H)
        scores = {"abs": (np.abs(yc - pc), np.abs(yt - pt), pt, pt),
                  "cqr": (np.maximum(d["qlo_cal"] - yc, yc - d["qhi_cal"]), np.maximum(d["qlo_test"] - yt, yt - d["qhi_test"]), d["qlo_test"], d["qhi_test"])}
        print(f"\n==================== {label(net, model)} ====================")
        hdr = (f"{'score':<5}{'method':<22}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | "
               f"{'winT':>6}{'inf%':>6}")
        print(hdr); print("-" * len(hdr))
        for sname, (sc, st, lo0, hi0) in scores.items():
            srt_all = [np.sort(sc[okc])]; srt_reg = [np.sort(sc[okc & (gc == r)]) for r in range(3)]
            q_all = [float(np.quantile(srt_all[0], 1 - ALPHA))]; q_reg = [float(np.quantile(s_, 1 - ALPHA)) for s_ in srt_reg]
            cap = float(srt_all[0][-1]); scale = float(np.std(srt_all[0]))   # step unit for quantile tracking
            configs = [("batch", None), ("batch per regime", None)]
            for g_ in (0.002, 0.005, 0.02, 0.05):
                configs += [(f"ACI g={g_}", ("aci", g_, H, False)), (f"R-ACI g={g_}", ("aci", g_, H, True))]
            for e_ in (0.02, 0.1, 0.5):
                configs += [(f"QT eta={e_}", ("qt", e_, H, False)), (f"R-QT eta={e_}", ("qt", e_, H, True))]
            configs += [("ACI g=0.05 no delay", ("aci", 0.05, 1, False)), ("R-ACI g=0.05 no delay", ("aci", 0.05, 1, True)),
                        ("QT eta=0.1 no delay", ("qt", 0.1, 1, False)), ("R-QT eta=0.1 no delay", ("qt", 0.1, 1, True))]
            for name, cfg in configs:
                if cfg is None:
                    pad = np.full(st.shape, q_all[0]) if name == "batch" else np.array(q_reg)[gt]
                else:
                    kind, rate, dl, pr = cfg
                    pad = run_online(st, okt, gt, srt_reg if pr else srt_all, q_reg if pr else q_all, kind, rate, dl, pr, scale)
                inf = ~np.isfinite(pad); padc = np.clip(np.where(np.isneginf(pad), -cap, pad), -cap, cap)
                lo, hi = lo0 - padc, hi0 + padc; hi = np.maximum(hi, lo)
                ins = (st <= pad); wk = winkler(yt, lo, hi); wd = hi - lo
                cs = [float(ins[okt & (gt == r)].mean()) for r in range(3)]
                wtc = float(ins[okt & (wt == 1)].mean())
                row = dict(net=net, model=model, score=sname, method=name, cov=float(ins[okt].mean()), width=float(wd[okt].mean()),
                           wink=float(wk[okt].mean()), cov_regime=cs, cov_window_transition=wtc, inf_share=float(inf[okt].mean()))
                out.append(row)
                print(f"{sname:<5}{name:<22}{row['cov']:6.3f}{row['width']:7.2f}{row['wink']:7.2f} | " + "".join(f"{c:6.3f}" for c in cs)
                      + f" | {wtc:6.3f}{100 * row['inf_share']:6.2f}", flush=True)
                if name == "batch":
                    L, acf = memory_lag((st > pad).astype(float), okt)
                    row["memory_lag"] = L; row["acf_12"] = float(acf[11]); row["acf_1"] = float(acf[0])
                    mem = (sname, L, float(acf[0]), float(acf[11]))
                    print(f"      misses stay correlated for about {L} steps ({5 * L} min); autocorrelation at 1 step {acf[0]:.2f}, at 12 steps {acf[11]:.2f}; "
                          f"delay-to-memory ratio {H / L:.2f}")
    json.dump(out, open("results_audit/online_delay_study.json", "w"), indent=1)
    horizons()


def horizons():
    """PeMS04, original GRU runs at 15, 30 and 60 minutes: how the benefit of ACI changes as the
    feedback delay grows with the horizon. Absolute-residual score, gamma = 0.05."""
    from train_stid import load_raw
    from pairs import origin_regime
    speed = load_raw("pems04")[0]; T, N = speed.shape; tr, ca = int(0.6 * T), int(0.8 * T)
    print("\n==================== PeMS04, original GRU, by horizon (absolute-residual score) ====================")
    print(f"{'horizon':<9}{'method':<22}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | {'inf%':>6}")
    res = []
    for h, f in ((3, "predictions/predictions_gru_h3.npz"), (6, "predictions/predictions_gru_h6.npz"), (12, "predictions/predictions_gru_pems04_h12.npz")):
        if not os.path.exists(f):
            continue
        d = np.load(f); sh = lambda k: d[k].reshape(-1, N)
        yc, pc, yt, pt = sh("y_cal"), sh("yhat_cal"), sh("y_test"), sh("yhat_test")
        gc, gt = origin_regime(speed[tr:tr + len(yc)]), origin_regime(speed[ca:ca + len(yt)])
        sc, st = np.abs(yc - pc), np.abs(yt - pt); ok = np.ones(st.shape, bool)
        srt = [np.sort(sc.ravel())]; q0 = [float(np.quantile(srt[0], 1 - ALPHA))]; cap = float(srt[0][-1])
        for name, cfg in (("batch", None), ("ACI g=0.05", (h,)), ("ACI g=0.05 no delay", (1,))):
            pad = np.full(st.shape, q0[0]) if cfg is None else run_online(st, ok, gt, srt, q0, "aci", 0.05, cfg[0], False, 1.0)
            inf = ~np.isfinite(pad); padc = np.clip(np.where(np.isneginf(pad), -cap, pad), -cap, cap)
            lo, hi = pt - padc, np.maximum(pt + padc, pt - padc); ins = st <= pad; wk = winkler(yt, lo, hi)
            cs = [float(ins[gt == r].mean()) for r in range(3)]
            res.append(dict(horizon=h, method=name, cov=float(ins.mean()), width=float((hi - lo).mean()), wink=float(wk.mean()), cov_regime=cs, inf_share=float(inf.mean())))
            print(f"{5 * h:>3} min  {name:<22}{ins.mean():6.3f}{(hi - lo).mean():7.2f}{wk.mean():7.2f} | " + "".join(f"{c:6.3f}" for c in cs) + f" | {100 * inf.mean():6.2f}", flush=True)
            if cfg is None:
                L, acf = memory_lag((st > pad).astype(float), ok)
                print(f"         misses stay correlated for about {L} steps; delay-to-memory ratio {h / L:.2f}")
    json.dump(res, open("results_audit/online_delay_horizons.json", "w"), indent=1)


if __name__ == "__main__":
    main()
