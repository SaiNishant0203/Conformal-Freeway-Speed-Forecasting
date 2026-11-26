"""Comparison of calibration methods, all at alpha = 0.10, 60-minute horizon.

Methods
  global      split CP, single pad
  FD          Mondrian on per-sensor FD state (paper's method)
  rawocc      Mondrian on network-wide raw-occupancy thresholds (same class proportions as FD)
  speed       Mondrian on origin-speed thresholds (same class proportions as FD)
  yhat10      Mondrian on deciles of the point forecast yhat
  normCP      normalized CP, score |r|/sigma(x); sigma from gradient boosting on (spd, yhat, occ)
  normCP+o~   as normCP, plus normalized occupancy o/o_c as a feature
  sigbin10    Mondrian on deciles of predicted error sigma(x)  ("binned predicted error")
  CQR         conformalized quantile regression (GRU 5%/95% quantile model, where one exists)
  ACI         adaptive conformal inference per sensor (Gibbs & Candes 2021), 12-step feedback delay
  clim        state-wise climatology: [Q05, Q95] of future speed in each FD state (ignores the forecast)

Methods that learn sigma(x) fit it on the first half of the calibration period and
conformalize on the second half (so the guarantee is preserved). All others use the
full calibration split. Evaluation is always on the full test split.

Run: python code/baselines.py > results_v2/baselines.txt
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import pred_file, quantile_file

ALPHA = 0.10
H = 12
NAMES = ["free", "trans", "cong"]


def winkler(y, lo, hi, a=ALPHA):
    return (hi - lo) + (2 / a) * np.clip(lo - y, 0, None) + (2 / a) * np.clip(y - hi, 0, None)


def mondrian_pad(s_cal, g_cal, g_test):
    q = {g: split_conformal_quantile(s_cal[g_cal == g], ALPHA) for g in np.unique(g_cal)}
    return np.array([q[g] for g in g_test])


def match_thresholds(ref_var, fd_ref, low_is_cong):
    """3-class thresholds on ref_var with the FD class proportions."""
    p_c, p_t = (fd_ref == 2).mean(), (fd_ref == 1).mean()
    if low_is_cong:
        return np.quantile(ref_var, [p_c, p_c + p_t])
    return np.quantile(ref_var, [1 - p_c, 1 - p_c - p_t])


def assign(var, thr, low_is_cong):
    g = np.zeros(len(var), dtype=int)
    if low_is_cong:
        g[var < thr[1]] = 1; g[var < thr[0]] = 2
    else:
        g[var > thr[1]] = 1; g[var > thr[0]] = 2
    return g


def deciles(ref, x):
    e = np.quantile(ref, np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(e, x)


def aci(y, yhat, sid, s_cal, n_sensors, gamma):
    """Per-sensor ACI on a time-major test array. Error indicator for step t becomes
    known only at t+H, so the update uses the error from H steps earlier."""
    T = len(y) // n_sensors
    Y, P = y.reshape(T, n_sensors), yhat.reshape(T, n_sensors)
    srt = np.sort(s_cal); n = len(srt)
    a_t = np.full(n_sensors, ALPHA)
    pads = np.empty((T, n_sensors)); err = np.zeros((T, n_sensors))
    for t in range(T):
        lvl = np.clip(1 - a_t, 0, 1)
        k = np.clip(np.ceil((n + 1) * lvl).astype(int), 1, n)
        pad = np.where(lvl >= 1, np.inf, srt[k - 1])
        pad = np.where(a_t >= 1, 0.0, pad)
        pads[t] = pad
        err[t] = np.abs(Y[t] - P[t]) > pad
        if t - H >= 0:
            a_t = a_t + gamma * (ALPHA - err[t - H])
    return pads.reshape(-1)


def summarize(name, y, lo, hi, gfd, gsp, rows):
    ins = (y >= lo) & (y <= hi)
    w = np.minimum(hi, 200) - np.maximum(lo, -200)  # guard inf pads from ACI
    wk = winkler(y, np.maximum(lo, -200), np.minimum(hi, 200))
    r = dict(name=name, cov=ins.mean(), width=w.mean(), wink=wk.mean())
    for g in range(3):
        m = gfd == g
        r[f"cov_{g}"], r[f"w_{g}"], r[f"wk_{g}"] = ins[m].mean(), w[m].mean(), wk[m].mean()
    r["worst_fd"] = min(r[f"cov_{g}"] for g in range(3))
    r["worst_sp"] = min(ins[gsp == g].mean() for g in range(3))
    rows.append(r)


def run(net, qfile=None):
    d = dict(np.load(pred_file(net)))
    oc = d["o_c"]; N = len(oc)
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    sc = np.abs(yc - pc)
    gfc = fd_regime_from_pems(d["occ_cal"], oc[d["sid_cal"]])
    gft = fd_regime_from_pems(d["occ_test"], oc[d["sid_test"]])
    thr_v = match_thresholds(d["spd_cal"], gfc, True)
    gvc, gvt = assign(d["spd_cal"], thr_v, True), assign(d["spd_test"], thr_v, True)
    rows = []

    q = split_conformal_quantile(sc, ALPHA)
    summarize("global", yt, pt - q, pt + q, gft, gvt, rows)

    q = mondrian_pad(sc, gfc, gft); summarize("FD", yt, pt - q, pt + q, gft, gvt, rows)

    thr_o = match_thresholds(d["occ_cal"], gfc, False)
    q = mondrian_pad(sc, assign(d["occ_cal"], thr_o, False), assign(d["occ_test"], thr_o, False))
    summarize("rawocc", yt, pt - q, pt + q, gft, gvt, rows)

    q = mondrian_pad(sc, gvc, gvt); summarize("speed", yt, pt - q, pt + q, gft, gvt, rows)

    q = mondrian_pad(sc, deciles(pc, pc), deciles(pc, pt)); summarize("yhat10", yt, pt - q, pt + q, gft, gvt, rows)

    # --- learned sigma(x): fit on first half of calibration period, conformalize on second half
    Tc = len(yc) // N; half = (Tc // 2) * N
    A, B = slice(0, half), slice(half, None)
    otc, ott = d["occ_cal"] / oc[d["sid_cal"]], d["occ_test"] / oc[d["sid_test"]]
    for label, Xc, Xt in [("normCP", np.c_[d["spd_cal"], pc, d["occ_cal"]], np.c_[d["spd_test"], pt, d["occ_test"]]),
                          ("normCP+o~", np.c_[d["spd_cal"], pc, d["occ_cal"], otc], np.c_[d["spd_test"], pt, d["occ_test"], ott])]:
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
        m.fit(Xc[A], sc[A])
        sig_c = np.maximum(m.predict(Xc[B]), 0.5); sig_t = np.maximum(m.predict(Xt), 0.5)
        qn = split_conformal_quantile(sc[B] / sig_c, ALPHA)
        summarize(label, yt, pt - qn * sig_t, pt + qn * sig_t, gft, gvt, rows)
        if label == "normCP":
            e = np.quantile(sig_c, np.linspace(0, 1, 11)[1:-1])
            q = mondrian_pad(sc[B], np.searchsorted(e, sig_c), np.searchsorted(e, sig_t))
            summarize("sigbin10", yt, pt - q, pt + q, gft, gvt, rows)

    if qfile:
        qd = np.load(qfile)
        assert np.array_equal(qd["y_test"], yt)
        s = np.maximum(qd["qlo_cal"] - yc, yc - qd["qhi_cal"])
        qq = split_conformal_quantile(s, ALPHA)
        summarize("CQR", yt, qd["qlo_test"] - qq, qd["qhi_test"] + qq, gft, gvt, rows)

    for gam in (0.005, 0.05):
        pad = aci(yt, pt, d["sid_test"], sc, N, gam)
        summarize(f"ACI g={gam}", yt, pt - pad, pt + pad, gft, gvt, rows)

    lo = np.array([np.quantile(yc[gfc == g], ALPHA / 2) for g in range(3)])
    hi = np.array([np.quantile(yc[gfc == g], 1 - ALPHA / 2) for g in range(3)])
    summarize("clim(FD)", yt, lo[gft], hi[gft], gft, gvt, rows)

    print(f"\n==================== {net} (n_test={len(yt):,}) ====================")
    print(f"speed thresholds (mph) cong<{thr_v[0]:.1f}, trans<{thr_v[1]:.1f};  raw-occ thresholds cong>{thr_o[0]:.4f}, trans>{thr_o[1]:.4f}")
    hdr = (f"{'method':<11}{'cov':>6}{'width':>7}{'Wink':>7} | {'cov F':>6}{'cov T':>6}{'cov C':>6} | "
           f"{'w F':>6}{'w T':>6}{'w C':>6} | {'Wk F':>6}{'Wk T':>6}{'Wk C':>6} | {'worstFD':>7}{'worstSp':>8}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<11}{r['cov']:6.3f}{r['width']:7.2f}{r['wink']:7.2f} | "
              + "".join(f"{r[f'cov_{g}']:6.3f}" for g in range(3)) + " | "
              + "".join(f"{r[f'w_{g}']:6.1f}" for g in range(3)) + " | "
              + "".join(f"{r[f'wk_{g}']:6.1f}" for g in range(3)) + " | "
              + f"{r['worst_fd']:7.3f}{r['worst_sp']:8.3f}")


if __name__ == "__main__":
    for net in sys.argv[1:] or ["pems04", "pems08"]:
        run(net, quantile_file(net))
