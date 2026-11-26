"""Reproduce the main tables from the saved 60-minute predictions: point error by FD state,
global vs FD-conditional coverage, and the cross-conditioning comparison between FD states
and speed states. Two speed partitions are reported (matched to the FD class proportions,
and tertiles) because the result depends on which one is used.

Run from the project root:  python code/reproduce_tables.py > results_v2/reproduce_tables.txt
"""
import sys
import numpy as np
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import pred_file

ALPHA = 0.10
NAMES = ["free", "trans", "cong"]


def load(net):
    return dict(np.load(pred_file(net)))


def fd_state(occ, sid, o_c, band=0.15):
    return fd_regime_from_pems(occ, o_c[sid], band)


def cov_width(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean()), float((hi - lo).mean())


def mondrian(s_cal, g_cal, g_test, fallback=None):
    q = {}
    for g in np.unique(g_test):
        m = g_cal == g
        q[g] = split_conformal_quantile(s_cal[m], ALPHA) if m.sum() else np.nan
    return np.array([q[g] for g in g_test]), q


def speed_state_matched(spd_ref, spd, fd_ref):
    """Speed thresholds chosen so class proportions (on ref split) match the FD proportions.
    Low speed = congested. Returns labels 0 free / 1 trans / 2 cong."""
    p_cong = (fd_ref == 2).mean()
    p_trans = (fd_ref == 1).mean()
    t_cong = np.quantile(spd_ref, p_cong)            # below -> congested
    t_trans = np.quantile(spd_ref, p_cong + p_trans)  # below -> transition
    g = np.zeros_like(spd, dtype=int)
    g[spd < t_trans] = 1
    g[spd < t_cong] = 2
    return g, (t_cong, t_trans)


def speed_state_tertile(spd_ref, spd):
    t1, t2 = np.quantile(spd_ref, [1 / 3, 2 / 3])
    g = np.zeros_like(spd, dtype=int)
    g[spd < t2] = 1
    g[spd < t1] = 2
    return g, (t1, t2)


def report(net):
    d = load(net)
    yc, yhc, yt, yht = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    sc = np.abs(yc - yhc)
    gc = fd_state(d["occ_cal"], d["sid_cal"], d["o_c"])
    gt = fd_state(d["occ_test"], d["sid_test"], d["o_c"])
    print(f"\n==================== {net} ====================")
    print(f"n_cal={len(yc):,}  n_test={len(yt):,}")

    # point error by FD state
    ae = np.abs(yt - yht)
    print("\nPoint MAE by FD state (test):")
    for g in range(3):
        m = gt == g
        print(f"  {NAMES[g]:<6} MAE={ae[m].mean():6.2f}  n={m.sum():>9,}")
    print(f"  all    MAE={ae.mean():6.2f}")

    # global split CP
    q = split_conformal_quantile(sc, ALPHA)
    lo, hi = yht - q, yht + q
    inside = (yt >= lo) & (yt <= hi)
    print(f"\nGlobal split CP: pad={q:.2f}  width={2*q:.2f}  coverage={inside.mean():.3f}")
    for g in range(3):
        print(f"  {NAMES[g]:<6} cov={inside[gt == g].mean():.3f}")

    # FD Mondrian
    qt, qd = mondrian(sc, gc, gt)
    inside = np.abs(yt - yht) <= qt
    print(f"\nFD Mondrian: all cov={inside.mean():.3f} width={2*qt.mean():.2f}")
    for g in range(3):
        print(f"  {NAMES[g]:<6} cov={inside[gt == g].mean():.3f} width={2*qd[g]:.2f}")

    # speed-state definitions: which one did the paper use?
    for label, fn in [("MATCHED-to-FD proportions", lambda: (speed_state_matched(d["spd_cal"], d["spd_cal"], gc)[0],
                                                                speed_state_matched(d["spd_cal"], d["spd_test"], gc))),
                      ("TERTILES", lambda: (speed_state_tertile(d["spd_cal"], d["spd_cal"])[0],
                                             speed_state_tertile(d["spd_cal"], d["spd_test"])))]:
        vc, (vt, thr) = fn()
        print(f"\n--- speed states: {label}; thresholds (mph) = {np.round(thr, 2)}")
        print("  test class counts free/trans/cong:", [int((vt == g).sum()) for g in range(3)])
        print(f"  agreement with FD (diagonal): {(vt == gt).mean():.3f}")
        # joint crosstab counts (FD rows, speed cols)
        ct = np.array([[((gt == a) & (vt == b)).sum() for b in range(3)] for a in range(3)])
        print("  crosstab test (rows FD free/trans/cong, cols speed free/trans/cong):")
        for a in range(3):
            print("   ", NAMES[a].ljust(6), " ".join(f"{c:>9,}" for c in ct[a]))
        # cross-conditioning
        for cal_name, cal_c, cal_t in [("speed", vc, vt), ("FD", gc, gt)]:
            qt_, _ = mondrian(sc, cal_c, cal_t)
            ins = np.abs(yt - yht) <= qt_
            for ev_name, ev in [("FD", gt), ("speed", vt)]:
                covs = [ins[ev == g].mean() for g in range(3)]
                short = max(0.0, 0.90 - min(covs)) * 100
                print(f"  cal={cal_name:<5} eval={ev_name:<5} cov free/trans/cong = "
                      + " ".join(f"{c:.3f}" for c in covs) + f"  shortfall={short:.1f}pp"
                      + (f"  agg cov={ins.mean():.3f} width={2*qt_.mean():.2f}" if cal_name == ev_name else ""))


if __name__ == "__main__":
    for net in sys.argv[1:] or ["pems04", "pems08"]:
        report(net)
