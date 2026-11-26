"""Joint-state (FD x speed) calibration and the two-component mixture correction
(paper Sections 4.4 and 5.5). Two things differ from the submitted version:

  1. The speed partition g_V is the one matched to FD class proportions (the definition the
     paper states and Table 3 uses), so the joint table and the cross-conditioning table now agree.
  2. The mixture can be fit either on the same calibration data used for its threshold
     ("same", as in the paper) or on a disjoint first half of the calibration period with the
     threshold on the second half ("split"), which keeps the finite-sample guarantee.

Variants reported:
  joint             joint Mondrian, no mixture
  joint+mix(same)   mixture on outcomes y in flagged states, fit and calibrated on the same data
  joint+mix(split)  as above, fit on cal half A, threshold on cal half B
  joint+mix gated   split version, mixture allowed only in FD-congested states
  joint+rmix gated  mixture on residuals y - yhat instead of outcomes, so the set stays
                    centred on the forecast, split and gated

Run: python code/joint_mixture.py > results_v2/joint_mixture.txt
"""
import sys
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import pred_file

ALPHA = 0.10
N_MIN = 500          # calibration obs needed for a joint state to get its own pad
NAMES = ["free", "trans", "cong"]


def speed_state(spd, thr):
    g = np.zeros(len(spd), dtype=int); g[spd < thr[1]] = 1; g[spd < thr[0]] = 2
    return g


def n_modes(v, frac=0.10, max_n=20000):
    if len(v) > max_n:
        v = np.random.default_rng(0).choice(v, max_n, replace=False)
    k = gaussian_kde(v)  # Scott's rule
    xs = np.linspace(v.min(), v.max(), 512); ds = k(xs)
    return len(find_peaks(ds, height=frac * ds.max())[0])


def fit_gmm(v):
    gm = GaussianMixture(2, random_state=0).fit(v.reshape(-1, 1))
    return gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())


def mix_score(v, mu, sd):
    return np.min(np.abs(v[:, None] - mu[None, :]) / sd[None, :], axis=1)


def union_len(a1, b1, a2, b2):
    """Total length of [a1,b1] U [a2,b2] (elementwise)."""
    ov = np.clip(np.minimum(b1, b2) - np.maximum(a1, a2), 0, None)
    return (b1 - a1) + (b2 - a2) - ov


def run(net, n_min=N_MIN, verbose=True):
    d = dict(np.load(pred_file(net)))
    oc = d["o_c"]; N = len(oc)
    yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
    fc = fd_regime_from_pems(d["occ_cal"], oc[d["sid_cal"]])
    ft = fd_regime_from_pems(d["occ_test"], oc[d["sid_test"]])
    p_c, p_t = (fc == 2).mean(), (fc == 1).mean()
    thr = np.quantile(d["spd_cal"], [p_c, p_c + p_t])
    jc = 3 * fc + speed_state(d["spd_cal"], thr)
    jt = 3 * ft + speed_state(d["spd_test"], thr)
    sc = np.abs(yc - pc)
    Tc = len(yc) // N; half = (Tc // 2) * N
    inA = np.zeros(len(yc), bool); inA[:half] = True

    fd_pad = {g: split_conformal_quantile(sc[fc == g], ALPHA) for g in range(3)}
    base_pad, own = {}, {}
    for j in range(9):
        m = jc == j
        own[j] = m.sum() >= n_min
        base_pad[j] = split_conformal_quantile(sc[m], ALPHA) if own[j] else fd_pad[j // 3]

    def plain_sets(mask_states=()):
        lo = pt - np.array([base_pad[j] for j in range(9)])[jt]
        hi = pt + np.array([base_pad[j] for j in range(9)])[jt]
        return lo, hi

    results = {}
    lo, hi = plain_sets()
    results["joint"] = ((yt >= lo) & (yt <= hi), hi - lo, set())

    for label, split, gated, on_resid in [("joint+mix(same)", False, False, False),
                                          ("joint+mix(split)", True, False, False),
                                          ("joint+mix gated", True, True, False),
                                          ("joint+rmix gated", True, True, True)]:
        ins = (yt >= lo) & (yt <= hi); wid = hi - lo; fired = set()
        for j in range(9):
            if not own[j] or (gated and j // 3 != 2):
                continue
            m = jc == j
            fit_m = m & inA if split else m
            cal_m = m & ~inA if split else m
            vfit = (yc - pc)[fit_m] if on_resid else yc[fit_m]
            if fit_m.sum() < 50 or cal_m.sum() < 50 or n_modes(vfit) < 2:
                continue
            mu, sd = fit_gmm(vfit)
            vcal = (yc - pc)[cal_m] if on_resid else yc[cal_m]
            Q = split_conformal_quantile(mix_score(vcal, mu, sd), ALPHA)
            mt = jt == j
            c = pt[mt] if on_resid else 0.0
            a1, b1 = c + mu[0] - Q * sd[0], c + mu[0] + Q * sd[0]
            a2, b2 = c + mu[1] - Q * sd[1], c + mu[1] + Q * sd[1]
            y = yt[mt]
            ins[mt] = ((y >= a1) & (y <= b1)) | ((y >= a2) & (y <= b2))
            wid[mt] = union_len(a1, b1, a2, b2)
            fired.add(j)
        results[label] = (ins, wid, fired)

    if verbose:
        print(f"\n==================== {net} (n_min={n_min}) ====================")
        print(f"speed thresholds (matched to FD proportions): cong < {thr[0]:.1f} mph, trans < {thr[1]:.1f} mph")
        print("joint states with own pad:", [f"{NAMES[j//3]}/{NAMES[j%3]}" for j in range(9) if own[j]])
        print("fallback to FD parent pad:", [f"{NAMES[j//3]}/{NAMES[j%3]} (n_cal={int((jc==j).sum())})" for j in range(9) if not own[j]])
        for label, (ins, wid, fired) in results.items():
            print(f"\n  {label}: aggregate cov={ins.mean():.3f} width={wid.mean():.2f}; "
                  f"mixture fired in {sorted(f'{NAMES[j//3]}/{NAMES[j%3]}' for j in fired) or 'none'}")
            print("   FD-state cov:", " ".join(f"{NAMES[g]}={ins[ft == g].mean():.3f}" for g in range(3)),
                  "| width:", " ".join(f"{NAMES[g]}={wid[ft == g].mean():.1f}" for g in range(3)))
            print(f"   {'state (FD/speed)':<18}{'n_test':>8}{'cov':>7}{'width':>7}")
            for j in range(9):
                m = jt == j
                print(f"   {NAMES[j//3] + '/' + NAMES[j%3]:<18}{m.sum():>8,}{ins[m].mean():7.3f}{wid[m].mean():7.1f}")
    return results


if __name__ == "__main__":
    for net in ["pems04", "pems08"]:
        run(net)
    print("\n\n==== sensitivity of plain joint Mondrian to n_min ====")
    for net in ["pems04", "pems08"]:
        for nm in (200, 500, 1000, 2000):
            r = run(net, nm, verbose=False)["joint"]
            print(f"  {net} n_min={nm:>5}: cov={r[0].mean():.3f} width={r[1].mean():.2f}")
