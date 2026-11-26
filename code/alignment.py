"""Per-sensor alignment test and a check on the shape of the congested-state distributions.

Alignment: for each sensor, bin test observations on an occupancy axis, compute the
coverage of global split-CP intervals per bin, and define the sensor's breakdown point as
the first bin (in increasing occupancy) whose coverage falls below 0.75 with >= 30 obs.
Compare the dispersion of breakdown points (coefficient of variation, scale free) across
axes:
  raw        raw occupancy
  o/o_c      normalized by FD critical occupancy (the paper's claim)
  o/med      normalized by the sensor's median occupancy   (control: any per-sensor scale)
  o/p95      normalized by the sensor's 95th-pct occupancy (control)
Scale factors come from the training-free calibration split (o_c from the saved file).

Run: python code/alignment.py > results_v2/alignment.txt
"""
import sys
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import pred_file

ALPHA, THR, MIN_N, NB = 0.10, 0.75, 30, 40


def breakdown(x, ins, lo, hi):
    ed = np.linspace(lo, hi, NB + 1)
    b = np.clip(np.searchsorted(ed, x) - 1, 0, NB - 1)
    n = np.bincount(b, minlength=NB); c = np.bincount(b, weights=ins, minlength=NB)
    ok = n >= MIN_N
    cov = np.where(ok, c / np.maximum(n, 1), np.nan)
    below = np.where(ok & (cov < THR))[0]
    return 0.5 * (ed[below[0]] + ed[below[0] + 1]) if len(below) else np.nan


def cv(v):
    v = v[np.isfinite(v)]
    return np.std(v) / np.mean(v), len(v), np.median(v), np.percentile(v, [10, 90])


def modes(v, frac=0.10):
    v = v[np.isfinite(v)]
    if len(v) > 20000:
        v = np.random.default_rng(0).choice(v, 20000, replace=False)
    k = gaussian_kde(v)  # Scott's rule
    xs = np.linspace(v.min(), v.max(), 512); ds = k(xs)
    p, _ = find_peaks(ds, height=frac * ds.max())
    return len(p), np.round(xs[p], 1)


def run(net):
    d = dict(np.load(pred_file(net)))
    oc = d["o_c"]; N = len(oc)
    q = split_conformal_quantile(np.abs(d["y_cal"] - d["yhat_cal"]), ALPHA)
    yt, pt, ot, st = d["y_test"], d["yhat_test"], d["occ_test"], d["sid_test"]
    ins = (np.abs(yt - pt) <= q).astype(float)
    med = np.array([np.median(d["occ_cal"][d["sid_cal"] == s]) for s in range(N)])
    p95 = np.array([np.percentile(d["occ_cal"][d["sid_cal"] == s], 95) for s in range(N)])
    axes = {"raw": np.ones(N), "o/o_c": oc, "o/med": med, "o/p95": p95}
    print(f"\n=========== {net}: global pad {q:.2f}, breakdown = first bin with coverage < {THR} ===========")
    bp = {}
    for name, scale in axes.items():
        x = ot / scale[st]
        hi = np.percentile(x, 99.5)
        bp[name] = np.array([breakdown(x[st == s], ins[st == s], 0, hi) for s in range(N)])
        c, n, m, (p10, p90) = cv(bp[name])
        print(f"  {name:<6} sensors with a breakdown: {n:>3}/{N}  median={m:.3f}  10-90%=[{p10:.3f},{p90:.3f}]  CV={c:.3f}")
    both = np.isfinite(bp["raw"]) & np.isfinite(bp["o/o_c"])
    r, n = np.corrcoef(bp["raw"][both], oc[both])[0, 1], both.sum()
    print(f"  corr(raw breakdown occupancy, o_c) over {n} sensors = {r:.3f}")
    print(f"  CV ratio raw / (o/o_c) = {cv(bp['raw'])[0] / cv(bp['o/o_c'])[0]:.2f}x reduction")

    # --- modes of outcome vs residual, pooled vs within sensor, FD congested state
    g = fd_regime_from_pems(ot, oc[st])
    m = g == 2
    y, r_ = yt[m], (yt - pt)[m]
    z = np.empty_like(y)
    for s in np.unique(st[m]):
        k = st[m] == s
        z[k] = (y[k] - y[k].mean()) / (y[k].std() + 1e-9)
    print(f"\n  FD-congested test obs n={m.sum():,}; KDE peaks >= 10% of max (Scott bandwidth):")
    for lab, v in [("future speed, pooled mph", y), ("future speed, per-sensor z", z), ("residual y-yhat, pooled mph", r_)]:
        k, loc = modes(v); print(f"    {lab:<30} modes={k} at {loc}")
    counts = np.bincount(st[m], minlength=N)
    big = np.where(counts >= 300)[0]
    km = [modes(y[st[m] == s])[0] for s in big]
    kr = [modes(r_[st[m] == s])[0] for s in big]
    print(f"    within-sensor ({len(big)} sensors with >=300 congested obs): "
          f"future speed multimodal in {np.mean(np.array(km) >= 2):.0%}, residual multimodal in {np.mean(np.array(kr) >= 2):.0%}")


if __name__ == "__main__":
    for net in ["pems04", "pems08"]:
        run(net)
