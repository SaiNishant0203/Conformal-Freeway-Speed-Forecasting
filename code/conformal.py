
import numpy as np

def split_conformal_quantile(scores, alpha):
    n = len(scores)
    if n == 0: raise ValueError("empty calibration group")
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])

def abs_residual(y, yhat): return np.abs(y - yhat)

class SplitConformal:
    def __init__(self, alpha=0.10): self.alpha, self.q_ = alpha, None
    def calibrate(self, y, yhat):
        self.q_ = split_conformal_quantile(abs_residual(y, yhat), self.alpha); return self
    def interval(self, yhat): return yhat - self.q_, yhat + self.q_

class MondrianConformal:
    def __init__(self, alpha=0.10): self.alpha, self.q_ = alpha, {}
    def calibrate(self, y, yhat, groups):
        s = abs_residual(y, yhat)
        for g in np.unique(groups):
            self.q_[int(g)] = split_conformal_quantile(s[groups == g], self.alpha)
        return self
    def interval(self, yhat, groups):
        q = np.array([self.q_[int(g)] for g in groups]); return yhat - q, yhat + q

def coverage_report(y, lo, hi, groups, names=None):
    inside, width = (y >= lo) & (y <= hi), hi - lo
    names = names or {g: str(g) for g in np.unique(groups)}
    print(f"{'regime':<12}{'n':>8}{'coverage':>11}{'width':>10}")
    for g in np.unique(groups):
        m = groups == g
        print(f"{names[int(g)]:<12}{m.sum():>8}{inside[m].mean():>10.1%}{width[m].mean():>10.2f}")
    print(f"{'MARGINAL':<12}{len(y):>8}{inside.mean():>10.1%}{width.mean():>10.2f}")

def estimate_critical_occupancy(flow, occ, n_bins=40):
    lo, hi = np.nanpercentile(occ, [1, 99]); ed = np.linspace(lo, hi, n_bins + 1)
    c = 0.5 * (ed[:-1] + ed[1:]); bf = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (occ >= ed[b]) & (occ < ed[b + 1])
        if m.sum() >= 20: bf[b] = np.nanquantile(flow[m], 0.90)
    return float(c[np.nanargmax(bf)]) if not np.all(np.isnan(bf)) else float(np.nanmedian(occ))

def fd_regime_from_pems(occ, o_c, band=0.15):
    lo, hi = o_c * (1 - band), o_c * (1 + band)
    r = np.ones_like(occ, dtype=int); r[occ < lo] = 0; r[occ > hi] = 2; return r
