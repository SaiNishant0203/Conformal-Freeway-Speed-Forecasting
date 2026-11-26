"""The eight dataset-forecaster pairs and a common loader for the audit scripts.

Each pair gives point forecasts and 5%/95% quantile forecasts at a 60-minute horizon on the
calibration and test splits, as [steps, sensors] arrays, with the raw speed series alongside.
"""
import sys
import numpy as np
sys.path.insert(0, "code")
from train_stid import load_raw

H, DAY, ALPHA = 12, 288, 0.10
PAIRS = [("pems04", "gru", "predictions_v2"), ("pems04", "stid", "predictions_stid_gpu"),
         ("pems08", "gru", "predictions_v2"), ("pems08", "stid", "predictions_stid_gpu"),
         ("metr_la", "gru", "predictions_gru_gpu"), ("metr_la", "stid", "predictions_stid_gpu"),
         ("pems_bay", "gru", "predictions_gru_gpu"), ("pems_bay", "stid", "predictions_stid_gpu")]
NICE = {"pems04": "PeMS04", "pems08": "PeMS08", "metr_la": "METR-LA", "pems_bay": "PEMS-BAY", "gru": "GRU", "stid": "STID"}


def label(net, model):
    return f"{NICE[net]} {NICE[model]}"


def origin_regime(spd):
    """0 free (>= 55 mph), 1 transition, 2 congested (< 25 mph), from the speed at the forecast origin."""
    g = np.ones(spd.shape, dtype=np.int8); g[spd >= 55] = 0; g[spd < 25] = 2
    return g


def window_regime(speed, t0, t1, h=H):
    """Regime of the forecast window (t, t+H], as in Wang et al. (arXiv 2606.18367): free if every
    reading in the window is above 55 mph, congested if every reading is below 25 mph, transition
    otherwise. Missing readings (0) are ignored; -1 if the whole window is missing."""
    win = np.stack([speed[t0 + k:t1 + k] for k in range(1, h + 1)], 0)          # [h, steps, N]
    ok = win > 0
    lo = np.where(ok, win, np.inf).min(0); hi = np.where(ok, win, -np.inf).max(0)
    g = np.ones(lo.shape, dtype=np.int8); g[lo > 55] = 0; g[hi < 25] = 2; g[~ok.any(0)] = -1
    return g


def load(net, model, folder, h=H):
    d = np.load(f"{folder}/predictions_{model}_{net}_h{h}.npz"); q = np.load(f"{folder}/quantile_{model}_{net}_h{h}.npz")
    speed, _, _, _, tod, dow = load_raw(net)
    T, N = speed.shape; tr, ca = int(0.6 * T), int(0.8 * T)
    out = dict(net=net, model=model, N=N, T=T, tr=tr, ca=ca, speed=speed, tod=tod, dow=dow, h=h)
    for s, (a, b) in (("cal", (tr, ca - h)), ("test", (ca, T - h))):
        sh = (b - a, N)
        for k, src in (("y", d), ("yhat", d), ("spd", d), ("qlo", q), ("qhi", q)):
            out[f"{k}_{s}"] = src[f"{k}_{s}"].reshape(sh)
        out[f"ok_{s}"] = (out[f"y_{s}"] > 0) & (out[f"spd_{s}"] > 0)
        out[f"g_{s}"] = origin_regime(out[f"spd_{s}"])
        out[f"t_{s}"] = np.arange(a, b)
    assert np.array_equal(out["spd_test"], speed[ca:T - h]) and np.array_equal(out["y_test"], speed[ca + h:T])
    return out


def winkler(y, lo, hi, alpha=ALPHA):
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


def cq(scores, level):
    """Conformal quantile: the ceil((n+1) * level)-th smallest score."""
    s = np.sort(np.asarray(scores).ravel()); n = len(s)
    return float(s[min(int(np.ceil((n + 1) * level)), n) - 1])
