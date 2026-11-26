"""Figures.

  figures_v2/fd_scatter.png           flow-occupancy scatter for one District 4 sensor, coloured by
                                     FD state, with the 90th-percentile flow envelope and o_c
  figures_v2/coverage_vs_occupancy.png  pooled coverage of global split-CP intervals against
                                     occupancy normalized by each sensor's o_c
  figures_v2/alignment_per_sensor.png   per-sensor coverage curves on raw vs o/o_c axes, and the
                                     scale-free spread of per-sensor breakdown points
  figures_v2/congested_residuals.png    residual (y - yhat) density in the FD-congested state

fd_scatter needs the raw data in Dataset/; the others only need the saved predictions.

Run: python code/make_figures.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems, estimate_critical_occupancy
import os
from paths import pred_file, FIG_DIR

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
THR, MIN_N, NB = 0.75, 30, 40
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6})


def sensor_curves(x, ins, st, N, hi):
    ed = np.linspace(0, hi, NB + 1); ctr = 0.5 * (ed[:-1] + ed[1:])
    curves, bps = [], np.full(N, np.nan)
    for s in range(N):
        m = st == s
        b = np.clip(np.searchsorted(ed, x[m]) - 1, 0, NB - 1)
        n = np.bincount(b, minlength=NB); c = np.bincount(b, weights=ins[m], minlength=NB)
        cov = np.where(n >= MIN_N, c / np.maximum(n, 1), np.nan)
        below = np.where(cov < THR)[0]
        if len(below):
            bps[s] = ctr[below[0]]
        curves.append(cov)
    return ctr, np.array(curves), bps


def load(net):
    d = dict(np.load(pred_file(net)))
    q = split_conformal_quantile(np.abs(d["y_cal"] - d["yhat_cal"]), 0.10)
    ins = (np.abs(d["y_test"] - d["yhat_test"]) <= q).astype(float)
    return d, ins


def fig_alignment():
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
    for r, (net, lab) in enumerate([("pems04", "District 4"), ("pems08", "District 8")]):
        d, ins = load(net); oc = d["o_c"]; N = len(oc); st, ot = d["sid_test"], d["occ_test"]
        rel = {}
        for c, (scale, xl) in enumerate([(np.ones(N), "raw occupancy"), (oc, "normalized occupancy o / o_c")]):
            x = ot / scale[st]; hi = np.percentile(x, 99.5)
            ctr, curves, bps = sensor_curves(x, ins, st, N, hi)
            a = ax[r, c]
            has = np.isfinite(bps)
            for cv in curves[has]:
                a.plot(ctr, cv, color=MUTED, lw=0.4, alpha=0.25)
            a.plot(ctr, np.nanmedian(curves[has], axis=0), color=BLUE if c == 0 else ORANGE, lw=2,
                   label="median over sensors")
            a.axhline(0.90, color=INK, lw=0.8, ls="--")
            if c == 1:
                a.axvline(1.0, color=INK, lw=0.8, ls=":")
            a.set_ylim(0, 1.02); a.set_xlim(0, hi)
            a.set_xlabel(xl); a.set_ylabel("coverage" if c == 0 else "")
            a.set_title(f"{lab}: per-sensor coverage vs {xl.split(' o')[0]}", fontsize=9, loc="left")
            rel[c] = bps[has] / np.median(bps[has])
        a = ax[r, 2]
        bins = np.linspace(0, 3, 31)
        for c, col, name in [(0, BLUE, "raw"), (1, ORANGE, "o / o_c")]:
            v = rel[c]; cvv = v.std() / v.mean()
            a.hist(v, bins=bins, histtype="step", lw=2, color=col, label=f"{name} (CV = {cvv:.2f})")
        a.set_xlabel("breakdown point / its median"); a.set_ylabel("sensors")
        a.set_title(f"{lab}: spread of breakdown points", fontsize=9, loc="left")
        a.legend(frameon=False, fontsize=8)
    fig.savefig(f"{FIG_DIR}/alignment_per_sensor.png", dpi=200)


def fig_resid():
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2), constrained_layout=True)
    for net, lab, col in [("pems04", "District 4", BLUE), ("pems08", "District 8", ORANGE)]:
        d = dict(np.load(pred_file(net)))
        g = fd_regime_from_pems(d["occ_test"], d["o_c"][d["sid_test"]])
        r = (d["y_test"] - d["yhat_test"])[g == 2]
        v = np.random.default_rng(0).choice(r, min(20000, len(r)), replace=False)
        xs = np.linspace(-45, 45, 400)
        ax.plot(xs, gaussian_kde(v)(xs), color=col, lw=2, label=f"{lab} (n = {len(r):,})")
    ax.axvline(0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("residual y - yhat (mph), FD-congested state, 60-min horizon")
    ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{FIG_DIR}/congested_residuals.png", dpi=200)


def fig_fd_scatter(net="pems04", sensor=5, n_bins=40):
    raw = np.load(f"Dataset/{net}.npz")["data"]
    tr = int(0.6 * raw.shape[0])
    flow, occ = raw[:tr, sensor, 0], raw[:tr, sensor, 1]
    oc = estimate_critical_occupancy(flow, occ, n_bins)
    g = fd_regime_from_pems(occ, oc)
    lo, hi = np.nanpercentile(occ, [1, 99]); ed = np.linspace(lo, hi, n_bins + 1)
    ctr = 0.5 * (ed[:-1] + ed[1:]); env = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (occ >= ed[b]) & (occ < ed[b + 1])
        if m.sum() >= 20:
            env[b] = np.quantile(flow[m], 0.90)
    fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
    for k, col, lab in [(0, BLUE, "free flow"), (1, AQUA, "transition"), (2, ORANGE, "congested")]:
        ax.scatter(occ[g == k], flow[g == k], s=3, color=col, alpha=0.35, linewidths=0, label=lab)
    ax.plot(ctr, env, color=INK, lw=1.5, label="90th-pct flow envelope")
    ax.scatter([oc], [np.nanmax(env)], marker="D", s=40, color=INK, zorder=3)
    ax.axvline(oc, color=INK, lw=0.8, ls="--")
    ax.set_xlim(0, hi * 1.05)
    ax.set_xlabel("occupancy"); ax.set_ylabel("flow (veh / 5 min)")
    ax.set_title(f"District 4, sensor {sensor}, training split (o_c = {oc:.3f})", fontsize=9, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.savefig(f"{FIG_DIR}/fd_scatter.png", dpi=200)


def fig_coverage_vs_occupancy():
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ed = np.linspace(0, 2.5, 26); ctr = 0.5 * (ed[:-1] + ed[1:])
    for net, lab, col in [("pems04", "District 4", BLUE), ("pems08", "District 8", ORANGE)]:
        d, ins = load(net)
        x = d["occ_test"] / d["o_c"][d["sid_test"]]
        b = np.searchsorted(ed, x) - 1; ok = (b >= 0) & (b < len(ctr))
        n = np.bincount(b[ok], minlength=len(ctr)); c = np.bincount(b[ok], weights=ins[ok], minlength=len(ctr))
        ax.plot(ctr[n >= 200], (c / np.maximum(n, 1))[n >= 200], color=col, lw=2, marker="o", ms=3, label=lab)
    ax.axhline(0.90, color=INK, lw=0.8, ls="--"); ax.axvline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_ylim(0, 1.02); ax.set_xlabel("normalized occupancy o / o_c (pooled over sensors)")
    ax.set_ylabel("coverage"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{FIG_DIR}/coverage_vs_occupancy.png", dpi=200)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_fd_scatter(); fig_coverage_vs_occupancy(); fig_alignment(); fig_resid()
