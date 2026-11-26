"""Two figures for the README.

  figures_v2/example_episode.png    one congestion episode at District 4 sensor 79 (test split):
                                    split conformal vs Mondrian conformal on the FD state
  figures_v2/coverage_by_state.png  coverage of the same two methods in each FD state,
                                    both networks

Both only need the saved predictions in predictions_v2/.

Run: python code/make_readme_figures.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, "code")
from conformal import split_conformal_quantile, fd_regime_from_pems
from paths import pred_file, FIG_DIR

BLUE, ORANGE, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#c0392b", "#0b0b0b", "#52514e", "#e4e3df"
ALPHA = 0.10
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6})


def load(net):
    d = dict(np.load(pred_file(net)))
    oc = d["o_c"]
    gc = fd_regime_from_pems(d["occ_cal"], oc[d["sid_cal"]])
    gt = fd_regime_from_pems(d["occ_test"], oc[d["sid_test"]])
    s = np.abs(d["y_cal"] - d["yhat_cal"])
    q = split_conformal_quantile(s, ALPHA)
    qm = np.array([split_conformal_quantile(s[gc == k], ALPHA) for k in range(3)])
    return d, gt, q, qm


def fig_example(sensor=79, start=313, stop=519):
    d, gt, q, qm = load("pems04")
    N = len(d["o_c"])
    idx = np.arange(start, stop) * N + sensor
    y, yh, g = d["y_test"][idx], d["yhat_test"][idx], gt[idx]
    hrs = np.arange(len(idx)) / 12
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, sharey=True, constrained_layout=True)
    panels = [(np.full_like(yh, q), BLUE, "Split conformal: one pad for every point"),
              (qm[g], ORANGE, "Mondrian conformal on the FD state: one pad per traffic state")]
    for a, (pad, col, name) in zip(ax, panels):
        lo, hi = yh - pad, yh + pad
        miss = (y < lo) | (y > hi)
        for t in np.where(g == 2)[0]:
            a.axvspan(hrs[t] - 1 / 24, hrs[t] + 1 / 24, color=MUTED, alpha=0.10, lw=0)
        a.fill_between(hrs, lo, hi, color=col, alpha=0.22, lw=0)
        a.plot(hrs, yh, color=col, lw=1.3)
        a.plot(hrs, y, color=INK, lw=1.3)
        a.scatter(hrs[miss], y[miss], s=13, color=RED, zorder=5)
        a.set_title(f"{name}.  Coverage in this window: {1 - miss.mean():.0%}", loc="left", fontsize=10)
        a.set_ylabel("speed (mph)")
        a.set_ylim(0, 90)
        a.legend(handles=[Patch(color=col, alpha=0.3, label="90% interval"),
                          plt.Line2D([], [], color=col, lw=1.3, label="forecast (60 min ahead)"),
                          plt.Line2D([], [], color=INK, lw=1.3, label="actual speed"),
                          plt.Line2D([], [], color=RED, marker="o", ls="", ms=4, label="outside the interval"),
                          Patch(color=MUTED, alpha=0.15, label="forecast made in congestion")],
                 loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, frameon=False)
    ax[1].set_xlabel("time (hours)")
    fig.savefig(f"{FIG_DIR}/example_episode.png", dpi=150)
    plt.close(fig)


def fig_coverage_by_state():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, constrained_layout=True)
    names = ["free flow", "transition", "congested"]
    x = np.arange(3)
    for a, (net, lab) in zip(ax, [("pems04", "District 4"), ("pems08", "District 8")]):
        d, gt, q, qm = load(net)
        r = np.abs(d["y_test"] - d["yhat_test"])
        split = [np.mean(r[gt == k] <= q) for k in range(3)]
        mond = [np.mean(r[gt == k] <= qm[k]) for k in range(3)]
        for off, vals, col, name in [(-0.2, split, BLUE, "split conformal"),
                                     (0.2, mond, ORANGE, "Mondrian on FD state")]:
            b = a.bar(x + off, vals, 0.38, color=col, label=name)
            a.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
        a.axhline(1 - ALPHA, color=INK, lw=0.8, ls="--")
        a.set_xticks(x, names)
        a.set_ylim(0, 1.08)
        a.set_title(lab, loc="left", fontsize=10)
        print(lab, "split", np.round(split, 3), "mondrian", np.round(mond, 3))
    ax[0].set_ylabel("coverage (target 0.90, dashed)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, fontsize=9, frameon=False)
    fig.savefig(f"{FIG_DIR}/coverage_by_state.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_example()
    fig_coverage_by_state()
