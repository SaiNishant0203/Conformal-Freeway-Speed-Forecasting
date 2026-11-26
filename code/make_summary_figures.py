"""Summary figures across datasets and forecasters, built from the JSON files written by
speed_regime_study.py (speed regimes: free >= 55 mph, congested < 25 mph at the forecast origin).

  figures_summary/winkler_by_method.png   interval score by method, weak (GRU) vs strong (STID)
                                          forecaster, one panel per dataset
  figures_summary/formula_vs_measured.png coverage of the single split-conformal pad in each
                                          regime: closed-form prediction against measurement

Run: python code/make_summary_figures.py
"""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
NETS = [("pems04", "PeMS04"), ("pems08", "PeMS08"), ("metr_la", "METR-LA"), ("pems_bay", "PEMS-BAY")]
METHODS = [("split CP", "Split"), ("Mondrian", "Mondrian"), ("normalized CP", "Normalized"), ("CQR", "CQR"),
           ("CQR + Mondrian", "CQR + Mondrian"), ("rq + Mondrian", "Two-stage")]

data = {}
for f in glob.glob("results_*/speed_regimes_*_*.json"):
    d = json.load(open(f)); data[(d["net"], d["model"])] = d
os.makedirs("figures_summary", exist_ok=True)

nets = [(k, lab) for k, lab in NETS if (k, "gru") in data or (k, "stid") in data]
fig, axes = plt.subplots(1, len(nets), figsize=(3.3 * len(nets), 3.4), constrained_layout=True, squeeze=False)
for ax, (net, lab) in zip(axes[0], nets):
    x = np.arange(len(METHODS)); w = 0.38
    for j, (model, col, name) in enumerate([("gru", BLUE, "GRU (weak)"), ("stid", ORANGE, "STID (strong)")]):
        if (net, model) not in data:
            continue
        by = {r["method"]: r["wink"] for r in data[(net, model)]["rows"]}
        v = [by.get(m, np.nan) for m, _ in METHODS]
        ax.bar(x + (j - 0.5) * (w + 0.03), v, w, color=col, label=f"{name}, MAE {data[(net, model)]['mae']:.2f}")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in METHODS], fontsize=7.5, rotation=30, ha="right")
    ax.set_title(lab, fontsize=9, loc="left"); ax.set_ylabel("Winkler score (lower is better)" if ax is axes[0][0] else "")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=7)
fig.savefig("figures_summary/winkler_by_method.png", dpi=200)

fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
pts = []
for (net, model), d in sorted(data.items()):
    meas = next(r for r in d["rows"] if r["method"] == "split CP")["cov_regime"]
    for g in range(3):
        pts.append((d["formula_mixture"][g], meas[g], g))
pts = np.array(pts)
for g, col, name in [(0, BLUE, "free flow"), (1, AQUA, "transition"), (2, ORANGE, "congested")]:
    m = pts[:, 2] == g
    ax.scatter(pts[m, 0], pts[m, 1], s=42, color=col, edgecolor="white", linewidth=1.2, label=name, zorder=3)
ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls="--")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("coverage predicted from calibration errors"); ax.set_ylabel("coverage measured on the test split")
gap = np.abs(pts[:, 0] - pts[:, 1])
ax.set_title(f"Split conformal coverage by regime, {len(data)} dataset-forecaster pairs\n"
             f"mean absolute gap {gap.mean():.3f}", fontsize=8.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig("figures_summary/formula_vs_measured.png", dpi=200)
print("pairs:", sorted(data.keys())); print("mean abs gap by regime:", [round(float(gap[pts[:, 2] == g].mean()), 3) for g in range(3)])
