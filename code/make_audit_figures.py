"""Figures for the audit experiments, from the JSON files in results_audit/.

  figures_audit/online_delay.png      interval score of batch and online calibration with and
                                      without the feedback delay, eight dataset-forecaster pairs
  figures_audit/memory_vs_horizon.png how long misses stay correlated, against the forecast horizon
  figures_audit/outcome_selection.png coverage inside transition-risk bins, split by whether a
                                      transition actually followed

Run: python code/make_audit_figures.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pairs import PAIRS, label, NICE

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
os.makedirs("figures_audit", exist_ok=True)

# --- online calibration with and without the feedback delay (absolute-residual score)
rows = json.load(open("results_audit/online_delay_study.json"))
fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
x = np.arange(len(PAIRS)); w = 0.2
series = [("Split conformal", BLUE, lambda m: m["batch"]["wink"]),
          ("Batch, per regime", AQUA, lambda m: m["batch per regime"]["wink"]),
          ("ACI, 60-min feedback delay (best setting)", ORANGE, lambda m: min(v["wink"] for k, v in m.items() if k.startswith("ACI") and "no delay" not in k)),
          ("ACI, one-step feedback (not available in practice)", YELLOW, lambda m: m["ACI g=0.05 no delay"]["wink"])]
for j, (name, col, fn) in enumerate(series):
    vals = []
    for net, model, _ in PAIRS:
        m = {r["method"]: r for r in rows if r["net"] == net and r["model"] == model and r["score"] == "abs"}
        vals.append(fn(m))
    ax.bar(x + (j - 1.5) * (w + 0.02), vals, w, color=col, label=name)
ax.set_xticks(x); ax.set_xticklabels([label(n, m) for n, m, _ in PAIRS], fontsize=7.5, rotation=20, ha="right")
ax.set_ylabel("Winkler score (lower is better)"); ax.grid(axis="x", visible=False)
ax.set_ylim(0, ax.get_ylim()[1] * 1.22); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
ax.set_title("Online calibration at a 60-minute horizon: the feedback delay removes the benefit", fontsize=9, loc="left")
fig.savefig("figures_audit/online_delay.png", dpi=200)

# --- memory of misses against horizon
fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
cols = {"pems04": BLUE, "pems08": ORANGE, "metr_la": AQUA, "pems_bay": YELLOW}
if os.path.exists("results_audit/horizon_study.json"):
    hz = json.load(open("results_audit/horizon_study.json"))
    for net in cols:
        pts = sorted([(r["h"], r["memory_lag"]) for r in hz if r["net"] == net])
        if pts:
            ax.plot([5 * p[0] for p in pts], [5 * p[1] for p in pts], marker="o", ms=5, lw=2, color=cols[net], label=f"{NICE[net]} STID")
ax.plot([0, 65], [0, 65], color=INK, lw=0.8, ls="--"); ax.text(47, 52, "memory = delay", fontsize=7.5, color=MUTED, rotation=38)
ax.set_xlim(0, 65); ax.set_ylim(0, 65); ax.set_xlabel("forecast horizon = feedback delay (minutes)")
ax.set_ylabel("time for which misses stay correlated (minutes)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Feedback arrives after the\ncorrelation in misses has faded", fontsize=9, loc="left")
fig.savefig("figures_audit/memory_vs_horizon.png", dpi=200)

# --- outcome selection inside risk bins
tr = json.load(open("results_audit/transition_risk_cqr.json"))
show = [("metr_la", "stid"), ("pems_bay", "stid")]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True, sharey=True)
for ax, (net, model) in zip(axes, show):
    r = next(x for x in tr if x["net"] == net and x["model"] == model and "by_bin" in x)
    b = r["by_bin"]; k = np.arange(1, len(b) + 1)
    f = lambda key: np.array([np.nan if v[key] is None else v[key] for v in b], float)
    ax.plot(k, f("cov"), color=INK, lw=2, marker="o", ms=4, label="all points in the bin")
    ax.plot(k, f("cov_other"), color=BLUE, lw=2, marker="o", ms=4, label="no transition followed")
    ax.plot(k, f("cov_trans"), color=ORANGE, lw=2, marker="o", ms=4, label="a transition followed")
    ax.axhline(0.90, color=MUTED, lw=0.8, ls="--"); ax.set_ylim(0, 1.02); ax.set_xticks(k)
    ax.set_xlabel("predicted transition risk (decile, low to high)"); ax.set_title(label(net, model), fontsize=9, loc="left")
axes[0].set_ylabel("coverage"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig("figures_audit/outcome_selection.png", dpi=200)
print("figures written")
