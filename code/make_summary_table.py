"""Markdown tables across datasets and forecasters, from the JSON files written by
speed_regime_study.py. Run: python code/make_summary_table.py > results_summary/tables.md"""
import glob
import json

NETS = [("pems04", "PeMS04"), ("pems08", "PeMS08"), ("metr_la", "METR-LA"), ("pems_bay", "PEMS-BAY")]
MODELS = [("gru", "GRU"), ("stid", "STID")]
METHODS = ["split CP", "Mondrian", "normalized CP", "CQR", "CQR + Mondrian", "refined CQR+Mond.", "rq + Mondrian", "climatology"]
NICE = {"split CP": "Split conformal", "Mondrian": "Mondrian on regime", "normalized CP": "Normalized conformal", "CQR": "CQR",
        "CQR + Mondrian": "CQR, per-regime correction", "refined CQR+Mond.": "Refined CQR, per-regime correction",
        "rq + Mondrian": "Two-stage on point forecast", "climatology": "Climatology (no forecast)"}
data = {}
for f in glob.glob("results_*/speed_regimes_*_*.json"):
    d = json.load(open(f)); data[(d["net"], d["model"])] = d
pairs = [(n, m) for n, _ in NETS for m, _ in MODELS if (n, m) in data]
lab = lambda n, m: f"{dict(NETS)[n]} {dict(MODELS)[m]}"

print("Speed regimes at the forecast origin: free flow at 55 mph or more, congested below 25 mph. 60-minute horizon, target coverage 0.90.\n")
print("**Test error and regime shares**\n")
print("| Dataset, forecaster | Test points | MAE (mph) | MAE free / transition / congested | Share free / transition / congested |")
print("|---|---|---|---|---|")
for n, m in pairs:
    d = data[(n, m)]
    print(f"| {lab(n, m)} | {d['n_test']:,} | {d['mae']:.2f} | " + " / ".join(f"{v:.2f}" for v in d["mae_regime"]) + " | " + " / ".join(f"{v:.3f}" for v in d["shares"]) + " |")

print("\n**Winkler interval score (lower is better)**\n")
print("| Method | " + " | ".join(lab(n, m) for n, m in pairs) + " |"); print("|---|" + "---|" * len(pairs))
for meth in METHODS:
    row = []
    for n, m in pairs:
        r = next((r for r in data[(n, m)]["rows"] if r["method"] == meth), None)
        best = min(x["wink"] for x in data[(n, m)]["rows"])
        row.append("n/a" if r is None else (f"**{r['wink']:.2f}**" if abs(r["wink"] - best) < 1e-9 else f"{r['wink']:.2f}"))
    print(f"| {NICE[meth]} | " + " | ".join(row) + " |")

print("\n**Coverage in the congested regime (target 0.90)**\n")
print("| Method | " + " | ".join(lab(n, m) for n, m in pairs) + " |"); print("|---|" + "---|" * len(pairs))
for meth in METHODS:
    row = []
    for n, m in pairs:
        r = next((r for r in data[(n, m)]["rows"] if r["method"] == meth), None)
        row.append("n/a" if r is None else f"{r['cov_regime'][2]:.3f}")
    print(f"| {NICE[meth]} | " + " | ".join(row) + " |")

print("\n**Split conformal coverage by regime: measured against the formula (free / transition / congested)**\n")
print("| Dataset, forecaster | Measured | Formula | Rule of thumb |"); print("|---|---|---|---|")
gaps = []
for n, m in pairs:
    d = data[(n, m)]; meas = next(r for r in d["rows"] if r["method"] == "split CP")["cov_regime"]
    gaps += [abs(a - b) for a, b in zip(meas, d["formula_mixture"])]
    f3 = lambda v: " / ".join(f"{x:.3f}" for x in v)
    print(f"| {lab(n, m)} | {f3(meas)} | {f3(d['formula_mixture'])} | {f3(d['formula_rule'])} |")
print(f"\nMean absolute gap between measured and mixture formula over {len(gaps)} regime values: {sum(gaps) / len(gaps):.3f}")
