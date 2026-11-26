"""Markdown tables for the audit experiments, from the JSON files in results_audit/.
Run: python code/make_audit_tables.py > results_audit/tables.md"""
import json
import os
from collections import defaultdict
from pairs import PAIRS, label, NICE

J = lambda name: json.load(open(f"results_audit/{name}.json"))
pairs = [(n, m) for n, m, _ in PAIRS]

# 1. Wang regimes
rows = J("wang_regime_eval")
print("### Table A. Coverage in transition windows (window-based regimes of Wang et al.), 60 minutes, target 0.90\n")
print("| Dataset, forecaster | Share of transition windows | Split conformal | CQR | CQR, per-regime correction | Historical baseline | Same, 30 most congestion-prone sensors: CQR, per-regime |")
print("|---|---|---|---|---|---|---|")
for n, m in pairs:
    a = {r["method"]: r for r in rows if r["net"] == n and r["model"] == m and r["subset"] == "all sensors"}
    b = {r["method"]: r for r in rows if r["net"] == n and r["model"] == m and r["subset"] != "all sensors"}
    c = lambda d, k: f"{d[k]['cov_window_regime'][1]:.3f}"
    print(f"| {label(n, m)} | {a['CQR']['shares'][1]:.3f} | {c(a, 'split conformal')} | {c(a, 'CQR')} | {c(a, 'CQR, per-regime correction')} | {c(a, 'historical baseline')} | {c(b, 'CQR, per-regime correction')} |")
print("\nFor reference, Wang et al. report 60-minute transition coverage of 0.68 to 0.80 for three foundation models on METR-LA and 0.55 to 0.72 on PEMS-BAY, rising to at most 0.85 and 0.77 with their corrections (their Table 2; 30 sensors per dataset; read from the arXiv HTML and not yet checked against the PDF).\n")

# 2. Outcome selection
tr = J("transition_risk_cqr")
print("### Table B. Calibrating on predicted transition risk\n")
print("| Dataset, forecaster | Classifier AUC | Transition-window coverage: CQR per origin regime | CQR per risk bin | Winkler: CQR per origin regime | CQR per risk bin |")
print("|---|---|---|---|---|---|")
for n, m in pairs:
    a = {r["correction"]: r for r in tr if r["net"] == n and r["model"] == m}
    print(f"| {label(n, m)} | {a['CQR']['auc']:.3f} | {a['CQR per origin regime']['cov_window'][1]:.3f} | {a['CQR per transition-risk bin']['cov_window'][1]:.3f} | {a['CQR per origin regime']['wink']:.2f} | {a['CQR per transition-risk bin']['wink']:.2f} |")
print("\n### Table C. Coverage inside risk deciles, split by what happened next (CQR per risk bin)\n")
for n, m in (("metr_la", "stid"), ("pems_bay", "stid")):
    r = next(x for x in tr if x["net"] == n and x["model"] == m and "by_bin" in x)
    print(f"**{label(n, m)}**\n")
    print("| Risk decile | " + " | ".join(str(b["bin"] + 1) for b in r["by_bin"]) + " |"); print("|---|" + "---|" * len(r["by_bin"]))
    f = lambda v: "n/a" if v is None else f"{v:.2f}"
    print("| Share of windows that were transitions | " + " | ".join(f"{b['share']:.2f}" for b in r["by_bin"]) + " |")
    print("| Coverage, all points | " + " | ".join(f"{b['cov']:.2f}" for b in r["by_bin"]) + " |")
    print("| Coverage, a transition followed | " + " | ".join(f(b["cov_trans"]) for b in r["by_bin"]) + " |")
    print("| Coverage, no transition followed | " + " | ".join(f(b["cov_other"]) for b in r["by_bin"]) + " |\n")

print("### Table C2. Coverage by predicted transition-risk decile: lowest decile value, and value in the highest-risk decile\n")
print("| Dataset, forecaster | Split conformal | Mondrian on origin regime | CQR | CQR per origin regime | CQR per risk bin |"); print("|---|---|---|---|---|---|")
for n, m in pairs:
    r = next(x for x in tr if x["net"] == n and x["model"] == m and "coverage_by_risk_decile" in x)["coverage_by_risk_decile"]
    f = lambda v: f"{min(v):.3f} / {v[-1]:.3f}"
    print(f"| {label(n, m)} | " + " | ".join(f(r[k]) for k in ("split conformal", "Mondrian on origin regime", "CQR", "CQR per origin regime", "CQR per transition-risk bin")) + " |")
print()

# 3. Online with delay
on = J("online_delay_study"); by = defaultdict(dict)
for r in on: by[(r["net"], r["model"], r["score"])][r["method"]] = r
print("### Table D. Online calibration with a 60-minute feedback delay (Winkler score)\n")
print("| Dataset, forecaster | Score | Minutes misses stay correlated | Delay-to-memory ratio | Split or CQR | Batch per regime | ACI, delayed (best of four step sizes) | Per-sensor per-regime tracker, delayed (best of three) | ACI, one-step feedback |")
print("|---|---|---|---|---|---|---|---|---|")
for n, m in pairs:
    for sc, nm in (("abs", "absolute residual"), ("cqr", "CQR")):
        d = by[(n, m, sc)]; b = d["batch"]
        aci = min(v["wink"] for k, v in d.items() if k.startswith("ACI") and "no delay" not in k)
        rqt = min(v["wink"] for k, v in d.items() if k.startswith("R-QT") and "no delay" not in k)
        print(f"| {label(n, m)} | {nm} | {5 * b['memory_lag']} | {12 / b['memory_lag']:.1f} | {b['wink']:.2f} | {d['batch per regime']['wink']:.2f} | {aci:.2f} | {rqt:.2f} | {d['ACI g=0.05 no delay']['wink']:.2f} |")
print("\nStep sizes were picked after seeing test results, which favours the online methods.\n")

# 4. Formula decomposition
fd = J("formula_decomposition")
print("### Table E. Where the rule of thumb's error comes from (split conformal coverage by regime)\n")
print("| Regime | Mean gap: Laplace rule | Common shape, no parametric form | Regime's own calibration distribution | Mean size of: drift | shape difference | Laplace misfit |")
print("|---|---|---|---|---|---|---|")
for reg in ("free", "transition", "congested"):
    rr = [x for x in fd if x["regime"] == reg]; mean = lambda f: sum(f(x) for x in rr) / len(rr)
    print(f"| {reg} | {mean(lambda x: abs(x['measured'] - x['laplace'])):.3f} | {mean(lambda x: abs(x['measured'] - x['common'])):.3f} | {mean(lambda x: abs(x['measured'] - x['own'])):.3f} | "
          f"{mean(lambda x: abs(x['drift'])):.3f} | {mean(lambda x: abs(x['shape'])):.3f} | {mean(lambda x: abs(x['parametric'])):.3f} |")
print("\n| Dataset, forecaster | Distance between congested and pooled residual shape (KS) | Shape term, congested | Laplace term, congested |")
print("|---|---|---|---|")
for n, m in pairs:
    x = next(r for r in fd if r["net"] == n and r["model"] == m and r["regime"] == "congested")
    print(f"| {label(n, m)} | {x['ks']:.3f} | {x['shape']:+.3f} | {x['parametric']:+.3f} |")

# 5. Error bars
bb = J("block_bootstrap"); q = lambda n, m, name: next(r for r in bb if r["net"] == n and r["model"] == m and r["quantity"] == name)
print("\n### Table F. Day-block bootstrap, 95% intervals (congested regime at the origin)\n")
print("| Dataset, forecaster | Test days | Split conformal | CQR | CQR, per-regime correction |"); print("|---|---|---|---|---|")
fmt = lambda r: f"{r['estimate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
for n, m in pairs:
    print(f"| {label(n, m)} | {q(n, m, 'CQR, congested coverage')['days']} | {fmt(q(n, m, 'split conformal, congested coverage'))} | {fmt(q(n, m, 'CQR, congested coverage'))} | {fmt(q(n, m, 'CQR per-regime, congested coverage'))} |")

# 6. Multi-group
mg = J("multigroup_study"); cs = ["pooled", "regime", "regime x hour", "sensor x regime", "multigroup"]
for key, title in (("min_regime_hour", "lowest coverage over regime x 4-hour cells"), ("sensor_min", "coverage of the worst sensor"), ("wink", "Winkler score")):
    print(f"\n### Table G ({title}), CQR score with five ways of computing the correction\n")
    print("| Dataset, forecaster | " + " | ".join(cs) + " |"); print("|---|" + "---|" * len(cs))
    for n, m in pairs:
        a = {r["correction"]: r for r in mg if r["net"] == n and r["model"] == m}
        print(f"| {label(n, m)} | " + " | ".join(f"{a[c][key]:.3f}" if key != "wink" else f"{a[c][key]:.2f}" for c in cs) + " |")

# 7. Diagnostic
er = J("ert_diagnostic"); ms = ["split conformal", "Mondrian on regime", "CQR", "CQR per regime", "CQR per sensor x regime"]
print("\n### Table H. Typical distance of conditional coverage from 0.90 (classifier-based diagnostic, lower is better)\n")
print("| Dataset, forecaster | " + " | ".join(ms) + " |"); print("|---|" + "---|" * len(ms))
for n, m in pairs:
    a = {r["method"]: r["gap"] for r in er if r["net"] == n and r["model"] == m}
    print(f"| {label(n, m)} | " + " | ".join(f"{a[k]:.3f}" for k in ms) + " |")

# 8. Horizons
if os.path.exists("results_audit/horizon_study.json"):
    hz = J("horizon_study")
    print("\n### Table I. STID at 15, 30 and 60 minutes\n")
    print("| Dataset | Horizon (min) | MAE | Split conformal congested coverage | CQR per-regime congested coverage | Minutes misses stay correlated | Delay-to-memory ratio | Winkler: Mondrian | ACI, delayed | ACI, one-step feedback |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in hz:
        print(f"| {NICE[r['net']]} | {5 * r['h']} | {r['mae']:.2f} | {r['split_cov'][2]:.3f} | {r['cqr_reg_cov'][2]:.3f} | {5 * r['memory_lag']} | {r['ratio']:.2f} | {r['mondrian_wink']:.2f} | {r['aci_delayed']['wink']:.2f} | {r['aci_no_delay']['wink']:.2f} |")

# 9. Breakdown prediction
if os.path.exists("results_audit/breakdown_prediction.json"):
    bp = J("breakdown_prediction")
    print("\n### Table J. Predicting a transition within the hour from a free-flow origin (PeMS04, PeMS08)\n")
    print("| Dataset | Features | With STID forecast | AUC | Average precision | Share caught at 5% false alarms |"); print("|---|---|---|---|---|---|")
    for r in bp:
        if r["target"] == "onset":
            print(f"| {NICE[r['net']]} | {r['features']} | {'yes' if r['forecast'] else 'no'} | {r['auc']:.3f} | {r['ap']:.3f} | {r['tpr5']:.3f} |")
