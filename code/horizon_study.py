"""The same questions at 15, 30 and 60 minutes, with the STID forecaster.

For each dataset and horizon h (3, 6, 12 steps):
  - forecast error by regime and split-conformal coverage by regime, with the Laplace rule of thumb
  - CQR and CQR with a per-regime correction
  - how long misses stay correlated (the lag where the autocorrelation of the split-conformal
    miss indicator falls below 1/e) and the delay-to-memory ratio h / lag
  - adaptive conformal inference with feedback delayed by h steps, against the same method with
    one-step feedback (not available in practice) and against batch calibration

Run: python code/horizon_study.py > results_audit/horizon_study.txt
"""
import json
import os
import numpy as np
from pairs import load, label, winkler, cq, ALPHA
from online_delay_study import run_online, memory_lag

os.makedirs("results_audit", exist_ok=True)
out = []
for net in ("pems04", "pems08", "metr_la", "pems_bay"):
    for h in (3, 6, 12):
        f = f"predictions_stid_gpu/quantile_stid_{net}_h{h}.npz"
        if not os.path.exists(f):
            continue
        d = load(net, "stid", "predictions_stid_gpu", h)
        yc, pc, yt, pt = d["y_cal"], d["yhat_cal"], d["y_test"], d["yhat_test"]
        okc, okt, gc, gt = d["ok_cal"], d["ok_test"], d["g_cal"], d["g_test"]
        sa, sat = np.abs(yc - pc), np.abs(yt - pt)
        s, st = np.maximum(d["qlo_cal"] - yc, yc - d["qhi_cal"]), np.maximum(d["qlo_test"] - yt, yt - d["qhi_test"])
        q = cq(sa[okc], 1 - ALPHA); b = np.array([sa[okc & (gc == g)].mean() for g in range(3)])
        cov = lambda ins: [float(ins[okt & (gt == g)].mean()) for g in range(3)]
        row = dict(net=net, h=h, mae=float(sat[okt].mean()), mae_regime=[float(sat[okt & (gt == g)].mean()) for g in range(3)],
                   split_cov=cov(sat <= q), laplace=[float(v) for v in 1 - np.exp(-q / b)])
        e = cq(s[okc], 1 - ALPHA); eg = np.array([cq(s[okc & (gc == g)], 1 - ALPHA) for g in range(3)])
        wk = lambda cr: float(winkler(yt, d["qlo_test"] - cr, np.maximum(d["qhi_test"] + cr, d["qlo_test"] - cr))[okt].mean())
        row.update(cqr_cov=cov(st <= e), cqr_wink=wk(np.full(st.shape, e)), cqr_reg_cov=cov(st <= eg[gt]), cqr_reg_wink=wk(eg[gt]),
                   split_wink=float(winkler(yt, pt - q, pt + q)[okt].mean()))
        L, acf = memory_lag((sat > q).astype(float), okt); row.update(memory_lag=L, ratio=h / L, acf_at_h=float(acf[h - 1]))
        srt = [np.sort(sa[okc])]; cap = float(srt[0][-1])
        for name, delay in (("aci_delayed", h), ("aci_no_delay", 1)):
            best = None
            for g_ in (0.002, 0.005, 0.02, 0.05):
                pad = run_online(sat, okt, gt, srt, [q], "aci", g_, delay, False, 1.0)
                padc = np.clip(np.where(np.isneginf(pad), -cap, pad), -cap, cap)
                w = float(winkler(yt, pt - padc, np.maximum(pt + padc, pt - padc))[okt].mean())
                if best is None or w < best[0]:
                    best = (w, g_, cov(sat <= pad))
            row[name] = dict(wink=best[0], gamma=best[1], cov=best[2])
        qg = np.array([cq(sa[okc & (gc == g)], 1 - ALPHA) for g in range(3)])
        row["mondrian_wink"] = float(winkler(yt, pt - qg[gt], pt + qg[gt])[okt].mean())
        out.append(row)
        print(f"\n=== {label(net, 'stid')}, {5 * h} min: MAE {row['mae']:.2f} (" + " / ".join(f"{v:.2f}" for v in row["mae_regime"]) + ")")
        print("  split conformal coverage free/trans/cong " + " / ".join(f"{v:.3f}" for v in row["split_cov"]) + "   rule of thumb " + " / ".join(f"{v:.3f}" for v in row["laplace"]))
        print("  CQR coverage " + " / ".join(f"{v:.3f}" for v in row["cqr_cov"]) + f" (Winkler {row['cqr_wink']:.2f});  per regime " + " / ".join(f"{v:.3f}" for v in row["cqr_reg_cov"]) + f" ({row['cqr_reg_wink']:.2f})")
        print(f"  misses stay correlated for {L} steps; delay-to-memory ratio {h / L:.2f}; autocorrelation at the feedback delay {acf[h - 1]:.2f}")
        print(f"  absolute-residual score, Winkler: split {row['split_wink']:.2f}, Mondrian {row['mondrian_wink']:.2f}, ACI delayed {row['aci_delayed']['wink']:.2f} "
              f"(gamma {row['aci_delayed']['gamma']}), ACI with one-step feedback {row['aci_no_delay']['wink']:.2f} (gamma {row['aci_no_delay']['gamma']})", flush=True)
json.dump(out, open("results_audit/horizon_study.json", "w"), indent=1)
