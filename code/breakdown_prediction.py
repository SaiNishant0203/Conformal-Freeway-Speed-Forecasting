"""Does the fundamental diagram help to foresee a regime change within the hour?

Traffic-flow theory says a breakdown is preceded by occupancy rising toward its critical value
while speed is still high. If that is visible in the data, occupancy relative to each sensor's
critical occupancy should predict "the coming hour contains a transition" better than speed
alone, most of all for points that are still in free flow.

Targets (window regime as in Wang et al., origin regime by speed):
  onset      origin in free flow, and the coming window is a transition
  any        any origin, and the coming window is a transition
Feature sets (as in fd_informed.py), each with and without the STID forecast:
  A  speed history and time
  B  A + raw occupancy and flow and their trends
  C  B + fundamental-diagram features (occupancy / critical occupancy, flow / capacity, time
     spent beyond critical occupancy, speed / free-flow speed)
Classifier: gradient boosting, trained on the calibration period, scored on the test period.
Scores: area under the ROC curve, average precision, and the share of transitions caught when
5% of non-transition points are flagged.

Run: PRED_DIR=predictions_stid_gpu MODEL=stid python code/breakdown_prediction.py > results_audit/breakdown_prediction.txt
"""
import json
import os
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
sys.path.insert(0, "code")
from fd_informed import build
from pairs import window_regime, origin_regime, H
from train_stid import load_raw

os.makedirs("results_audit", exist_ok=True)
out = []; rng = np.random.default_rng(0)
for net in ("pems04", "pems08"):
    d, Xc, Xt, N = build(net)
    speed = load_raw(net)[0]; T = speed.shape[0]; tr, ca = int(0.6 * T), int(0.8 * T)
    wc, wt = window_regime(speed, tr, ca - H).ravel(), window_regime(speed, ca, T - H).ravel()
    gc, gt = origin_regime(d["spd_cal"]), origin_regime(d["spd_test"])
    sets = {"A speed": list(range(10)), "B + occupancy, flow": list(range(14)), "C + FD": list(range(21))}
    print(f"\n==================== {net} ====================")
    print(f"{'target':<7}{'features':<22}{'forecast':>9} | {'base rate':>9}{'AUC':>7}{'avg prec':>9}{'caught at 5% false alarms':>27}")
    for target, mc, mt in (("onset", gc == 0, gt == 0), ("any", np.ones(len(gc), bool), np.ones(len(gt), bool))):
        yc_, yt_ = (wc[mc] == 1).astype(int), (wt[mt] == 1).astype(int)
        idx = np.flatnonzero(mc); sub = rng.choice(len(idx), min(500000, len(idx)), replace=False)
        for name, cols in sets.items():
            for with_fc in (False, True):
                cc = [c for c in cols if with_fc or c != 1]
                clf = HGBC(max_iter=200, learning_rate=0.1, random_state=0).fit(Xc[idx[sub]][:, cc], yc_[sub])
                p = clf.predict_proba(Xt[mt][:, cc])[:, 1]
                fpr, tpr, _ = roc_curve(yt_, p)
                row = dict(net=net, target=target, features=name, forecast=with_fc, base=float(yt_.mean()), auc=float(roc_auc_score(yt_, p)),
                           ap=float(average_precision_score(yt_, p)), tpr5=float(np.interp(0.05, fpr, tpr)))
                out.append(row)
                print(f"{target:<7}{name:<22}{'yes' if with_fc else 'no':>9} | {row['base']:9.3f}{row['auc']:7.3f}{row['ap']:9.3f}{row['tpr5']:27.3f}", flush=True)
json.dump(out, open("results_audit/breakdown_prediction.json", "w"), indent=1)
