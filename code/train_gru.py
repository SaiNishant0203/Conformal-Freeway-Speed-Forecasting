"""Train the base forecaster and write the prediction files the analysis scripts read.

Model: one-layer GRU (64 hidden units) on the last 12 speed readings of a sensor, linear
head, speed H steps ahead as target. One set of weights per network, shared by all sensors.
Two models are trained separately:
  point     one output, squared error            -> predictions_gru_<net>_h<H>.npz
  quantile  two outputs (5% and 95%), pinball    -> quantile_gru_<net>_h<H>.npz

Split is chronological: 60% train, 20% calibration, 20% test. The last 10% of the training
period is held out for early stopping, so the calibration and test splits are never seen
during training. Critical occupancy o_c is estimated per sensor on the training split.

Run from the repo root:
  python code/train_gru.py --net pems04 --out predictions_v2
  python code/train_gru.py --net pems08 --out predictions_v2

Training is slow on a laptop CPU (roughly 4 minutes per epoch for District 4). Progress is
saved after every epoch, so an interrupted run continues where it stopped when the same
command is run again. --budget-min N makes a run stop cleanly after about N minutes.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "code")
from conformal import estimate_critical_occupancy

L = 12  # history length (5-minute steps)
QUANTILES = (0.05, 0.95)
START = time.time()


class GRUForecaster(nn.Module):
    def __init__(self, n_out=1, hidden=64):
        super().__init__()
        self.g = nn.GRU(1, hidden, batch_first=True)
        self.h = nn.Linear(hidden, n_out)

    def forward(self, x):
        o, _ = self.g(x.unsqueeze(-1))
        return self.h(o[:, -1])


def windows(speed, t0, t1, H):
    """History and target for every origin t in [t0, t1) and every sensor, time-major."""
    t = np.arange(t0, t1)
    X = np.stack([speed[t - L + 1 + k] for k in range(L)], axis=-1)  # [n_t, N, L]
    return X.reshape(-1, L), speed[t + H].reshape(-1)


def pinball(pred, y):
    loss = 0.0
    for j, q in enumerate(QUANTILES):
        e = y - pred[:, j]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss


def mse(pred, y):
    return ((pred[:, 0] - y) ** 2).mean()


@torch.no_grad()
def predict(model, X, bs=65536):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def out_of_time(args, need_seconds):
    return args.budget_min is not None and time.time() - START + need_seconds > args.budget_min * 60


def fit(Xtr, ytr, Xva, yva, kind, args):
    """Returns (model, log, finished). Resumes from a checkpoint if one exists."""
    ckpt_path = f"{args.out}/ckpt_{args.net}_{kind}.pt"
    loss_fn = mse if kind == "point" else pinball
    torch.manual_seed(args.seed)
    model = GRUForecaster(1 if kind == "point" else 2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    st = dict(epoch=0, best=float("inf"), best_state=None, bad=0, log=[], done=False)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); g.set_state(ck["gen"])
        st = ck["st"]
        print(f"  [{kind}] resuming after epoch {st['epoch']}", flush=True)

    def save():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), gen=g.get_state(), st=st), ckpt_path)

    last = 0.0
    while not st["done"]:
        if out_of_time(args, 1.15 * last + 30):
            return model, st["log"], False
        t0 = time.time(); model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        va = float(loss_fn(predict(model, Xva), yva))
        last = time.time() - t0
        st["epoch"] += 1
        st["log"].append(dict(kind=kind, epoch=st["epoch"], val_loss=va, seconds=round(last, 1)))
        print(f"  [{kind}] epoch {st['epoch']:2d}  val loss {va:.5f}  ({last:.0f}s)", flush=True)
        if va < st["best"] * (1 - args.min_rel_improve):
            st["best"], st["bad"] = va, 0
            st["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            st["bad"] += 1
        st["done"] = st["bad"] >= args.patience or st["epoch"] >= args.epochs
        save()
    model.load_state_dict(st["best_state"])
    return model, st["log"], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=["pems04", "pems08"])
    ap.add_argument("--out", default="predictions_v2")
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--kind", default="both", choices=["point", "quantile", "both"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-rel-improve", type=float, default=0.005,
                    help="an epoch counts as an improvement if validation loss drops by this fraction")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-min", type=float, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    raw = np.load(f"Dataset/{args.net}.npz")["data"]
    T, N, _ = raw.shape
    flow, occ, speed = raw[..., 0], raw[..., 1].astype(np.float32), raw[..., 2].astype(np.float32)
    tr, ca, H = int(0.6 * T), int(0.8 * T), args.H
    va = tr - H - int(0.1 * tr)                      # first validation origin
    mu, sd = float(speed[:tr].mean()), float(speed[:tr].std())
    z = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    un = lambda p: (p.numpy() * sd + mu).astype(np.float32)

    Xtr, ytr = windows(speed, L - 1, va - H, H)      # targets end before validation starts
    Xva, yva = windows(speed, va, tr - H, H)
    Xca, yca = windows(speed, tr, ca - H, H)
    Xte, yte = windows(speed, ca, T - H, H)
    print(f"{args.net}: T={T} N={N}  train {len(ytr):,}  val {len(yva):,}  cal {len(yca):,}  test {len(yte):,}", flush=True)

    meta = lambda t0, t1: dict(occ=occ[t0:t1].reshape(-1), sid=np.tile(np.arange(N), t1 - t0), spd=speed[t0:t1].reshape(-1))
    mc, mt = meta(tr, ca - H), meta(ca, T - H)
    settings = dict(vars(args), history=L, hidden=64, layers=1, optimizer="Adam", point_loss="MSE",
                    quantile_loss=f"pinball {QUANTILES}", speed_mean=mu, speed_std=sd,
                    n_train=len(ytr), n_val=len(yva), n_cal=len(yca), n_test=len(yte), torch=torch.__version__)
    settings.pop("budget_min")
    Ztr, Zytr, Zva, Zyva = z(Xtr), z(ytr), z(Xva), z(yva)

    for kind in (["point", "quantile"] if args.kind == "both" else [args.kind]):
        target = f"{args.out}/{'predictions' if kind == 'point' else 'quantile'}_gru_{args.net}_h{H}.npz"
        if os.path.exists(target):
            print(f"  [{kind}] {target} exists, skipping", flush=True)
            continue
        model, log, finished = fit(Ztr, Zytr, Zva, Zyva, kind, args)
        if not finished or out_of_time(args, 120):
            print(f"PAUSED [{kind}]: run the same command again to continue", flush=True)
            return
        torch.save(model.state_dict(), f"models/gru_{args.net}_h{H}_{kind}.pt")
        pc, pt = un(predict(model, z(Xca))), un(predict(model, z(Xte)))
        if kind == "point":
            o_c = np.array([estimate_critical_occupancy(flow[:tr, s], raw[:tr, s, 1]) for s in range(N)], dtype=np.float32)
            np.savez_compressed(target, y_cal=yca, yhat_cal=pc[:, 0], occ_cal=mc["occ"], sid_cal=mc["sid"], spd_cal=mc["spd"],
                                y_test=yte, yhat_test=pt[:, 0], occ_test=mt["occ"], sid_test=mt["sid"], spd_test=mt["spd"],
                                o_c=o_c, H=H, model="gru")
        else:
            np.savez_compressed(target, qlo_cal=pc[:, 0], qhi_cal=pc[:, 1], y_cal=yca, occ_cal=mc["occ"], sid_cal=mc["sid"],
                                qlo_test=pt[:, 0], qhi_test=pt[:, 1], y_test=yte, occ_test=mt["occ"], sid_test=mt["sid"])
        with open(f"{args.out}/training_log_{args.net}_{kind}.json", "w") as f:
            json.dump(dict(settings=settings, epochs=log), f, indent=1)
        os.remove(f"{args.out}/ckpt_{args.net}_{kind}.pt")
        print(f"DONE [{kind}]: wrote {target}", flush=True)


if __name__ == "__main__":
    main()
