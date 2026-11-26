"""Train a stronger forecaster: STID (Shao et al., "Spatial-Temporal Identity: A Simple yet
Effective Baseline for Multivariate Time Series Forecasting", CIKM 2022, arXiv 2208.05233).

STID is an MLP that sees, for each sensor, the last hour of readings together with three
learned embeddings: one per sensor, one per time of day (288 five-minute slots) and one per
day of week. It reaches the accuracy of graph neural networks on the standard traffic
benchmarks at a fraction of the cost, which makes it trainable on a laptop CPU.

Differences from the GRU in train_gru.py: it knows which sensor it is looking at, it knows
the time, and on PeMS04/08 it reads flow and occupancy as well as speed. The target is the
same (speed 60 minutes ahead), the split is the same, and the output files have the same
layout, so every analysis script runs on them unchanged.

Datasets: pems04, pems08 (flow, occupancy, speed) and metr_la (speed only; zeros are missing
readings and are masked out of the loss).

Run from the repo root:
  python code/train_stid.py --net pems04 --out predictions_stid
Progress is saved after every epoch; run the same command again to continue an interrupted
run. --budget-min N stops cleanly after about N minutes.
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

L, DAY = 12, 288
QUANTILES = (0.05, 0.95)
START = time.time()


def load_raw(net):
    """Returns speed [T, N], inputs [T, N, C] (speed is the last channel), flow and occupancy
    (None if the dataset has none), time-of-day and day-of-week index per step."""
    if net in ("pems04", "pems08"):
        raw = np.load(f"Dataset/{net}.npz")["data"].astype(np.float32)
        T = raw.shape[0]; t = np.arange(T)
        return raw[..., 2], raw, raw[..., 0], raw[..., 1], t % DAY, (t // DAY) % 7
    d = np.load(f"Dataset/{net}.npz"); sp = d["speed"]; T = sp.shape[0]
    t = np.arange(T) + int(d["start_tod"])
    return sp, sp[..., None], None, None, t % DAY, (t // DAY + int(d["start_dow"])) % 7


class STID(nn.Module):
    def __init__(self, n_nodes, in_dim, n_out=1, emb=32, layers=3, drop=0.15):
        super().__init__()
        self.node = nn.Parameter(torch.empty(n_nodes, emb)); nn.init.xavier_uniform_(self.node)
        self.tod = nn.Parameter(torch.empty(DAY, emb)); nn.init.xavier_uniform_(self.tod)
        self.dow = nn.Parameter(torch.empty(7, emb)); nn.init.xavier_uniform_(self.dow)
        self.inp = nn.Linear(in_dim, emb)
        hid = 4 * emb
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, hid))
                                     for _ in range(layers)])
        self.out = nn.Linear(hid, n_out)

    def forward(self, x, tod, dow):                       # x [B, N, L*C]; tod, dow [B]
        B, N, _ = x.shape
        h = torch.cat([self.inp(x), self.node.expand(B, N, -1),
                       self.tod[tod][:, None].expand(B, N, -1), self.dow[dow][:, None].expand(B, N, -1)], -1)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)                                # [B, N, n_out]


def masked(loss_elem, mask):
    return (loss_elem * mask).sum() / mask.sum().clamp(min=1)


def point_loss(pred, y, mask):
    return masked((pred[..., 0] - y).abs(), mask)


def quantile_loss(pred, y, mask):
    tot = 0.0
    for j, q in enumerate(QUANTILES):
        e = y - pred[..., j]
        tot = tot + masked(torch.maximum(q * e, (q - 1) * e), mask)
    return tot


def out_of_time(args, need):
    return args.budget_min is not None and time.time() - START + need > args.budget_min * 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", default="predictions_stid")
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--kind", default="both", choices=["point", "quantile", "both"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--milestones", default="12,24", help="epochs at which the learning rate is halved")
    ap.add_argument("--device", default="cpu", help="cpu, or xpu for an Intel GPU, or cuda")
    ap.add_argument("--budget-min", type=float, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    miles = [int(v) for v in args.milestones.split(",")]
    os.makedirs(args.out, exist_ok=True)

    speed, inputs, flow, occ, tod, dow = load_raw(args.net)
    T, N, C = inputs.shape; H = args.H
    tr, ca = int(0.6 * T), int(0.8 * T); va = tr - H - int(0.1 * tr)
    mu, sd = inputs[:tr].reshape(-1, C).mean(0), inputs[:tr].reshape(-1, C).std(0)
    smu, ssd = float(mu[-1]), float(sd[-1])
    Z = torch.tensor((inputs - mu) / sd).to(dev)                               # [T, N, C]
    Y = torch.tensor((speed - smu) / ssd).to(dev); M = torch.tensor((speed > 0).astype(np.float32)).to(dev)
    tod_t, dow_t = torch.tensor(tod).to(dev), torch.tensor(dow).to(dev)

    def batch(ts):                                         # ts: tensor of origin indices
        ts = ts.to(dev)
        x = torch.stack([Z[ts - L + 1 + k] for k in range(L)], 2).reshape(len(ts), N, L * C)
        return x, tod_t[ts], dow_t[ts], Y[ts + H], M[ts + H]

    segs = dict(train=(L - 1, va - H), val=(va, tr - H), cal=(tr, ca - H), test=(ca, T - H))
    print(f"{args.net}: T={T} N={N} C={C}  origins: " + "  ".join(f"{k} {b - a:,}" for k, (a, b) in segs.items()), flush=True)

    @torch.no_grad()
    def run(model, a, b, loss_fn=None):
        model.eval(); outs, tot, cnt = [], 0.0, 0.0
        for i in range(a, b, 256):
            ts = torch.arange(i, min(i + 256, b)); x, td, dw, y, m = batch(ts); p = model(x, td, dw)
            outs.append(p.cpu())
            if loss_fn is not None:
                tot += float(loss_fn(p, y, m)) * float(m.sum()); cnt += float(m.sum())
        return torch.cat(outs), tot / max(cnt, 1)

    settings = dict(vars(args), model="STID", history=L, emb=32, layers=3, dropout=0.15, optimizer="Adam",
                    lr_schedule=f"halved at epochs {miles}", point_loss="masked MAE",
                    quantile_loss=f"masked pinball {QUANTILES}", inputs=C, torch=torch.__version__)
    settings.pop("budget_min")

    for kind in (["point", "quantile"] if args.kind == "both" else [args.kind]):
        target = f"{args.out}/{'predictions' if kind == 'point' else 'quantile'}_stid_{args.net}_h{H}.npz"
        if os.path.exists(target):
            print(f"  [{kind}] {target} exists, skipping", flush=True); continue
        ck_path = f"{args.out}/ckpt_stid_{args.net}_{kind}.pt"
        loss_fn = point_loss if kind == "point" else quantile_loss
        torch.manual_seed(args.seed)
        model = STID(N, L * C, 1 if kind == "point" else 2).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, miles, 0.5)
        g = torch.Generator().manual_seed(args.seed)
        st = dict(epoch=0, best=float("inf"), best_state=None, bad=0, log=[], done=False)
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, weights_only=False, map_location=dev)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            g.set_state(ck["gen"]); st = ck["st"]
            print(f"  [{kind}] resuming after epoch {st['epoch']}", flush=True)
        a, b = segs["train"]; last = 0.0
        while not st["done"]:
            if out_of_time(args, 1.2 * last + 20):
                print(f"PAUSED [{kind}] after epoch {st['epoch']}", flush=True); return
            t0 = time.time(); model.train()
            perm = a + torch.randperm(b - a, generator=g)
            for i in range(0, len(perm), args.batch):
                x, td, dw, y, m = batch(perm[i:i + args.batch])
                opt.zero_grad(); loss_fn(model(x, td, dw), y, m).backward(); opt.step()
            sched.step()
            _, vl = run(model, *segs["val"], loss_fn)
            last = time.time() - t0; st["epoch"] += 1
            st["log"].append(dict(kind=kind, epoch=st["epoch"], val_loss=vl, seconds=round(last, 1)))
            if vl < st["best"]:
                st["best"], st["bad"] = vl, 0
                st["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                st["bad"] += 1
            st["done"] = st["bad"] >= args.patience or st["epoch"] >= args.epochs
            if st["epoch"] % 5 == 0 or st["done"]:
                print(f"  [{kind}] epoch {st['epoch']:3d}  val {vl:.5f}  best {st['best']:.5f}  ({last:.0f}s/epoch)", flush=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=g.get_state(), st=st), ck_path)
        if out_of_time(args, 60):
            print(f"PAUSED [{kind}] before writing predictions", flush=True); return
        model.load_state_dict(st["best_state"])
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"{args.out}/stid_{args.net}_h{H}_{kind}.pt")
        un = lambda p: (p.numpy() * ssd + smu).astype(np.float32)
        pc, pt = un(run(model, *segs["cal"])[0]), un(run(model, *segs["test"])[0])       # [n_t, N, n_out]
        meta = {}
        for name, (t0_, t1_) in (("cal", segs["cal"]), ("test", segs["test"])):
            meta[f"y_{name}"] = speed[t0_ + H:t1_ + H].reshape(-1)
            meta[f"sid_{name}"] = np.tile(np.arange(N), t1_ - t0_)
            meta[f"spd_{name}"] = speed[t0_:t1_].reshape(-1)
            meta[f"tod_{name}"] = np.repeat(tod[t0_:t1_], N).astype(np.int16)
            meta[f"occ_{name}"] = occ[t0_:t1_].reshape(-1) if occ is not None else np.zeros((t1_ - t0_) * N, np.float32)
        if kind == "point":
            o_c = (np.array([estimate_critical_occupancy(flow[:tr, s], occ[:tr, s]) for s in range(N)], dtype=np.float32)
                   if occ is not None else np.ones(N, np.float32))
            np.savez_compressed(target, yhat_cal=pc[..., 0].reshape(-1), yhat_test=pt[..., 0].reshape(-1),
                                o_c=o_c, H=H, model="stid", **meta)
        else:
            np.savez_compressed(target, qlo_cal=pc[..., 0].reshape(-1), qhi_cal=pc[..., 1].reshape(-1),
                                qlo_test=pt[..., 0].reshape(-1), qhi_test=pt[..., 1].reshape(-1), **meta)
        with open(f"{args.out}/training_log_stid_{args.net}_{kind}.json", "w") as f:
            json.dump(dict(settings=settings, epochs=st["log"]), f, indent=1)
        os.remove(ck_path)
        print(f"DONE [{kind}]: wrote {target} after {st['epoch']} epochs", flush=True)


if __name__ == "__main__":
    main()
