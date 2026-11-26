"""The GRU of train_gru.py for datasets that train_gru.py cannot read (METR-LA: speed only,
missing readings stored as zeros), with optional GPU support.

Same model (one-layer GRU, 64 hidden units, last 12 speed readings of one sensor), same split,
same losses and the same stopping rule as train_gru.py. Targets recorded as zero are masked
out of the loss. Batches are built from whole time steps (10 steps x all sensors, about the
same number of sequences per batch as train_gru.py).

Run: python code/train_gru_any.py --net metr_la --out predictions_gru_gpu --device xpu
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
sys.path.insert(0, "code")
from train_gru import GRUForecaster, QUANTILES
from train_stid import load_raw

L = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", default="predictions_gru_gpu")
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-rel-improve", type=float, default=0.005)
    ap.add_argument("--batch-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device); os.makedirs(args.out, exist_ok=True)
    speed, _, flow, occ, tod, dow = load_raw(args.net)
    T, N = speed.shape; H = args.H
    tr, ca = int(0.6 * T), int(0.8 * T); va = tr - H - int(0.1 * tr)
    mu, sd = float(speed[:tr].mean()), float(speed[:tr].std())
    Z = torch.tensor((speed - mu) / sd).to(dev); M = torch.tensor((speed > 0).astype(np.float32)).to(dev)

    def batch(ts):
        ts = ts.to(dev)
        x = torch.stack([Z[ts - L + 1 + k] for k in range(L)], 2).reshape(-1, L)          # [B*N, L]
        return x, Z[ts + H].reshape(-1), M[ts + H].reshape(-1)

    def loss_fn(kind, p, y, m):
        if kind == "point":
            return (((p[:, 0] - y) ** 2) * m).sum() / m.sum().clamp(min=1)
        tot = 0.0
        for j, q in enumerate(QUANTILES):
            e = y - p[:, j]; tot = tot + (torch.maximum(q * e, (q - 1) * e) * m).sum() / m.sum().clamp(min=1)
        return tot

    @torch.no_grad()
    def run(model, kind, a, b):
        model.eval(); outs, tot, cnt = [], 0.0, 0.0
        for i in range(a, b, 64):
            x, y, m = batch(torch.arange(i, min(i + 64, b))); p = model(x); outs.append(p.cpu())
            tot += float(loss_fn(kind, p, y, m)) * float(m.sum()); cnt += float(m.sum())
        return torch.cat(outs), tot / max(cnt, 1)

    for kind in ("point", "quantile"):
        target = f"{args.out}/{'predictions' if kind == 'point' else 'quantile'}_gru_{args.net}_h{H}.npz"
        if os.path.exists(target):
            continue
        torch.manual_seed(args.seed)
        model = GRUForecaster(1 if kind == "point" else 2).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr); g = torch.Generator().manual_seed(args.seed)
        best, best_state, bad, log = float("inf"), None, 0, []
        for ep in range(1, args.epochs + 1):
            t0 = time.time(); model.train()
            perm = (L - 1) + torch.randperm(va - H - (L - 1), generator=g)
            for i in range(0, len(perm), args.batch_steps):
                x, y, m = batch(perm[i:i + args.batch_steps])
                opt.zero_grad(); loss_fn(kind, model(x), y, m).backward(); opt.step()
            _, vl = run(model, kind, va, tr - H)
            log.append(dict(kind=kind, epoch=ep, val_loss=vl, seconds=round(time.time() - t0, 1)))
            print(f"  [{kind}] epoch {ep:2d}  val loss {vl:.5f}  ({time.time() - t0:.0f}s)", flush=True)
            if vl < best * (1 - args.min_rel_improve):
                best, bad = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        un = lambda p: (p.numpy() * sd + mu).astype(np.float32)
        pc, pt = un(run(model, kind, tr, ca - H)[0]), un(run(model, kind, ca, T - H)[0])
        meta = {}
        for name, (a, b) in (("cal", (tr, ca - H)), ("test", (ca, T - H))):
            meta[f"y_{name}"] = speed[a + H:b + H].reshape(-1); meta[f"sid_{name}"] = np.tile(np.arange(N), b - a)
            meta[f"spd_{name}"] = speed[a:b].reshape(-1)
            meta[f"occ_{name}"] = occ[a:b].reshape(-1) if occ is not None else np.zeros((b - a) * N, np.float32)
        if kind == "point":
            np.savez_compressed(target, yhat_cal=pc[:, 0], yhat_test=pt[:, 0], o_c=np.ones(N, np.float32), H=H, model="gru", **meta)
        else:
            np.savez_compressed(target, qlo_cal=pc[:, 0], qhi_cal=pc[:, 1], qlo_test=pt[:, 0], qhi_test=pt[:, 1], **meta)
        with open(f"{args.out}/training_log_gru_{args.net}_{kind}.json", "w") as f:
            json.dump(dict(settings=dict(vars(args), model="GRU", hidden=64, history=L), epochs=log), f, indent=1)
        print(f"DONE [{kind}]: wrote {target}", flush=True)


if __name__ == "__main__":
    main()
