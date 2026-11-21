""""No fine-tuning" comparator: frozen encoder features + the same GMA aggregator, trained with the EAGLE schedule
(1 slide per step, K tiles sampled per slide, drop 0.5, AdamW, warm-up 2 epochs, cosine lr, cosine weight decay,
CrossEntropy class weights [1-pos_weight, pos_weight], 20 epochs). Features: per-slide h5 with dataset `features`
(infer.py --feat_dir dumps, or TRIDENT-style <slide>.h5). Then scores every slide of the table with all its tiles.

--lr defaults to the end-to-end value (1e-6). An aggregator alone learns slowly at that rate, so a conventional
frozen-feature rate (e.g. --lr 1e-4) should be reported next to it; the flag makes both runs explicit."""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import eagle_utils as utils
import modules

p = argparse.ArgumentParser()
p.add_argument("--feat_dir", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--slide_data", default="/work/nb295/eagle_glioma/slide_data.csv")
p.add_argument("--train_splits", nargs="+", default=["train"])
p.add_argument("--target", default="IDH")
p.add_argument("--ndim", type=int, default=1536)
p.add_argument("--k", type=int, default=2208)
p.add_argument("--drop", type=float, default=0.5)
p.add_argument("--pos_weight", type=float, default=0.7)
p.add_argument("--lr", type=float, default=1e-6)
p.add_argument("--lr_end", type=float, default=1e-8)
p.add_argument("--warmup_epochs", type=int, default=2)
p.add_argument("--weight_decay", type=float, default=0.04)
p.add_argument("--weight_decay_end", type=float, default=0.4)
p.add_argument("--nepochs", type=int, default=20)
p.add_argument("--seed", type=int, default=1634)
p.add_argument("--suffix", default=".h5")


def fpath(a, slide):
    for s in (a.suffix, "_patches.h5", ".h5"):
        f = os.path.join(a.feat_dir, slide + s)
        if os.path.exists(f):
            return f
    return None


def main():
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alls = D.load_slides(a.slide_data, target=None)
    alls["f"] = [fpath(a, str(s)) for s in alls.slide]
    print(f"slides {len(alls)}, with features {int(alls.f.notna().sum())}", flush=True)
    alls = alls[alls.f.notna()].reset_index(drop=True)
    tr = alls[alls.split.isin(a.train_splits) & alls[a.target].notna()].reset_index(drop=True)
    y = tr[a.target].astype(int).values
    nslides = int(len(tr) * (1 - a.drop))
    torch.manual_seed(a.seed)
    gma = modules.GMA(ndim=a.ndim, dropout=True, n_classes=2).to(dev)
    crit = nn.CrossEntropyLoss() if a.pos_weight == 0.5 else nn.CrossEntropyLoss(torch.Tensor([1 - a.pos_weight, a.pos_weight]))
    crit = crit.to(dev)
    opt = optim.AdamW(utils.get_params_groups(gma))
    lrs = utils.cosine_scheduler(a.lr, a.lr_end, a.nepochs, nslides, warmup_epochs=a.warmup_epochs)
    wds = utils.cosine_scheduler(a.weight_decay, a.weight_decay_end, a.nepochs, nslides)
    conv = os.path.join(a.outdir, "convergence.csv")
    open(conv, "w").write("epoch,loss\n")
    print(f"train slides {len(tr)} (pos {int(y.sum())}), steps/epoch {nslides}, lr {a.lr}", flush=True)
    for epoch in range(1, a.nepochs + 1):
        rng = np.random.RandomState(a.seed + epoch)
        order = rng.choice(len(tr), size=nslides, replace=False)
        run, t0 = 0.0, time.time()
        gma.train()
        for i, si in enumerate(order):
            it = nslides * (epoch - 1) + i
            for j, g in enumerate(opt.param_groups):
                g["lr"] = lrs[it]
                if j == 0:
                    g["weight_decay"] = wds[it]
            with h5py.File(tr.f[si], "r") as h:
                n = h["features"].shape[-2]
                sel = np.sort(rng.choice(n, size=a.k, replace=n < a.k))
                uniq, inv = np.unique(sel, return_inverse=True)
                ft = h["features"]
                if ft.ndim == 3:
                    x = ft[0][sel]
                elif n > 4 * a.k:
                    x = ft[uniq][inv]
                else:
                    x = ft[:][sel]
            x = torch.from_numpy(np.asarray(x, np.float32)).to(dev)
            opt.zero_grad()
            _, _, out = gma(x)
            loss = crit(out, torch.LongTensor([y[si]]).to(dev))
            loss.backward()
            opt.step()
            run += loss.item()
        open(conv, "a").write(f"{epoch},{run / nslides}\n")
        print(f"epoch {epoch}/{a.nepochs} loss {run / nslides:.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save({"epoch": a.nepochs + 1, "slide_model": gma.state_dict()}, os.path.join(a.outdir, f"checkpoint_slide_{a.nepochs:03}.pth"))
    gma.eval()
    rows = []
    with torch.no_grad():
        for r in alls.itertuples(index=False):
            with h5py.File(r.f, "r") as h:
                ft = h["features"]
                x = torch.from_numpy(np.asarray(ft[0] if ft.ndim == 3 else ft[:], np.float32)).to(dev)
            _, _, out = gma(x)
            rows.append((r.slide, float(F.softmax(out, 1)[0, 1]), x.shape[0]))
    pd.DataFrame(rows, columns=["slide", "score", "n_tiles"]).to_csv(os.path.join(a.outdir, "scores.csv"), index=False)
    print("FROZEN_DONE", os.path.join(a.outdir, "scores.csv"))


if __name__ == "__main__":
    main()
