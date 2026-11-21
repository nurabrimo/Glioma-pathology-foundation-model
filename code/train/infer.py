"""Full-slide inference as in the paper: ALL tissue tiles of a slide -> encoder -> GMA -> softmax P(positive).

Writes <out>/scores_shard<i>of<n>.csv (slide, score, n_tiles, seconds); resumable (slides already in the CSV or with
a finished feature file are skipped). Optional per-slide h5 (<feat_dir>/<slide>.h5: coords [x, y, level, mult],
features fp16, attention (raw, pre-softmax), sign) for attention maps and for the frozen-encoder baseline.

  --encoder  gigapath | eagle_lung | path to checkpoint_tile_XXX.pth / checkpoint_last.pth
  --slide_ckpt  checkpoint_slide_XXX.pth / checkpoint_last.pth / frozen-baseline GMA; omit to only dump features
Merge shards with:  python infer.py --merge <out>
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import encoder as E
import modules

p = argparse.ArgumentParser()
p.add_argument("--out", default=None)
p.add_argument("--slide_data", default="/work/nb295/eagle_glioma/slide_data.csv")
p.add_argument("--tiles", nargs="+", default=["/work/nb295/eagle_glioma/tiles"])
p.add_argument("--splits", nargs="*", default=None)
p.add_argument("--cohorts", nargs="*", default=None)
p.add_argument("--encoder", default="gigapath")
p.add_argument("--slide_ckpt", default=None)
p.add_argument("--feat_dir", default=None, help="save per-slide h5 (features, attention) here")
p.add_argument("--from_feats", default=None, help="score from existing feature h5 files instead of reading slides")
p.add_argument("--save_feats", type=int, default=1, help="with --feat_dir: 1 = features + attention, 0 = attention only")
p.add_argument("--amp", type=int, default=0, help="fp16 autocast (paper deployment: full precision)")
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--workers", type=int, default=10)
p.add_argument("--max_tiles", type=int, default=0, help="0 = all tiles (paper); >0 = random subset (tests only)")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshards", type=int, default=1)
p.add_argument("--backend", default="auto")
p.add_argument("--merge", default=None)


def load_gma(path, dev):
    m = modules.GMA(ndim=E.NDIM, dropout=True, n_classes=2)
    ck = E.safe_load(path, mmap=False)
    m.load_state_dict(E.clean_state(ck["slide_model"] if "slide_model" in ck else ck))
    return m.to(dev).eval()


@torch.no_grad()
def score_features(gma, feats):
    A_raw, sign, logits = gma(feats)
    return float(F.softmax(logits, dim=1)[0, 1]), A_raw, sign[:, 1] - sign[:, 0]


def merge(out):
    fs = sorted(glob.glob(os.path.join(out, "scores_shard*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("slide", keep="last")
    df.to_csv(os.path.join(out, "scores.csv"), index=False)
    print(f"merged {len(fs)} shards -> {len(df)} slides -> {os.path.join(out, 'scores.csv')}")


def main():
    a = p.parse_args()
    if a.merge:
        return merge(a.merge)
    os.makedirs(a.out, exist_ok=True)
    if a.feat_dir:
        os.makedirs(a.feat_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False  # full precision, as deployed
    df = D.load_slides(a.slide_data, target=None, splits=a.splits, cohorts=a.cohorts)
    df = df.iloc[a.shard::a.nshards].reset_index(drop=True)
    csv = os.path.join(a.out, f"scores_shard{a.shard}of{a.nshards}.csv")
    done = set(pd.read_csv(csv).slide.astype(str)) if os.path.exists(csv) else set()
    if not os.path.exists(csv):
        open(csv, "w").write("slide,score,n_tiles,seconds\n")
    gma = load_gma(a.slide_ckpt, dev) if a.slide_ckpt else None
    enc = None
    if not a.from_feats:
        tiles = D.load_tiles(a.tiles, df.slide.astype(str))
        enc = E.get_encoder(a.encoder).to(dev).eval()
    print(f"shard {a.shard}/{a.nshards}: {len(df)} slides, {len(done)} done, device {dev}, amp {a.amp}", flush=True)
    tot_tiles, tot_t = 0, 0.0
    for r in df.itertuples(index=False):
        slide = str(r.slide)
        fpath = os.path.join(a.feat_dir, slide + ".h5") if a.feat_dir else None
        if slide in done or (gma is None and fpath and os.path.exists(fpath)):
            continue
        t0 = time.time()
        try:
            if a.from_feats:
                with h5py.File(os.path.join(a.from_feats, slide + ".h5"), "r") as h:
                    feats = torch.from_numpy(h["features"][:].astype(np.float32)).to(dev)
                coords = None
            else:
                if slide not in tiles:
                    print(f"[skip] {slide}: no tiles", flush=True)
                    continue
                x, y, lv, mu = tiles[slide]
                if a.max_tiles and len(x) > a.max_tiles:
                    sel = np.sort(np.random.RandomState(0).choice(len(x), a.max_tiles, replace=False))
                    x, y, lv, mu = x[sel], y[sel], lv[sel], mu[sel]
                ds = D.SlideTiles(D.resolve_path(slide, r.path), (x, y, lv, mu), backend=a.backend)
                dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)
                out = []
                with torch.no_grad():
                    for xb in dl:
                        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp)):
                            f = enc(xb.to(dev, non_blocking=True))
                        # copy to CPU: the encoder output is a view of the full token tensor (B,197,1536);
                        # keeping it on the GPU pins ~1.2 MB per tile and runs large slides out of memory
                        out.append(f.float().cpu())
                feats = torch.cat(out).to(dev)
                coords = np.stack([x, y, lv, mu], 1).astype(np.float64)
            n = feats.shape[0]
            score, A, sign = (np.nan, None, None) if gma is None else score_features(gma, feats)
            if fpath:
                tmp = fpath + ".tmp"
                with h5py.File(tmp, "w") as h:
                    if coords is not None:
                        h.create_dataset("coords", data=coords)
                    if a.save_feats and not a.from_feats:
                        h.create_dataset("features", data=feats.cpu().numpy().astype(np.float16))
                    if A is not None:
                        h.create_dataset("attention", data=A.astype(np.float32))
                        h.create_dataset("sign", data=np.asarray(sign, np.float32))
                        h.attrs["score"] = score
                os.replace(tmp, fpath)
            dt = time.time() - t0
            tot_tiles += n; tot_t += dt
            if gma is not None:
                open(csv, "a").write(f"{slide},{score:.6f},{n},{dt:.1f}\n")
            print(f"{slide} n={n} score={score:.4f} {dt:.1f}s ({n / dt:.0f} tiles/s)", flush=True)
        except Exception as e:
            print(f"[error] {slide}: {e!r}", flush=True)
    if tot_t:
        print(f"SHARD_DONE {a.shard}: {tot_tiles} tiles in {tot_t:.0f}s = {tot_tiles / tot_t:.0f} tiles/s", flush=True)


if __name__ == "__main__":
    main()
