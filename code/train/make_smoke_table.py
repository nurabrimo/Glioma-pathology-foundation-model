"""Smoke-test tables only (NOT for results): a few TCGA slides, 224 px @ 0.5 um/px tiles derived from the existing
TRIDENT 20x/512 px tissue patches (2 x 2 tiles inside each patch), alternating fake labels."""
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd

out = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
pat = "/cwork/nb295/recast_feats/tcga_ffpe/trident/20x_512px_0px_overlap/patches/*_patches.h5"
wsi = {os.path.basename(f)[:-4]: f for f in glob.glob("/cwork/nb295/mechanopath/wsi/*/*.svs")}
rows, tiles = [], []
for f in sorted(glob.glob(pat)):
    name = os.path.basename(f).replace("_patches.h5", "")
    if name not in wsi:
        continue
    with h5py.File(f, "r") as h:
        c = h["coords"][:]
        p0 = float(h["coords"].attrs["patch_size_level0"])
    if len(c) < 800:
        continue
    s0 = p0 * 224 / 512  # level-0 footprint of a 224 px tile at 20x
    xs = np.concatenate([c[:, 0] + dx for dx in (0, s0) for dy in (0, s0)])
    ys = np.concatenate([c[:, 1] + dy for dx in (0, s0) for dy in (0, s0)])
    tiles.append(pd.DataFrame({"slide": name, "x": np.round(xs).astype(np.int64), "y": np.round(ys).astype(np.int64),
                               "level": np.int8(0), "mult": np.float32(s0 / 224)}))
    k = len(rows)
    rows.append(dict(slide=name, path=wsi[name], cohort="tcga", patient=name[:12], institution=name[5:7], IDH=k % 2,
                     idh_variant="R132H" if k % 2 else "", rapid=k % 2, split="train" if k < n - 2 else "val",
                     tissue_tiles=len(xs)))
    if len(rows) >= n:
        break
pd.DataFrame(rows).to_csv(os.path.join(out, "slide_data.csv"), index=False)
pd.concat(tiles).to_parquet(os.path.join(out, "tiles", "tcga.parquet"), index=False)
print(pd.DataFrame(rows)[["slide", "split", "IDH", "tissue_tiles"]].to_string())
print("mult values", pd.concat(tiles).mult.unique())
