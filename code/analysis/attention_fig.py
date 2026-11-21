"""Fig. 4: attention maps for silent-set slides (runs on DCC, needs the slides and infer_e020/h5).

Cases (slide level, fine-tuned model, epoch 20): TP = highest-scoring IDH-mutant, TN = lowest-scoring IDH-wildtype,
FP = highest-scoring IDH-wildtype, FN = lowest-scoring IDH-mutant. Each row: thumbnail, attention overlay
(within-slide percentile), and the 4 most-attended tiles.

  python attention_fig.py <infer_dir> <slide_data.csv> <out.png>
"""
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from scipy.stats import rankdata

R, SD, OUT = sys.argv[1:4]
S = pd.read_csv(os.path.join(R, "scores.csv")); D = pd.read_csv(SD, low_memory=False)
X = D[D.split == "silent"].merge(S[["slide", "score"]], on="slide")
X = X[[os.path.exists(os.path.join(R, "h5", s + ".h5")) for s in X.slide]]
mut, wt = X[X.IDH == 1].sort_values("score"), X[X.IDH == 0].sort_values("score")
cases = [("True positive", mut.iloc[-1]), ("True negative", wt.iloc[0]),
         ("False positive", wt.iloc[-1]), ("False negative", mut.iloc[0])]
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(4, 6, width_ratios=[1.3, 1.3, .62, .62, .62, .62], wspace=.06, hspace=.3)
rows = []
for i, (lab, r) in enumerate(cases):
    with h5py.File(os.path.join(R, "h5", r.slide + ".h5")) as h:
        c, A = h["coords"][:], h["attention"][:]
    sl = openslide.OpenSlide(r.path)
    W0, H0 = sl.dimensions
    th = sl.get_thumbnail((1400, 1400)); f = W0 / th.size[0]
    x, y, mu = c[:, 0], c[:, 1], c[:, 3]
    fp = 224 * mu  # tile footprint at level 0 (all tiles are read at level 0)
    pct = rankdata(A) / len(A)
    # crop to tissue bounding box
    x0, y0 = (x.min() - fp.max()) / f, (y.min() - fp.max()) / f
    x1, y1 = (x.max() + 2 * fp.max()) / f, (y.max() + 2 * fp.max()) / f
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(th.size[0], x1), min(th.size[1], y1)
    ax = fig.add_subplot(gs[i, 0]); ax.imshow(th); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    ax.set_title(f"{lab}: {r.histology}, {r.idh_variant}\nscore {r.score:.4f} ({r.institution})", loc="left", fontsize=6.5)
    ax = fig.add_subplot(gs[i, 1]); ax.imshow(th, alpha=.35)
    hm = np.full((th.size[1], th.size[0]), np.nan)
    for xi, yi, fi, p in zip(x / f, y / f, fp / f, pct):
        hm[int(yi):int(yi + fi) + 1, int(xi):int(xi + fi) + 1] = p
    im = ax.imshow(hm, cmap="inferno", vmin=0, vmax=1, alpha=.85, interpolation="nearest")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    if i == 0:
        ax.set_title("attention (within-slide percentile)", fontsize=6.5)
    top = np.argsort(-A)[:4]
    for j, t in enumerate(top):
        tile = sl.read_region((int(x[t]), int(y[t])), 0, (int(round(fp[t])), int(round(fp[t])))).convert("RGB").resize((224, 224))
        ax = fig.add_subplot(gs[i, 2 + j]); ax.imshow(tile); ax.axis("off")
        if i == 0 and j == 0:
            ax.set_title("top-attended tiles", loc="left", fontsize=6.5)
    rows.append(dict(case=lab, slide=r.slide, patient=r.patient, institution=r.institution, histology=r.histology,
                     who_class=r.who_class, idh_variant=r.idh_variant, score=r.score, n_tiles=len(A)))
cb = fig.add_axes([.31, .07, .17, .008]); fig.colorbar(im, cax=cb, orientation="horizontal").set_label("attention percentile", fontsize=6)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
pd.DataFrame(rows).to_csv(OUT.replace(".png", "_cases.csv"), index=False)
print(pd.DataFrame(rows).to_string())
