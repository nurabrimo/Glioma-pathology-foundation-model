"""EAGLE-glioma step 2: tile coordinates in the format EAGLE's training/datasets.py expects.

One row per tile: slide, x, y (level-0 pixel coordinates of the top-left corner), level, mult.
A tile is read as read_region((x, y), level, (round(224*mult),)*2) and resized to 224 px when mult != 1, i.e. a
224 px tile at 20x (0.5 um/px), non-overlapping grid, tissue only.

level / mult follow EAGLE's IRT_Pipeline/extract_tissue_new.py::find_level exactly, including its 20 % tolerance
(MAX_PIXEL_DIFFERENCE = 0.2): a pyramid level whose resolution is within 20 % of 0.5 um/px is used as is (mult = 1).
The effective resolution of every slide is written to the meta table (eff_mpp).

Tissue: TRIDENT (HEST segmenter) tissue contours, GeoJSON polygons in level-0 coordinates. The mask is rasterised at
4 x 4 sub-cells per tile and a tile is kept when >= 50 % of its area is tissue (--min_tissue).

Outputs: <out>/tiles/<cohort>.parquet (sorted by slide), <out>/tiles/<cohort>_meta.csv (per slide: mpp, W, H, level,
mult, eff_mpp, tissue_tiles, row offset `start`, status).
"""
import argparse, json, os, sys, traceback
from multiprocessing import Pool
import numpy as np
import pandas as pd

PATCH = 224; TARGET_MPP = 0.5; MAX_PIXEL_DIFFERENCE = 0.2; SUB = 4


def find_level(level_downsamples, mpp, base_mpp, patchsize=PATCH):
    """EAGLE find_level (extract_tissue_new.py), unchanged logic."""
    downsample = mpp / base_mpp; level = None
    for i in range(len(level_downsamples))[::-1]:
        if abs(downsample / level_downsamples[i] * patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize or downsample > level_downsamples[i]:
            level = i; mult = downsample / level_downsamples[level]; break
    if level is None: raise Exception(f"Requested resolution ({mpp} mpp) is too high")
    mult = np.round(mult * patchsize) / patchsize
    if abs(mult * patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize: mult = 1.
    return level, float(mult)


def slide_mpp(s):
    p = s.properties
    for k in ("openslide.mpp-x", "aperio.MPP"):
        if p.get(k):
            try:
                v = float(p[k])
                if 0.05 < v < 5: return v, k
            except ValueError: pass
    for k in ("openslide.objective-power", "aperio.AppMag"):
        if p.get(k):
            try: return 10.0 / float(p[k]), k + " (10/objective)"
            except ValueError: pass
    return None, "none"


def one(job):
    slide, path, gj, min_tissue = job
    r = dict(slide=slide, status="ok", mpp=np.nan, mpp_source="", W=0, H=0, level=-1, mult=np.nan, eff_mpp=np.nan, tissue_tiles=0)
    try:
        import openslide, cv2
        if not os.path.exists(path): r["status"] = "missing_file"; return r, None
        if gj is None or not os.path.exists(gj): r["status"] = "no_contours"; return r, None
        s = openslide.OpenSlide(path); W, H = s.dimensions; mpp, src = slide_mpp(s)
        r.update(W=W, H=H, mpp=mpp if mpp else np.nan, mpp_source=src)
        if mpp is None: r["status"] = "no_mpp"; return r, None
        level, mult = find_level(list(s.level_downsamples), TARGET_MPP, mpp); ld = float(s.level_downsamples[level]); s.close()
        step = PATCH * mult * ld  # level-0 footprint of one tile
        r.update(level=level, mult=mult, eff_mpp=mpp * ld * mult)
        nx, ny = int(W // step), int(H // step)
        if nx < 1 or ny < 1: r["status"] = "too_small"; return r, None
        mask = np.zeros((ny * SUB, nx * SUB), np.uint8); sc = SUB / step
        feats = json.load(open(gj)).get("features", [])
        for f in feats:
            g = f["geometry"]; polys = [g["coordinates"]] if g["type"] == "Polygon" else (g["coordinates"] if g["type"] == "MultiPolygon" else [])
            for rings in polys:
                if not rings: continue
                cv2.fillPoly(mask, [np.round(np.asarray(rings[0], float)[:, :2] * sc).astype(np.int32)], 1)
                for hole in rings[1:]:
                    cv2.fillPoly(mask, [np.round(np.asarray(hole, float)[:, :2] * sc).astype(np.int32)], 0)
        frac = mask.reshape(ny, SUB, nx, SUB).mean(axis=(1, 3))
        iy, ix = np.nonzero(frac >= min_tissue)
        if len(ix) == 0: r["status"] = "no_tissue"; return r, None
        x = np.round(ix * step).astype(np.int32); y = np.round(iy * step).astype(np.int32)
        r["tissue_tiles"] = int(len(x))
        return r, (x, y)
    except Exception as e:
        r["status"] = "error: " + repr(e)[:150]; traceback.print_exc(file=sys.stderr); return r, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/work/nb295/eagle_glioma/slide_labels.csv")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--contours", nargs="+", required=True, help="directories with <stem>.geojson (first hit wins)")
    ap.add_argument("--out", default="/work/nb295/eagle_glioma/tiles")
    ap.add_argument("--min_tissue", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = pd.read_csv(a.labels, low_memory=False); D = D[(D.cohort == a.cohort) & D.IDH.notna()]
    jobs = []
    for r in D.itertuples(index=False):
        stem = os.path.splitext(os.path.basename(r.path))[0]
        gj = next((os.path.join(c, stem + ".geojson") for c in a.contours if os.path.exists(os.path.join(c, stem + ".geojson"))), None)
        jobs.append((r.slide, r.path, gj, a.min_tissue))
    print(a.cohort, "slides", len(jobs), flush=True)
    meta, parts = [], []
    with Pool(a.workers) as pool:
        for k, (r, xy) in enumerate(pool.imap(one, jobs, chunksize=4)):
            meta.append(r)
            if xy is not None:
                n = len(xy[0])
                parts.append(pd.DataFrame({"slide": pd.Categorical([r["slide"]] * n), "x": xy[0], "y": xy[1],
                                           "level": np.int8(r["level"]), "mult": np.float32(r["mult"])}))
            if (k + 1) % 100 == 0: print(k + 1, "/", len(jobs), flush=True)
    M = pd.DataFrame(meta)
    ok = M[M.status == "ok"].copy(); ok["start"] = ok.tissue_tiles.cumsum() - ok.tissue_tiles
    M = M.merge(ok[["slide", "start"]], on="slide", how="left")
    T = pd.concat(parts, ignore_index=True); T["slide"] = T.slide.astype(str)
    T.to_parquet(os.path.join(a.out, f"{a.cohort}.parquet"), index=False)
    M.to_csv(os.path.join(a.out, f"{a.cohort}_meta.csv"), index=False)
    print("tiles", len(T), "| status", M.status.str.slice(0, 20).value_counts().to_dict())
    print("level/mult", ok.groupby(["level", "mult"]).size().to_dict())
    print("eff_mpp", ok.eff_mpp.describe().round(4).to_dict()); print("tissue_tiles", ok.tissue_tiles.describe().round(0).to_dict())
    print("TILES_DONE", a.cohort)


if __name__ == "__main__":
    main()
