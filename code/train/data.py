"""Slide / tile tables and datasets. Tile semantics identical to EAGLE training/datasets.py:
read_region(location=(x, y) [level-0 px], size=round(224*mult), level) then LANCZOS resize to 224 if mult != 1."""
import glob
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import extra_transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXTS = [".svs", ".ndpi", ".tif", ".tiff", ".dcm", ".mrxs", ".scn"]


def resolve_path(slide, path):
    p = str(path)
    if os.path.isfile(p):
        return p
    for e in EXTS:  # EAGLE convention: path = directory, file = <slide>.svs
        q = os.path.join(p, f"{slide}{e}")
        if os.path.isfile(q):
            return q
    g = glob.glob(os.path.join(p, f"{slide}*"))
    if g:
        return g[0]
    raise FileNotFoundError(f"{slide}: {p}")


def load_slides(slide_data, target="IDH", splits=None, cohorts=None):
    df = pd.read_csv(slide_data, low_memory=False)
    if splits:
        df = df[df.split.isin(splits)]
    if cohorts:
        df = df[df.cohort.isin(cohorts)]
    if target is not None:
        df = df[df[target].notna()].copy()
        df["target"] = df[target].astype(int)
    return df.reset_index(drop=True)


def load_tiles(tile_files, slides):
    """-> dict slide -> (x, y, level, mult) numpy arrays."""
    files = []
    for t in tile_files:
        files += sorted(glob.glob(os.path.join(t, "*.parquet"))) if os.path.isdir(t) else sorted(glob.glob(t))
    assert files, f"no tile tables in {tile_files}"
    want = set(map(str, slides))
    out = {}
    for f in files:
        if f.endswith(".parquet"):
            try:
                T = pd.read_parquet(f, filters=[("slide", "in", list(want))])
            except Exception:
                T = pd.read_parquet(f)
        else:
            T = pd.read_csv(f)
        T["slide"] = T["slide"].astype(str)
        T = T[T.slide.isin(want)]
        if not len(T):
            continue
        for s, idx in T.groupby("slide", sort=False).indices.items():
            out[s] = (T.x.values[idx].astype(np.int64), T.y.values[idx].astype(np.int64),
                      T.level.values[idx].astype(np.int16), T.mult.values[idx].astype(np.float32))
    return out


class Reader:
    """Per-process slide handle cache. cucim for .svs/.tif (as EAGLE), OpenSlide otherwise or on failure."""

    def __init__(self, backend="auto", max_open=256):
        self.backend, self.max_open, self.h = backend, max_open, OrderedDict()

    def _open(self, path):
        if path in self.h:
            self.h.move_to_end(path)
            return self.h[path]
        ext = os.path.splitext(path)[1].lower()
        use_cucim = self.backend == "cucim" or (self.backend == "auto" and ext in (".svs", ".tif", ".tiff"))
        obj = None
        if use_cucim:
            try:
                from cucim import CuImage
                obj = ("cucim", CuImage(path))
            except Exception:
                obj = None
        if obj is None:
            import openslide
            obj = ("openslide", openslide.OpenSlide(path))
        self.h[path] = obj
        if len(self.h) > self.max_open:
            _, old = self.h.popitem(last=False)
            try:
                old[1].close()
            except Exception:
                pass
        return obj

    def read(self, path, x, y, level, size):
        kind, s = self._open(path)
        if kind == "cucim":
            try:
                a = np.asarray(s.read_region(location=(int(x), int(y)), size=(size, size), level=int(level)))
                if a.shape[0] == size and a.shape[1] == size:
                    return Image.fromarray(a[..., :3])
            except Exception:
                pass
            import openslide  # fall back for this handle
            self.h[path] = ("openslide", openslide.OpenSlide(path))
            kind, s = self.h[path]
        return s.read_region((int(x), int(y)), int(level), (size, size)).convert("RGB")


def read_tile(reader, path, x, y, level, mult, tilesize=224):
    size = int(np.round(tilesize * mult))
    img = reader.read(path, x, y, level, size)
    if mult != 1:
        img = img.resize((tilesize, tilesize), Image.LANCZOS)
    return img


def train_transform():
    return transforms.Compose([transforms.RandomHorizontalFlip(),
                               extra_transforms.RandomRectRotation(),
                               extra_transforms.GaussianBlur(),
                               transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class TrainTiles(data.Dataset):
    """EAGLE training_tile_dataset_binary: each epoch a random (1-drop) fraction of the training slides, k tiles per
    slide (without replacement if the slide has >= k tiles, else with replacement), reseeded with seed+epoch.
    Index = step * k + j. Augmentation RNG is seeded per (seed, epoch, index) so a resumed run sees the same tiles."""

    def __init__(self, dfs, tiles, k, tilesize=224, drop=0.0, seed=1634, backend="auto"):
        self.master = dfs.reset_index(drop=True)
        self.paths = [resolve_path(s, p) for s, p in zip(self.master.slide.astype(str), self.master.path)]
        self.tiles = tiles
        self.k, self.tilesize, self.seed = k, tilesize, seed
        self.nslides = int(len(dfs) * (1 - drop))
        self.transform = train_transform()
        self.backend = backend
        self.reader = None
        self.epoch = None
        self.failed = 0

    def makeData(self, epoch):
        rng = np.random.RandomState(self.seed + epoch)
        self.epoch = epoch
        self.order = rng.choice(len(self.master), size=self.nslides, replace=False)  # _sample_slides
        self.sel = np.empty((self.nslides, self.k), np.int64)                          # _sample_tile_data
        for i, si in enumerate(self.order):
            n = len(self.tiles[str(self.master.slide.iloc[si])][0])
            self.sel[i] = rng.choice(n, size=self.k, replace=n < self.k)

    def get_target(self, step):
        return int(self.master.target.iloc[self.order[step]])

    def get_slide(self, step):
        return str(self.master.slide.iloc[self.order[step]])

    def __len__(self):
        return self.nslides * self.k

    def __getitem__(self, index):
        if self.reader is None:
            self.reader = Reader(self.backend)
        step, j = divmod(int(index), self.k)
        si = self.order[step]
        x, y, level, mult = self.tiles[str(self.master.slide.iloc[si])]
        t = self.sel[step, j]
        h = (self.seed * 1000003 + self.epoch * 7919 + int(index)) % (2 ** 31)
        random.seed(h)
        torch.manual_seed(h)
        try:
            img = read_tile(self.reader, self.paths[si], x[t], y[t], level[t], float(mult[t]), self.tilesize)
        except Exception as e:  # unreadable region: white tile, reported
            print(f"[data] read failed {self.paths[si]} ({x[t]},{y[t]}): {e!r}", flush=True)
            img = Image.new("RGB", (self.tilesize, self.tilesize), (255, 255, 255))
        return self.transform(img)


class StepBatchSampler:
    """Yields, for each step >= start_step, this rank's share of the slide's k tiles in batches of load_bs.
    Rank r of W takes tile positions r, r+W, ... (k must be divisible by W)."""

    def __init__(self, nsteps, k, rank, world, load_bs, start_step=0):
        assert k % world == 0, f"k={k} must be divisible by the number of processes ({world})"
        self.nsteps, self.k, self.rank, self.world, self.bs, self.start = nsteps, k, rank, world, load_bs, start_step
        self.per_step = int(np.ceil((k // world) / load_bs))

    def __iter__(self):
        for step in range(self.start, self.nsteps):
            idx = list(range(step * self.k + self.rank, (step + 1) * self.k, self.world))
            for s in range(0, len(idx), self.bs):
                yield idx[s:s + self.bs]

    def __len__(self):
        return (self.nsteps - self.start) * self.per_step


class SlideTiles(data.Dataset):
    """All tissue tiles of one slide, no augmentation (inference)."""

    def __init__(self, path, xylm, tilesize=224, backend="auto"):
        self.path, (self.x, self.y, self.level, self.mult) = path, xylm
        self.tilesize, self.backend, self.reader = tilesize, backend, None
        self.transform = eval_transform()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.reader is None:
            self.reader = Reader(self.backend)
        img = read_tile(self.reader, self.path, self.x[i], self.y[i], self.level[i], float(self.mult[i]), self.tilesize)
        return self.transform(img)
