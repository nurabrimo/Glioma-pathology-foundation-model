"""Tile reading throughput (augmented training tiles) for cucim vs OpenSlide at a given worker count."""
import sys, time, torch
sys.path.insert(0, ".")
import data as D
sm, workers = sys.argv[1], int(sys.argv[2])
dfs = D.load_slides(sm + "/slide_data.csv", target="IDH", splits=["train"])
tiles = D.load_tiles([sm + "/tiles"], dfs.slide.astype(str))
for backend in ("cucim", "openslide"):
    for w in sorted({workers, max(1, workers // 2)}):
        ds = D.TrainTiles(dfs, tiles, 2208, drop=0.5, backend=backend); ds.makeData(1 if backend == "cucim" else 2)
        sp = D.StepBatchSampler(ds.nslides, 2208, 0, 1, 96)
        dl = torch.utils.data.DataLoader(ds, batch_sampler=sp, num_workers=w, prefetch_factor=4)
        t = time.time(); n = 0
        for x in dl:
            n += len(x)
            if n >= 2208 * 2: break
        dt = time.time() - t
        print(f"{backend} workers={w}: {n} tiles in {dt:.1f}s = {n/dt:.0f} tiles/s (incl. worker start-up)", flush=True)
        del dl
