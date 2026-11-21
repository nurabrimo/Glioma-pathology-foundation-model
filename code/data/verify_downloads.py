"""Verify the GDC downloads against the manifest (size for all, md5 for all unless --no_md5). Writes download_verify.csv."""
import argparse, hashlib, os
from multiprocessing import Pool
import pandas as pd


def chk(a):
    fid, fn, md5, size, root, do_md5 = a; p = os.path.join(root, fid, fn)
    if not os.path.exists(p): return fid, fn, "missing"
    if os.path.getsize(p) != size: return fid, fn, "size_mismatch"
    if do_md5:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        if h.hexdigest() != md5: return fid, fn, "md5_mismatch"
    return fid, fn, "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/work/nb295/eagle_glioma/tcga_download_manifest.txt")
    ap.add_argument("--root", default="/cwork/nb295/eagle_glioma/tcga_dx_extra")
    ap.add_argument("--out", default="/work/nb295/eagle_glioma/download_verify.csv")
    ap.add_argument("--no_md5", action="store_true"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); M = pd.read_csv(a.manifest, sep="\t")
    with Pool(a.workers) as pool:
        R = pool.map(chk, [(r.id, r.filename, r.md5, r.size, a.root, not a.no_md5) for r in M.itertuples()], chunksize=2)
    V = pd.DataFrame(R, columns=["file_id", "file_name", "state"]); V.to_csv(a.out, index=False)
    print("VERIFY", V.state.value_counts().to_dict())
