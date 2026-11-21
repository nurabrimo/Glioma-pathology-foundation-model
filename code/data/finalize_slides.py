"""EAGLE-glioma step 3: splits + final slide table + QC.

TCGA split (patients with known IDH status), unit = canonical institution so that no institution spans two splits:
  train ~60 %  | val ~20 % (internal validation) | pretrial ~10 % (threshold tuning) | silent ~10 % (locked one-shot test)
"Case Western" and "Case Western - St Joes" are kept in the same split (same university system; conservative).
Search: N random institution->split assignments (fixed seed); cost = sum |patient share - target| + 0.5 * sum |IDH-mutant
prevalence - overall|; pretrial and silent must each hold >= 25 mutant and >= 25 wild-type patients from >= 3 institutions.
Among assignments within 0.02 of the best cost, the one whose `silent` split has the latest median year of diagnosis is
chosen (institution separation wins over a strict temporal split, which TCGA cannot provide).
EBRAINS and CPTAC: split = external.

Outputs: <out>/slide_data.csv (usable slides only), <out>/slide_data_all.csv (every slide with drop reason),
         <out>/tcga_split_institutions.csv, <out>/qc.txt
"""
import argparse, io, os, sys
import numpy as np
import pandas as pd

TARGET = {"train": 0.6, "val": 0.2, "pretrial": 0.1, "silent": 0.1}
SPLITS = list(TARGET)
GROUP_MERGE = {"Case Western - St Joes": "Case Western"}


def tcga_split(P, seed, n_iter):
    P = P.copy(); P["split_group"] = P.institution.replace(GROUP_MERGE)
    G = P.groupby("split_group").agg(n=("patient", "size"), mut=("IDH", "sum"), yr=("year_dx", "median"))
    n, mut = G.n.values.astype(float), G.mut.values.astype(float); N, prev = n.sum(), mut.sum() / n.sum()
    tgt = np.array([TARGET[s] for s in SPLITS]); rng = np.random.default_rng(seed); cands = []
    yrs = {g: P[P.split_group == g].year_dx.dropna().values for g in G.index}
    for _ in range(n_iter):
        a = rng.choice(4, size=len(G), p=tgt)
        sn = np.array([n[a == k].sum() for k in range(4)]); sm = np.array([mut[a == k].sum() for k in range(4)])
        if (sn == 0).any(): continue
        ok = all(sm[k] >= 25 and sn[k] - sm[k] >= 25 and (a == k).sum() >= 3 for k in (2, 3))
        if not ok: continue
        cost = np.abs(sn / N - tgt).sum() + 0.5 * np.abs(sm / sn - prev).sum()
        cands.append((cost, tuple(a)))
    assert cands, "no feasible split"
    cands = sorted(set(cands)); best = cands[0][0]
    near = [c for c in cands if c[0] <= best + 0.02]

    def silent_year(c):
        ys = np.concatenate([yrs[g] for g, k in zip(G.index, c[1]) if k == 3]); return np.median(ys) if len(ys) else -1
    pick = max(near, key=lambda c: (silent_year(c), -c[0]))
    G["split"] = [SPLITS[k] for k in pick[1]]
    return P.split_group.map(G.split), P.split_group, G, dict(best_cost=best, chosen_cost=pick[0], n_feasible=len(cands), n_near=len(near), silent_median_year=silent_year(pick))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/work/nb295/eagle_glioma")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--n_iter", type=int, default=200000)
    ap.add_argument("--min_tiles", type=int, default=1)
    a = ap.parse_args()
    D = pd.read_csv(os.path.join(a.dir, "slide_labels.csv"), low_memory=False)
    metas = []
    for c in ("tcga", "ebrains", "cptac"):
        f = os.path.join(a.dir, "tiles", f"{c}_meta.csv")
        if os.path.exists(f): metas.append(pd.read_csv(f))
    M = pd.concat(metas) if metas else pd.DataFrame(columns=["slide", "status", "mpp", "level", "mult", "eff_mpp", "tissue_tiles", "W", "H", "start"])
    D = D.merge(M[["slide", "status", "mpp", "level", "mult", "eff_mpp", "tissue_tiles", "W", "H", "start"]], on="slide", how="left")
    D["drop_reason"] = ""
    D.loc[D.IDH.isna(), "drop_reason"] = "IDH status unknown"
    m = (D.drop_reason == "") & D.status.isna(); D.loc[m, "drop_reason"] = "tiling pending (slide not yet downloaded/segmented)"
    m = (D.drop_reason == "") & (D.status != "ok"); D.loc[m, "drop_reason"] = "tiling: " + D.loc[m, "status"].astype(str)
    m = (D.drop_reason == "") & (D.tissue_tiles < a.min_tiles); D.loc[m, "drop_reason"] = "too few tissue tiles"

    T = D[(D.cohort == "tcga") & D.IDH.notna()]
    P = T.drop_duplicates("patient")[["patient", "institution", "IDH", "year_dx"]]
    sp, grp, G, info = tcga_split(P, a.seed, a.n_iter)
    P = P.assign(split=sp.values, split_group=grp.values)
    D = D.merge(P[["patient", "split", "split_group"]], on="patient", how="left")
    D.loc[D.cohort != "tcga", "split"] = "external"; D.loc[D.cohort != "tcga", "split_group"] = D.institution
    D.loc[D.IDH.isna(), "split"] = np.nan
    D["target"] = D.IDH
    cols = ["slide", "path", "cohort", "patient", "institution", "project", "histology", "who_class", "year_dx", "IDH", "idh_variant", "rapid",
            "codel", "mpp", "tissue_tiles", "split", "target", "split_group", "institution_raw", "grade", "level", "mult", "eff_mpp", "W", "H",
            "start", "file_id", "label_source", "variant_source", "note", "drop_reason"]
    D[cols].to_csv(os.path.join(a.dir, "slide_data_all.csv"), index=False)
    U = D[D.drop_reason == ""][cols[:-1]]
    U.to_csv(os.path.join(a.dir, "slide_data.csv"), index=False)
    G.to_csv(os.path.join(a.dir, "tcga_split_institutions.csv"))

    out = io.StringIO()
    def pr(*x): print(*x, file=out)
    pr("split search:", info)
    leak = D[D.cohort == "tcga"].dropna(subset=["split"]).groupby("institution").split.nunique().gt(1).sum()
    pleak = D.dropna(subset=["split"]).groupby("patient").split.nunique().gt(1).sum()
    pr("institutions spanning >1 split:", int(leak), "| patients spanning >1 split:", int(pleak))
    pr("\nTCGA institutions per split:\n" + G.sort_values(["split", "n"], ascending=[True, False]).to_string())

    def tab(X, title):
        rows = []
        for (c, s), g in X.groupby(["cohort", "split"]):
            p = g.drop_duplicates("patient"); mu = p[p.IDH == 1]; kv = mu[mu.rapid.notna()]
            rows.append(dict(cohort=c, split=s, slides=len(g), patients=len(p), institutions=g.institution.nunique(), IDH_mut_pts=len(mu),
                             IDH_mut_pct=round(100 * len(mu) / max(len(p), 1), 1), mut_slides=int((g.IDH == 1).sum()),
                             mut_with_variant=len(kv), R132H=int((kv.rapid == 1).sum()),
                             ihc_sens=round(float((kv.rapid == 1).mean()), 3) if len(kv) else np.nan,
                             median_year=p.year_dx.median(), median_tiles=g.tissue_tiles.median()))
        pr("\n" + title + "\n" + pd.DataFrame(rows).to_string(index=False))
        for c, g in X.groupby("cohort"):
            p = g.drop_duplicates("patient"); mu = p[p.IDH == 1]; kv = mu[mu.rapid.notna()]
            pr(f"  {c} total: slides {len(g)}, patients {len(p)}, IDH-mut {len(mu)} ({100*len(mu)/max(len(p),1):.1f} %), "
               f"R132H / mutants with variant = {int((kv.rapid==1).sum())}/{len(kv)}" + (f" = {(kv.rapid==1).mean():.3f}" if len(kv) else ""))
    tab(U, "USABLE slides (slide_data.csv)")
    tab(D[D.IDH.notna()], "ALL slides with known IDH (incl. pending / dropped)")
    pr("\ndrop reasons (slides):\n" + D[D.drop_reason != ""].groupby(["cohort", "drop_reason"]).agg(slides=("slide", "size"), patients=("patient", "nunique")).to_string())
    lost = set(D[D.IDH.notna()].patient) - set(U.patient)
    pr("patients with known IDH but no usable slide:", len(lost))
    if len(U):
        pr("\nlevel/mult by cohort:\n" + U.groupby(["cohort", "level", "mult"]).size().to_string())
        pr("\neffective mpp by cohort:\n" + U.groupby("cohort").eff_mpp.describe().round(4).to_string())
        pr("\ntissue tiles by cohort:\n" + U.groupby("cohort").tissue_tiles.describe().round(0).to_string())
    pr("\nTCGA variants (patients):\n" + D[(D.cohort == "tcga") & (D.IDH == 1)].drop_duplicates("patient").idh_variant.value_counts().to_string())
    open(os.path.join(a.dir, "qc.txt"), "w").write(out.getvalue()); print(out.getvalue())


if __name__ == "__main__":
    main()
