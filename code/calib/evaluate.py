"""Evaluation of a slide-score table against slide_data.csv, following the EAGLE paper's analyses.

Outputs in --out:
  auc.csv            AUC + 1,000-bootstrap 95% CI: overall per evaluation set, and by cohort, institution, project,
                     histology, IDH variant (variant-positive vs all negative), tissue-amount deciles
  rapid.csv          rapid test (simulated IDH1-R132H IHC) sensitivity / specificity / PPV / NPV with bootstrap CIs
  thresholds.json    reference rapid NPV / PPV (+CI) and the two operating points chosen on the PRETRIAL split only
  workflow.csv       assisted-workflow NPV / PPV / sens / spec / test reduction (+ bootstrap CIs) at those locked
                     thresholds on every set (silent = the locked one-shot application)
  roc_<set>.png, heat_<set>_<metric>_<window>.png, merged.csv

Threshold rule (the paper's Results: "conservative" point matches the rapid test's NPV and PPV, "least conservative"
point stays inside the rapid test's 95% CI):
  most conservative  = maximum test reduction with assisted NPV >= rapid NPV and PPV >= rapid PPV (point estimates)
  least conservative = maximum test reduction with assisted NPV >= rapid NPV lower CI and PPV >= rapid PPV lower CI
searched on the union of the paper's two grids (th0 in [0, 0.5] and [0, 0.05], th1 in [0.5, 1] and [0.95, 1], 201
steps each). Ties: smallest th0, then largest th1.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U

p = argparse.ArgumentParser()
p.add_argument("--scores", required=True)
p.add_argument("--slide_data", default="/work/nb295/eagle_glioma/slide_data.csv")
p.add_argument("--out", required=True)
p.add_argument("--target", default="IDH")
p.add_argument("--rapid", default="rapid")
p.add_argument("--variant", default="idh_variant")
p.add_argument("--tissue", default=None, help="tissue-amount column (default: n_tiles from the scores file)")
p.add_argument("--pretrial", default="pretrial")
p.add_argument("--silent", default="silent")
p.add_argument("--rapid_ref_splits", nargs="+", default=["train", "val", "pretrial"],
               help="splits giving the reference rapid-test NPV/PPV: the paper used historical rapid-test data, so the "
                    "default is every development patient outside the locked silent split (no model scores involved)")
p.add_argument("--level", default="patient", choices=["patient", "slide"],
               help="patient: one row per patient, score = mean of that patient's slide scores, so bootstrap rows are "
                    "independent (the paper had ~1 slide per patient); slide: one row per slide, naive bootstrap")
p.add_argument("--patient", default="patient")
p.add_argument("--rapid_ref", default=None, help="override: npv,npv_lo,npv_hi,ppv,ppv_lo,ppv_hi")
p.add_argument("--exclude_splits", nargs="*", default=["train"])
p.add_argument("--groups", nargs="*", default=["cohort", "institution", "project", "histology", "who_class", "scanner"])
p.add_argument("--min_n", type=int, default=20)
p.add_argument("--nboot", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no_plots", type=int, default=0)


def boot_auc(y, s, n=1000, seed=0):
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rs = np.random.RandomState(seed)
    v = []
    for _ in range(n):
        i = rs.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.quantile(v, 0.025), np.quantile(v, 0.975)


def boot_binary(y, pred, n=1000, seed=0):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    f = dict(se=U.se_score, sp=U.sp_score, ppv=U.ppv_score, npv=U.npv_score)
    out = {k: g(y, pred) for k, g in f.items()}
    rs = np.random.RandomState(seed)
    B = {k: [] for k in f}
    for _ in range(n):
        i = rs.randint(0, len(y), len(y))
        for k, g in f.items():
            B[k].append(g(y[i], pred[i]))
    for k in f:
        out[k + "_lo"], out[k + "_hi"] = np.nanquantile(B[k], 0.025), np.nanquantile(B[k], 0.975)
    return out


def choose(df, ref, a):
    th0 = np.unique(np.concatenate([np.linspace(0, 0.5, 201), np.linspace(0, 0.05, 201)]))
    th1 = np.unique(np.concatenate([np.linspace(0.5, 1, 201), np.linspace(0.95, 1, 201)]))
    X, Y, npv, ppv, sen, spe, bac, red = U.simulation_fast(df, th0, th1, a.target, a.rapid, "score")
    res = {}
    for name, (nmin, pmin) in dict(most_conservative=(ref["npv"], ref["ppv"]),
                                   least_conservative=(ref["npv_lo"], ref["ppv_lo"])).items():
        ok = (np.nan_to_num(npv, nan=-1) >= nmin - 1e-12) & (np.nan_to_num(ppv, nan=-1) >= pmin - 1e-12)
        if not ok.any():
            res[name] = None
            continue
        best = np.where(ok, red, -1).max()
        ii, jj = np.nonzero(ok & (red >= best - 1e-12))
        k = np.lexsort((-th1[jj], th0[ii]))[0]  # smallest th0, then largest th1
        i, j = ii[k], jj[k]
        res[name] = dict(th0=float(th0[i]), th1=float(th1[j]), npv=float(npv[i, j]), ppv=float(ppv[i, j]),
                         se=float(sen[i, j]), sp=float(spe[i, j]), reduction=float(red[i, j]))
    return res


def main():
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = pd.read_csv(a.scores); S["slide"] = S.slide.astype(str)
    D = pd.read_csv(a.slide_data, low_memory=False); D["slide"] = D.slide.astype(str)
    M = D.merge(S[[c for c in S.columns if c == "slide" or c not in D.columns]], on="slide", how="inner")
    M = M[M[a.target].notna() & M.score.notna()].copy()
    M[a.target] = M[a.target].astype(int)
    M = M[~M.split.isin(a.exclude_splits)]
    tissue = a.tissue or ("n_tiles" if "n_tiles" in M.columns else "tissue_tiles")

    def to_patient(X):
        """One row per patient: mean slide score, summed tissue, first value of the patient-level columns."""
        X = X.copy(); X["n_slides"] = 1
        num = {c: "sum" for c in ("n_slides", "n_tiles", "tissue_tiles") if c in X.columns}
        if "score" in X.columns:
            num["score"] = "mean"
        key = ["cohort", a.patient] if "cohort" in X.columns else [a.patient]
        bad = X.groupby(key)[a.target].transform("nunique") > 1
        if bad.any():  # discordant labels across a patient's slides (2 EBRAINS patients): excluded, listed
            X[bad].to_csv(os.path.join(a.out, "excluded_discordant_patients.csv"), index=False)
            print(f"excluded {bad.sum()} slides from {X[bad].groupby(key).ngroups} patients with discordant labels")
            X = X[~bad]
        G = X.groupby(key, sort=False)
        assert (G[a.target].nunique() <= 1).all() and (G["split"].nunique() <= 1).all(), "label/split differs within a patient"
        return G.agg({**{c: "first" for c in X.columns if c not in num and c not in key}, **num}).reset_index()
    if a.level == "patient":
        M = to_patient(M)
    M.to_csv(os.path.join(a.out, "merged.csv"), index=False)
    print(f"level={a.level}; scored rows with label (excl. {a.exclude_splits}): {len(M)}; missing scores: {len(D) - len(D.merge(S, on='slide'))}")

    sets = {f"split={s}": M[M.split == s] for s in sorted(M.split.dropna().unique())}
    if "cohort" in M.columns:
        for c in sorted(M.cohort.dropna().unique()):
            sets[f"cohort={c}"] = M[M.cohort == c]

    # ---- AUC tables
    rows = []
    for sname, X in sets.items():
        def add(gt, g, Z, y=None, s=None):
            y = Z[a.target].values if y is None else y
            s = Z.score.values if s is None else s
            auc, lo, hi = boot_auc(y, s, a.nboot, a.seed) if len(y) >= a.min_n else (np.nan,) * 3
            rows.append(dict(set=sname, group_type=gt, group=g, n=len(y), n_pos=int(np.sum(y)), auc=auc, lo=lo, hi=hi))
        add("all", "all", X)
        for gcol in a.groups:
            if gcol in X.columns and X[gcol].nunique() > 1:
                for g, Z in X.groupby(gcol):
                    add(gcol, g, Z)
        if a.variant in X.columns:
            neg = X[X[a.target] == 0]
            for g, Z in X[X[a.target] == 1].groupby(X[a.variant].fillna("unknown")):
                add("variant_vs_negatives", g, None, np.r_[np.ones(len(Z), int), np.zeros(len(neg), int)], np.r_[Z.score.values, neg.score.values])
        if tissue in X.columns and len(X) >= 10 * a.min_n:
            dec = pd.qcut(X[tissue], 10, duplicates="drop")
            for k, (g, Z) in enumerate(X.groupby(dec, observed=True)):
                add("tissue_decile", f"{k + 1}: {g}", Z)
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(a.out, "auc.csv"), index=False)
    print(A[A.group_type == "all"].round(3).to_string(index=False))

    # ---- rapid test
    has_rapid = a.rapid in M.columns and M[a.rapid].notna().any()
    R, th, Wf = None, {}, None
    if has_rapid:
        MR = M[M[a.rapid].notna()].copy(); MR[a.rapid] = MR[a.rapid].astype(int)
        rs = {k: v[v[a.rapid].notna()] for k, v in sets.items()}
        rr = []
        # reference rapid-test performance: from the label table alone (independent of which slides were scored)
        DR = D[D[a.target].notna() & D[a.rapid].notna() & D.split.isin(a.rapid_ref_splits)].copy()
        DR[a.target] = DR[a.target].astype(int)
        if a.level == "patient":
            DR = to_patient(DR)
        for sname, X in list(rs.items()) + [("ref=" + "+".join(a.rapid_ref_splits), DR)]:
            if len(X):
                rr.append(dict(set=sname, n=len(X), n_pos=int(X[a.target].sum()), **boot_binary(X[a.target], X[a.rapid].astype(int), a.nboot, a.seed)))
        R = pd.DataFrame(rr); R.to_csv(os.path.join(a.out, "rapid.csv"), index=False)
        print(R.round(3).to_string(index=False))
        if a.rapid_ref:
            v = [float(x) for x in a.rapid_ref.split(",")]
            ref = dict(npv=v[0], npv_lo=v[1], npv_hi=v[2], ppv=v[3], ppv_lo=v[4], ppv_hi=v[5], source="given")
        else:
            r = R[R.set.str.startswith("ref=")].iloc[0]
            ref = {k: float(r[k]) for k in ("npv", "npv_lo", "npv_hi", "ppv", "ppv_lo", "ppv_hi")}; ref["source"] = r.set
        pre = MR[MR.split == a.pretrial]
        th = dict(rapid_reference=ref, level=a.level)
        if len(pre):
            th.update(choose(pre, ref, a)); th["pretrial_n"] = int(len(pre))
        json.dump(th, open(os.path.join(a.out, "thresholds.json"), "w"), indent=1)
        print(json.dumps(th, indent=1))
        pts = [(k, v) for k, v in th.items() if isinstance(v, dict) and "th0" in v]
        if pts:
            w = []
            for sname, X in rs.items():
                X = X.assign(**{a.rapid: X[a.rapid].astype(int)})
                if len(X) < a.min_n:
                    continue
                t0, t1 = [v["th0"] for _, v in pts], [v["th1"] for _, v in pts]
                P = U.get_performance_assisted(X, t0, t1, a.target, a.rapid, "score")
                B = U.get_performance_assisted_bootstrapped(X, t0, t1, a.nboot, a.target, a.rapid, "score", seed=a.seed)
                P = P.merge(B, on=["th0", "th1"]); P.insert(0, "point", [k for k, _ in pts]); P.insert(0, "n", len(X)); P.insert(0, "set", sname)
                P["locked"] = sname != f"split={a.pretrial}"
                w.append(P)
            Wf = pd.concat(w); Wf.to_csv(os.path.join(a.out, "workflow.csv"), index=False)
            print(Wf[["set", "n", "point", "th0", "th1", "npv", "ppv", "se", "sp", "reduction", "reduction_lo", "reduction_hi"]].round(4).to_string(index=False))

    # ---- figures
    if not a.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for sname, X in sets.items():
            if X[a.target].nunique() < 2:
                continue
            f, ax = plt.subplots(figsize=(4, 4))
            y, s = X[a.target].values, X.score.values
            fpr, tpr, _ = roc_curve(y, s)
            rs_ = np.random.RandomState(a.seed); grid = np.linspace(0, 1, 201); curves = []
            for _ in range(a.nboot):
                i = rs_.randint(0, len(y), len(y))
                if len(np.unique(y[i])) == 2:
                    fb, tb, _ = roc_curve(y[i], s[i]); curves.append(np.interp(grid, fb, tb))
            ax.fill_between(grid, np.quantile(curves, 0.025, 0), np.quantile(curves, 0.975, 0), alpha=0.25, lw=0)
            r = A[(A.set == sname) & (A.group_type == "all")].iloc[0]
            ax.plot(fpr, tpr, lw=1.5, label=f"AUC {r.auc:.3f} ({r.lo:.3f}-{r.hi:.3f}), n={len(y)}")
            ax.plot([0, 1], [0, 1], "k:", lw=0.7); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
            ax.set_title(sname, fontsize=9); ax.legend(loc="lower right", fontsize=7); f.tight_layout()
            f.savefig(os.path.join(a.out, f"roc_{sname.replace('=', '_')}.png"), dpi=200); plt.close(f)
        if has_rapid and "rapid_reference" in th:
            ref = th["rapid_reference"]
            mark = [(v["th0"], v["th1"], k.split("_")[0]) for k, v in th.items() if isinstance(v, dict) and "th0" in v]
            for sname in (f"split={a.pretrial}", f"split={a.silent}"):
                X = sets.get(sname)
                if X is None or X[a.rapid].isna().all() or X[a.target].nunique() < 2:
                    continue
                X = X[X[a.rapid].notna()].assign(**{a.rapid: lambda d: d[a.rapid].astype(int)})
                for win, (se_max, sp_min) in dict(full=(0.5, 0.5), small=(0.05, 0.95)).items():
                    for metric in ("reduction", "npv", "ppv"):
                        U.plot_simulation(X, metric, (ref["npv"], ref["ppv"]), (ref["npv_lo"], ref["npv_hi"], ref["ppv_lo"], ref["ppv_hi"]),
                                          se_max=se_max, sp_min=sp_min, N=201, target_col=a.target, rapid_col=a.rapid, eagle_col="score",
                                          savepath=os.path.join(a.out, f"heat_{sname.replace('=', '_')}_{metric}_{win}.png"),
                                          title=f"{sname} {metric}", mark=[m for m in mark if m[0] <= se_max and m[1] >= sp_min])
    print("EVAL_DONE", a.out)


if __name__ == "__main__":
    main()
