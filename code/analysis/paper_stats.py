"""Paper statistics and figures from the patient-level evaluation tables (runs locally on the copied results/).

  python code/analysis/paper_stats.py <project_root>

Outputs results/paper_stats/*.csv and paper/figures/Fig2_roc.png, Fig3_workflow.png, ExtendedDataFig1_cohort.png,
Fig1_overview.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
OUT = os.path.join(RES, "paper_stats")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
NB = 2000
rng = np.random.default_rng(0)

# warm palette (RECAST "cozy" style)
C = dict(ft="#B5523B", fr4="#E09F3E", fr6="#9E9E7E", ihc="#5B7065", ink="#3A2E2A", grid="#E8DED2", bg="#FFFBF5")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": C["ink"], "text.color": C["ink"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"],
                     "font.family": "DejaVu Sans", "savefig.dpi": 300, "figure.facecolor": "white"})

MODELS = {"ft": ("final_e020/eval/merged.csv", "Fine-tuned GigaPath (EAGLE recipe)"),
          "fr4": ("frozen_comparator/gma_lr1e-4/eval/merged.csv", "Frozen GigaPath, lr 1e-4"),
          "fr6": ("frozen_comparator/gma_lr1e-6/eval/merged.csv", "Frozen GigaPath, lr 1e-6")}
M = {k: pd.read_csv(os.path.join(RES, f), low_memory=False) for k, (f, _) in MODELS.items()}
key = ["cohort", "patient"]
base = M["ft"].copy()
for k in ("fr4", "fr6"):
    base = base.merge(M[k][key + ["score"]].rename(columns={"score": "score_" + k}), on=key, how="left")
base = base.rename(columns={"score": "score_ft"})
assert base[["score_fr4", "score_fr6"]].notna().all().all(), "patients differ between models"

SETS = {"Internal validation": base.split == "val", "Pretrial": base.split == "pretrial",
        "Silent (locked)": base.split == "silent", "EBRAINS": base.cohort == "ebrains",
        "CPTAC-GBM": base.cohort == "cptac", "All external": base.split == "external"}


def strat_boot_idx(y):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(NB):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


# 1. paired bootstrap AUC differences -------------------------------------------------------------------------------
rows = []
for name, m in SETS.items():
    d = base[m]; y = d.IDH.values.astype(int)
    s = {k: d["score_" + k].values for k in MODELS}
    auc = {k: roc_auc_score(y, v) for k, v in s.items()}
    bs = {k: [] for k in ("fr4", "fr6")}
    for ix in strat_boot_idx(y):
        a_ft = roc_auc_score(y[ix], s["ft"][ix])
        for k in bs:
            bs[k].append(a_ft - roc_auc_score(y[ix], s[k][ix]))
    for k in bs:
        b = np.array(bs[k])
        rows.append(dict(set=name, n=len(y), n_pos=int(y.sum()), comparator=MODELS[k][1], auc_ft=auc["ft"],
                         auc_comp=auc[k], delta=auc["ft"] - auc[k], lo=np.quantile(b, .025), hi=np.quantile(b, .975),
                         p_two_sided=min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))))
dAUC = pd.DataFrame(rows); dAUC.to_csv(os.path.join(OUT, "delta_auc_finetuned_vs_frozen.csv"), index=False)
print(dAUC.round(3).to_string())


# 2. harder within-histology comparisons ----------------------------------------------------------------------------
def auc_ci(y, s):
    a = roc_auc_score(y, s)
    b = [roc_auc_score(y[ix], s[ix]) for ix in strat_boot_idx(y)]
    return a, np.quantile(b, .025), np.quantile(b, .975)


E = base[base.cohort == "ebrains"]; T = base[(base.cohort == "tcga")]
hard = {
    "EBRAINS glioblastoma histology: IDH-mut vs IDH-wt": E[E.histology.str.startswith("Glioblastoma", na=False)],
    "EBRAINS astrocytoma histology (diffuse+anaplastic): IDH-mut vs IDH-wt": E[E.histology.str.contains(
        "^(Diffuse|Anaplastic) astrocytoma", na=False, regex=True)],
    "TCGA (val+pretrial+silent) TCGA-GBM project": T[T.project == "TCGA-GBM"],
    "TCGA (val+pretrial+silent) TCGA-LGG project": T[T.project == "TCGA-LGG"],
    "TCGA (val+pretrial+silent) grade 2-3 only": T[T.grade.isin(["G2", "G3"])],
    "TCGA (val+pretrial+silent) grade 4 only": T[T.grade == "G4"],
}
rows = []
for name, d in hard.items():
    y = d.IDH.values.astype(int)
    if y.min() == y.max():
        continue
    r = dict(subset=name, n=len(y), n_pos=int(y.sum()))
    for k in MODELS:
        a, lo, hi = auc_ci(y, d["score_" + k].values)
        r.update({f"auc_{k}": a, f"lo_{k}": lo, f"hi_{k}": hi})
    rows.append(r)
H = pd.DataFrame(rows); H.to_csv(os.path.join(OUT, "auc_within_histology.csv"), index=False)
print(H.round(3).to_string())

# 3. assisted workflow on the locked silent set: every case the workflow got wrong ----------------------------------
th = json.load(open(os.path.join(RES, "final_e020/eval/thresholds.json")))
W = pd.read_csv(os.path.join(RES, "final_e020/eval/workflow.csv"))
err = []
S = base[base.split == "silent"].copy()
for pt in ("most_conservative", "least_conservative"):
    t0, t1 = th[pt]["th0"], th[pt]["th1"]
    call = np.where(S.score_ft < t0, 0, np.where(S.score_ft > t1, 1, S.rapid))
    route = np.where(S.score_ft < t0, "model-negative", np.where(S.score_ft > t1, "model-positive", "stain"))
    wrong = call != S.IDH.values
    e = S[wrong][["patient", "institution", "project", "histology", "who_class", "IDH", "idh_variant", "rapid",
                  "score_ft", "n_slides", "n_tiles"]].copy()
    e["route"] = route[wrong]; e["point"] = pt
    err.append(e)
    # stain-only errors on the same patients, for comparison
    print(pt, "silent: stain-only errors", int((S.rapid != S.IDH).sum()), "assisted errors", int(wrong.sum()),
          "routes", pd.Series(route).value_counts().to_dict())
ERR = pd.concat(err); ERR.to_csv(os.path.join(OUT, "silent_workflow_errors.csv"), index=False)
print(ERR.to_string())

# 4. Figures ---------------------------------------------------------------------------------------------------------
# Fig 2: ROC curves, fine-tuned vs frozen, four sets
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1))
for ax, name in zip(axs, ["Internal validation", "Silent (locked)", "EBRAINS", "CPTAC-GBM"]):
    d = base[SETS[name]]; y = d.IDH.values.astype(int)
    for k, col, lw in (("fr6", C["fr6"], 1.0), ("fr4", C["fr4"], 1.0), ("ft", C["ft"], 1.6)):
        fpr, tpr, _ = roc_curve(y, d["score_" + k])
        ax.plot(fpr, tpr, color=col, lw=lw, label=f"{['Frozen 1e-6', 'Frozen 1e-4', 'Fine-tuned'][['fr6', 'fr4', 'ft'].index(k)]} "
                                              f"{roc_auc_score(y, d['score_' + k]):.3f}")
    if name != "EBRAINS":  # simulated stain operating point
        r = d.dropna(subset=["rapid"])
        ax.scatter(1 - ((r.rapid == 0) & (r.IDH == 0)).sum() / (r.IDH == 0).sum(),
                   ((r.rapid == 1) & (r.IDH == 1)).sum() / (r.IDH == 1).sum(), marker="D", s=18, color=C["ihc"],
                   zorder=5, label="R132H stain (sim.)")
    ax.plot([0, 1], [0, 1], ls=":", color=C["grid"], lw=0.8)
    ax.set_title(f"{name}\n{len(y)} patients, {y.sum()} mutant")
    ax.set_xlabel("1 − specificity"); ax.set_aspect("equal"); ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
    ax.legend(loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axs[0].set_ylabel("Sensitivity")
fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(FIG, "Fig2_roc.png")); plt.close(fig)

# Fig 3: (a) silent score distribution with locked thresholds; (b) stains avoided across sets with CI
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
lg = lambda v: np.log10(np.clip(v, 1e-6, 1 - 1e-9))
bins = np.linspace(-6, 0, 37)
for lab, col, txt in ((0, C["ihc"], "IDH-wildtype"), (1, C["ft"], "IDH-mutant")):
    a1.hist(lg(S[S.IDH == lab].score_ft), bins=bins, color=col, alpha=.75, label=f"{txt} (n={int((S.IDH == lab).sum())})")
miss = S[(S.IDH == 1) & (S.rapid == 0)]
a1.scatter(lg(miss.score_ft), np.full(len(miss), -0.6), marker="^", color=C["ink"], s=14, clip_on=False,
           label=f"mutant, stain-negative (n={len(miss)})")
for pt, ls in (("most_conservative", "-"), ("least_conservative", "--")):
    a1.axvline(np.log10(th[pt]["th0"]), color=C["ink"], ls=ls, lw=.9)
a1.axvline(np.log10(th["most_conservative"]["th1"]), color=C["ink"], ls="-", lw=.9)
a1.set_xlabel("Model score (log10), silent set, patient level"); a1.set_ylabel("Patients")
a1.set_title("a  Silent set with locked thresholds\n(solid: most, dashed: least conservative)", loc="left")
a1.legend(frameon=False, loc="upper center")
for s_ in ("top", "right"):
    a1.spines[s_].set_visible(False)
order = ["split=pretrial", "split=val", "split=silent", "cohort=cptac"]
names = ["Pretrial\n(threshold set)", "Validation", "Silent\n(locked)", "CPTAC-GBM"]
x = np.arange(len(order)); w = .38
for j, (pt, col) in enumerate((("most_conservative", C["ft"]), ("least_conservative", C["fr4"]))):
    ww = W[W.point == pt].set_index("set").loc[order]
    a2.bar(x + (j - .5) * w, ww.reduction * 100, w, color=col,
           yerr=[(ww.reduction - ww.reduction_lo) * 100, (ww.reduction_hi - ww.reduction) * 100],
           error_kw=dict(lw=.7, capsize=2, ecolor=C["ink"]), label=pt.replace("_", " "))
    for xi, (_, r) in zip(x, ww.iterrows()):
        a2.text(xi + (j - .5) * w, 2, f"NPV\n{r.npv:.2f}", ha="center", va="bottom", fontsize=5.5, color="white")
a2.set_xticks(x, names); a2.set_ylabel("Immunostains avoided (%)"); a2.set_ylim(0, 100)
a2.set_title("b  Immunostains avoided\n(95% bootstrap CI)", loc="left"); a2.legend(frameon=False, loc="upper left")
for s_ in ("top", "right"):
    a2.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "Fig3_workflow.png")); plt.close(fig)


# Extended Data Fig 1: cohort flow
def box(ax, x, y, w, h, txt, fc, fs=6.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=C["ink"], lw=.7))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=.7, color=C["ink"]))


coh = pd.read_csv(os.path.join(RES, "cohort/slide_data.csv"), low_memory=False)
fig, ax = plt.subplots(figsize=(9.5, 3.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
t = coh[coh.cohort == "tcga"]
box(ax, .22, .9, .42, .14, f"TCGA-GBM + TCGA-LGG diagnostic slides with IDH label\n{len(t)} slides, "
    f"{t.patient.nunique()} patients, 30 canonical institutions", C["bg"])
for i, (sp, lab) in enumerate((("train", "Train"), ("val", "Validation"), ("pretrial", "Pretrial\n(thresholds)"),
                               ("silent", "Silent\n(locked, once)"))):
    d = t[t.split == sp]; xx = .055 + i * .11
    box(ax, xx, .55, .1, .3, f"{lab}\n{len(d)} slides\n{d.patient.nunique()} pts\n{d.institution.nunique()} inst.\n"
        f"{d.groupby('patient').IDH.first().mean() * 100:.0f}% mut", "#F6E3D3" if sp != "train" else "#EADBC8")
    arrow(ax, .22, .83, xx, .70)
ax.text(.22, .32, "Split by institution: no hospital in more than one set\n(UCSF, MD Anderson, Besta, UNC aliases merged)",
        ha="center", fontsize=6.3, style="italic")
for xx, cname, lab in ((.6, "ebrains", "EBRAINS Digital Brain\nTumour Atlas (Vienna)\nWHO 2016 integrated diagnosis"),
                       (.86, "cptac", "CPTAC-GBM (IDC DICOM)\nmulti-site\nsequencing labels")):
    d = coh[coh.cohort == cname]
    box(ax, xx, .9, .22, .16, lab, C["bg"])
    box(ax, xx, .55, .22, .3, f"External test\n{len(d)} slides\n{d.patient.nunique()} pts\n"
        f"{d.groupby('patient').IDH.first().mean() * 100:.1f}% mut\n"
        + ("AUC only\n(no variant, no stain simulation)" if cname == "ebrains" else "AUC + locked workflow"), "#E4EAE4")
    arrow(ax, xx, .82, xx, .70)
ax.text(.73, .32, "Never used for training, model selection or thresholds", ha="center", fontsize=6.3, style="italic")
ax.text(.5, .12, "Patient-level evaluation: mean of slide scores. 2 EBRAINS patients with discordant labels across slides "
        "excluded (4 slides).", ha="center", fontsize=6.3)
fig.savefig(os.path.join(FIG, "ExtendedDataFig1_cohort.png"), bbox_inches="tight"); plt.close(fig)

# Fig 1: study overview
fig, ax = plt.subplots(figsize=(9.5, 2.8)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
steps = [(.09, "H&E diagnostic slide\nall tissue tiles\n224 px, 0.5 µm/px", C["bg"]),
         (.29, "Prov-GigaPath ViT-g\n(1.13 B params)\nfine-tuned end to end", "#F6E3D3"),
         (.49, "Gated-attention MIL\naggregator\n→ P(IDH-mutant)", "#F6E3D3"),
         (.71, "Assisted workflow\nscore < t₀ → negative\nscore > t₁ → positive\nelse → R132H stain", "#E4EAE4"),
         (.91, "Sequencing\nconfirms all\n(reference)", C["bg"])]
for x0, txt, fc in steps:
    box(ax, x0, .62, .17, .42, txt, fc)
for (a_, _, _), (b_, _, _) in zip(steps[:-1], steps[1:]):
    arrow(ax, a_ + .086, .62, b_ - .086, .62)
ax.text(.29, .14, "Training: TCGA train split, 2,208 random tiles per slide per step, 20 epochs, AdamW lr 1e-6\n"
        "two-pass chunked gradient = authors' 24-GPU step on 1–4 GPUs", ha="center", fontsize=6.4, va="center")
ax.text(.81, .14, "t₀, t₁ chosen on pretrial set\n(NPV, PPV ≥ stain alone), then locked\nand applied once to the silent set",
        ha="center", fontsize=6.4, va="center")
fig.savefig(os.path.join(FIG, "Fig1_overview.png"), bbox_inches="tight"); plt.close(fig)
print("figures written to", FIG)
