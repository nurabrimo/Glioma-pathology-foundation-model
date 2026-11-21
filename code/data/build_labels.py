"""EAGLE-glioma step 1: slide-level label table (no splits, no tile counts yet).

Biomarker: IDH (IDH1 R132* or IDH2 R172* hotspot mutation). Rapid-test analogue: IDH1 R132H immunohistochemistry,
simulated as rapid = 1 iff the tumour carries IDH1 p.R132H (0 for wild type and for non-canonical mutants).

Sources
  TCGA   slides : GDC API (all open-access diagnostic SVS of TCGA-GBM / TCGA-LGG; id, md5, size, year of diagnosis, TSS)
         IDH    : Ceccarelli et al. 2016 (cBioPortal lgggbm_tcga_pub, IDH_STATUS in $MP_ROOT/data/processed/cohort.csv);
                  variant from the same study's MAF (/work/nb295/recast/data/cbio/data_mutations.txt), then GDC open
                  ssm_occurrences as a second source. Patients whose status is NA in Ceccarelli get the GDC call when
                  the case has open somatic mutation calls (label_source records which).
  EBRAINS: ~/recast labels_ebrains.csv (integrated WHO-2016 diagnosis implies IDH status; no variant -> rapid NA)
  CPTAC  : GDC open MAFs (/work/nb295/recast/ext/cptac/gdc_maf) + cBioPortal gbm_cptac_2021 mutations.

Output: <out>/slide_labels.csv, <out>/tcga_gdc_files.csv, <out>/tcga_download_manifest.txt (slides not yet on DCC)
"""
import argparse, glob, gzip, json, os, re
import numpy as np
import pandas as pd
import requests

GDC = "https://api.gdc.cancer.gov"
ALIASES = {  # same map as ~/recast/scripts/fedplan_partition.py
    "University of California San Francisco": "UCSF",
    "MD Anderson": "MD Anderson Cancer Center",
    "Milan - Italy, Fondazione IRCCS Instituto Neuroligico C. Besta": "Fondazione-Besta",
    "University of North Carolina": "UNC",
}
PROJ = ["TCGA-GBM", "TCGA-LGG"]


def gdc(endpoint, filters, fields, size=20000, expand=None):
    p = {"filters": json.dumps(filters), "fields": ",".join(fields), "size": size, "format": "JSON"}
    if expand: p["expand"] = expand
    r = requests.post(f"{GDC}/{endpoint}", json=p, timeout=300); r.raise_for_status()
    d = r.json()["data"]; assert d["pagination"]["total"] <= size, d["pagination"]
    return d["hits"]


def F(field, vals):
    return {"op": "in", "content": {"field": field, "value": vals}}


def hotspot(gene, aa):
    """Return canonical variant name for an IDH hotspot change, else None."""
    aa = str(aa).replace("p.", "")
    if gene == "IDH1" and re.match(r"^R132[A-Z]$", aa): return f"IDH1_{aa}"
    if gene == "IDH2" and re.match(r"^R172[A-Z]$", aa): return f"IDH2_{aa}"
    return None


def tcga(a):
    files = gdc("files", {"op": "and", "content": [F("cases.project.project_id", PROJ), F("data_format", ["SVS"]),
                                                    F("experimental_strategy", ["Diagnostic Slide"]), F("access", ["open"])]},
                ["file_id", "file_name", "md5sum", "file_size", "cases.submitter_id", "cases.project.project_id"])
    Fd = pd.DataFrame([dict(file_id=h["file_id"], file_name=h["file_name"], md5=h["md5sum"], size=h["file_size"],
                            patient=h["cases"][0]["submitter_id"], project=h["cases"][0]["project"]["project_id"]) for h in files])
    cases = gdc("cases", F("project.project_id", PROJ),
                ["submitter_id", "diagnoses.year_of_diagnosis", "tissue_source_site.name", "tissue_source_site.code",
                 "available_variation_data", "diagnoses.primary_diagnosis"], size=3000)
    C = pd.DataFrame([dict(patient=h["submitter_id"],
                           year_dx=next((d.get("year_of_diagnosis") for d in h.get("diagnoses", []) if d.get("year_of_diagnosis")), np.nan),
                           gdc_dx=next((d.get("primary_diagnosis") for d in h.get("diagnoses", []) if d.get("primary_diagnosis")), None),
                           tss_code=h.get("tissue_source_site", {}).get("code"), tss_name=h.get("tissue_source_site", {}).get("name"),
                           has_ssm="ssm" in (h.get("available_variation_data") or [])) for h in cases]).set_index("patient")
    occ = gdc("ssm_occurrences", {"op": "and", "content": [F("case.project.project_id", PROJ),
                                                            F("ssm.consequence.transcript.gene.symbol", ["IDH1", "IDH2"])]},
              ["case.submitter_id", "ssm.consequence.transcript.aa_change", "ssm.consequence.transcript.gene.symbol",
               "ssm.consequence.transcript.is_canonical"], size=5000)
    gv = {}
    for h in occ:
        for c in h["ssm"]["consequence"]:
            t = c["transcript"]; v = hotspot(t["gene"]["symbol"], t.get("aa_change"))
            if v and t.get("is_canonical"): gv.setdefault(h["case"]["submitter_id"], set()).add(v)
    # Ceccarelli MAF
    M = pd.read_csv(a.cbio_maf, sep="\t", comment="#", low_memory=False, usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "HGVSp_Short"])
    M = M[M.Hugo_Symbol.isin(["IDH1", "IDH2"])]
    cv = {}
    for g, s, aa in M.itertuples(index=False):
        v = hotspot(g, aa)
        if v: cv.setdefault(s[:12], set()).add(v)
    seq = set(open(a.cbio_maf).readline().replace("#sequenced_samples:", "").split())
    seq = {s[:12] for s in seq}
    coh = pd.read_csv(a.cohort).set_index("patient")
    tss2inst = coh.dropna(subset=["institution"]).groupby("tss_code").institution.first().to_dict()

    rows = []
    for r in Fd.itertuples(index=False):
        p = r.patient; c = C.loc[p] if p in C.index else None
        ce = coh.loc[p] if p in coh.index else None
        st = ce.IDH_STATUS if ce is not None and isinstance(ce.IDH_STATUS, str) else None
        var_c = sorted(cv.get(p, [])); var_g = sorted(gv.get(p, []))
        var = var_c or var_g
        if st in ("Mutant", "WT"):
            idh = int(st == "Mutant"); src = "Ceccarelli2016"
        elif var_g:
            idh, src = 1, "GDC_ssm"
        elif c is not None and bool(c.has_ssm):
            idh, src = 0, "GDC_ssm"
        else:
            idh, src = np.nan, "none"
        note = ""
        if idh == 1:
            variant = "+".join(var) if var else "MUT_unknown_variant"
            rapid = (1 if "IDH1_R132H" in var else 0) if var else np.nan
            vsrc = "Ceccarelli_MAF" if var_c else ("GDC_ssm" if var_g else "none")
        elif idh == 0:
            variant, rapid, vsrc = "WT", 0, ""
            if var: note = f"status WT but hotspot call {'+'.join(var)} in {'Ceccarelli MAF' if var_c else 'GDC ssm'}"
        else:
            variant, rapid, vsrc = None, np.nan, ""
        tss = p.split("-")[1]
        inst_raw = tss2inst.get(tss) or (c.tss_name if c is not None else None)
        codel = np.nan
        if ce is not None and isinstance(ce.IDH_CODEL_SUBTYPE, str):
            codel = int(ce.IDH_CODEL_SUBTYPE == "IDHmut-codel")
        who = {0: "GBM-IDHwt"}.get(idh) if idh == 0 else (None if not idh == 1 or np.isnan(codel) else ("Oligo-IDHmut-codel" if codel else "Astro-IDHmut"))
        if idh == 0 and ce is not None and isinstance(ce.who2021_like, str) and ce.who2021_like != "GBM_IDHwt":
            who = "IDHwt-other(" + ce.who2021_like + ")"
        rows.append(dict(slide=r.file_name[:-4], file_id=r.file_id, file_name=r.file_name, md5=r.md5, size=r.size, cohort="tcga",
                         patient=p, institution_raw=inst_raw, institution=ALIASES.get(str(inst_raw).strip(), str(inst_raw).strip()) if inst_raw else None,
                         tss_code=tss, project=r.project,
                         histology=(ce.HISTOLOGICAL_DIAGNOSIS if ce is not None else None) or (c.gdc_dx if c is not None else None),
                         grade=ce.GRADE if ce is not None else None, who_class=who,
                         year_dx=c.year_dx if c is not None else np.nan, IDH=idh, idh_variant=variant, rapid=rapid, codel=codel,
                         label_source=src, variant_source=vsrc, note=note))
    D = pd.DataFrame(rows)
    Fd.to_csv(os.path.join(a.out, "tcga_gdc_files.csv"), index=False)
    return D


def ebrains(a):
    L = pd.read_csv(a.ebrains_labels); L = L[L.idh_mut.notna()].copy()
    paths = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(a.ebrains_root, "*", "*", "*.ndpi"))}
    cls = {0.0: "GBM-IDHwt", 1.0: "Astro-IDHmut", 2.0: "Oligo-IDHmut-codel"}
    rows = []
    for r in L.itertuples(index=False):
        rows.append(dict(slide=r.slide, path=paths.get(r.slide), cohort="ebrains", patient=r.patient, institution="MedUni Vienna (EBRAINS DBTA)",
                         project="EBRAINS-DBTA", histology=r.diagnosis_2016, grade=r.GRADE if isinstance(r.GRADE, str) else None,
                         who_class=cls.get(r.cns5_idx, "IDHwt-astrocytoma-2016(unresolved)" if r.scope == "ambiguous" else None),
                         year_dx=np.nan, IDH=int(r.idh_mut), idh_variant="WT" if r.idh_mut == 0 else "MUT_unknown_variant",
                         rapid=np.nan, codel=r.codel, label_source="DBTA integrated diagnosis (WHO 2016)", variant_source="",
                         note=("recurrence;" if r.recurrence == 1 else "") + (str(r.note) if isinstance(r.note, str) else "")))
    return pd.DataFrame(rows)


def cptac(a):
    S = pd.read_csv(a.cptac_series)
    man = pd.read_csv(os.path.join(a.cptac_dir, "gdc_maf_manifest.csv"))
    var, has_maf = {}, set()
    for r in man.itertuples(index=False):
        f = os.path.join(a.cptac_dir, "gdc_maf", r.file_name)
        if not os.path.exists(f): continue
        has_maf.add(r.case)
        try:
            m = pd.read_csv(f, sep="\t", comment="#", low_memory=False, usecols=["Hugo_Symbol", "HGVSp_Short"])
        except Exception as e:
            print("MAF read fail", f, e); continue
        for g, aa in m[m.Hugo_Symbol.isin(["IDH1", "IDH2"])].itertuples(index=False):
            v = hotspot(g, aa)
            if v: var.setdefault(r.case, set()).add(v)
    cb = json.load(open(os.path.join(a.cptac_dir, "cbio", "mutations.json")))
    # sequenced set = cBioPortal "<study>_sequenced" sample list (samples.json lists every sample, sequenced or not)
    cb_samples = set()
    slp = os.path.join(a.cptac_dir, "cbio", "sample_lists.json")
    if os.path.exists(slp):
        for sl in json.load(open(slp)):
            if str(sl.get("sampleListId", "")).endswith("_sequenced"): cb_samples |= set(sl.get("sampleIds", []))
    print("CPTAC cBioPortal sequenced samples:", len(cb_samples))
    cbv = {}
    for m in cb:
        v = hotspot(m["gene"]["hugoGeneSymbol"], m.get("proteinChange"))
        if v: cbv.setdefault(m["patientId"], set()).add(v)
    rows = []
    for r in S.itertuples(index=False):
        p = r.patient; v = sorted(var.get(p, set()) | cbv.get(p, set()))
        if v: idh, variant, rapid = 1, "+".join(v), int("IDH1_R132H" in v)
        elif p in has_maf or p in cb_samples: idh, variant, rapid = 0, "WT", 0
        else: idh, variant, rapid = np.nan, None, np.nan
        src = "+".join([s for s, ok in (("GDC_MAF", p in has_maf), ("cBioPortal gbm_cptac_2021", p in cb_samples)) if ok]) or "none"
        rows.append(dict(slide=r.stem, path=os.path.join(a.cptac_root, r.wsi), cohort="cptac", patient=p, institution="CPTAC-GBM (multi-site)",
                         project="CPTAC-GBM", histology="Glioblastoma", grade="G4", who_class="GBM-IDHwt" if idh == 0 else ("Astro-IDHmut" if idh == 1 else None),
                         year_dx=np.nan, IDH=idh, idh_variant=variant, rapid=rapid, codel=0 if idh == idh else np.nan, label_source=src,
                         variant_source=src if idh == 1 else "", note=f"series {r.series}"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/nb295/eagle_glioma")
    ap.add_argument("--cohort", default=os.path.expanduser("~/mechanopath_full/data/processed/cohort.csv"))
    ap.add_argument("--cbio_maf", default="/work/nb295/recast/data/cbio/data_mutations.txt")
    ap.add_argument("--tcga_wsi", default="/cwork/nb295/mechanopath/wsi")
    ap.add_argument("--tcga_extra", default="/cwork/nb295/eagle_glioma/tcga_dx_extra")
    ap.add_argument("--ebrains_labels", default="/work/nb295/recast/ext/ebrains/labels_ebrains.csv")
    ap.add_argument("--ebrains_root", default="/cwork/nb295/recast_data/ebrains/dbta")
    ap.add_argument("--cptac_dir", default="/work/nb295/recast/ext/cptac")
    ap.add_argument("--cptac_series", default="/cwork/nb295/recast_feats/cptac_gbm/series_map.csv")
    ap.add_argument("--cptac_root", default="/cwork/nb295/recast_ext/cptac/idc")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.tcga_extra, exist_ok=True)
    T = tcga(a)
    have = {os.path.basename(p): p for p in glob.glob(os.path.join(a.tcga_wsi, "*", "*.svs"))}
    T["path"] = [have.get(f) or os.path.join(a.tcga_extra, i, f) for i, f in zip(T.file_id, T.file_name)]
    T["on_dcc"] = T.file_name.isin(have)
    need = T[(~T.on_dcc) & T.IDH.notna()]
    need[["file_id", "file_name", "md5", "size"]].assign(state="released").rename(columns={"file_id": "id", "file_name": "filename"}).to_csv(
        os.path.join(a.out, "tcga_download_manifest.txt"), sep="\t", index=False)
    E = ebrains(a); C = cptac(a)
    D = pd.concat([T, E, C], ignore_index=True)
    D.to_csv(os.path.join(a.out, "slide_labels.csv"), index=False)
    print("TCGA DX slides", len(T), "patients", T.patient.nunique(), "| IDH known slides", int(T.IDH.notna().sum()),
          "patients", T[T.IDH.notna()].patient.nunique(), "| on DCC", int(T.on_dcc.sum()), "| to download", len(need), f"{need['size'].sum()/1e9:.0f} GB")
    print(T.groupby(["label_source"]).patient.nunique().to_dict())
    print("institution missing:", int(T.institution.isna().sum()), "| year missing:", int(T.year_dx.isna().sum()))
    print("TCGA notes:", T[T.note != ""].drop_duplicates("patient")[["patient", "IDH", "note"]].to_string(index=False))
    for c, G in D.groupby("cohort"):
        P = G.drop_duplicates("patient"); K = P[P.IDH.notna()]; Mu = K[K.IDH == 1]
        print(f"{c}: slides {len(G)} (IDH known {int(G.IDH.notna().sum())}), patients {len(P)} (IDH known {len(K)}), mutant {len(Mu)},",
              "variants", Mu.idh_variant.value_counts().to_dict(), "| path missing", int(G.path.isna().sum()))


if __name__ == "__main__":
    main()
