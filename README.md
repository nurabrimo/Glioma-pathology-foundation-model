# Glioma pathology foundation model (EAGLE protocol for IDH)

Replication, for IDH mutation in adult diffuse glioma, of Campanella *et al.*, "Real-world deployment of a fine-tuned
pathology foundation model for lung cancer biomarker detection", *Nature Medicine* 2025
(doi [10.1038/s41591-025-03780-x](https://doi.org/10.1038/s41591-025-03780-x); original code
[chadvanderbilt/EAGLE](https://github.com/chadvanderbilt/EAGLE)).

The authors' released recipe is followed unchanged:

- **Model.** Prov-GigaPath ViT-g encoder plus the authors' gated-attention aggregator.
- **Training.** End to end, 2,208 tiles per slide per step, AdamW at learning rate 1e-6, 20 epochs.
- **Calibration.** The authors' two-threshold procedure. The rapid test it calibrates against is a simulated IDH1
  R132H immunostain (positive exactly when the tumour carries R132H).

## Results (patient level, epoch-20 model)

| Set | Patients (mutant) | AUC (95% CI) |
|---|---|---|
| TCGA internal validation (held-out institutions) | 151 (69) | 0.945 (0.902–0.978) |
| TCGA pretrial (threshold selection) | 86 (50) | 0.956 (0.905–0.998) |
| TCGA silent (locked, scored once) | 89 (49) | 0.949 (0.896–0.989) |
| EBRAINS Digital Brain Tumour Atlas | 856 (291) | 0.944 (0.926–0.960) |
| CPTAC-GBM | 152 (10) | 0.907 (0.816–0.979) |

- **Assisted workflow on the locked silent set** (most conservative thresholds, *t*₀ = 0.023 and *t*₁ = 0.998):
  - 55.1% of immunostains avoided;
  - NPV 0.929 (stain alone 0.889);
  - PPV 0.979 (stain alone 1.000), because of one false positive.
- **Frozen GigaPath with the same aggregator, learning rate 1e-4:** statistically indistinguishable from end-to-end
  fine-tuning. ΔAUC ranges from −0.003 to +0.011 and every 95% CI includes 0.
- **Frozen GigaPath at EAGLE's learning rate (1e-6):** underfits. Fine-tuning beats it by +0.06 to +0.19 AUC.

## Layout

- **`code/data/`:** cohort labels, GDC download and checksum check, tissue segmentation, 224 px tiles at 0.5 µm/px,
  and the institution-separated splits (`finalize_slides.py`).
- **`code/train/`:**
  - `train_e2e.py`: end-to-end fine-tuning. It uses a two-pass chunked gradient that equals the authors' 24-GPU step
    on 1–4 GPUs; `test_grad_equiv.py` checks this.
  - `infer.py`: full-slide inference.
  - `train_frozen_gma.py`: frozen-encoder comparator.
- **`code/calib/`:**
  - `utils.py` (fetched): the authors' calibration functions.
  - `evaluate.py`: AUCs, subgroups, the rapid-test reference, threshold search, and the locked application.
- **`code/analysis/`:**
  - `paper_stats.py`: paired ΔAUC, within-histology AUCs, silent-set errors and figures.
  - `attention_fig.py`: attention maps.
- **`code/slurm/`:** job scripts for the Duke Compute Cluster.
- **`results/cohort/`, `results/tiles/`:** slide labels, splits and tile coordinates (TCGA and CPTAC).
- **`results/final_e020/`:** slide and patient scores and the full evaluation output for the final model.
- **`results/frozen_comparator/`:** the same outputs for the frozen-encoder baselines.
- **`results/paper_stats/`, `results/training/`, `results/figures/`:** analysis tables, training curves and figures.

## Not included

- **Slides.** Download them from the GDC, the NCI Imaging Data Commons and EBRAINS.
- **Prov-GigaPath weights.** They are gated on Hugging Face.
- **Fine-tuned model weights.**
- **Per-slide EBRAINS labels and scores.** These are covered by the EBRAINS data-use agreement; only summary numbers
  are given here.

## Reproduce

0. Run `bash code/fetch_eagle_files.sh` to download the authors' files this code imports (see Licence).
1. Run `code/data` in numbered order to build the labels, download the slides and make the tiles.
2. Train: `sbatch code/slurm/train_e2e.sbatch`.
3. Score: `RUN=idh_gigapath EPOCH=020 sbatch --array=0-47%16 code/slurm/infer.sbatch`.
4. Evaluate: `bash code/slurm/evaluate.sh idh_gigapath 020`.
5. Make the paper statistics and figures: `python code/analysis/paper_stats.py <root>`.

Paths in the job scripts point to the Duke Compute Cluster and need to be changed for other systems.

## Licence

The EAGLE code is covered by MSK's EAGLE Terms of Use, which do not allow it to be redistributed. The files this
project imports from it are therefore not in this repository. These are `modules.py`, `extra_transforms.py`,
`training/utils.py` and `calibration/utils.py`. `code/fetch_eagle_files.sh` downloads them from the authors' repository
at commit `627f4e0`.

The reported results used `calibration/utils.py` with two fixes:

- confusion matrices declared with `labels=[0, 1]`, so a single-class bootstrap sample does not fail;
- a vectorised threshold grid search.

Use of the EAGLE files is subject to the authors' terms.
