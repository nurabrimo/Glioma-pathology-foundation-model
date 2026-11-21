#!/bin/bash
#SBATCH --job-name=eg_frozen
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-core-gpu-ferc-s-h36-6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/work/nb295/eagle_glioma/logs/frozen_%x_%j.out
# "No fine-tuning" comparator on dumped frozen-encoder features (see infer.sbatch feature-dump mode; train slides must be dumped too: SEL="").
#   TAG=frozen_gigapath sbatch slurm/frozen_gma.sbatch               # lr 1e-6 (identical to end-to-end)
#   TAG=frozen_gigapath LR=1e-4 sbatch slurm/frozen_gma.sbatch       # conventional frozen-feature rate
source $HOME/recast/dcc/config.sh
PY=$CONDA_ENV/bin/python
cd $HOME/eagle_glioma
F=/work/nb295/eagle_glioma/runs/${TAG:?set TAG}; LR=${LR:-1e-6}
$PY train/train_frozen_gma.py --feat_dir $F/h5 --outdir $F/gma_lr$LR --lr $LR ${EXTRA:-} && \
$PY calib/evaluate.py --scores $F/gma_lr$LR/scores.csv --out $F/gma_lr$LR/eval
