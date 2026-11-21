#!/bin/bash
# CPU only, a few minutes: run inside an sbatch --wrap or dcc/77_py.sbatch.
#   bash slurm/evaluate.sh idh_gigapath 020
source $HOME/recast/dcc/config.sh
R=/work/nb295/eagle_glioma/runs/$1/infer_e${2:-020}
$CONDA_ENV/bin/python $HOME/eagle_glioma/train/infer.py --merge $R && \
$CONDA_ENV/bin/python $HOME/eagle_glioma/calib/evaluate.py --scores $R/scores.csv --out $R/eval ${@:3}
