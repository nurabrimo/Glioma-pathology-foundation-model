#!/bin/bash
# 5 CPUs per GPU (3 loader workers per rank are enough: a rank needs ~40 tiles/s, a worker reads ~75/s); the account
# has a 400-CPU group limit shared with all other projects.
# bash slurm/submit_train.sh <partition> <gres, e.g. gpu:6000_ada:4> [begin]     (env: RUN, INIT, CHUNK, CKPT_MIN, EXTRA)
PART=$1; GRES=$2; BEGIN=${3:-now}
N=${GRES##*:}; TL=7-00:00:00; [ "$PART" = gpu-common ] && TL=2-00:00:00
sbatch -p $PART --gres=$GRES --cpus-per-task=$((5*N)) --mem=$((28*N))G --time=$TL --begin=$BEGIN \
  -J eg_${RUN:?set RUN} --export=ALL,EG_PART=$PART,EG_GRES=$GRES $HOME/eagle_glioma/slurm/train_e2e.sbatch
