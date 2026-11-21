#!/bin/bash
#SBATCH --job-name=eg_infer
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-core-gpu-ferc-s-h36-6
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/work/nb295/eagle_glioma/logs/infer_%x_%A_%a.out
# Full-slide inference, all tissue tiles, sharded; resumable (finished slides are skipped). Any GPU works (2080 Ti: BS<=64).
# Fine-tuned model, paper setting (full precision):
#   RUN=idh_gigapath EPOCH=020 sbatch --array=0-47%16 slurm/infer.sbatch                 # val pretrial silent external
#   RUN=idh_gigapath EPOCH=020 FEATS=1 SEL="--splits silent" sbatch --array=0-3 slurm/infer.sbatch   # + attention h5
# Frozen-encoder feature dump (no aggregator):
#   ENCODER=gigapath EG_FEAT_DIR=/cwork/nb295/eagle_glioma/feats_gigapath_frozen AMP=1 SEL="" sbatch --array=0-63%16 slurm/infer.sbatch
source $HOME/recast/dcc/config.sh   # NB: exports FEAT_DIR (RECAST features); use EG_FEAT_DIR here
PY=$CONDA_ENV/bin/python
cd $HOME/eagle_glioma/train
N=${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}; I=${SLURM_ARRAY_TASK_ID:-0}   # set NSHARDS when resubmitting a subset of array ids
SEL=${SEL---splits val pretrial silent external}
if [ -n "$RUN" ]; then
  R=/work/nb295/eagle_glioma/runs/$RUN; EPOCH=${EPOCH:-020}
  ENC=$R/checkpoint_tile_$EPOCH.pth; AGG="--slide_ckpt $R/checkpoint_slide_$EPOCH.pth"; OUT=$R/infer_e$EPOCH${AMP:+_amp}
  FD=""; [ -n "$FEATS" ] && FD="--feat_dir $OUT/h5 --save_feats ${SAVE_FEATS:-0}"
else
  ENC=${ENCODER:?set RUN or ENCODER}; AGG=""; FD="--feat_dir ${EG_FEAT_DIR:?set EG_FEAT_DIR}"; OUT=$EG_FEAT_DIR/_logs
fi
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); BS=${BS:-$([ $MEM -lt 16000 ] && echo 32 || echo 128)}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; hostname
$PY infer.py --encoder $ENC $AGG --out $OUT $FD $SEL --shard $I --nshards $N --amp ${AMP:-0} --batch_size $BS --workers $((SLURM_CPUS_PER_TASK-1))
# after all shards:  bash slurm/evaluate.sh <RUN> <EPOCH>
