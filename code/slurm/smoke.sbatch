#!/bin/bash
#SBATCH --job-name=eg_smoke
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-core-gpu-ferc-s-h36-6
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/work/nb295/eagle_glioma/logs/smoke_%x_%j.out
# usage: sbatch --gres=gpu:a6000:1 -J eg_smoke_a6000 slurm/smoke.sbatch
source $HOME/recast/dcc/config.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f /work/nb295/eagle_glioma/smoke/SMOKE_OK ] && { echo "smoke already passed elsewhere"; exit 0; }
PY=$CONDA_ENV/bin/python
cd $HOME/eagle_glioma/train
TAG=${SLURM_JOB_NAME}
SM=/work/nb295/eagle_glioma/smoke; RUN=/work/nb295/eagle_glioma/runs/$TAG
CHUNK=${CHUNK:-0}; INIT=${INIT:-gigapath}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; hostname
[ -f $SM/slide_data.csv ] || $PY make_smoke_table.py $SM 8
rm -rf $RUN; mkdir -p $RUN
echo "=== tile reading benchmark"; $PY bench_read.py $SM 10
COMMON="--slide_data $SM/slide_data.csv --tiles $SM/tiles --outdir $RUN --init $INIT --workers 10 --chunk $CHUNK --ckpt_minutes 0"
echo "=== train run 1 (2 steps, then stop)"; $PY train_e2e.py $COMMON --max_steps 2 || exit 1
echo "=== train run 2 (resume, 3 steps)"; $PY train_e2e.py $COMMON --max_steps 3 || exit 1
cat $RUN/convergence.csv; cat $RUN/steps.csv
echo "=== run 3 (resume; 2 more steps with AdamW state present)"; $PY train_e2e.py $COMMON --max_steps 2 || exit 1; cat $RUN/steps.csv
echo "=== 2-rank loop on one GPU (gloo; mechanics only)"; cp $RUN/steps.csv $RUN/steps_1gpu.csv
[ $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) -gt 45000 ] && EG_DIST_BACKEND=gloo $CONDA_ENV/bin/torchrun --nproc_per_node=2 train_e2e.py $COMMON --chunk 4 --load_bs 48 --workers 5 --max_steps 2 2>&1 | grep -E "world=|loss:|resumed|Error|error" | tail -6
echo "=== inference fp32"; $PY infer.py --slide_data $SM/slide_data.csv --tiles $SM/tiles --splits val --encoder $RUN/checkpoint_last.pth --slide_ckpt $RUN/checkpoint_last.pth --out $RUN/infer_fp32 --feat_dir $RUN/feats --workers 10
echo "=== inference amp"; $PY infer.py --slide_data $SM/slide_data.csv --tiles $SM/tiles --splits val --encoder $RUN/checkpoint_last.pth --slide_ckpt $RUN/checkpoint_last.pth --out $RUN/infer_amp --amp 1 --workers 10
$PY infer.py --merge $RUN/infer_fp32; cat $RUN/infer_fp32/scores.csv $RUN/infer_amp/scores_shard0of1.csv
echo "=== frozen GMA on dumped features"; $PY train_frozen_gma.py --feat_dir $RUN/feats --slide_data $SM/slide_data.csv --train_splits val --outdir $RUN/frozen --nepochs 2 --drop 0 && cat $RUN/frozen/scores.csv
ls -la $RUN; rm -f $RUN/checkpoint_last.pth $RUN/checkpoint_tile_*.pth
grep -q FROZEN_DONE /work/nb295/eagle_glioma/logs/smoke_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out && touch $SM/SMOKE_OK
echo SMOKE_DONE
