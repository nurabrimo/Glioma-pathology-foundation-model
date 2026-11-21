#!/bin/bash
#SBATCH --job-name=eg_train
#SBATCH --nodes=1
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-core-gpu-ferc-s-h36-6
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH --output=/work/nb295/eagle_glioma/logs/train_%x_%j.out
# End-to-end fine-tuning (EAGLE recipe). Submit with slurm/submit_train.sh (sets partition / gres / cpus / mem / time).
# Several variants (partition x GPU type x GPU count) may be queued for the same RUN: <run>/ACTIVE_JOB is a lock, only one
# trains at a time; a variant with more GPUs asks a running smaller one to checkpoint and takes over; others resubmit
# themselves with a delay. Pre-emption (scavenger, no grace time) -> Slurm requeues, the run resumes from
# <run>/checkpoint_last.pth (written every CKPT_MIN minutes and at each epoch end). Time limit -> USR1 300 s before,
# the trainer checkpoints after the current step and the job resubmits itself.
source $HOME/recast/dcc/config.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # weights are in $HF_HOME already
PY=$CONDA_ENV/bin/python
cd $HOME/eagle_glioma/train
RUN=${RUN:?set RUN}; INIT=${INIT:-gigapath}; CHUNK=${CHUNK:-0}; CKPT_MIN=${CKPT_MIN:-20}
OUT=/work/nb295/eagle_glioma/runs/$RUN; mkdir -p $OUT; LOCK=$OUT/ACTIVE_JOB
NGPU=$(nvidia-smi -L | wc -l)
resubmit() { [ -f $OUT/DONE ] || bash $HOME/eagle_glioma/slurm/submit_train.sh "$EG_PART" "$EG_GRES" "$1"; }
echo "$(date) job $SLURM_JOB_ID host $(hostname) gpus $NGPU ($EG_PART $EG_GRES)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f $OUT/DONE ] && { echo "run already finished"; exit 0; }
# ---- lock
if [ -f $LOCK ]; then
  read OJ ON < $LOCK
  if [ "$OJ" != "$SLURM_JOB_ID" ] && [ "$(squeue -h -j $OJ -o %T 2>/dev/null)" = "RUNNING" ]; then
    if [ "$NGPU" -gt "${ON:-0}" ]; then
      echo "job $OJ trains with $ON GPUs; asking it to checkpoint and hand over"
      scancel -b -s USR1 $OJ
      for i in $(seq 1 90); do [ -f $LOCK ] && [ "$(cut -d" " -f1 $LOCK)" = "$OJ" ] && [ "$(squeue -h -j $OJ -o %T 2>/dev/null)" = "RUNNING" ] || break; sleep 10; done
      if [ -f $LOCK ] && [ "$(cut -d" " -f1 $LOCK)" = "$OJ" ] && [ "$(squeue -h -j $OJ -o %T 2>/dev/null)" = "RUNNING" ]; then echo "no hand-over; retry later"; resubmit now+60minutes; exit 0; fi
    else
      echo "job $OJ already trains this run with $ON GPUs; retry later"; resubmit now+90minutes; exit 0
    fi
  fi
fi
echo "$SLURM_JOB_ID $NGPU" > $LOCK
sleep 5; [ "$(cut -d" " -f1 $LOCK)" = "$SLURM_JOB_ID" ] || { echo "lost the lock race"; resubmit now+30minutes; exit 0; }
# ---- train
WORKERS=$(( SLURM_CPUS_PER_TASK / NGPU - 2 )); [ $WORKERS -lt 2 ] && WORKERS=2
ARGS="--outdir $OUT --init $INIT --chunk $CHUNK --load_bs 96 --workers $WORKERS --ckpt_minutes $CKPT_MIN --prefetch 8 ${EXTRA:-}"
echo "workers/rank $WORKERS args: $ARGS"
if [ $NGPU -gt 1 ]; then
  $CONDA_ENV/bin/torchrun --nproc_per_node=$NGPU train_e2e.py $ARGS &
  PID=$!
  trap 'echo "$(date) signal: asking trainers to checkpoint"; pkill -USR1 -f "[-]u train_e2e.py --outdir $OUT "' USR1
else
  $PY train_e2e.py $ARGS &
  PID=$!
  trap 'echo "$(date) signal: asking trainer to checkpoint"; kill -USR1 $PID' USR1
fi
wait $PID; RC=$?
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
echo "$(date) trainer exit $RC"
[ "$(cut -d" " -f1 $LOCK 2>/dev/null)" = "$SLURM_JOB_ID" ] && rm -f $LOCK
if [ -f $OUT/DONE ]; then echo RUN_DONE; elif [ $RC -eq 0 ]; then resubmit now; else echo "trainer failed (rc $RC): not resubmitting"; fi
exit $RC
