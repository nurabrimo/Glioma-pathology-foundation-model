#!/bin/bash -l
#SBATCH --job-name=eg_seg
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-core-gpu-ferc-s-h36-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-7%4
#SBATCH --output=/hpc/home/nb295/eagle_glioma/logs/seg_%A_%a.out
# TRIDENT tissue segmentation only (HEST segmenter, same settings as the 880 slides already processed for RECAST)
# for the newly downloaded TCGA DX slides. Lists are written here from the download manifest (complete files only).
source $HOME/recast/dcc/config.sh 2>/dev/null; export PATH="$CONDA_ENV/bin:$PATH"; set -eo pipefail
export HF_HOME=/work/$NETID/hf_cache PYTHONWARNINGS=ignore
WSI=/cwork/nb295/eagle_glioma/tcga_dx_extra; JOB=/cwork/nb295/eagle_glioma/trident_tcga_extra; mkdir -p $JOB/lists
K=$SLURM_ARRAY_TASK_ID; LIST=$JOB/lists/$(printf "%02d" $K).csv
awk -v k=$K -v out=$WSI 'BEGIN{print "wsi"} NR>1 && ((NR-2)%8)==k{f=out"/"$1"/"$2; cmd="stat -c %s \""f"\" 2>/dev/null"; s=""; cmd|getline s; close(cmd); if(s==$4) print $1"/"$2}' \
    /work/nb295/eagle_glioma/tcga_download_manifest.txt > $LIST
echo "shard $K n=$(($(wc -l < $LIST)-1)) node=$(hostname)"
# stale locks from pre-empted runs make TRIDENT skip slides silently
while read -r w; do b=$(basename "$w" .svs); find $JOB -name "$b*.lock" -delete 2>/dev/null || true; done < <(tail -n +2 $LIST)
cd $HOME/trident
python run_batch_of_slides.py --task seg --wsi_dir $WSI --custom_list_of_wsis $LIST --job_dir $JOB --max_workers 4 --skip_errors --gpu 0 --segmenter hest
echo SEG_DONE shard=$K
