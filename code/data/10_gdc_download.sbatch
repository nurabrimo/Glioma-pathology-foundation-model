#!/bin/bash -l
#SBATCH --job-name=eg_gdc
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/hpc/home/nb295/eagle_glioma/logs/gdc_%A_%a.out
# Download the TCGA-GBM/LGG diagnostic slides that are not yet on DCC (open access, no token).
# Manifest: /work/nb295/eagle_glioma/tcga_download_manifest.txt (from data/build_labels.py). Resumable: rerun the array.
source $HOME/recast/dcc/config.sh 2>/dev/null
OUT=/cwork/nb295/eagle_glioma/tcga_dx_extra; M=/work/nb295/eagle_glioma/tcga_download_manifest.txt
mkdir -p $OUT /work/nb295/eagle_glioma/gdc_shards
S=/work/nb295/eagle_glioma/gdc_shards/m_${SLURM_ARRAY_TASK_ID}.txt
# shard k takes rows with (row index mod 8) == k; skip files already complete (size matches)
awk -v k=$SLURM_ARRAY_TASK_ID -v out=$OUT 'NR==1{print;next} ((NR-2)%8)==k{f=out"/"$1"/"$2; cmd="stat -c %s \""f"\" 2>/dev/null"; s=""; cmd|getline s; close(cmd); if(s!=$4) print}' $M > $S
echo "shard $SLURM_ARRAY_TASK_ID: $(($(wc -l < $S)-1)) files to fetch"
for try in 1 2 3; do
  $MP_ROOT/bin/gdc-client download -m $S -d $OUT -n 8 --retry-amount 10 --wait-time 10 --no-related-files --no-annotations && break
  echo "retry $try"; sleep 60
done
echo GDC_SHARD_DONE $SLURM_ARRAY_TASK_ID
