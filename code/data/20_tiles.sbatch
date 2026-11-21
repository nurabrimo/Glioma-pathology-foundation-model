#!/bin/bash -l
#SBATCH --job-name=eg_tiles
#SBATCH --partition=common
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc/home/nb295/eagle_glioma/logs/tiles_%x_%j.out
# COHORT=tcga|ebrains|cptac sbatch -J tiles_$COHORT data/20_tiles.sbatch
source $HOME/recast/dcc/config.sh 2>/dev/null; export PATH="$CONDA_ENV/bin:$PATH"; cd $HOME/eagle_glioma; set -eo pipefail
F=/cwork/nb295/recast_feats
case ${COHORT:?} in
  tcga)    C="$F/tcga_ffpe/trident/contours_geojson /cwork/nb295/eagle_glioma/trident_tcga_extra/contours_geojson";;
  ebrains) C="$F/ebrains_glioma/trident/contours_geojson";;
  cptac)   C="$F/cptac_gbm/trident/contours_geojson";;
esac
python -u data/make_tiles.py --cohort $COHORT --contours $C --workers 16
