#!/bin/bash -l
#SBATCH --job-name=eg_final
#SBATCH --partition=common
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/hpc/home/nb295/eagle_glioma/logs/final_%j.out
# After download + segmentation: verify md5, tile TCGA, write slide_data.csv + qc.txt.
source $HOME/recast/dcc/config.sh 2>/dev/null; export PATH="$CONDA_ENV/bin:$PATH"; cd $HOME/eagle_glioma
python -u data/verify_downloads.py --workers 16
echo "contours for new slides: $(ls /cwork/nb295/eagle_glioma/trident_tcga_extra/contours_geojson 2>/dev/null | wc -l)"
python -u data/make_tiles.py --cohort tcga --workers 16 --contours /cwork/nb295/recast_feats/tcga_ffpe/trident/contours_geojson /cwork/nb295/eagle_glioma/trident_tcga_extra/contours_geojson
python -u data/finalize_slides.py
echo EG_FINAL_DONE
