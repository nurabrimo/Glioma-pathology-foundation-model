#!/bin/bash
# The EAGLE Terms of Use (MSK) do not allow redistribution of their code, so the files this project imports from it
# are not in this repository. This script downloads them from the authors' repository at the commit we used.
#   bash code/fetch_eagle_files.sh
# NB: our results used calib/utils.py with two fixes (confusion matrices declared with labels=[0, 1] so single-class
# bootstrap samples do not fail; vectorised threshold grid search); see the manuscript Methods.
set -e
cd "$(dirname "$0")"
U=https://raw.githubusercontent.com/chadvanderbilt/EAGLE/627f4e09239c5a160842cd7365c1ce4436859193
curl -fsSL $U/training/modules.py          -o train/modules.py
curl -fsSL $U/training/extra_transforms.py -o train/extra_transforms.py
curl -fsSL $U/training/utils.py            -o train/eagle_utils.py
curl -fsSL $U/calibration/utils.py         -o calib/utils.py
cp calib/utils.py calib/eagle_calib_orig.py
curl -fsSL $U/calibration/example_data.csv -o calib/example_data.csv
echo "fetched EAGLE files at 627f4e09239c5a160842cd7365c1ce4436859193 (use is subject to the EAGLE Terms of Use)"
