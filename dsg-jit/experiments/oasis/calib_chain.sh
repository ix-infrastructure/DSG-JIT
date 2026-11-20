#!/usr/bin/env bash
# Calibration / IMU-model error sensitivity sweep (viability question for the calibration module).
cd ~/repos/oasis-experiments
E=/data/tkocher/datasets/euroc
CELLS=""; for arm in toff rbc tbc rst isf imis; do for lv in 1 2 3; do CELLS="$CELLS $lv:$arm"; done; done
for spec in MH_01_easy:MH01:3682 MH_03_medium:MH03:2700; do
  IFS=: read -r name stem nf <<<"$spec"
  echo "=== c$stem start $(date +%T) ==="
  python3 phase2_suite.py --tag c$stem --cells $CELLS --seeds 0 1 2 3 4 5 6 7 8 9 --clean-runs 10 \
    --src $E/$name --gt $E/$name/mav0/state_groundtruth_estimate0/data.csv --stamps $stem --n-frames $nf \
    --gen-script calib_perturb.py --gen-workers 1 --skip-done
done
echo "=== calib chain complete $(date +%T) ==="
