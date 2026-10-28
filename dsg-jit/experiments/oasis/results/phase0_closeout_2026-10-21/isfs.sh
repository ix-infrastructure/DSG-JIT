#!/usr/bin/env bash
# Phase 0.3: IMU scale re-sweep with sourced tiers (gyro 1/3/9 %, accel 1/3/4 %), MH01/MH03, 10 seeds.
# Reuses the calib sweep's clean runs (same tag, --skip-done) and its concurrency (3).
cd ~/repos/oasis-experiments
E=/data/tkocher/datasets/euroc
for spec in MH_01_easy:MH01:3682 MH_03_medium:MH03:2700; do
  IFS=: read -r name stem nf <<<"$spec"
  echo "=== c$stem isfs start $(date +%T) ==="
  python3 phase2_suite.py --tag c$stem --cells 1:isfs 2:isfs 3:isfs --seeds 0 1 2 3 4 5 6 7 8 9 --clean-runs 10 \
    --src $E/$name --gt $E/$name/mav0/state_groundtruth_estimate0/data.csv --stamps $stem --n-frames $nf \
    --gen-script calib_perturb.py --gen-workers 1 --skip-done
done
echo "=== isfs complete $(date +%T) ==="
# Phase 0.4 follow-on, if staged
[ -x /data/tkocher/p04.sh ] && /data/tkocher/p04.sh
