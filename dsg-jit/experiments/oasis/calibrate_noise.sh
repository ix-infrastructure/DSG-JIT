#!/usr/bin/env bash
# Find the noise range over which ATE actually responds. One run per scale --
# this locates the dynamic range, it does not measure effects.
set -uo pipefail
cd ~/repos/ORB_SLAM3
INJ=~/repos/oasis-experiments/inject_imu_noise.py
NOISY=/data/tkocher/datasets/euroc_noisy
for S in 1 5 10 20 50; do
  DST="$NOISY/calib_s${S}_seed00"
  python3 "$INJ" --src /data/tkocher/datasets/euroc/MH_01_easy --dst "$DST" \
      --mode gaussian --scale "$S" --seed 0 --force >/dev/null
  echo "=== $(date +%H:%M:%S) scale=${S}x ==="
  ./run_oasis.sh "$DST" MH01 "calib_s${S}_seed00" EuRoC_oasis.yaml >/dev/null 2>&1 \
    && echo "    ok" || echo "    FAILED"
done
echo "=== calibration complete ==="
