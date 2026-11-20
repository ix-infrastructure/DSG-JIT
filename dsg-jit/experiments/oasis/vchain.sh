#!/usr/bin/env bash
# After the calibration sweep: clean baselines on the Vicon sequences (blur viability), then blur-split rescoring.
cd ~/repos/oasis-experiments; E=/data/tkocher/datasets/euroc; R=/data/tkocher/oasis_results
while pgrep -f calib_chain.sh >/dev/null; do sleep 60; done
for spec in V1_03_difficult:V103:V13 V2_03_difficult:V203:V23 V1_01_easy:V101:V11 V2_01_easy:V201:V21; do
  IFS=: read -r name stem tag <<<"$spec"
  nf=$(ls $E/$name/mav0/cam0/data | wc -l)
  python3 phase2_suite.py --tag v$tag --cells --seeds 0 --clean-runs 10 --src $E/$name \
    --gt $E/$name/mav0/state_groundtruth_estimate0/data.csv --stamps $stem --n-frames $nf --skip-done
  python3 rescore_real.py --tag v$tag --gt $E/$name/mav0/state_groundtruth_estimate0/data.csv --images $E/$name/mav0/cam0/data \
    --levels $R/blur_vicon/${name}_blur_per_frame.csv --value-col blur_rate_px_s --dark-level 500 > $R/v${tag}_blur_rescored.jsonl
done
echo "=== vicon chain complete $(date +%T) ==="
