#!/usr/bin/env bash
# Wait for Phase 2 to finish, then run C0/C2/C4 x 10 on the real low-light sequences.
cd ~/repos/oasis-experiments
while ps -p 898628 >/dev/null; do sleep 60; done
E=/data/tkocher/datasets/euroc
for spec in MH_04_difficult:MH04:2033 MH_05_difficult:MH05:2273 MH_03_medium:MH03:2700; do
  IFS=: read -r name stem nf <<<"$spec"
  echo "=== r$stem start $(date +%T) ==="
  python3 phase2_suite.py --tag r$stem --cells 0:C2 0:C4 0:C2s 0:C4s --seeds 0 1 2 3 4 5 6 7 8 9 --clean-runs 10 \
    --src $E/$name --gt $E/$name/mav0/state_groundtruth_estimate0/data.csv --stamps $stem --n-frames $nf \
    --gen-script real_correct_sequence.py --gen-args "--m-ref 76.6" --skip-done
done
echo "=== real chain complete $(date +%T) ==="
