#!/usr/bin/env bash
# Phase 0.2: real IMU miscalibration on TUM-VI room1-6. cal / nos (scale+misalignment uncorrected) / raw (plus turn-on bias), 10 seeds.
cd ~/repos/oasis-experiments
C=/data/tkocher/oasis_results/_configs; T=/data/tkocher/datasets/tumvi; V=/data/tkocher/datasets/tumvi_variants; E=/data/tkocher/datasets/tumvi_euroc; O=/data/tkocher/oasis_results
echo "$(date +%T) convert rooms 2-6"
python3 -I tumvi_to_euroc.py --src $T --variants $V --tumvi-yaml $C/TUM-VI.yaml --base-yaml $C/EuRoC_oasis.yaml --dst $E --rooms 2,3,4,5,6 || exit 1
run() { # room cond seed
  rid=tv$1_$2_s$3; nf=$(wc -l < $V/room$1/times.txt)
  ./run_oasis.sh $E/room$1/$2 $V/room$1/times.txt $rid $E/room$1/config.yaml > /dev/null 2>&1
  python3 -I phase2_eval_run.py $O/$rid --gt $T/dataset-room$1_512_16/mav0/mocap0/data.csv --n-frames $nf > $O/$rid/record.json
  echo "$(date +%T) $rid $(python3 -c "import json;r=json.load(open('$O/$rid/record.json'));print('ATE',r['ate_m'],'cov',round(r['frac_tracked'],3),'maps',r['maps_in_atlas'],'badimu',r['reset_bad_imu'],'valid',r['valid'])")"
}
jobs=()
for s in $(seq -w 0 9); do for r in 1 2 3 4 5 6; do for c in cal nos raw; do jobs+=("$r $c $s"); done; done; done
n=0; for j in "${jobs[@]}"; do run $j & n=$((n+1)); [ $((n % 6)) -eq 0 ] && wait; done; wait
echo "=== tv complete $(date +%T) ==="
