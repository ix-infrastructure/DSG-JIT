#!/usr/bin/env bash
# Phase 0.6: does the night gain (C2s) generalise to the other 4Seasons night recordings?
# Both are test-split sequences without reference poses, so outcomes are GT-free: maps, coverage,
# tracking failures (ATE stays null). 15 seeds each, C0 vs C2s, 6-way as s4seeds.sh.
#   m2N  maximilianeum_2_test  different route
#   o3N  old_town_3_test       same route as old_town_4_train, 14 min earlier
cd ~/repos/oasis-experiments
S=/data/tkocher/datasets/4seasons; G=$S/gen; O=/data/tkocher/oasis_results
echo "=== p06 start $(date +%T) ==="
for spec in m2N:maximilianeum_2_night o3N:old_town_3_night; do
  IFS=: read -r tag name <<<"$spec"
  python3 -I fourseasons_to_euroc.py --src $S/$name --calib $S/calibration --dst $S/${name}_euroc --base-yaml $O/_configs/EuRoC.yaml || exit 1
  python3 -I real_correct_sequence.py --src $S/${name}_euroc --dst $G/${name}_C2s --yaml $S/${name}_euroc/config.yaml --arm C2s --m-ref 76.6 --workers 12 --force || exit 1
  cp $S/${name}_euroc/stamps.txt $G/${name}_C2s/
done
run() { # data_dir rid
  NF=$(wc -l < $1/stamps.txt)
  ./run_oasis.sh $1 $1/stamps.txt $2 $1/config.yaml > /dev/null 2>&1
  python3 -I phase2_eval_run.py $O/$2 --gt /dev/null --n-frames $NF --frame-dt 0.0333333 > $O/$2/record.json
  [ -f $1/LOWLIGHT_MANIFEST.json ] && cp $1/LOWLIGHT_MANIFEST.json $O/$2/
  echo "$(date +%T) $2 $(python3 -c "import json;r=json.load(open('$O/$2/record.json'));print('cov',round(r['frac_tracked'],3),'fails',r['fail_track_local_map'],'maps',r['maps_in_atlas'],'badimu',r['reset_bad_imu'])" 2>&1)"
}
jobs=()
for i in $(seq 0 14); do
  for spec in m2N:maximilianeum_2_night o3N:old_town_3_night; do
    IFS=: read -r tag name <<<"$spec"
    jobs+=("$S/${name}_euroc s4${tag}_L0_clean_r$i")
    jobs+=("$G/${name}_C2s s4${tag}_L0_C2s_s$i")
  done
done
n=0; for j in "${jobs[@]}"; do set -- $j; run $1 $2 & n=$((n+1)); [ $((n % 6)) -eq 0 ] && wait; done; wait
rm -rf $G/maximilianeum_2_night_C2s/mav0/cam? $G/old_town_3_night_C2s/mav0/cam?
echo "=== p06 complete $(date +%T) ==="
