#!/usr/bin/env bash
# Phase 0 follow-ups.
#  (a) old_town_3 night: regenerate the C2s frames and profile the region where every C2s seed breaks (~frame 9300).
#  (b) Phase 0.5b: synthetic night WITH exposure-matched motion blur (day 0.12 ms -> night 21 ms), 15 seeds each
#      C0 / C2s, otherwise identical to p05.sh. Question: does blur close the gap to the real night
#      (real: C0 12/30 good, C2s 20/30 good at ~27 m; synthetic without blur: C0 0/15, C2s 15/15 at 7.4 m)?
cd ~/repos/oasis-experiments
S=/data/tkocher/datasets/4seasons; G=$S/gen; O=/data/tkocher/oasis_results
echo "=== p07 start $(date +%T) ==="
python3 -I real_correct_sequence.py --src $S/old_town_3_night_euroc --dst $G/old_town_3_night_C2s --yaml $S/old_town_3_night_euroc/config.yaml --arm C2s --m-ref 76.6 --workers 12 --force
python3 -I /data/tkocher/o3break.py 8400 10200 20 > $O/o3break.txt 2>&1
echo "$(date +%T) o3break profile -> $O/o3break.txt"
rm -rf $G/old_town_3_night_C2s/mav0/cam0 $G/old_town_3_night_C2s/mav0/cam1

DAY=$S/old_town_1_day; NF=$(wc -l < $DAY/stamps.txt)
python3 -I synth_night.py --day $DAY --night $S/old_town_4_night --dst $G/synth_night_blur --blur-exp-ms 21 --workers 12 --force || exit 1
python3 -I real_correct_sequence.py --src $G/synth_night_blur --dst $G/synth_night_blur_C2s --yaml $G/synth_night_blur/config.yaml --arm C2s --m-ref 76.6 --workers 12 --force || exit 1
cp $G/synth_night_blur/stamps.txt $G/synth_night_blur_C2s/
cp $G/synth_night_blur/LOWLIGHT_MANIFEST.json $O/synth_night_blur_manifest.json
run() { # data_dir rid
  ./run_oasis.sh $1 $1/stamps.txt $2 $1/config.yaml > /dev/null 2>&1
  python3 -I phase2_eval_run.py $O/$2 --gt $DAY/mav0/state_groundtruth_estimate0/data.csv --n-frames $NF --frame-dt 0.0333333 > $O/$2/record.json
  [ -f $1/LOWLIGHT_MANIFEST.json ] && cp $1/LOWLIGHT_MANIFEST.json $O/$2/
  echo "$(date +%T) $2 $(python3 -c "import json;r=json.load(open('$O/$2/record.json'));print('ATE',r['ate_m'],'cov',round(r['frac_tracked'],3),'fails',r['fail_track_local_map'],'maps',r['maps_in_atlas'])" 2>&1)"
}
jobs=()
for i in $(seq 0 14); do
  jobs+=("$G/synth_night_blur s4B_L0_C0_s$i")
  jobs+=("$G/synth_night_blur_C2s s4B_L0_C2s_s$i")
done
n=0; for j in "${jobs[@]}"; do set -- $j; run $1 $2 & n=$((n+1)); [ $((n % 6)) -eq 0 ] && wait; done; wait
rm -rf $G/synth_night_blur/mav0/cam0 $G/synth_night_blur/mav0/cam1 $G/synth_night_blur_C2s/mav0/cam0 $G/synth_night_blur_C2s/mav0/cam1
echo "=== p07 complete $(date +%T) ==="
