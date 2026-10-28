#!/usr/bin/env bash
# Phase 0.5: synthetic night (black offset + scale + 8-bit quantization + measured sigma) from the
# old_town day recording, calibrated to old_town_4 night quantiles (synth_night.py). One synthetic
# sequence, 15 SLAM seeds each without correction (C0) and with the saturation-safe gain (C2s), the
# same design as the real night runs. Done when synthetic C0 reproduces the real night failure rate (~60%).
cd ~/repos/oasis-experiments
S=/data/tkocher/datasets/4seasons; G=$S/gen; O=/data/tkocher/oasis_results
DAY=$S/old_town_1_day; NF=$(wc -l < $DAY/stamps.txt)
echo "=== p05 start $(date +%T) ==="
python3 -I synth_night.py --day $DAY --night $S/old_town_4_night --dst $G/synth_night --workers 12 --force || exit 1
python3 -I real_correct_sequence.py --src $G/synth_night --dst $G/synth_night_C2s --yaml $G/synth_night/config.yaml --arm C2s --m-ref 76.6 --workers 12 --force || exit 1
cp $G/synth_night/stamps.txt $G/synth_night_C2s/
cp $G/synth_night/LOWLIGHT_MANIFEST.json $O/synth_night_manifest.json
run() { # data_dir rid
  ./run_oasis.sh $1 $1/stamps.txt $2 $1/config.yaml > /dev/null 2>&1
  python3 -I phase2_eval_run.py $O/$2 --gt $DAY/mav0/state_groundtruth_estimate0/data.csv --n-frames $NF --frame-dt 0.0333333 > $O/$2/record.json
  [ -f $1/LOWLIGHT_MANIFEST.json ] && cp $1/LOWLIGHT_MANIFEST.json $O/$2/
  echo "$(date +%T) $2 $(python3 -c "import json;r=json.load(open('$O/$2/record.json'));print('ATE',r['ate_m'],'cov',round(r['frac_tracked'],3),'fails',r['fail_track_local_map'],'maps',r['maps_in_atlas'])" 2>&1)"
}
jobs=()
for i in $(seq 0 14); do
  jobs+=("$G/synth_night s4S_L0_C0_s$i")
  jobs+=("$G/synth_night_C2s s4S_L0_C2s_s$i")
done
n=0; for j in "${jobs[@]}"; do set -- $j; run $1 $2 & n=$((n+1)); [ $((n % 6)) -eq 0 ] && wait; done; wait
rm -rf $G/synth_night/mav0/cam0 $G/synth_night/mav0/cam1 $G/synth_night_C2s/mav0/cam0 $G/synth_night_C2s/mav0/cam1
echo "=== p05 complete $(date +%T) ==="
