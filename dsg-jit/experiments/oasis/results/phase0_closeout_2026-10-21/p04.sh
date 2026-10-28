#!/usr/bin/env bash
# Phase 0.4: are the 4Seasons-night "bad imu" resets an IMU problem or a symptom of visual loss?
# LocalMapping.cc:138 resets a new map (mbBadImu) when it barely moves in its first 10 s after IMU init.
# Ablation: the same night input through visual-only stereo (stereo_euroc, no IMU), C0 and gain (C2s),
# 10 seeds each, same 6-way concurrency as s4seeds.sh. If visual-only also fails at night and gain also
# fixes it, the resets are a downstream symptom of visual tracking loss.
cd ~/repos/oasis-experiments
N=/data/tkocher/datasets/4seasons/old_town_4_night; G=/data/tkocher/datasets/4seasons/gen; O=/data/tkocher/oasis_results
echo "=== p04 start $(date +%T) ==="
echo "$(date +%T) gen C2s"; python3 -I real_correct_sequence.py --src $N --dst $G/night_C2s --yaml $N/config.yaml --arm C2s --m-ref 76.6 --workers 12 --force || exit 1
run() { # data_dir config rid
  mkdir -p $O/$3
  docker run --rm --user "$(id -u):$(id -g)" -v /data:/data:ro -v $O/$3:/results -w /results oasis-orbslam3:24.04 \
    /opt/ORB_SLAM3/Examples/Stereo/stereo_euroc /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt $2 $1 $N/stamps.txt $3 \
    > $O/$3/slam_stdout.log 2>&1
  python3 -I phase2_eval_run.py $O/$3 --gt $N/mav0/state_groundtruth_estimate0/data.csv --n-frames 24658 --frame-dt 0.0333333 > $O/$3/record.json
  echo "$(date +%T) $3 $(python3 -c "import json;r=json.load(open('$O/$3/record.json'));print('ATE',r['ate_m'],'cov',round(r['frac_tracked'],3),'fails',r['fail_track_local_map'],'maps',r['maps_in_atlas'])" 2>&1)"
}
jobs=()
for i in $(seq 0 9); do
  jobs+=("$N $N/config.yaml s4V_L0_clean_r$i")
  jobs+=("$G/night_C2s $G/night_C2s/config.yaml s4V_L0_C2s_s$i")
done
n=0; for j in "${jobs[@]}"; do set -- $j; run $1 $2 $3 & n=$((n+1)); [ $((n % 6)) -eq 0 ] && wait; done; wait
rm -rf $G/night_C2s/mav0/cam0 $G/night_C2s/mav0/cam1
echo "=== p04 complete $(date +%T) ==="
# Phase 0.5 and 0.6 follow-on, if staged
[ -x /data/tkocher/p05.sh ] && /data/tkocher/p05.sh
[ -x /data/tkocher/p06.sh ] && /data/tkocher/p06.sh
