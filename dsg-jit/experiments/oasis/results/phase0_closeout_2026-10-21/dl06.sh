#!/usr/bin/env bash
# Phase 0.6 data: the only other 4Seasons night recordings (test split: IMU + undistorted stereo, no reference poses).
#   maximilianeum_2_test  recording_2021-05-10_20-59-00  different route
#   old_town_3_test       recording_2021-05-10_21-18-00  same route as old_town_4_train, 14 min earlier
D=/data/tkocher/datasets/4seasons; cd $D || exit 1
B=https://vision.cs.tum.edu/webshare/g/4seasons-dataset/dataset
for R in recording_2021-05-10_20-59-00 recording_2021-05-10_21-18-00; do
  for t in imu stereo_images_undistorted; do
    f=${R}_$t.zip
    echo "$(date +%T) get $f"; curl -sSfL -C - -o $f "$B/$R/$f" || { echo "FAILED $f"; continue; }
    unzip -qo $f && echo "$(date +%T) unzipped $f" && rm -f $f
  done
done
ln -sfn recording_2021-05-10_20-59-00 maximilianeum_2_night
ln -sfn recording_2021-05-10_21-18-00 old_town_3_night
du -sh $D; echo "=== dl06 complete $(date +%T) ==="
