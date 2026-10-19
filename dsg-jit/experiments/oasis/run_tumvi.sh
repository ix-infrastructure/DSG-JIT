#!/usr/bin/env bash
# Run ORB_SLAM3 stereo-inertial on a TUM-VI sequence with a chosen IMU file.
#
#   ./run_tumvi.sh <sequence_dir> <times_file> <imu_csv> <run_id> [config]
#
# Uses the image's stereo_inertial_tum_vi (it applies CLAHE to every frame, as
# ORB-SLAM3's published TUM-VI results do), not stereo_inertial_euroc, so the
# IMU file can be swapped without copying images. Same per-run CWD and log
# conventions as run_oasis.sh.
set -euo pipefail
SEQ="${1:?sequence dir, e.g. /data/tkocher/datasets/tumvi/dataset-room1_512_16}"
TIMES="${2:?times file (ns per line)}"
IMU="${3:?IMU csv in EuRoC format}"
RUN_ID="${4:?run id}"
CONFIG="${5:-TUM-VI.yaml}"
case "$CONFIG" in
  /*) CFG_PATH="$CONFIG" ;;
  *)  CFG_PATH="/opt/ORB_SLAM3/Examples/Stereo-Inertial/${CONFIG}" ;;
esac
RESULTS_ROOT="${RESULTS_DIR:-/data/tkocher/oasis_results}"
OUTDIR="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "$OUTDIR"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /data:/data:ro \
  -v "$OUTDIR":/results \
  -w /results \
  oasis-orbslam3:24.04 \
  /opt/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_tum_vi \
      /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
      "$CFG_PATH" \
      "$SEQ/mav0/cam0/data" \
      "$SEQ/mav0/cam1/data" \
      "$TIMES" \
      "$IMU" \
      "$RUN_ID" > "$OUTDIR/slam_stdout.log" 2>&1
