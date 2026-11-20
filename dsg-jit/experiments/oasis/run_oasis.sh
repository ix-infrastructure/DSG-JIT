#!/usr/bin/env bash
# Run OASIS / ORB_SLAM3 in dataready (offline batch) mode.
#
#   ./run_oasis.sh <dataset_dir> <timestamp_stem> <run_id> [config]
#
#   config: a file name in Examples/Stereo-Inertial/ inside the image
#           (default EuRoC_oasis.yaml), OR an absolute path under /data --
#           /data is mounted at the same path, so per-run configs (e.g. Phase 2's
#           scaled-threshold arm) need no extra mount.
#
# Constraints learned the hard way:
#  1. The trajectory prefix MUST be a bare name, not a path. The binary builds
#     "f_" + prefix + ".txt", so an absolute prefix yields "f_/results/foo.txt"
#     and the write silently fails. Outputs land in the CWD.
#  2. CellManager::printStats() appends to ./cellManager.txt in the CWD, and the
#     image is root-owned. Both are solved by a per-run, caller-owned CWD.
#  3. ORB-SLAM3's console output is the only record of tracking loss, map
#     resets and OASIS dropped frames, so it is kept in slam_stdout.log.
set -euo pipefail
DATASET="${1:?dataset dir, e.g. /data/tkocher/datasets/euroc/MH_01_easy}"
STAMPS="${2:?timestamp stem, e.g. MH01}"
RUN_ID="${3:?run id, e.g. baseline_MH01_run01}"
CONFIG="${4:-EuRoC_oasis.yaml}"
case "$CONFIG" in
  /*) CFG_PATH="$CONFIG" ;;
  *)  CFG_PATH="/opt/ORB_SLAM3/Examples/Stereo-Inertial/${CONFIG}" ;;
esac
# A stem names a file shipped in the image; an absolute path (under /data) is
# used as-is, for datasets that are not EuRoC (e.g. 4Seasons).
case "$STAMPS" in
  /*) STAMPS_PATH="$STAMPS" ;;
  *)  STAMPS_PATH="/opt/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC_TimeStamps/${STAMPS}.txt" ;;
esac
RESULTS_ROOT="${RESULTS_DIR:-/data/tkocher/oasis_results}"
OUTDIR="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "$OUTDIR"
# Phase 1: OASIS_EXPORT=1 selects the export build (fork branch phase1-export) and writes the estimation
# problem (keyframes, map points, observations, IMU preintegration) to $OUTDIR/export. Unset = Phase 0 image.
IMAGE="${OASIS_IMAGE:-oasis-orbslam3:24.04}"
EXPORT_ARGS=()
if [ -n "${OASIS_EXPORT:-}" ]; then
  IMAGE="${OASIS_IMAGE:-oasis-orbslam3:24.04-p1export}"
  EXPORT_ARGS=(-e OASIS_EXPORT=/results/export)
fi
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /data:/data:ro \
  -v "$OUTDIR":/results \
  -w /results \
  "${EXPORT_ARGS[@]}" \
  "$IMAGE" \
  /opt/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc \
      /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
      "$CFG_PATH" \
      "$DATASET" \
      "$STAMPS_PATH" \
      "$RUN_ID" > "$OUTDIR/slam_stdout.log" 2>&1
echo "--- outputs in ${OUTDIR} ---"
ls -la "$OUTDIR"
