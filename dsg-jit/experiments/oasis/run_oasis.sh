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
RESULTS_ROOT="${RESULTS_DIR:-/data/tkocher/oasis_results}"
OUTDIR="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "$OUTDIR"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /data:/data:ro \
  -v "$OUTDIR":/results \
  -w /results \
  oasis-orbslam3:24.04 \
  /opt/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc \
      /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
      "$CFG_PATH" \
      "$DATASET" \
      "/opt/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC_TimeStamps/${STAMPS}.txt" \
      "$RUN_ID" > "$OUTDIR/slam_stdout.log" 2>&1
echo "--- outputs in ${OUTDIR} ---"
ls -la "$OUTDIR"
