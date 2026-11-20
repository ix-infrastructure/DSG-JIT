# ORB-SLAM3 estimation-problem export (Phase 1)

The OASIS ORB-SLAM3 fork on brd-lambda (`~/repos/ORB_SLAM3`, branch `phase1-export`, based on `8b66f54`)
carries these uncommitted changes; this directory is the copy of record.

- `orbslam3_phase1_export.patch` -- adds `System::SaveEstimationProblem()` and calls it from
  `stereo_inertial_euroc` when `OASIS_EXPORT=1`. It writes keyframes, map points, observations, IMU
  preintegration and raw IMU samples per map to `$OUTDIR/export/*.csv` (read with `../orb_export.py`).
- `Dockerfile.oasis`, `dockerignore` -- build environment (ubuntu 24.04, OpenCV 4.6.0). Copy
  `dockerignore` to `.dockerignore` in the ORB-SLAM3 checkout.

```bash
cd ORB_SLAM3 && git checkout 8b66f54 -b phase1-export
git apply /path/to/orbslam3_phase1_export.patch
cp /path/to/Dockerfile.oasis . && cp /path/to/dockerignore .dockerignore
docker build -f Dockerfile.oasis -t oasis-orbslam3:24.04-p1export .
OASIS_EXPORT=1 ../run_oasis.sh <dataset_dir> <timestamp_stem> <run_id> [config]
```
