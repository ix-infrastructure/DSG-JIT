#!/usr/bin/env python3
"""Present a TUM-VI room sequence as pinhole EuRoC sequences, one per IMU variant.

The OASIS ORB-SLAM3 fork segfaults on TUM-VI's KannalaBrandt8 (fisheye)
camera model before the first frame, with either runner and even with the
image's own TUM-VI files (exit 139). Pinhole input (EuRoC, rectified
4Seasons) works, so the fisheye stereo pair is rectified to pinhole here and
run through the proven stereo_inertial_euroc path.

  images   16-bit PNG -> 8-bit (>> 8, as cv::IMREAD_GRAYSCALE does), CLAHE(3, 8x8)
           as ORB-SLAM3's stereo_inertial_tum_vi applies to every frame, then
           cv2.fisheye.stereoRectify(balance=0) remap; written once per room
  variants <dst>/room<N>/<cal|nos|raw>/mav0/{cam0,cam1 -> shared images, imu0/data.csv
           from tumvi_imu_variants.py}
  GT       mav0/mocap0/data.csv is already IMU pose in IMU time (TUM-VI calibrated
           export), EuRoC column layout; used directly by phase2_eval_run --gt
  config   <dst>/room<N>/config.yaml from a pinhole EuRoC template: rectified K,
           zero distortion, 512x512 @ 20 Hz, Stereo.T_c1_c2 = pure-x baseline,
           IMU.T_b_c1 = T_b_c1 * [R1^T] (rectifying rotation folded in), TUM-VI
           IMU noise values

    tumvi_to_euroc.py --src /data/tkocher/datasets/tumvi --variants /data/tkocher/datasets/tumvi_variants \
        --tumvi-yaml TUM-VI.yaml --base-yaml EuRoC_oasis.yaml --dst /data/tkocher/datasets/tumvi_euroc
"""
import argparse, glob, os, re
from multiprocessing import Pool
import numpy as np
import cv2

TUM_NOISE = {"IMU.NoiseGyro": 0.00016, "IMU.NoiseAcc": 0.0028, "IMU.GyroWalk": 0.000022, "IMU.AccWalk": 0.00086}


def read_cfg(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    g = lambda k: fs.getNode(k).real()
    cam = lambda i: (np.array([[g(f"Camera{i}.fx"), 0, g(f"Camera{i}.cx")], [0, g(f"Camera{i}.fy"), g(f"Camera{i}.cy")], [0, 0, 1]]),
                     np.array([g(f"Camera{i}.k{j}") for j in range(1, 5)]))
    return cam(1), cam(2), fs.getNode("Stereo.T_c1_c2").mat(), fs.getNode("IMU.T_b_c1").mat()


def fmt(T):
    return ",\n         ".join(",".join(f"{v:.12f}" for v in row) for row in T)


def work(job):
    src, dst, maps = job
    clahe = cv2.createCLAHE(3.0, (8, 8))
    im = (cv2.imread(src, cv2.IMREAD_UNCHANGED) >> 8).astype(np.uint8)
    cv2.imwrite(dst, cv2.remap(clahe.apply(im), maps[0], maps[1], cv2.INTER_LINEAR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--variants", required=True)
    ap.add_argument("--tumvi-yaml", required=True); ap.add_argument("--base-yaml", required=True)
    ap.add_argument("--dst", required=True); ap.add_argument("--rooms", default="1,2,3,4,5,6")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    (K1, D1), (K2, D2), T12, Tb1 = read_cfg(a.tumvi_yaml)
    T21 = np.linalg.inv(T12)                       # OpenCV wants X2 = R X1 + T
    size = (512, 512)
    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(K1, D1, K2, D2, size, T21[:3, :3], T21[:3, 3],
                                                  flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0)
    maps = {c: cv2.fisheye.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_32FC1)
            for c, K, D, R, P in (("cam0", K1, D1, R1, P1), ("cam1", K2, D2, R2, P2))}
    fx, fy, cx, cy = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
    baseline = -P2[0, 3] / P2[0, 0]
    T1r = np.eye(4); T1r[:3, :3] = R1.T
    T_b_rect = Tb1 @ T1r
    T_c1_c2 = np.eye(4); T_c1_c2[0, 3] = baseline

    cfg = open(a.base_yaml).read()
    sub = {"Camera1.fx": fx, "Camera1.fy": fy, "Camera1.cx": cx, "Camera1.cy": cy,
           "Camera2.fx": fx, "Camera2.fy": fy, "Camera2.cx": cx, "Camera2.cy": cy,
           "Camera1.k1": 0.0, "Camera1.k2": 0.0, "Camera1.p1": 0.0, "Camera1.p2": 0.0,
           "Camera2.k1": 0.0, "Camera2.k2": 0.0, "Camera2.p1": 0.0, "Camera2.p2": 0.0,
           "Camera.width": 512, "Camera.height": 512, "Camera.fps": 20, "IMU.Frequency": 200.0, **TUM_NOISE}
    for k, v in sub.items():
        cfg, nsub = re.subn(rf"^{re.escape(k)}:.*$", f"{k}: {v}", cfg, flags=re.M)
        assert nsub == 1, k
    for key, T in (("Stereo.T_c1_c2", T_c1_c2), ("IMU.T_b_c1", T_b_rect)):
        cfg, nsub = re.subn(rf"({re.escape(key)}:\s*!!opencv-matrix.*?data:\s*\[).*?(\])", lambda mm: mm.group(1) + fmt(T) + mm.group(2), cfg, flags=re.S)
        assert nsub == 1, key
    print(f"rectified fx {fx:.2f} cx {cx:.1f} cy {cy:.1f}, baseline {baseline:.4f} m, "
          f"R1 {np.degrees(np.linalg.norm(cv2.Rodrigues(R1)[0])):.2f} deg")

    with Pool(a.workers) as pool:
        for r in a.rooms.split(","):
            seq = os.path.join(a.src, f"dataset-room{r}_512_16", "mav0")
            out = os.path.join(a.dst, f"room{r}")
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "config.yaml"), "w").write(cfg)
            jobs = []
            for c in ("cam0", "cam1"):
                d = os.path.join(out, "images", c); os.makedirs(d, exist_ok=True)
                jobs += [(f, os.path.join(d, os.path.basename(f)), maps[c]) for f in sorted(glob.glob(os.path.join(seq, c, "data", "*.png")))
                         if not os.path.exists(os.path.join(d, os.path.basename(f)))]
            pool.map(work, jobs, chunksize=32)
            for v in ("cal", "nos", "raw"):
                m = os.path.join(out, v, "mav0")
                for c in ("cam0", "cam1"):
                    os.makedirs(os.path.join(m, c), exist_ok=True)
                    link = os.path.join(m, c, "data")
                    if not os.path.islink(link):
                        os.symlink(os.path.join(out, "images", c), link)
                os.makedirs(os.path.join(m, "imu0"), exist_ok=True)
                open(os.path.join(m, "imu0", "data.csv"), "w").write(open(os.path.join(a.variants, f"room{r}", f"imu_{v}.csv")).read())
            print(f"room{r}: {len(jobs)} images written -> {out}", flush=True)


if __name__ == "__main__":
    main()
