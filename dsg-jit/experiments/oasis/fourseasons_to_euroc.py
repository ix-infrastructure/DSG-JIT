#!/usr/bin/env python3
"""Present a 4Seasons recording as a EuRoC ASL sequence for stereo_inertial_euroc.

4Seasons (Wenzel et al. 2020) ships stereo images that are already undistorted
AND stereo-rectified (pinhole, identical K, pure-x baseline), a 2000 Hz
ADIS16465 IMU and RTK-GNSS/VIO reference poses of cam0. Conversion:

  cam0, cam1   symlinked image directories (frames named <ts_ns>.png already)
  imu0         IMU averaged over blocks of --decimate samples (2000 -> 200 Hz,
               the rate ORB-SLAM3's EuRoC settings and noise values assume)
  GT           result.txt (TUM, cam0 poses) moved to the IMU origin so it
               matches ORB-SLAM3's body-frame trajectory output
  stamps       frame timestamp list for the binary
  config.yaml  ORB-SLAM3 settings from the calibration

The rectified cam0 is the raw cam0 rotated by R1 (here about 1.5 deg). R1 is
not shipped; it is recovered with cv2.fisheye.stereoRectify(balance=0) on the
camchain, which reproduces the shipped rectified intrinsics to < 0.01 px. It
must go into IMU.T_b_c1: leaving it out is a 1.5 deg camera-IMU rotation
error, which the calibration sweep (4.4) shows costs tens of mm.
"""
import argparse, os, re
import numpy as np
import cv2
import yaml


def quat_to_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fmt(T):
    return ",\n         ".join(",".join(f"{v:.12f}" for v in row) for row in T)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="recording_* directory")
    ap.add_argument("--calib", required=True, help="4Seasons calibration directory")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--base-yaml", required=True, help="EuRoC ORB-SLAM3 yaml used as template")
    ap.add_argument("--decimate", type=int, default=10)
    a = ap.parse_args()

    cc = yaml.safe_load(open(os.path.join(a.calib, "camchain.yaml")))
    K = lambda c: np.array([[c["intrinsics"][0], 0, c["intrinsics"][2]], [0, c["intrinsics"][1], c["intrinsics"][3]], [0, 0, 1]])
    T10 = np.array(cc["T_cam1_cam0"])
    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
        K(cc["cam0"]), np.array(cc["cam0"]["distortion_coeffs"]), K(cc["cam1"]), np.array(cc["cam1"]["distortion_coeffs"]),
        tuple(cc["cam0"]["resolution"]), T10[:3, :3], T10[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0)
    shipped = [float(v) for v in open(os.path.join(a.calib, "undistorted_calib_0.txt")).readline().split()[1:5]]
    dev = np.abs(np.array([P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]]) - shipped).max()
    assert dev < 0.05, f"rectification not reproduced (max deviation {dev:.3f} px)"
    fx, fy, cx, cy = shipped
    baseline = -P2[0, 3] / P2[0, 0]

    T_c0_i = np.array(cc["cam0"]["T_cam_imu"])
    T_r_i = np.eye(4); T_r_i[:3, :3] = R1 @ T_c0_i[:3, :3]; T_r_i[:3, 3] = R1 @ T_c0_i[:3, 3]
    T_b_c1 = np.linalg.inv(T_r_i)
    T_c1_c2 = np.eye(4); T_c1_c2[0, 3] = baseline

    m = os.path.join(a.dst, "mav0")
    for c in ("cam0", "cam1"):
        os.makedirs(os.path.join(m, c), exist_ok=True)
        link = os.path.join(m, c, "data")
        if not os.path.islink(link):
            os.symlink(os.path.realpath(os.path.join(a.src, "undistorted_images", c)), link)

    imu = np.loadtxt(os.path.join(a.src, "imu.txt"))
    n = len(imu) // a.decimate * a.decimate
    blk = imu[:n].reshape(-1, a.decimate, 7)
    ts = blk[:, :, 0].astype(np.int64).mean(1).round().astype(np.int64)
    os.makedirs(os.path.join(m, "imu0"), exist_ok=True)
    with open(os.path.join(m, "imu0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for t, row in zip(ts, blk[:, :, 1:].mean(1)):
            f.write(f"{t}," + ",".join(f"{v:.9g}" for v in row) + "\n")
    rate = 1e9 / np.median(np.diff(ts))

    # GT: result.txt is T_w_cam0 (raw cam0). IMU origin in cam0 = t of T_cam_imu.
    fr = np.array([int(l.split()[0]) for l in open(os.path.join(a.src, "times.txt"))])
    gt = np.loadtxt(os.path.join(a.src, "result.txt"))
    gts = fr[np.searchsorted(fr, np.round(gt[:, 0] * 1e9).astype(np.int64) - 1_000_000)]   # snap to frame ns
    os.makedirs(os.path.join(m, "state_groundtruth_estimate0"), exist_ok=True)
    with open(os.path.join(m, "state_groundtruth_estimate0", "data.csv"), "w") as f:
        f.write("#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n")
        for t, r in zip(gts, gt):
            R = quat_to_R(*r[4:8]); p = r[1:4] + R @ T_c0_i[:3, 3]
            f.write(f"{t},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{r[7]:.9f},{r[4]:.9f},{r[5]:.9f},{r[6]:.9f}\n")
    open(os.path.join(a.dst, "stamps.txt"), "w").write("\n".join(str(t) for t in fr) + "\n")

    cfg = open(a.base_yaml).read()
    sub = {"Camera1.fx": fx, "Camera1.fy": fy, "Camera1.cx": cx, "Camera1.cy": cy,
           "Camera2.fx": fx, "Camera2.fy": fy, "Camera2.cx": cx, "Camera2.cy": cy,
           "Camera1.k1": 0.0, "Camera1.k2": 0.0, "Camera1.p1": 0.0, "Camera1.p2": 0.0,
           "Camera2.k1": 0.0, "Camera2.k2": 0.0, "Camera2.p1": 0.0, "Camera2.p2": 0.0,
           "Camera.width": 800, "Camera.height": 400, "Camera.fps": 30, "IMU.Frequency": round(rate, 1)}
    for k, v in sub.items():
        cfg, nsub = re.subn(rf"^{re.escape(k)}:.*$", f"{k}: {v}", cfg, flags=re.M)
        assert nsub == 1, k
    for key, T in (("Stereo.T_c1_c2", T_c1_c2), ("IMU.T_b_c1", T_b_c1)):
        cfg, nsub = re.subn(rf"({re.escape(key)}:\s*!!opencv-matrix.*?data:\s*\[).*?(\])", lambda mm: mm.group(1) + fmt(T) + mm.group(2), cfg, flags=re.S)
        assert nsub == 1, key
    open(os.path.join(a.dst, "config.yaml"), "w").write(cfg)
    print(f"{a.dst}: {len(fr)} frames, IMU {rate:.1f} Hz ({len(ts)} samples), GT {len(gt)} poses, "
          f"baseline {baseline:.4f} m, rectifying rotation {np.degrees(np.linalg.norm(cv2.Rodrigues(R1)[0])):.2f} deg")


if __name__ == "__main__":
    main()
