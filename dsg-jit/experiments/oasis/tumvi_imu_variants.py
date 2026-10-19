#!/usr/bin/env python3
"""Real (not injected) IMU miscalibration from TUM-VI.

The exported TUM-VI sequences ship IMU data already corrected with the
published IMU calibration (Schubert et al. 2018, eqs. 1-2):

    x_calibrated = M x_raw - b      (M = accel_scale / gyro_scale, b = bias)

so the uncorrected data are recoverable exactly. Per room this writes three
IMU files in the dataset's EuRoC CSV format:

  cal   as shipped (calibrated)
  nos   M^-1 x          scale and misalignment uncorrected, no added bias
  raw   M^-1 (x + b)    fully raw: scale, misalignment and the large
                        reproducible turn-on bias (accel up to 1.3 m/s^2)

plus times.txt (camera timestamps, ns) for stereo_inertial_tum_vi.

Values: https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
("Time Alignment, Hand-Eye Calibration, IMU Axis Scaling and Misalignment",
estimated on calib-imu1). Checked on lambda: shipped data give |g| 9.77-9.83
m/s^2 and near-zero gyro in the quietest 2 s of each room; raw gives 10.5.

    tumvi_imu_variants.py --src /data/tkocher/datasets/tumvi --dst /data/tkocher/datasets/tumvi_variants
"""
import argparse, os
import numpy as np

MA = np.array([[1.00422, 0, 0], [-7.82123e-05, 1.00136, 0], [-0.0097745, -0.000976476, 0.970467]])
MG = np.array([[0.943611, 0.00148681, 0.000824366], [0.000369694, 1.09413, -0.00273521],
               [-0.00175252, 0.00834754, 1.01588]])
BA = np.array([-1.30318, -0.391441, 0.380509])
BG = np.array([0.0283122, 0.00723077, 0.0165292])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--rooms", default="1,2,3,4,5,6")
    a = ap.parse_args()
    iMa, iMg = np.linalg.inv(MA), np.linalg.inv(MG)
    for r in a.rooms.split(","):
        seq = os.path.join(a.src, f"dataset-room{r}_512_16", "mav0")
        out = os.path.join(a.dst, f"room{r}"); os.makedirs(out, exist_ok=True)
        hdr = open(os.path.join(seq, "imu0", "data.csv")).readline().strip()
        d = np.loadtxt(os.path.join(seq, "imu0", "data.csv"), delimiter=",", comments="#")
        ts, w, acc = d[:, 0].astype(np.int64), d[:, 1:4], d[:, 4:7]
        variants = {"cal": (w, acc),
                    "nos": ((iMg @ w.T).T, (iMa @ acc.T).T),
                    "raw": ((iMg @ (w + BG).T).T, (iMa @ (acc + BA).T).T)}
        for name, (ww, aa) in variants.items():
            with open(os.path.join(out, f"imu_{name}.csv"), "w") as f:
                f.write(hdr + "\n")
                for t, g, x in zip(ts, ww, aa):
                    f.write(f"{t},{g[0]:.10f},{g[1]:.10f},{g[2]:.10f},{x[0]:.10f},{x[1]:.10f},{x[2]:.10f}\n")
        cam = np.loadtxt(os.path.join(seq, "cam0", "data.csv"), delimiter=",", comments="#", dtype=str)
        open(os.path.join(out, "times.txt"), "w").write("\n".join(c[0] for c in cam) + "\n")
        print(f"room{r}: {len(ts)} IMU samples, {len(cam)} frames -> {out}")


if __name__ == "__main__":
    main()
