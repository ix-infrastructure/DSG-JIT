#!/usr/bin/env python3
"""Inject one calibration / IMU-model error into a EuRoC sequence for a SLAM run.

Unlike image corruptions, these need no invented noise model: a wrong
calibration value or a mis-scaled IMU axis is exactly what a drifted or poorly
calibrated rig delivers, so the injection itself is not a modelling
assumption -- only the MAGNITUDES are, and those come from the tiers in
reports/Sensor noise models for SLAM.md (realistic / mishandled / stress).

Two kinds of error:
  * BELIEF errors -- the rig is fine, ORB-SLAM3's YAML is wrong:
      rbc  camera-IMU rotation        (IMU.T_b_c1, rotation about a random axis)
      tbc  camera-IMU translation     (IMU.T_b_c1, offset in a random direction)
      rst  stereo extrinsic rotation  (Stereo.T_c1_c2; changes the startup rectification)
  * SENSOR errors -- the YAML is right, the IMU data violates it:
      toff camera-IMU time offset     (IMU timestamps shifted by +/-dt)
      isf  IMU scale-factor error     (each gyro and accel axis scaled by 1 +/- s)
      imis IMU axis misalignment      (small skew: off-diagonal terms of size m rad)

Each seed draws ONE error direction (axis / sign pattern) that is reused at
every level, so levels are paired on direction exactly as Phase 2 paired noise
draws. Images, ground truth and other files are symlinked, not copied.

Interface matches lowlight_sequence.py so phase2_suite.py can drive it:
--arm is the error type, --level 1..3 the tier.
"""
import argparse, json, os, re, shutil, sys
import numpy as np

# Tier magnitudes. rbc/tbc/rst/toff: report's realistic / mishandled / stress
# tiers. isf/imis: the reports give no tiers (datasheet alignment 0.05 deg,
# scale factor not given); these are marked PROVISIONAL in the manifest.
TIERS = {
    "toff": ("ms", [5.0, 15.0, 30.0]),
    "rbc":  ("deg", [0.25, 0.8, 2.0]),
    "tbc":  ("cm", [1.0, 3.0, 6.0]),
    "rst":  ("deg", [0.2, 1.0, 2.5]),
    "isf":  ("pct", [0.5, 1.0, 2.0]),
    "imis": ("deg", [0.05, 0.5, 2.0]),
}
PROVISIONAL = {"isf", "imis"}
ARM_ID = {k: i for i, k in enumerate(TIERS)}
LINK = ["cam0", "cam1", "state_groundtruth_estimate0", "leica0", "body.yaml"]


def rotvec(axis, ang):
    """Rodrigues: rotation matrix for angle ang (rad) about unit axis."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k


def unit(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def read_mat(text, key):
    m = re.search(rf"{re.escape(key)}:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", text, re.S)
    return np.array([float(x) for x in m.group(1).replace("\n", " ").split(",")]).reshape(4, 4), m


def write_mat(text, key, T):
    _, m = read_mat(text, key)
    rows = [",".join(f"{v:.15f}" for v in T[r]) for r in range(4)]
    body = (",\n         ").join(rows)
    return text[:m.start(1)] + body + text[m.end(1):]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--yaml", required=True, help="base ORB-SLAM3 YAML")
    p.add_argument("--arm", required=True, choices=list(TIERS))
    p.add_argument("--level", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=1, help="unused; interface compatibility")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"{a.dst} exists (use --force)")
        shutil.rmtree(a.dst)
    sm, dm = os.path.join(a.src, "mav0"), os.path.join(a.dst, "mav0")
    os.makedirs(os.path.join(dm, "imu0"))
    for e in LINK:
        if os.path.exists(os.path.join(sm, e)):
            os.symlink(os.path.realpath(os.path.join(sm, e)), os.path.join(dm, e))

    unit_name, mags = TIERS[a.arm]
    mag = mags[a.level - 1]
    rng = np.random.default_rng([a.seed, ARM_ID[a.arm]])      # direction depends on seed only, not level
    cfg = open(a.yaml).read()
    imu = np.loadtxt(os.path.join(sm, "imu0", "data.csv"), delimiter=",", comments="#", dtype=np.float64)
    ts = np.loadtxt(os.path.join(sm, "imu0", "data.csv"), delimiter=",", comments="#", usecols=0, dtype=np.int64)
    detail = {}

    if a.arm in ("rbc", "tbc", "rst"):
        key = "Stereo.T_c1_c2" if a.arm == "rst" else "IMU.T_b_c1"
        T, _ = read_mat(cfg, key)
        if a.arm == "tbc":
            d = unit(rng); T[:3, 3] += d * mag / 100.0; detail["direction"] = d.tolist()
        else:
            ax = unit(rng); T[:3, :3] = T[:3, :3] @ rotvec(ax, np.radians(mag)); detail["axis"] = ax.tolist()
        cfg = write_mat(cfg, key, T)
    elif a.arm == "toff":
        sign = 1 if a.seed % 2 == 0 else -1                  # IMU late for even seeds, early for odd
        ts = ts + sign * int(round(mag * 1e6)); detail["sign"] = sign
    elif a.arm == "isf":
        s = rng.choice([-1.0, 1.0], size=6) * mag / 100.0
        imu[:, 1:7] *= 1.0 + s; detail["scale_errors"] = s.tolist()
    elif a.arm == "imis":
        m = np.radians(mag)
        for c0 in (1, 4):                                      # gyro block, accel block
            S = np.eye(3)
            off = rng.choice([-1.0, 1.0], size=6) * m
            S[0, 1], S[0, 2], S[1, 0], S[1, 2], S[2, 0], S[2, 1] = off
            imu[:, c0:c0 + 3] = imu[:, c0:c0 + 3] @ S.T
            detail["gyro_skew" if c0 == 1 else "accel_skew"] = off.tolist()

    with open(os.path.join(dm, "imu0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for t, row in zip(ts, imu[:, 1:7]):
            f.write(f"{t}," + ",".join(f"{v:.17g}" for v in row) + "\n")
    if os.path.exists(os.path.join(sm, "imu0", "sensor.yaml")):
        shutil.copy2(os.path.join(sm, "imu0", "sensor.yaml"), os.path.join(dm, "imu0", "sensor.yaml"))
    open(os.path.join(a.dst, "config.yaml"), "w").write(cfg)

    man = dict(source=os.path.abspath(a.src), arm=a.arm, level=a.level, seed=a.seed,
               magnitude=mag, unit=unit_name, provisional_tier=a.arm in PROVISIONAL, **detail)
    json.dump(man, open(os.path.join(a.dst, "LOWLIGHT_MANIFEST.json"), "w"), indent=1)
    print(f"{a.dst}: {a.arm} L{a.level} = {mag} {unit_name}  seed {a.seed}")


if __name__ == "__main__":
    main()
