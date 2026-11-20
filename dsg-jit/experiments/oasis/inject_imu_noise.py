#!/usr/bin/env python3
"""Inject noise into an EuRoC sequence's IMU stream.

Creates a variant dataset directory that SYMLINKS the imagery (2.6 GB) and
rewrites only imu0/data.csv (5 MB). A full noise sweep therefore costs
megabytes rather than hundreds of gigabytes.

Noise is parameterised as a MULTIPLE of the sensor's own documented noise
density (imu0/sensor.yaml), not as a bare sigma. EuRoC's ADIS16448 has
        gyro  1.6968e-04 rad/s/sqrt(Hz)  ->  2.400e-03 rad/s at 200 Hz
        accel 2.0000e-03 m/s^2/sqrt(Hz)  ->  2.828e-02 m/s^2 at 200 Hz
against signal std of ~0.15 rad/s and ~1.0 m/s^2, i.e. the real sensor noise
is only 1-4% of the signal. "--scale 1" is therefore a no-op in practice;
degradation needs multiples. Absolute sigmas can still be forced if wanted.

Modes:
  gaussian  i.i.d. white Gaussian on gyro and accel (default; matches the
            noise model used in the SE(3) pose-denoising experiments)
  euroc     white noise PLUS bias random walk, i.e. the full model EuRoC
            documents. Kept available because reviewers will ask whether
            results survive temporally-correlated noise.
"""
import argparse, os, re, shutil, sys
import numpy as np

GYRO_COLS, ACCEL_COLS = slice(1, 4), slice(4, 7)
LINK_ENTRIES = ["cam0", "cam1", "state_groundtruth_estimate0", "leica0", "body.yaml"]


def read_sensor_yaml(path):
    """Minimal scalar scrape -- avoids a pyyaml dependency."""
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^\s*(\w+)\s*:\s*([-+0-9.eE]+)\s*(?:#.*)?$", line)
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="pristine sequence dir (contains mav0/)")
    p.add_argument("--dst", required=True, help="variant dir to create")
    p.add_argument("--mode", choices=["gaussian", "euroc"], default="gaussian")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiplier on the sensor's documented noise density")
    p.add_argument("--sigma-gyro", type=float, default=None, help="absolute override, rad/s")
    p.add_argument("--sigma-accel", type=float, default=None, help="absolute override, m/s^2")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    src_mav0 = os.path.join(a.src, "mav0")
    if not os.path.isdir(src_mav0):
        sys.exit(f"ERROR: no mav0/ under {a.src}")
    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"ERROR: {a.dst} exists (use --force)")
        shutil.rmtree(a.dst)

    dst_mav0 = os.path.join(a.dst, "mav0")
    os.makedirs(os.path.join(dst_mav0, "imu0"))

    # Imagery and ground truth are unmodified -> symlink, never copy.
    for e in LINK_ENTRIES:
        s = os.path.join(src_mav0, e)
        if os.path.exists(s):
            os.symlink(os.path.abspath(s), os.path.join(dst_mav0, e))

    sensor_yaml = os.path.join(src_mav0, "imu0", "sensor.yaml")
    shutil.copy2(sensor_yaml, os.path.join(dst_mav0, "imu0", "sensor.yaml"))
    spec = read_sensor_yaml(sensor_yaml)
    rate = spec.get("rate_hz", 200.0)

    # Continuous noise density -> discrete per-sample sigma.
    sg = a.sigma_gyro  if a.sigma_gyro  is not None else \
        spec["gyroscope_noise_density"] * np.sqrt(rate) * a.scale
    sa = a.sigma_accel if a.sigma_accel is not None else \
        spec["accelerometer_noise_density"] * np.sqrt(rate) * a.scale

    src_csv = os.path.join(src_mav0, "imu0", "data.csv")
    with open(src_csv) as f:
        header = f.readline().rstrip("\n")
    data = np.loadtxt(src_csv, delimiter=",", skiprows=1)
    n = len(data)

    rng = np.random.default_rng(a.seed)
    out = data.copy()
    out[:, GYRO_COLS]  += rng.normal(0.0, sg, (n, 3))
    out[:, ACCEL_COLS] += rng.normal(0.0, sa, (n, 3))

    if a.mode == "euroc":
        # Bias random walk: sigma_rw * sqrt(dt) per step, cumulatively summed.
        dt = 1.0 / rate
        bg = spec["gyroscope_random_walk"] * np.sqrt(dt) * a.scale
        ba = spec["accelerometer_random_walk"] * np.sqrt(dt) * a.scale
        out[:, GYRO_COLS]  += np.cumsum(rng.normal(0.0, bg, (n, 3)), axis=0)
        out[:, ACCEL_COLS] += np.cumsum(rng.normal(0.0, ba, (n, 3)), axis=0)

    # Timestamps MUST stay integer nanoseconds -- writing them as floats loses
    # precision and breaks the loader's image/IMU association.
    dst_csv = os.path.join(dst_mav0, "imu0", "data.csv")
    with open(dst_csv, "w") as f:
        f.write(header + "\n")
        for row in out:
            f.write("%d," % int(row[0]) + ",".join("%.17g" % v for v in row[1:7]) + "\n")

    with open(os.path.join(a.dst, "NOISE_MANIFEST.txt"), "w") as f:
        f.write(f"source        {os.path.abspath(a.src)}\n"
                f"mode          {a.mode}\nscale         {a.scale}\nseed          {a.seed}\n"
                f"sigma_gyro    {sg:.6e} rad/s\nsigma_accel   {sa:.6e} m/s^2\n"
                f"imu_samples   {n}\nrate_hz       {rate}\n")

    d = out[:, 1:7] - data[:, 1:7]
    print(f"{a.dst}")
    print(f"  mode={a.mode} scale={a.scale}x seed={a.seed}  n={n}")
    print(f"  sigma gyro={sg:.4e} rad/s   accel={sa:.4e} m/s^2")
    print(f"  applied delta std: gyro={d[:, :3].std():.4e}  accel={d[:, 3:].std():.4e}")
    print(f"  size on disk: {sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(a.dst) for x in fs if not os.path.islink(os.path.join(r, x))) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
