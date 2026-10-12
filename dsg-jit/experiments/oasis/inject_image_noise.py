#!/usr/bin/env python3
"""Inject sensor noise into an EuRoC sequence's stereo imagery.

Two classes of noise, which must be treated differently:

  FROZEN (fixed-pattern) -- a property of the physical sensor, drawn ONCE and
  applied identically to every frame of the sequence:
      --dsnu   additive offset field    O[y,x] ~ N(0, dsnu)     [gray levels]
      --prnu   multiplicative gain field G[y,x] ~ N(1, prnu)     [fraction]
  These are the two terms of the EMVA 1288 affine model (PRNU scales with
  signal, DSNU does not). Redrawing them per frame would simulate a sensor
  whose silicon changes every 50 ms, and would let multi-view averaging erase
  them -- understating the damage. A frozen pattern instead biases every
  observation of every landmark that lands on the affected pixels.

  TEMPORAL -- redrawn every frame:
      --temporal  additive N(0, temporal) per pixel per frame  [gray levels]

Image formation applied to each 8-bit frame x:
      y = clip( round( x * G + O + n_t ), 0, 255 )

cam0 and cam1 receive INDEPENDENT patterns by default, because they are two
physically separate sensors. --correlated gives both cameras the same fields.

The frozen fields are saved alongside the data (patterns/*.npy) so analysis
can use them as ground truth. A denoiser under test must not read them.

Imagery cannot be symlinked once perturbed, so a variant costs ~2.6 GB.
Everything is seeded, so variants are regenerated bit-identically on demand
rather than stored (see run_image_condition.sh).
"""
import argparse, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

try:
    import cv2
    def _read(p):  return cv2.imread(p, cv2.IMREAD_UNCHANGED)
    def _write(p, a): cv2.imwrite(p, a, [cv2.IMWRITE_PNG_COMPRESSION, 1])
except ImportError:                                   # pragma: no cover
    from PIL import Image
    def _read(p):  return np.asarray(Image.open(p))
    def _write(p, a): Image.fromarray(a).save(p, compress_level=1)

LINK_ENTRIES = ["state_groundtruth_estimate0", "leica0", "body.yaml"]
KIND_DSNU, KIND_PRNU, KIND_TEMPORAL = 0, 1, 2

# Worker state, populated once per process by _init (avoids pickling the
# full-resolution fields for every frame).
_W = {}


def _init(gain, offset, temporal, seed, cam_key, src_dir, dst_dir):
    _W.update(gain=gain, offset=offset, temporal=temporal, seed=seed,
              cam_key=cam_key, src=src_dir, dst=dst_dir)


def _process(job):
    idx, fn = job
    x = _read(os.path.join(_W["src"], fn))
    if x is None:
        return 0
    y = x.astype(np.float32)
    if _W["gain"] is not None:
        y *= _W["gain"]
    if _W["offset"] is not None:
        y += _W["offset"]
    if _W["temporal"] > 0:
        rng = np.random.default_rng(
            np.random.SeedSequence([_W["seed"], _W["cam_key"], KIND_TEMPORAL, idx]))
        y += rng.normal(0.0, _W["temporal"], y.shape).astype(np.float32)
    out = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    _write(os.path.join(_W["dst"], fn), out)
    return 1


def frozen_fields(shape, dsnu, prnu, seed, cam_key):
    """Draw the per-sensor fixed patterns exactly once."""
    gain = offset = None
    if prnu > 0:
        rng = np.random.default_rng(np.random.SeedSequence([seed, cam_key, KIND_PRNU]))
        gain = (1.0 + rng.normal(0.0, prnu, shape)).astype(np.float32)
    if dsnu > 0:
        rng = np.random.default_rng(np.random.SeedSequence([seed, cam_key, KIND_DSNU]))
        offset = rng.normal(0.0, dsnu, shape).astype(np.float32)
    return gain, offset


def perturb_camera(src_cam, dst_cam, args, cam_idx, pattern_dir):
    os.makedirs(os.path.join(dst_cam, "data"), exist_ok=True)
    for f in ("data.csv", "sensor.yaml"):
        s = os.path.join(src_cam, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_cam, f))

    files = sorted(f for f in os.listdir(os.path.join(src_cam, "data"))
                   if f.endswith(".png"))
    shape = _read(os.path.join(src_cam, "data", files[0])).shape
    cam_key = 0 if args.correlated else cam_idx
    gain, offset = frozen_fields(shape, args.dsnu, args.prnu, args.seed, cam_key)

    stats = {"images": len(files), "shape": list(shape)}
    if gain is not None:
        np.save(os.path.join(pattern_dir, f"cam{cam_idx}_prnu_gain.npy"), gain)
        stats["prnu_realized_std"] = float(gain.std())
    if offset is not None:
        np.save(os.path.join(pattern_dir, f"cam{cam_idx}_dsnu_offset.npy"), offset)
        stats["dsnu_realized_std"] = float(offset.std())

    with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init,
            initargs=(gain, offset, args.temporal, args.seed, cam_key,
                      os.path.join(src_cam, "data"),
                      os.path.join(dst_cam, "data"))) as ex:
        written = sum(ex.map(_process, enumerate(files), chunksize=64))
    stats["written"] = written
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="pristine sequence dir (contains mav0/)")
    p.add_argument("--dst", required=True, help="variant dir to create")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dsnu", type=float, default=0.0, help="frozen offset sigma, gray levels")
    p.add_argument("--prnu", type=float, default=0.0, help="frozen gain sigma, fraction (0.01 = 1%%)")
    p.add_argument("--temporal", type=float, default=0.0, help="per-frame sigma, gray levels")
    p.add_argument("--correlated", action="store_true", help="same fields on both cameras")
    p.add_argument("--link-imu", action="store_true", help="symlink imu0 instead of copying")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 4))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if a.dsnu <= 0 and a.prnu <= 0 and a.temporal <= 0:
        sys.exit("ERROR: no noise requested (set --dsnu, --prnu and/or --temporal)")
    src_mav0 = os.path.join(a.src, "mav0")
    if not os.path.isdir(src_mav0):
        sys.exit(f"ERROR: no mav0/ under {a.src}")
    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"ERROR: {a.dst} exists (use --force)")
        shutil.rmtree(a.dst)

    dst_mav0 = os.path.join(a.dst, "mav0")
    pattern_dir = os.path.join(a.dst, "patterns")
    os.makedirs(dst_mav0)
    os.makedirs(pattern_dir)

    for e in LINK_ENTRIES:
        s = os.path.join(src_mav0, e)
        if os.path.exists(s):
            os.symlink(os.path.abspath(s), os.path.join(dst_mav0, e))
    imu = os.path.join(src_mav0, "imu0")
    if a.link_imu:
        os.symlink(os.path.abspath(imu), os.path.join(dst_mav0, "imu0"))
    else:
        shutil.copytree(imu, os.path.join(dst_mav0, "imu0"))

    cams = {}
    for idx, cam in enumerate(["cam0", "cam1"]):
        s = os.path.join(src_mav0, cam)
        if os.path.isdir(s):
            cams[cam] = perturb_camera(s, os.path.join(dst_mav0, cam), a, idx, pattern_dir)

    manifest = {
        "source": os.path.abspath(a.src), "seed": a.seed,
        "dsnu_gray": a.dsnu, "prnu_frac": a.prnu, "temporal_gray": a.temporal,
        "correlated": a.correlated, "formation": "y = clip(round(x*G + O + n_t), 0, 255)",
        "cameras": cams,
    }
    with open(os.path.join(a.dst, "NOISE_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(a.dst)
    print(f"  dsnu={a.dsnu} gray  prnu={a.prnu*100:.2f}%  temporal={a.temporal} gray  "
          f"seed={a.seed}  correlated={a.correlated}")
    for cam, st in cams.items():
        print(f"  {cam}: {st['written']}/{st['images']} images"
              + (f"  dsnu_std={st['dsnu_realized_std']:.4f}" if "dsnu_realized_std" in st else "")
              + (f"  prnu_std={st['prnu_realized_std']:.5f}" if "prnu_realized_std" in st else ""))


if __name__ == "__main__":
    main()
