#!/usr/bin/env python3
"""Generate one darkened + corrected EuRoC sequence for a Phase 2 SLAM run.

Imports the darkening model and correction arms from phase1_detector_study so
the SLAM phase and the detector phase are guaranteed to use identical image
processing. (The denoising-experiment audit found two serious errors that both
came from copies of shared code drifting apart; this is deliberately not a copy.)

Darkening is seeded per (seed, frame, level, camera), exactly as in Phase 1, so
every arm of a given (seed, level) sees the SAME darkened realization -- arm
comparisons are paired on the noise draw -- and any sequence can be regenerated
bit-identically instead of stored.

Arm C1 changes no pixels: it writes the plain darkened frames plus a copy of the
ORB-SLAM3 YAML with FAST thresholds scaled by the attenuation a. Every other arm
writes corrected frames and the unmodified YAML.

Output layout (consumed by run_oasis.sh):
    <dst>/mav0/cam0|cam1/data/*.png      corrected frames (raw, unrectified)
    <dst>/mav0/{imu0,state_groundtruth_estimate0,...} -> symlinks to source
    <dst>/config.yaml                    YAML to run with
    <dst>/LOWLIGHT_MANIFEST.json
"""
import argparse, glob, json, os, re, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_detector_study import (darken, quant, restored_sigma, to_u8, nlm,  # noqa: E402
                                   load_cfg)

LINK = ["imu0", "state_groundtruth_estimate0", "leica0", "body.yaml"]
_W = {}


def _init(a, cfg, gamma):
    _W.update(a=a, cfg=cfg, gamma=gamma)
    _W["clahe"] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def correct_one(job):
    """Darken + correct a single frame of one camera for the requested arm."""
    fi, cam_idx, src, dst = job
    a = _W["a"]
    raw = cv2.imread(src, 0)
    rng = np.random.default_rng([a.seed, fi, a.level, cam_idx])          # == Phase 1 seeding
    y, att = darken(raw, a.level, rng, a.K, a.read_e, a.gain_max)
    sig = restored_sigma(a.level, a.K, a.read_e, a.gain_max)
    y8 = quant(y, 1.0)
    arm = a.arm
    if arm in ("C0", "C1"):
        out = to_u8(y8)
    elif arm == "C2":
        out = to_u8(y8 / att)
    elif arm == "C3":
        out = to_u8(255 * (y8 / 255) ** _W["gamma"])
    elif arm == "C4":
        out = nlm(to_u8(y8 / att), sig, 21)
    elif arm == "C5":
        out = nlm(to_u8(y8 / att), 1.3 * sig, 35)
    elif arm == "C6":
        out = _W["clahe"].apply(to_u8(y8))
    elif arm == "C8":
        out = to_u8(quant(y, 0.25) / att)
    else:
        raise ValueError(arm)
    cv2.imwrite(dst, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return att


def shared_gamma(a, files):
    """C3's gamma is shared across the stereo pair and held per sequence.

    Phase 1 chose gamma per frame from the joint median; for a sequence-level
    run that would flicker frame to frame (a documented failure mode of
    single-image enhancers), so here one gamma is fixed from the darkened
    sequence's overall median, sampled on every 20th frame."""
    meds = []
    for fi, (fl, fr) in list(enumerate(files))[::20]:
        for ci, f in ((0, fl), (1, fr)):
            rng = np.random.default_rng([a.seed, fi, a.level, ci])
            y, _ = darken(cv2.imread(f, 0), a.level, rng, a.K, a.read_e, a.gain_max)
            meds.append(np.median(quant(y, 1.0)))
    m = float(np.median(meds))
    return float(np.log(75 / 255) / np.log(max(m, 1.0) / 255)) if m < 75 else 1.0


def write_yaml(src_yaml, dst_yaml, ini, mn):
    t = open(src_yaml).read()
    t = re.sub(r"(ORBextractor\.iniThFAST:\s*)\d+", rf"\g<1>{ini}", t)
    t = re.sub(r"(ORBextractor\.minThFAST:\s*)\d+", rf"\g<1>{mn}", t)
    open(dst_yaml, "w").write(t)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--yaml", required=True, help="base ORB-SLAM3 YAML")
    p.add_argument("--level", type=int, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arm", required=True, choices=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C8"])
    p.add_argument("--K", type=float, default=0.0186); p.add_argument("--read-e", type=float, default=30.0)
    p.add_argument("--gain-max", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"{a.dst} exists (use --force)")
        shutil.rmtree(a.dst)
    sm = os.path.join(a.src, "mav0"); dm = os.path.join(a.dst, "mav0")
    for cam in ("cam0", "cam1"):
        os.makedirs(os.path.join(dm, cam, "data"))
        for f in ("data.csv", "sensor.yaml"):
            if os.path.exists(os.path.join(sm, cam, f)):
                shutil.copy2(os.path.join(sm, cam, f), os.path.join(dm, cam, f))
    for e in LINK:
        if os.path.exists(os.path.join(sm, e)):
            os.symlink(os.path.abspath(os.path.join(sm, e)), os.path.join(dm, e))

    fl = sorted(glob.glob(os.path.join(sm, "cam0", "data", "*.png")))
    names = {os.path.basename(f) for f in glob.glob(os.path.join(sm, "cam1", "data", "*.png"))}
    fl = [f for f in fl if os.path.basename(f) in names]
    pairs = [(f, os.path.join(sm, "cam1", "data", os.path.basename(f))) for f in fl]
    gamma = shared_gamma(a, pairs) if a.arm == "C3" else 1.0

    jobs = []
    for fi, (f0, f1) in enumerate(pairs):
        for ci, (cam, f) in enumerate((("cam0", f0), ("cam1", f1))):
            jobs.append((fi, ci, f, os.path.join(dm, cam, "data", os.path.basename(f))))
    cfg = load_cfg(a.yaml)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a, cfg, gamma)) as ex:
        atts = list(ex.map(correct_one, jobs, chunksize=64))
    att = float(atts[0])

    ini, mn = cfg["ini"], cfg["mn"]
    if a.arm == "C1":
        ini, mn = max(1, round(cfg["ini"] * att)), max(1, round(cfg["mn"] * att))
    write_yaml(a.yaml, os.path.join(a.dst, "config.yaml"), ini, mn)

    man = dict(source=os.path.abspath(a.src), level=a.level, seed=a.seed, arm=a.arm, attenuation=att,
               gain=min(2.0 ** a.level, a.gain_max), K=a.K, read_e=a.read_e,
               restored_sigma_dn=restored_sigma(a.level, a.K, a.read_e, a.gain_max),
               gamma=gamma, iniThFAST=ini, minThFAST=mn, frames=len(pairs))
    json.dump(man, open(os.path.join(a.dst, "LOWLIGHT_MANIFEST.json"), "w"), indent=1)
    print(f"{a.dst}: L{a.level} {a.arm} seed {a.seed}  a={att:.4f}  thresholds {ini}/{mn}  "
          f"gamma={gamma:.3f}  frames={len(pairs)}")


if __name__ == "__main__":
    main()
